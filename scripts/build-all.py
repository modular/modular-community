from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import os
import sys

import yaml

from scripts.common import (
    load_failed_compatibility,
    recipe_name_collisions,
    run_command_unchecked,
    save_failed_compatibility,
    eprint,
)

# Channels are in priority order
MODULAR_COMMUNITY_CHANNEL = "https://prefix.dev/modular-community"
MAX_CHANNEL = "https://conda.modular.com/max"
DEFAULT_CHANNELS = ["conda-forge", MAX_CHANNEL, MODULAR_COMMUNITY_CHANNEL]
LOCAL_OUTPUT = Path("output")


def _spec_name(spec: str) -> str:
    token = spec.strip().split()[0] if spec.strip() else ""
    if not token or token.startswith("${{"):
        return ""
    return token


def _requirement_names(node: Any) -> list[str]:
    names: list[str] = []
    if node is None:
        return names
    if isinstance(node, str):
        name = _spec_name(node)
        if name:
            names.append(name)
        return names
    if isinstance(node, list):
        for item in node:
            names.extend(_requirement_names(item))
        return names
    if isinstance(node, dict):
        for key in ("then", "else"):
            if key in node:
                names.extend(_requirement_names(node[key]))
        return names
    return names


def _package_name(recipe_file: Path) -> str | None:
    with recipe_file.open() as fh:
        recipe_data = yaml.safe_load(fh) or {}
    name = recipe_data.get("package", {}).get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _sibling_dependencies(recipe_file: Path, known_packages: set[str]) -> set[str]:
    with recipe_file.open() as fh:
        recipe_data = yaml.safe_load(fh) or {}
    reqs = recipe_data.get("requirements") or {}
    names: set[str] = set()
    for section in ("host", "run", "build"):
        names.update(_requirement_names(reqs.get(section)))
    self_name = recipe_data.get("package", {}).get("name")
    if isinstance(self_name, str):
        names.discard(self_name.strip())
    return names & known_packages


def order_recipes(recipe_dirs: list[Path]) -> list[Path]:
    """Build sibling packages before the recipes that depend on them."""
    package_to_dir: dict[str, Path] = {}
    for recipe_dir in recipe_dirs:
        name = _package_name(recipe_dir / "recipe.yaml")
        if name is None:
            eprint(f"Invalid package name in {recipe_dir / 'recipe.yaml'}")
            continue
        package_to_dir[name] = recipe_dir

    known = set(package_to_dir)
    dependents: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {name: 0 for name in known}
    for name, recipe_dir in package_to_dir.items():
        deps = _sibling_dependencies(recipe_dir / "recipe.yaml", known)
        indegree[name] = len(deps)
        for dep in deps:
            dependents[dep].add(name)

    ready = deque(sorted(name for name, degree in indegree.items() if degree == 0))
    ordered: list[str] = []
    while ready:
        name = ready.popleft()
        ordered.append(name)
        for dependent in sorted(dependents[name]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(ordered) != len(known):
        leftover = sorted(known - set(ordered))
        eprint(
            "Recipe dependency cycle or unresolved sibling deps; "
            f"appending in name order: {', '.join(leftover)}"
        )
        ordered.extend(leftover)

    return [package_to_dir[name] for name in ordered]


def _build_command(
    recipe_file: Path,
    extra_channels: list[str] | None,
    variant_config: str,
) -> list[str]:
    command = ["rattler-build", "build"]
    if LOCAL_OUTPUT.is_dir():
        command.extend(["--channel", str(LOCAL_OUTPUT.resolve())])
    for channel in DEFAULT_CHANNELS:
        command.extend(["--channel", channel])
    if extra_channels is not None:
        for channel in extra_channels:
            if channel in DEFAULT_CHANNELS:
                continue
            command.extend(["--channel", channel])
    command.extend(
        [
            "--variant-config",
            variant_config,
            "--skip-existing=all",
            "--recipe",
            str(recipe_file),
        ]
    )
    return command


def _record_success(
    recipe_dir: Path, failed_compatibility: dict[str, dict[str, str]] | None
) -> None:
    print(f"Successfully built recipe {recipe_dir.name}")
    if failed_compatibility is not None and recipe_dir.name in failed_compatibility:
        del failed_compatibility[recipe_dir.name]
        print(f"Removed {recipe_dir.name} from failed-compatibility.json")


def _record_failure(
    recipe_dir: Path,
    stderr: str,
    failed_compatibility: dict[str, dict[str, str]] | None,
) -> None:
    eprint(f"Error building recipe in {recipe_dir}: {stderr}")
    if failed_compatibility is not None:
        failed_compatibility[recipe_dir.name] = {
            "failed_at": datetime.now().isoformat()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all recipes.")
    parser.add_argument(
        "--channel",
        action="append",
        help="The channels to use for building.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=os.environ.get("DATA_FILE"),
        help="Path to where the data should be stored. Nothing will be stored if that flag is not provided.",
    )
    args = parser.parse_args()

    base_dir = Path("recipes")
    variant_config = "variants/variants.yaml"

    # Load existing failed compatibility data
    failed_compatibility = (
        None if args.data_file is None else load_failed_compatibility(args.data_file)
    )

    exit_code = 0
    default_channels_without_community = [
        c for c in DEFAULT_CHANNELS if c != MODULAR_COMMUNITY_CHANNEL
    ]

    recipe_dirs: list[Path] = []
    for recipe_dir in sorted(base_dir.iterdir(), key=lambda path: path.name.lower()):
        recipe_file = recipe_dir / "recipe.yaml"
        if not recipe_file.is_file():
            eprint(f"{recipe_dir} doesn't contain recipe.yaml")
            continue
        recipe_dirs.append(recipe_dir)

    pending = order_recipes(recipe_dirs)
    for build_pass in (1, 2):
        still_pending: list[Path] = []
        for recipe_dir in pending:
            recipe_file = recipe_dir / "recipe.yaml"
            if recipe_name_collisions(
                recipe_file, channels=default_channels_without_community
            ):
                eprint(
                    f"SKIPPING: {recipe_file} specifies a recipe whose name collides with another conda package in {default_channels_without_community}."
                )
                continue

            command = _build_command(recipe_file, args.channel, variant_config)
            result = run_command_unchecked(command)
            if result.returncode != 0:
                still_pending.append(recipe_dir)
                if build_pass == 2:
                    _record_failure(
                        recipe_dir, result.stderr, failed_compatibility
                    )
                    exit_code = 1
                else:
                    eprint(
                        f"First pass failed for {recipe_dir.name}; "
                        "retrying after remaining recipes"
                    )
            else:
                _record_success(recipe_dir, failed_compatibility)

        pending = still_pending
        if not pending:
            break

    if failed_compatibility is not None:
        # Save updated failed compatibility data
        save_failed_compatibility(args.data_file, failed_compatibility)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
