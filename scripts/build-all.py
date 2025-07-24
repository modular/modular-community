from pathlib import Path
from datetime import datetime
import argparse
from scripts.common import (
    load_failed_compatibility,
    run_command_unchecked,
    save_failed_compatibility,
    eprint,
)
import sys
import os

# Channels are in priority order
DEFAULT_CHANNELS = [
    "conda-forge",
    "https://conda.modular.com/max",
    "https://prefix.dev/modular-community",
]


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

    for recipe_dir in base_dir.iterdir():
        recipe_file = recipe_dir / "recipe.yaml"
        if not recipe_file.is_file():
            eprint(f"{recipe_dir} doesn't contain recipe.yaml")
            continue

        command = [
            "rattler-build",
            "build",
        ]

        for channel in DEFAULT_CHANNELS:
            command.extend(["--channel", channel])

        if args.channel is not None:
            for channel in args.channel:
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
        result = run_command_unchecked(command)
        if result.returncode != 0:
            eprint(f"Error building recipe in {recipe_dir}: {result.stderr}")
            exit_code = 1
            if failed_compatibility is not None:
                failed_compatibility[recipe_dir.name] = {
                    "failed_at": datetime.now().isoformat()
                }
        else:
            print(f"Successfully built recipe {recipe_dir.name}")
            if failed_compatibility is not None:
                if recipe_dir.name in failed_compatibility:
                    del failed_compatibility[recipe_dir.name]
                    print(f"Removed {recipe_dir.name} from failed-compatibility.json")

    if failed_compatibility is not None:
        # Save updated failed compatibility data
        save_failed_compatibility(args.data_file, failed_compatibility)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
