from typing import Any, TypedDict
import json
from pathlib import Path
import sys
import subprocess


class RecipeFailure(TypedDict):
    failed_at: str


def load_failed_compatibility(file_path: Path) -> dict[str, RecipeFailure]:
    if file_path.exists():
        with file_path.open("r") as file:
            return dict(json.load(file))
    return {}


def save_failed_compatibility(file_path: Path, data: dict[str, RecipeFailure]) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as file:
        json.dump(data, file, indent=4)


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def commit_push_changes(message: str, branch_name: str) -> None:
    """
    Commit and push changes to the specified branch with a given commit message.
    If there are no changes, do nothing.
    """

    # Switch to branch
    run_command(["git", "switch", branch_name])

    # Check if there are changes to commit
    result = run_command_unchecked(["git", "diff-index", "--quiet", "HEAD"])
    if result.returncode == 0:
        eprint("No changes to commit.")
        return

    # Commit, pull and push the changes
    run_command(["git", "pull", "origin", branch_name])
    run_command(["git", "commit", "--message", message, "--no-verify"])
    run_command(["git", "push", "--set-upstream", "origin", branch_name])


def run_command_unchecked(command: list[str]) -> subprocess.CompletedProcess[Any]:
    eprint(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    return result


def run_command(command: list[str]) -> subprocess.CompletedProcess[Any]:
    result = run_command_unchecked(command)
    if result.returncode != 0:
        eprint("Command failed")
        print(f"stdout: {result.stdout}")
        eprint(f"stderr: {result.stderr}")
        sys.exit(result.returncode)

    return result
