import sys
from pathlib import Path
import argparse

from scripts.common import (
    commit_push_changes,
    eprint,
    run_command,
    run_command_unchecked,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit changes to a specified file.")
    parser.add_argument("file", type=Path, help="The file to commit.")
    args = parser.parse_args()

    if not args.file.exists():
        eprint(f"{args.file} does not exist.")
        sys.exit(1)

    run_command(["git", "add", str(args.file)])

    # Check if there are changes to commit
    result = run_command_unchecked(["git", "diff-index", "--cached", "--quiet", "HEAD"])
    if result.returncode == 0:
        print("No changes to commit")
        sys.exit(0)

    # Commit and push changes
    commit_push_changes(f"Update {args.file.name}", "main")


if __name__ == "__main__":
    main()
