from scripts.common import run_command


def main() -> None:
    """
    Configures the global Git user name and email for the current environment.

    This function sets the global Git configuration for the user name and email
    to be used by Git commands. It is typically used in automated environments
    such as CI/CD pipelines to ensure that Git operations are performed with a
    consistent identity.

    The user name is set to "github-actions[bot]" and the email is set to
    "github-actions[bot]@users.noreply.github.com".

    Raises:
        subprocess.CalledProcessError: If any of the subprocess calls fail.
    """
    run_command(["git", "config", "--global", "user.name", "github-actions[bot]"])
    run_command(
        [
            "git",
            "config",
            "--global",
            "user.email",
            "github-actions[bot]@users.noreply.github.com",
        ]
    )


if __name__ == "__main__":
    main()
