"""Exercise build-all dependency resolution with local package channels."""

from contextlib import chdir
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


build_all = importlib.import_module("scripts.build-all")
upload = importlib.import_module("scripts.upload")


class BuildAllTest(unittest.TestCase):
    def write_channel(self, channel: Path, packages: list[dict]) -> None:
        for platform in ("linux-64", "noarch"):
            directory = channel / platform
            directory.mkdir(parents=True)
            records = {}
            for package in packages if platform == "linux-64" else []:
                record = {
                    **package,
                    "build": "0",
                    "build_number": 0,
                    "subdir": platform,
                }
                filename = f"{record['name']}-{record['version']}-0.tar.bz2"
                with tarfile.open(directory / filename, "w:bz2") as archive:
                    data = json.dumps(record).encode()
                    info = tarfile.TarInfo("info/index.json")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                records[filename] = record
            (directory / "repodata.json").write_text(
                json.dumps({"info": {"subdir": platform}, "packages": records})
            )

    def test_other_recipe_output_does_not_shadow_compatible_published_build(
        self,
    ) -> None:
        rattler = shutil.which("rattler-build")
        self.assertIsNotNone(rattler, "Run tests in the Pixi environment")
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp).resolve()
            self.write_channel(
                root / "published",
                [
                    {"name": "mojo-compiler", "version": "1.0.0", "depends": []},
                    {
                        "name": "mojo-snappy",
                        "version": "0.2.0",
                        "depends": ["mojo-compiler ==1.0.0"],
                    },
                ],
            )
            incompatible = [
                {
                    "name": "mojo-snappy",
                    "version": "0.2.0",
                    "depends": ["mojo-compiler ==1.1.0"],
                }
            ]
            # Cover both the previous shared layout and another recipe's output.
            self.write_channel(root / "output", incompatible)
            self.write_channel(root / "output" / "mojo-snappy", incompatible)
            recipe = root / "recipes" / "pyroquet" / "recipe.yaml"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(
                'package:\n  name: pyroquet\n  version: "0.1.1"\n'
                "requirements:\n  host:\n"
                "    - mojo-compiler ==1.0.0\n    - mojo-snappy ==0.2.0\n"
            )
            variants = root / "variants" / "variants.yaml"
            variants.parent.mkdir()
            variants.write_text("{}\n")
            results = []

            def solve(command: list[str]) -> subprocess.CompletedProcess[str]:
                # Use the real command assembled by build-all, but only solve:
                # no compiler execution or network packages are needed.
                result = subprocess.run(
                    [
                        rattler,
                        *command[1:],
                        "--render-only",
                        "--with-solve",
                        "--target-platform=linux-64",
                        "--build-platform=linux-64",
                        "--color=never",
                    ],
                    capture_output=True,
                    text=True,
                )
                results.append(result)
                return result

            with (
                patch.object(
                    build_all, "DEFAULT_CHANNELS", [(root / "published").as_uri()]
                ),
                patch.object(build_all, "recipe_name_collisions", return_value=False),
                patch.object(build_all, "run_command_unchecked", side_effect=solve),
                patch("sys.argv", ["build-all"]),
                patch.dict(os.environ, {"CONDA_BLD_PATH": str(root / "output")}),
            ):
                os.environ.pop("DATA_FILE", None)
                with self.assertRaises(SystemExit) as exited:
                    build_all.main()
            self.assertEqual(len(results), 1)
            self.assertEqual(exited.exception.code, 0, results[0].stderr)

    def test_upload_discovers_packages_in_recipe_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            package = Path("output/mojo-snappy/linux-64/mojo-snappy-0.2.0-0.conda")
            package.parent.mkdir(parents=True)
            package.touch()
            with (
                patch("sys.argv", ["upload", "--channel", "test-channel"]),
                patch.object(upload, "run_command_unchecked") as run,
            ):
                run.return_value.returncode = 0
                with self.assertRaises(SystemExit) as exited:
                    upload.main()
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(run.call_args.args[0][-1], str(package))


if __name__ == "__main__":
    unittest.main()
