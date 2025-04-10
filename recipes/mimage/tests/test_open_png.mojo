import mimage as mi
from pathlib import Path
from tests import compare_to_numpy


def main():
    hop_tensor = mi.imread("tests/images/hopper.png")
    hop_tensor = mi.imread(String("tests/images/hopper.png"))
    hop_tensor = mi.imread(Path("tests/images/hopper.png"))

    compare_to_numpy(hop_tensor, "tests/images/hopper.png")
