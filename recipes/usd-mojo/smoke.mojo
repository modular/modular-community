from std.sys import argv
from usd_mojo import Stage


def main() raises:
    var args = argv()
    var stage = Stage(args[1])
    var mesh = stage.mesh_snapshot("/World/Mesh")
    var points = mesh.points()
    if len(points) != 9 or points[3] != 1 or points[7] != 1:
        raise Error("Installed USD mesh differs")
    print("PASS: installed usd-mojo with resolved SDK")
