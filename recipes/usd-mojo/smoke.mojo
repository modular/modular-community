from std.sys import argv
from usd_mojo import (
    Stage,
    Layer,
    SceneSnapshot,
    SchemaSnapshot,
    poll_changes,
    ArraySnapshot,
    validate_stage,
)


def main() raises:
    var args = argv()
    var stage = Stage(args[1])
    var mesh = stage.mesh_snapshot("/World/Mesh")
    var points = mesh.points()
    if len(points) != 9 or points[3] != 1 or points[7] != 1:
        raise Error("Installed USD mesh differs")
    var scene = SceneSnapshot(stage)
    if len(scene.transforms()) != 32:
        raise Error("Installed scene snapshot differs")
    var schema = SchemaSnapshot(stage, "/World/Mesh")
    if schema.value("points").numbers[3] != 1:
        raise Error("Installed schema snapshot differs")
    var root = Layer(stage)
    if root.is_anonymous() or poll_changes(stage).generation != 0:
        raise Error("Installed layer/notice behavior differs")
    var array = ArraySnapshot(stage, "/World/Mesh", "points")
    if array.view[DType.float32]()[3] != 1:
        raise Error("Installed native array differs")
    if not validate_stage(
        stage, validators=[String("usd_mojo:MeshTopology")]
    ).is_valid():
        raise Error("Installed scene validation differs")
    print("PASS: installed usd-mojo with resolved SDK")
