from std.collections import Array, ImmSpan, List, Span
from std.testing import assert_equal
from std.sys.info import size_of
from axisweave.gpu import DeviceTransform, DeviceEinsum
from axisweave import (
    StaticRecipe,
    plan,
    execute,
    packing_plan,
    einsum_plan,
    mix_plan,
)


def main() raises:
    comptime assert size_of[DeviceTransform]() > 0
    comptime assert size_of[DeviceEinsum]() > 0
    var values: List[Float32] = [0, 1, 2, 3, 4, 5]
    var shape: List[Int] = [2, 3]
    var expected: List[Float32] = [0, 3, 1, 4, 2, 5]
    var recipe = plan("a b -> b a", Span(shape))
    assert_equal(execute(recipe, Span(values)), expected)
    var static = StaticRecipe["a b -> b a", 2, 3]()
    assert_equal(static.execute(Span(values)), expected)
    var shapes: List[List[Int]] = [[2, 3], [3, 2]]
    var inputs: List[List[Float32]] = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    var product = einsum_plan(
        Span(shapes), "row dim, dim col -> row col"
    ).execute(Span(inputs))
    var wanted: List[Float32] = [10, 13, 28, 40]
    assert_equal(product, wanted)
    var flat: List[List[Int]] = [[6], [6]]
    var packing = packing_plan(Span(flat), "*")
    var packed = packing.pack(Span(inputs))
    var views = packing.unpack_views(Span(packed))
    assert_equal(views[1].materialize(), inputs[1])
    var descriptors: Array[ImmSpan[Float32, origin_of(inputs[0])], 2] = [
        Span(inputs[0]).as_imm(),
        Span(inputs[1]).as_imm(),
    ]
    assert_equal(packing.pack(Span(descriptors)), packed)
    var split = List[Float32](length=12, fill=0)
    var outputs = Array[Span[mut=True, Float32, origin_of(split)], 2]()
    outputs[0] = Span(split)[:6]
    outputs[1] = Span(split)[6:]
    packing.unpack_into(Span(packed), Span(outputs))
    assert_equal(split, packed)
    var linear = mix_plan(
        Span(shapes[0]), Span(shapes[1]), "row dim, dim col -> row col"
    )
    assert_equal(linear.execute(Span(inputs[0]), Span(inputs[1])), wanted)
    var flags: List[Bool] = [True, False, True, False, False, False]
    var logical = StaticRecipe["row col -> row", 2, 3, operation=.reduce]()
    var truth_wanted: List[Bool] = [True, False]
    assert_equal(logical.execute[.any](Span(flags)), truth_wanted)
    var integer: List[Int64] = [9007199254740993, 1, 2, 3, 4, 5]
    var sums: List[Int64] = [9007199254740996, 12]
    assert_equal(logical.execute[.sum](Span(integer)), sums)
    var masks: List[List[Bool]] = [flags.copy(), flags.copy()]
    var mask_packed = packing.pack(Span(masks))
    assert_equal(packing.unpack(Span(mask_packed)), masks)
    var mask_views = packing.unpack_views(Span(mask_packed))
    var copied_mask = List[Bool](length=6, fill=False)
    mask_views[0].materialize_into(Span(copied_mask))
    assert_equal(copied_mask, flags)
    var transposed_mask = static.view(Span(flags))
    transposed_mask.materialize_into(Span(copied_mask))
    var mask_wanted: List[Bool] = [True, False, False, False, True, False]
    assert_equal(copied_mask, mask_wanted)
    var precise: List[Float64] = [16777217, 1, 2, 3, 4, 5]
    var wanted64: List[Float64] = [16777220, 12]
    assert_equal(logical.execute[.sum](Span(precise)), wanted64)
    print("PASS: installed axisweave CPU, Float64 and GPU API")
