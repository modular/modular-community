# Smoke test executed by the `tests:` block in recipe.yaml.
#
# Imports canonical public symbols from the installed projectodyssey.mojopkg
# and instantiates one to prove the package is not just parseable but
# actually usable. If any import or instantiation fails, rattler-build
# fails the package test and the recipe is rejected.
#
# Positive imports only: Mojo import failures are compile-time errors,
# so there is no equivalent of pytest.raises(ImportError) here.

from projectodyssey.tensor.tensor import Tensor
from projectodyssey.tensor.any_tensor import AnyTensor, zeros


def main() raises:
    # Compile-time-typed tensor.
    var _t = Tensor[DType.float32]([2, 3])

    # Runtime-typed tensor via the zeros() factory (canonical creation path).
    var _a = zeros([2, 3], DType.float32)

    print("projectodyssey package import test: OK")
