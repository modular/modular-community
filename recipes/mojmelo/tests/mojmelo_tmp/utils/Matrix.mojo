from .mojmelo_matmul import matmul
from std.memory import unsafe_memcpy, unsafe_memset_zero, Layout
from std.sys import simd_width_of, CompilationTarget
import std.math as math
import std.random as random
from mojmelo.utils.utils import cast

struct Matrix(Copyable, ImplicitlyCopyable, Sized):
    var height: Int
    var width: Int
    var size: Int
    var data: Pointer[Float32, MutUntrackedOrigin]
    var order: String
    comptime simd_width: Int = 4 * simd_width_of[DType.float32]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[DType.float32]()

    # initialize from Pointer
    @always_inline
    def __init__[src: DType = DType.float32](out self, data: Pointer[Scalar[src], MutUntrackedOrigin], height: Int, width: Int, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        if src == DType.float32:
            self.data = data.unsafe_bitcast[Float32]()
        else:
            self.data = cast[src=src, des=DType.float32, width=self.simd_width](data, self.size)
            data.unsafe_free()
        self.order = order.lower()

    # initialize by copying from Pointer
    @always_inline
    def __init__(out self, height: Int, width: Int, data: OptionalPointer[Float32, MutUntrackedOrigin] = None, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = alloc(Layout[Float32](count=self.size)).unsafe_leak()
        self.order = order.lower()
        if data:
            unsafe_memcpy(dest=self.data, src=data.value(), count=self.size)

    def __init__(out self, *, copy: Self):
        self.height = copy.height
        self.width = copy.width
        self.size = copy.size
        self.data = alloc(Layout[Float32](count=self.size)).unsafe_leak()
        self.order = copy.order
        unsafe_memcpy(dest=self.data, src=copy.data, count=self.size)

    def __init__(out self, *, deinit move: Self):
        self.height = move.height
        self.width = move.width
        self.size = move.size
        self.data = move.data
        self.order = move.order
        #move.height = move.width = move.size = 0
        #move.order = ''
        #move.data = Pointer[Float32, MutAnyOrigin]()

    @always_inline
    def __deinit__(deinit self):
        self.data.unsafe_free()

    @always_inline
    def __len__(self) -> Int:
        return self.size

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        var A = matmul.Matrix[DType.float32](self.data, (self.height, self.width))
        var B = matmul.Matrix[DType.float32](rhs.data, (rhs.height, rhs.width))
        var C = matmul.Matrix[DType.float32]((self.height, rhs.width))
        unsafe_memset_zero(C.data, self.height * rhs.width)
        matmul.matmul(self.height, self.width, rhs.width, C, A, B)
        return Matrix(C.data, self.height, rhs.width)

    @staticmethod
    def random(height: Int, width: Int, order: String = 'c') -> Matrix:
        random.seed()
        var mat = Matrix(height, width, order= order)
        random.rand(mat.data, mat.size, min=0.0, max=1.0)
        return mat^
