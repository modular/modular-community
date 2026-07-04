from .mojmelo_matmul import matmul
from std.memory import memcpy, memset_zero
from std.sys import simd_width_of, CompilationTarget
from std.algorithm import vectorize, parallelize
import std.math as math
import std.random as random

struct Matrix(Copyable, ImplicitlyCopyable, Sized):
    var height: Int
    var width: Int
    var size: Int
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var order: String
    comptime simd_width: Int = 4 * simd_width_of[DType.float32]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[DType.float32]()

    # initialize from UnsafePointer
    @always_inline
    def __init__[src: DType = DType.float32](out self, data: UnsafePointer[Scalar[src], MutAnyOrigin], height: Int, width: Int, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        if src == DType.float32:
            self.data = data.bitcast[Float32]()
        else:
            self.data = cast[src=src, des=DType.float32, width=self.simd_width](data, self.size).as_unsafe_any_origin()
            data.free()
        self.order = order.lower()

    # initialize by copying from UnsafePointer
    @always_inline
    def __init__(out self, height: Int, width: Int, data: OptionalUnsafePointer[Float32, MutAnyOrigin] = None, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = alloc[Float32](self.size).as_unsafe_any_origin()
        self.order = order.lower()
        if data:
            memcpy(dest=self.data, src=data.value(), count=self.size)

    def __init__(out self, *, copy: Self):
        self.height = copy.height
        self.width = copy.width
        self.size = copy.size
        self.data = alloc[Float32](self.size).as_unsafe_any_origin()
        self.order = copy.order
        memcpy(dest=self.data, src=copy.data, count=self.size)

    def __init__(out self, *, deinit take: Self):
        self.height = take.height
        self.width = take.width
        self.size = take.size
        self.data = take.data
        self.order = take.order
        #take.height = take.width = take.size = 0
        #take.order = ''
        #take.data = UnsafePointer[Float32, MutAnyOrigin]()

    # access an element
    @always_inline
    def __getitem__(self, row: Int, column: Int) raises -> Float32:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1 or loc < 0:
            raise Error("Location is out of range!")
        return self.data[loc]

    @always_inline
    def __del__(deinit self):
        self.data.free()

    @always_inline
    def __len__(self) -> Int:
        return self.size

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        var A = matmul.Matrix[DType.float32](self.data, (self.height, self.width))
        var B = matmul.Matrix[DType.float32](rhs.data, (rhs.height, rhs.width))
        var C = matmul.Matrix[DType.float32]((self.height, rhs.width))
        memset_zero(C.data, self.height * rhs.width)
        matmul.matmul(self.height, self.width, rhs.width, C, A, B)
        return Matrix(C.data, self.height, rhs.width)

    @staticmethod
    @always_inline
    def zeros(height: Int, width: Int, order: String = 'c') -> Matrix:
        var mat = Matrix(height, width, order= order)
        memset_zero(mat.data, mat.size)
        return mat^

    @staticmethod
    def random(height: Int, width: Int, order: String = 'c') -> Matrix:
        random.seed()
        var mat = Matrix(height, width, order= order)
        random.rand(mat.data, mat.size, min=0.0, max=1.0)
        return mat^

@always_inline
def cast[src: DType, des: DType, width: Int](data: UnsafePointer[Scalar[src], MutAnyOrigin], size: Int) -> UnsafePointer[Scalar[des], MutUntrackedOrigin]:
    var ptr = alloc[Scalar[des]](size)
    if size < 262144:

        def matrix_vectorize[simd_width: Int](idx: Int) {read}:
            ptr.store(idx, data.load[width=simd_width](idx).cast[des]())
        vectorize[width](size, matrix_vectorize)
    else:
        var n_vects = Int(math.ceil(size / width))
        @parameter
        def matrix_vectorize_parallelize(i: Int):
            var idx = i * width
            ptr.store(idx, data.load[width=width](idx).cast[des]())
        parallelize[matrix_vectorize_parallelize](n_vects)
    return ptr
