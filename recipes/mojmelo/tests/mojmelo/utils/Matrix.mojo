from .mojmelo_matmul import matmul
from memory import memcpy, memset_zero, UnsafePointer
import random

struct Matrix(Copyable, Movable, Sized):
    var height: Int
    var width: Int
    var size: Int
    var data: UnsafePointer[Float32]
    var order: String

    # initialize from UnsafePointer
    @always_inline
    fn __init__(out self, data: UnsafePointer[Float32], height: Int, width: Int, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = data
        self.order = order.lower()

    # initialize by copying from UnsafePointer
    @always_inline
    fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float32] = UnsafePointer[Float32](), order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = order.lower()
        if data:
            memcpy(self.data, data, self.size)

    fn __copyinit__(out self, other: Self):
        self.height = other.height
        self.width = other.width
        self.size = other.size
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = other.order
        memcpy(self.data, other.data, self.size)

    fn __moveinit__(out self, owned existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.size = existing.size
        self.data = existing.data
        self.order = existing.order
        existing.height = existing.width = existing.size = 0
        existing.order = ''
        existing.data = UnsafePointer[Float32]()

    # access an element
    @always_inline
    fn __getitem__(self, row: Int, column: Int) raises -> Float32:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        return self.data[loc]

    @always_inline
    fn __del__(owned self):
        if self.data:
            self.data.free()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn __mul__(self, rhs: Self) raises -> Self:
        if self.width != rhs.height:
            raise Error('Error: Cannot multiply matrices with shapes (' + String(self.height) + ', ' + String(self.width) + ') and (' + String(rhs.height) + ', ' + String(rhs.width) + ')')
        var A = matmul.Matrix[DType.float32](self.data, (self.height, self.width))
        var B = matmul.Matrix[DType.float32](rhs.data, (rhs.height, rhs.width))
        var C = matmul.Matrix[DType.float32]((self.height, rhs.width))
        memset_zero(C.data, self.height * rhs.width)
        matmul.matmul(self.height, self.width, rhs.width, C, A, B)
        return Matrix(C.data, self.height, rhs.width)

    @always_inline
    fn __imul__(mut self, rhs: Self) raises:
        self = self * rhs

    @staticmethod
    @always_inline
    fn zeros(height: Int, width: Int, order: String = 'c') -> Matrix:
        var mat = Matrix(height, width, order= order)
        memset_zero(mat.data, mat.size)
        return mat^

    @staticmethod
    @always_inline
    fn random(height: Int, width: Int, order: String = 'c') -> Matrix:
        random.seed()
        var mat = Matrix(height, width, order= order)
        random.rand(mat.data, mat.size, min=0.0, max=1.0)
        return mat^
