from .mojmelo_matmul import matmul
from memory import memcpy, memset_zero
import random

struct Matrix(Copyable, Movable, ImplicitlyCopyable, Sized):
    var height: Int
    var width: Int
    var size: Int
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var order: String

    # initialize from UnsafePointer
    @always_inline
    fn __init__[src: DType = DType.float32](out self, data: UnsafePointer[Scalar[src], MutAnyOrigin], height: Int, width: Int, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        if src == DType.float32:
            self.data = data.bitcast[Float32]()
        else:
            self.data = cast[src=src, des=DType.float32, width=self.simd_width](data, self.size)
            data.free()
        self.order = order.lower()

    # initialize by copying from UnsafePointer
    @always_inline
    fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](), order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = alloc[Float32](self.size)
        self.order = order.lower()
        if data:
            memcpy(dest=self.data, src=data, count=self.size)

    fn __copyinit__(out self, other: Self):
        self.height = other.height
        self.width = other.width
        self.size = other.size
        self.data = alloc[Float32](self.size)
        self.order = other.order
        memcpy(dest=self.data, src=other.data, count=self.size)

    fn __moveinit__(out self, deinit existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.size = existing.size
        self.data = existing.data
        self.order = existing.order
        #existing.height = existing.width = existing.size = 0
        #existing.order = ''
        #existing.data = UnsafePointer[Float32, MutAnyOrigin]()

    # access an element
    @always_inline
    fn __getitem__(self, row: Int, column: Int) raises -> Float32:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1 or loc < 0:
            raise Error("Location is out of range!")
        return self.data[loc]

    @always_inline
    fn __del__(deinit self):
        if self.data:
            self.data.free()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn __mul__(self, rhs: Self) raises -> Self:
        if self.width != rhs.height:
            raise Error('Error: Cannot multiply matrices with shapes (' + String(self.height) + ', ' + String(self.width) + ') and (' + String(rhs.height) + ', ' + String(rhs.width) + ')')

        if self.height == 1 and rhs.width == 1:
            # Dot product
            var mat = Self(1, 1)
            mat.data[0] = self.ele_mul(rhs.T()).sum()
            return mat^

        if self.height * self.width * rhs.width <= 4096:
            # matmul naive
            var mat = Self(self.height, rhs.width)
            for i in range(self.size):
                var rhsr = i % self.width
                for j in range(rhsr * rhs.width, rhsr * rhs.width + rhs.width):
                    if rhsr != 0:
                        mat.data[(Int(i / self.width) * mat.width) + (j % rhs.width)] += self.data[i] * rhs.data[j]
                    else:
                        mat.data[(Int(i / self.width) * mat.width) + (j % rhs.width)] = self.data[i] * rhs.data[j]
            return mat^
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
    fn random(height: Int, width: Int, order: String = 'c') -> Matrix:
        random.seed()
        var mat = Matrix(height, width, order= order)
        random.rand(mat.data, mat.size, min=0.0, max=1.0)
        return mat^
