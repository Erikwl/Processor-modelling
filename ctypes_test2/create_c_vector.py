from ctypes import *

class Int_vector(object):
    lib = cdll.LoadLibrary('./create_c_vector.so')
    lib.new_int_vector.restype = c_void_p
    lib.new_int_vector.argtypes = [c_int]
    lib.int_vector_delete.restype = None
    lib.int_vector_delete.argtypes = [c_void_p]
    lib.int_vector_getitem.restype = c_int
    lib.int_vector_getitem.argtypes = [c_void_p, c_int]
    lib.int_vector_setitem.restype = None
    lib.int_vector_setitem.argtypes = [c_void_p, c_int, c_int]

    def __init__(self, n):
        self.vector = Int_vector.lib.new_int_vector(c_int(n))

    def __del__(self):
        Int_vector.lib.int_vector_delete(self.vector)

    def getitem(self, i):
        return Int_vector.lib.int_vector_getitem(self.vector, c_int(i))

    def setitem(self, i, x):
        Int_vector.lib.int_vector_setitem(self.vector, c_int(i), c_int(x))


class Double_vector(object):
    lib = cdll.LoadLibrary('./create_c_vector.so')
    lib.new_double_vector.restype = c_void_p
    lib.new_double_vector.argtypes = [c_int]
    lib.double_vector_delete.restype = None
    lib.double_vector_delete.argtypes = [c_void_p]
    lib.double_vector_getitem.restype = c_double
    lib.double_vector_getitem.argtypes = [c_void_p, c_int]
    lib.double_vector_setitem.restype = None
    lib.double_vector_setitem.argtypes = [c_void_p, c_int, c_double]

    def __init__(self, n):
        self.vector = Double_vector.lib.new_double_vector(c_int(n))

    def __del__(self):
        Double_vector.lib.double_vector_delete(self.vector)

    def getitem(self, i):
        return Double_vector.lib.double_vector_getitem(self.vector, c_int(i))

    def setitem(self, i, x):
        Double_vector.lib.double_vector_setitem(self.vector, c_int(i), c_double(x))

class Double_double_vector(object):
    lib = cdll.LoadLibrary('./create_c_vector.so')
    lib.new_double_double_vector.restype = c_void_p
    lib.new_double_double_vector.argtypes = [c_int, c_int]
    lib.double_double_vector_delete.restype = None
    lib.double_double_vector_delete.argtypes = [c_void_p]
    lib.double_double_vector_getitem.restype = c_double
    lib.double_double_vector_getitem.argtypes = [c_void_p, c_int, c_int]
    lib.double_double_vector_setitem.restype = None
    lib.double_double_vector_setitem.argtypes = [c_void_p, c_int, c_int, c_double]

    def __init__(self, n, m):
        self.vector = Double_double_vector.lib.new_double_double_vector(c_int(n), c_int(m))

    def __del__(self):
        Double_double_vector.lib.double_double_vector_delete(self.vector)

    def getitem(self, i, j):
        return Double_double_vector.lib.double_double_vector_getitem(self.vector, c_int(i), c_int(j))

    def setitem(self, i, j, x):
        Double_double_vector.lib.double_double_vector_setitem(self.vector, c_int(i), c_int(j), c_double(x))


if __name__ == '__main__':
    a = Int_vector(5)
    print(1)
    a.setitem(2, 3)
    print(2)
    for i in range(5):

        print(i, a.getitem(i))
