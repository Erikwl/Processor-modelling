from ctypes import cdll
from ctypes import c_int, POINTER, c_byte, cast, c_void_p
import numpy as np
import ctypes
lib = cdll.LoadLibrary('./libfoo.so')

lib.Foo_bar.argtypes = [
    c_void_p
]

lib.Foo_bar.restype = None

class Foo(object):
    def __init__(self):
        self.obj = lib.Foo_new()

    def bar(self, n):
        lib.Foo_bar(self.obj, n)

# a = [1,2,3]
N = np.array([[0, 1], [2, 3]], dtype=np.int32)
# print(N, len(N))
a = N.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

# a_arr = (c_int * len(a))(*a)
print(type(a), a.contents)
# a_arr = cast(a, POINTER(c_int))
# print(type(a_arr))
# print(a_arr.contents)
f = Foo()
f.bar(a)
# lib.Foo_bar()

"""
run:
g++ -c -fPIC foo.cpp -o foo.o
g++ -shared -Wl,-soname,libfoo.so -o libfoo.so  foo.o
"""