from ctypes import *
import numpy as np
# classes           : 0, 1, ..., R - 1
# service centers   : 0, 1, ..., M - 1

lib = cdll.LoadLibrary('./mva_c.so')
lib.Mva_new.restype = c_void_p
lib.Mva_new.argtypes = [c_int, c_int, c_int]

lib.N_set.restype = None
lib.refs_set.restype = None
lib.visits_set.restype = None
lib.caps_set.restype = None
lib.service_times_set.restype = None

lib.N_set.argtypes = [c_void_p, c_int, c_int]
lib.refs_set.argtypes = [c_void_p, c_int, c_int]
lib.visits_set.argtypes = [c_void_p, c_int, c_int, c_double]
lib.caps_set.argtypes = [c_void_p, c_int, c_int]
lib.service_times_set.argtypes = [c_void_p, c_int, c_double]

lib.nrs_get.restype = c_double
lib.waits_get.restype = c_double
lib.throughputs_get.restype = c_double
lib.utils_get.restype = c_double
lib.probs_get.restype = c_double

lib.nrs_get.argtypes = [c_void_p, c_int]
lib.waits_get.argtypes = [c_void_p, c_int, c_int]
lib.throughputs_get.argtypes = [c_void_p, c_int]
lib.utils_get.argtypes = [c_void_p, c_int]
lib.probs_get.argtypes = [c_void_p, c_int, c_int]

lib.compute.restype = None
lib.compute.argtypes = [c_void_p]

def mva(N, refs, visits, caps, service_times):
    # Convert Python lists to C++ vectors
    R = len(N)
    M = len(visits)
    probs_size = min(np.sum(N) + 1, max(caps))
    mva_c = lib.Mva_new(c_int(R), c_int(M), c_int(probs_size))
    for i, x in enumerate(N):
        lib.N_set(mva_c, c_int(i), c_int(x))

    for i, x in enumerate(refs):
        lib.refs_set(mva_c, c_int(i), c_int(x))

    for i in range(M):
        for j, x in enumerate(visits[i]):
            lib.visits_set(mva_c, c_int(i), c_int(j), c_double(x))

    for i, x in enumerate(caps):
        lib.caps_set(mva_c, c_int(i), c_int(x))

    for i, x in enumerate(service_times):
        lib.service_times_set(mva_c, c_int(i), c_double(x))

    lib.compute(mva_c)

    nrs = np.array([lib.nrs_get(mva_c, i) for i in range(M)])
    waits = np.array([[lib.waits_get(mva_c, i, j) for j in range(R)] for i in range(M)])
    throughputs = np.array([lib.throughputs_get(mva_c, i) for i in range(R)])
    utils = np.array([lib.utils_get(mva_c, i) for i in range(M)])
    probs = np.array([[lib.probs_get(mva_c, i, j) for j in range(probs_size)] for i in range(M)])

    return [nrs, waits, throughputs, utils, probs]

    # return [[] for _ in range(5)]
    # N_arr = (c_int * len(N))(*N)
    # refs_arr = (c_int * len(refs))(*refs)
    # visits_arr = (POINTER(c_double) * len(visits))()
    # for i, sublist in enumerate(visits):
    #     visits_arr[i] = (c_double * len(sublist))(*sublist)
    # caps_arr = (c_int * len(caps))(*caps)
    # service_times_arr = (c_double * len(service_times))(*service_times)

    # print(type(N_arr))
    # print(type(refs_arr))
    # print(type(visits_arr))
    # print(type(caps_arr))
    # print(type(service_times_arr))

    # N_arr = cast(N_arr, POINTER(c_int))
    # refs_arr = cast(refs_arr, POINTER(c_double))
    # # visits_arr = cast(visits_arr, POINTER(POINTER(visits)))
    # caps_arr = cast(caps_arr, POINTER(c_int))
    # service_times_arr = cast(service_times_arr, POINTER(c_double))

    # print('\n')
    # print(type(N_arr))
    # print(type(refs_arr))
    # print(type(visits_arr))
    # print(type(caps_arr))
    # print(type(service_times_arr))

    # Call the C++ function
    # return lib.mva_c(mva_object, N_arr, refs_arr, visits_arr, caps_arr, service_times_arr)