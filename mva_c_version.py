from ctypes import *
import numpy as np
# classes           : 0, 1, ..., R - 1
# service centers   : 0, 1, ..., M - 1

lib = cdll.LoadLibrary('./mva_c.so')
lib.Mva_new.restype = c_void_p
lib.Mva_new.argtypes = [c_int, c_int, c_int, c_bool]

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

def mva(N, refs, visits, caps, service_times, complete_probs=False):
    # Convert Python lists to C++ vectors
    R = len(N)
    M = len(visits)
    probs_size = np.sum(N) + 1
    if complete_probs == False:
        probs_size = min(max(caps), probs_size)
    mva_c = lib.Mva_new(c_int(R), c_int(M), c_int(probs_size), c_bool(complete_probs))
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
