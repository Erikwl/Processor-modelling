import numpy as np
# # from discreteMarkovChain import markovChain
# import ctypes
# # import pathlib
# from ctypes import c_int, c_double, POINTER
# # from ctypes.util import find_library
# import os

# # # Get the directory path of the Python script
# # libname = pathlib.Path().absolute()

# # Load the shared library
# mva_c_lib = ctypes.CDLL('./mva_c.so')

# # # Get all functions from the library
# # functions = [attr for attr in dir(mva_c_lib) if callable(getattr(mva_c_lib, attr))]

# # # Print the list of function names
# # for function_name in functions:
# #     print(function_name)

# # mva_c_lib.total_customers.argtypes = [
# #     POINTER(c_int),
# #     c_int
# # ]

# # print('fasdf')

# # Define the argument.
# mva_c_lib.mva_c.argtypes = [
#     POINTER(c_int),  # vector<int> N
#     POINTER(c_int),  # vector<int> refs
#     POINTER(POINTER(c_double)),  # vector<vector<double>> visits
#     POINTER(c_int),  # vector<int> caps
#     POINTER(c_double),  # vector<double> service_times
# ]
# print('fasdf')
# # # Define the return types.
# # mva_c_lib.mva_c.restype = [
# #     POINTER(c_double),  # vector<double> nrs
# #     POINTER(POINTER(c_double)),  # vector<vector<double>> waits
# #     POINTER(c_double),  # vector<double> throughputs
# #     POINTER(c_double),  # vector<double> utils
# #     POINTER(POINTER(c_double)),  # vector<double> probs
# # ]

# # def find_inv_dists(Ps):
# #     mcs = [markovChain(P) for P in Ps]
# #     for P, mc in zip(Ps, mcs):
# #         mc.computePi('linear')
# #     return np.array(mc.pi for mc in mcs)

# # classes           : 0, 1, ..., R - 1
# # service centers   : 0, 1, ..., M - 1

# def mva(N, refs, visits, caps, service_times):
#     # Convert Python lists to C++ vectors
#     N_arr = (c_int * len(N))(*N)
#     refs_arr = (c_int * len(refs))(*refs)
#     visits_arr = (POINTER(c_double) * len(visits))()
#     for i, sublist in enumerate(visits):
#         visits_arr[i] = (c_double * len(sublist))(*sublist)
#     caps_arr = (c_int * len(caps))(*caps)
#     service_times_arr = (c_double * len(service_times))(*service_times)

#     # Call the C++ function
#     return mva_c_lib.mva_c(N_arr, refs_arr, visits_arr, caps_arr, service_times_arr)

def mva(N, refs, visits, caps, service_times):
    mus = 1 / service_times
    R = len(N)
    M = len(visits)

    # # Convert from 1 indexed to 0 indexed
    # refs -= np.ones(R, dtype='int')

    # Normalize visits using reference stations.
    for (r, ref) in enumerate(refs):
        visits[:,r] /= visits[ref,r]

    Rs = [[r for r in range(R) if visits[i,r] > 0] for i in range(M)]
    Ms = [[i for i in range(M) if visits[i,r] > 0] for r in range(R)]

    nr_of_states = int(np.product(N + np.ones(1)))

    # Mean number of customers for population vector N_ at center i.
    nrs = np.zeros((M, nr_of_states))
    utils = np.zeros((M, nr_of_states))
    probs = np.zeros((M, min(np.sum(N) + 1, max(caps)), nr_of_states))
    waits = np.zeros((M, R, nr_of_states))
    throughputs = np.zeros((R, nr_of_states))

    # print(M, caps, N)
    for i in range(M):
        probs[i, 0, 0] = 1

    N_products = []
    cur = 1
    for r in range(R):
        N_products.append(cur)
        cur *= (N[r] + 1)

    def total_customers(N_):
        totalN_ = 0
        cur = N_
        for Nr in N:
            totalN_ += cur % (Nr + 1)
            cur //= (Nr + 1)
        return totalN_

    def pop_vector(N_):
        vec = []

        cur = N_
        for Nr in N:
            vec.append(cur % (Nr + 1))
            cur //= (Nr + 1)
        return vec


    for N_ in range(nr_of_states):

        totalN_ = total_customers(N_)
        cur = N_

        for r, Nr in enumerate(N):
            Nr_ = cur % (Nr + 1)
            cur //= (Nr + 1)

            r_cus_removal = N_ - N_products[r]

            for i in Ms[r]:
                ci = caps[i]
                if Nr_:
                    waits[i, r, N_] = 1 / (ci * mus[i]) * ( \
                        1 \
                        + nrs[i, r_cus_removal] \
                        + np.sum((ci - n - 1) * probs[i, n, r_cus_removal]
                                 for n in range(1 + min(ci - 2, totalN_ - 1))))

            if Nr_:
                throughputs[r, N_] = Nr_ / np.sum(visits[i,r] * waits[i,r,N_] for i in Ms[r])
                # print('throughput ', r, throughputs[r,N_])

        for i in range(M):
            nrs[i,N_] = np.sum(visits[i,r] * throughputs[r,N_] * waits[i,r,N_] for r in Rs[i])
            utils[i,N_] = 1 / mus[i] * np.sum(throughputs[r,N_] * visits[i,r] for r in Rs[i])

            ci = caps[i]
            for n in range(1, 1 + min(ci - 1, totalN_)):
                # print(i, n, '\n')
                probs[i,n,N_] = 1 / (min(n, ci) * mus[i]) \
                    * np.sum(visits[i,r] * throughputs[r,N_] * probs[i,n - 1,N_ - N_products[r]]
                             for r in Rs[i])
                # r = 1
                # print(N_ - N_products[r])
                # print(probs[i,n - 1,N_ - N_products[r]])
                # print(visits[i,r])
                # print(throughputs[r,N_])
                # print(visits[i,r] * throughputs[r,N_] * probs[i,n - 1,N_ - N_products[r]], '\n\n\n')
                # print(probs[i,n,N_], throughputs[i:,N_], probs[i,n - 1,N_ - N_products[1]])
            probs[i,0,N_] = 1 - 1 / ci * (
                utils[i,N_] + np.sum((ci - n) * probs[i,n,N_] for n in range(1,1+min(ci-1,totalN_)))
            )

        # print(pop_vector(N_))
        # print(totalN_)
        # print(f'n_i(N) = \n{nrs[:,N_]}\n')
        # print(f'w_i,r(N) = \n{waits[:,:,N_]}\n')
        # print(f'x_l^*(r),r(N) = \n{throughputs[:,N_]}\n')
        # print(f'u_i(N) = \n{utils[:,N_]}')
        # print(f'p_i,n(N) =\n{probs[:,:,N_]}\n\n\n')
    N_ = nr_of_states - 1
    return [nrs[:,N_], waits[:,:,N_], throughputs[:,N_], utils[:,N_], probs[:,:,N_]]

    # print(f'n_i(N) = \n{nrs[:,nr_of_states-1]}\n')
    # print(f'w_i,r(N) = \n{waits[:,:,nr_of_states-1]}\n')
    # print(f'x_l^*(r),r(N) = \n{throughputs[:,nr_of_states-1]}\n')
    # print(f'u_i(N) = \n{utils[:,nr_of_states-1]}')
    # print(f'p_i,n(N) =\n{probs[:,:,nr_of_states-1]}')
