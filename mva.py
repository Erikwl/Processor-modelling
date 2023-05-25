import numpy as np
from discreteMarkovChain import markovChain

def find_inv_dists(Ps):
    mcs = [markovChain(P) for P in Ps]
    for P, mc in zip(Ps, mcs):
        mc.computePi('linear')
    return np.array(mc.pi for mc in mcs)

# classes           : 0, 1, ..., R - 1
# service centers   : 0, 1, ..., M - 1

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
        totalN = 0
        cur = N_
        for Nr in N:
            totalN += cur % (Nr + 1)
            cur //= (Nr + 1)
        return totalN

    def pop_vector(N_):
        vec = []

        cur = N_
        for Nr in N:
            vec.append(cur % (Nr + 1))
            cur //= (Nr + 1)
        return vec


    for N_ in range(nr_of_states):

        totalN = total_customers(N_)
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
                                 for n in range(1 + min(ci - 2, totalN - 1))))

            if Nr_:
                throughputs[r, N_] = Nr_ / np.sum(visits[i,r] * waits[i,r,N_] for i in Ms[r])
                # print('throughput ', r, throughputs[r,N_])

        for i in range(M):
            nrs[i,N_] = np.sum(visits[i,r] * throughputs[r,N_] * waits[i,r,N_] for r in Rs[i])
            utils[i,N_] = 1 / mus[i] * np.sum(throughputs[r,N_] * visits[i,r] for r in Rs[i])

            ci = caps[i]
            for n in range(1, 1 + min(ci - 1, totalN)):
                # print(i, n, '\n')
                probs[i,n,N_] = 1 / (min(n, ci) * mus[i]) \
                    * np.sum(visits[i,r] * throughputs[r,N_] * probs[i,n - 1,N_ - N_products[r]]
                             for r in Rs[i])
                r = 1
                # print(N_ - N_products[r])
                # print(probs[i,n - 1,N_ - N_products[r]])
                # print(visits[i,r])
                # print(throughputs[r,N_])
                # print(visits[i,r] * throughputs[r,N_] * probs[i,n - 1,N_ - N_products[r]], '\n\n\n')
                # print(probs[i,n,N_], throughputs[i:,N_], probs[i,n - 1,N_ - N_products[1]])
            probs[i,0,N_] = 1 - 1 / ci * (
                utils[i,N_] + np.sum((ci - n) * probs[i,n,N_] for n in range(1,1+min(ci-1,totalN)))
            )

        # print(pop_vector(N_))
        # print(totalN)
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
