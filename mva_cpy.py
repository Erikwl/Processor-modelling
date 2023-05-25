import numpy as np

def mva(N : np.array, refs : np.array, visits : np.array, caps : np.array, service_times : np.array):
    mus = 1 / service_times
    R = len(N)
    M = len(visits)

    Rs = np.zeros((2,2), dtype=int)
    Ms = np.zeros((2,2), dtype=int)

    nr_of_states = 0

    nrs = np.zeros((M, nr_of_states))
    utils = np.zeros((M, nr_of_states))
    probs = np.zeros((M, min(np.sum(N) + 1, max(caps)), nr_of_states))
    waits = np.zeros((M, R, nr_of_states))
    throughputs = np.zeros((R, nr_of_states))

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

        for i in range(M):
            nrs[i,N_] = np.sum(visits[i,r] * throughputs[r,N_] * waits[i,r,N_] for r in Rs[i])
            utils[i,N_] = 1 / mus[i] * np.sum(throughputs[r,N_] * visits[i,r] for r in Rs[i])

            ci = caps[i]
            for n in range(1, 1 + min(ci - 1, totalN_)):
                probs[i,n,N_] = 1 / (min(n, ci) * mus[i]) \
                    * np.sum(visits[i,r] * throughputs[r,N_] * probs[i,n - 1,N_ - N_products[r]]
                             for r in Rs[i])
                r = 1
            probs[i,0,N_] = 1 - 1 / ci * (
                utils[i,N_] + np.sum((ci - n) * probs[i,n,N_] for n in range(1,1+min(ci-1,totalN_)))
            )

    N_ = nr_of_states - 1
    return [nrs[:,N_], waits[:,:,N_], throughputs[:,N_], utils[:,N_], probs[:,:,N_]]
