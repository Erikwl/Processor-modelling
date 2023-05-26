from mva import mva
import numpy as np
from scipy import optimize
from constants import *
from main import model

# def subs_service_times(cur, core_ids, replacements):
#     for id, repl in zip(core_ids, replacements):
#         cur
    # print(cur, core_ids, new_service_times)
    # service_times = []
    # for core0_id in range(len(cur)):
    #     if core0_id in core_ids:
    #         service_times.append(new_service_times[core_ids.index(core0_id)])
    #     else:
    #         service_times.append(cur[core0_id])
    # # print(service_times)

    # return np.array(service_times)

def find_cores_service_times(args, core_ids, num_dram_requests, time):
    """All units are in microseconds. """
    def fun_root(*new_service_times):
        args[4][core_ids] = new_service_times
        mem_throughputs = mva(*args)[2][core_ids]
        return np.abs(num_dram_requests - mem_throughputs * time)
    def fun_min(*new_service_times):
        return np.sum(fun_root(new_service_times))

    # Initial guess for service time is given by:
    # service time of memory * capacity of core / capacity of memory.
    x0 = [args[4][-1] * args[3][core0_id] / args[3][-1] for core0_id in core_ids]
    bounds = [(0.001, np.inf)] * len(num_dram_requests)

    # Improvement of the guess is calculated by scipy.minimize.
    x0 = optimize.minimize(fun_min, x0, bounds=bounds, tol=1).x

    # Solution is further calculated by scipy.root.
    new_service_times = optimize.root(fun_root, x0).x
    args[4][core_ids] = new_service_times

    return args[4] if (fun_min(*new_service_times) < TOL) and np.all(new_service_times > 0) else None

def divide_proportional(vals, n):
    tot = sum(vals)
    lst = [int((val / tot) * n) for val in vals]
    remainder = n - sum(lst)

    # print(list(enumerate(vals)))
    highest_vals = sorted(enumerate(vals), reverse=True, key=lambda x : x[1])[:remainder]
    for i, _ in highest_vals:
        lst[i] += 1
    # for i, val in enumerate(vals):
    #     if remainder == 0:
    #         break
    #     if val in highest_vals:
    #         vals[i] += 1
    #         remainder -= 1
    return np.array(lst, dtype=int)

def find_all_params(num_dram_requests, waiting_times, time):
    R = len(num_dram_requests)
    M = R + 1
    def f(pops):
        args[0] = pops
        new_service_times = find_cores_service_times(args, range(R), num_dram_requests, time)
        if new_service_times is None:
            print('Something went wrong')
            exit(0)
        return mva(*args)[1][-1] - waiting_times

    args = model(R)
    args[3][:-1] = np.ones(R, dtype=int) # Capacities of cores

    total_pop = max(CAP_MEM + 1, R)
    pops = divide_proportional(waiting_times, total_pop)

    fpops = f(pops)
    if np.sum(fpops) > 0:
        return args
    best_lower_pops = pops
    best_lower_diff = np.sum(fpops)

    while True:
        for i in range(len(pops)):
            if fpops[i] == min(fpops):
                pops[i] += 1
        fpops = f(pops)
        if np.sum(fpops) > 0:
            break
        best_lower_pops = np.copy(pops)
        best_lower_diff = np.sum(fpops)

    if np.abs(best_lower_diff) > np.abs(np.sum(fpops)):
        return args
    args[0] = best_lower_pops
    return args


if __name__ == '__main__':
    num_dram_requests = np.array([5, 5])
    waiting_times = np.array([65, 69])
    time = 100
    print(find_all_params(num_dram_requests, waiting_times, time))

