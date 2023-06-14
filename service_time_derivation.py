from mva_c_version import mva
import numpy as np
from scipy import optimize
from constants import *
from main import model

def find_cores_service_times(args, core_ids, throughs):
    """All units are in microseconds. """
    def fun_root(*service_times):
        args_cpy = args.copy()
        args_cpy[4][core_ids] = service_times
        mem_throughputs = mva(*args_cpy)[2][core_ids]
        return np.abs(throughs - mem_throughputs)
    def fun_min(*service_times):
        return np.sum(fun_root(service_times))

    # Initial guess for service time is given by:
    # service time of memory * capacity of core / capacity of memory.
    x0 = [args[4][-1] * args[3][core0_id] / args[3][-1] for core0_id in core_ids]
    bounds = [(0.00001, np.inf)] * len(throughs)

    # Improvement of the guess is calculated by scipy.minimize.
    x1 = optimize.minimize(fun_min, x0, bounds=bounds).x

    # Solution is further calculated by scipy.root.
    service_times = optimize.root(fun_root, x1).x

    neg_service_times = [i for i in range(len(core_ids)) if service_times[i] < 0]
    return neg_service_times, service_times

def divide_proportional(vals, n):
    tot = sum(vals)
    lst = [int((val / tot) * n) for val in vals]
    remainder = n - sum(lst)

    # print(list(enumerate(vals)))
    highest_vals = sorted(enumerate(vals), reverse=True, key=lambda x : x[1])[:remainder]
    for i, _ in highest_vals:
        lst[i] += 1
    return np.array(lst, dtype=int)

def find_all_params(throughs, other_execution_data, other_data_name):
    R = len(throughs)
    M = R + 1
    tol = 0.5 if other_data_name == 'avg_latency' else 0.05

    def f(pops):
        args_cpy = args.copy()
        args_cpy[0] = pops
        neg_service_times, service_times = find_cores_service_times(args_cpy, range(R), throughs)
        args_cpy[4][:-1] = service_times
        waits, model_throughs = mva(*args_cpy)[1:3]
        nrs = np.multiply(model_throughs, waits[-1])
        if other_data_name == 'avg_latency':
            return neg_service_times, service_times, waits[-1] - other_execution_data
        return neg_service_times, service_times, nrs - other_execution_data

    args = model(R)
    args[3][:-1] = np.ones(R, dtype=int) # Capacities of cores

    pops = np.zeros(len(throughs), dtype=int)
    for i, through in enumerate(throughs):
        if through > 10e-7:
            pops[i] += 1

    neg_service_times, service_times, diff = f(pops)
    if not neg_service_times and np.sum(diff) > 0:
        args[0] = pops
        args[4][:-1] = service_times
        return args

    best_lower_pops = np.copy(pops)
    best_lower_diff = np.sum(diff)
    best_lower_service_times = np.copy(service_times)
    best_lower_neg_service_times = np.copy(neg_service_times)

    while True:
        if neg_service_times:
            for core in neg_service_times:
                pops[core] += 1
        else:
            for i in range(len(pops)):
                if diff[i] == min(diff):
                    pops[i] += 1
        old_diff = np.copy(diff)
        neg_service_times, service_times, diff = f(pops)

        if not neg_service_times \
            and (np.sum(abs(old_diff - diff)) < tol / 3 or max(pops) == MAX_POP_SIZE) \
            and (np.sum(pops) >= CAP_MEM + 2):
            break
        if not neg_service_times and np.sum(diff) > - tol * len(throughs):
            break
        best_lower_pops = np.copy(pops)
        best_lower_diff = np.sum(diff)
        best_lower_service_times = np.copy(service_times)
        best_lower_neg_service_times = np.copy(neg_service_times)

    if not best_lower_neg_service_times and np.abs(best_lower_diff) < np.abs(np.sum(diff)):
        args[0] = best_lower_pops
        args[4][:-1] = best_lower_service_times
        return args

    args[0] = pops
    args[4][:-1] = service_times
    return args


if __name__ == '__main__':
    num_dram_requests = np.array([5, 5])
    waiting_times = np.array([65, 69])
    time = 100
    print(find_all_params(num_dram_requests / time, waiting_times))

