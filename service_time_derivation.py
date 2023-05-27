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

def find_cores_service_times(args, core_ids, throughs):
    """All units are in microseconds. """
    def fun_root(*new_service_times):
        args[4][core_ids] = new_service_times
        mem_throughputs = mva(*args)[2][core_ids]
        return np.abs(throughs - mem_throughputs)
    def fun_min(*new_service_times):
        return np.sum(fun_root(new_service_times))

    # Initial guess for service time is given by:
    # service time of memory * capacity of core / capacity of memory.
    x0 = [args[4][-1] * args[3][core0_id] / args[3][-1] for core0_id in core_ids]
    bounds = [(0.00001, np.inf)] * len(throughs)

    # print(f'{x0 = }')
    # Improvement of the guess is calculated by scipy.minimize.
    x1 = optimize.minimize(fun_min, x0, bounds=bounds).x
    # print(f'{x1 = }')
    # print(f'{fun_min(x1) = }')
    # throughputs = mva(*args)[2]
    # for i, x in zip(core_ids, x0):
    #     if x == 0.00001:
    #         print(f'{throughputs = }, {throughs= }')
    #         print(f'Warning: throughput for core {i} is too high and could not be reached:')
    #         print(f'Desired throughput: {throughs[i]:.4f}')
    #         print(f'Gotten throughput: {throughputs[i]}')
    #         args[4][core_ids] = x0
    #         return 'too high', args[4]

    # Solution is further calculated by scipy.root.
    new_service_times = optimize.root(fun_root, x1).x
    if (fun_min(*new_service_times) > TOL) or np.any(new_service_times < 0):
        throughputs = mva(*args)[2]
        args[4][core_ids] = x1
        # print(f'\nWarning: throughput could not be reached:')
        # print(f'{x1 = }')
        # print(f'{new_service_times = }')
        # print(f'Pop = {args[0]}')
        # print(f'desired: {throughs}, achieved: {throughputs}\n')
        return 'too high', args[4]
    args[4][core_ids] = new_service_times

    return 'correct', args[4]
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

def find_all_params(throughs, waiting_times):
    R = len(throughs)
    M = R + 1
    def f(pops):
        args[0] = pops
        error, new_service_times = find_cores_service_times(args, range(R), throughs)
        # if error == 'too high':
        #     print('Something went wrong')
        #     exit(0)
        return mva(*args)[1][-1] - waiting_times

    args = model(R)
    args[3][:-1] = np.ones(R, dtype=int) # Capacities of cores

    # total_pop = max(CAP_MEM, R)
    # pops = divide_proportional(waiting_times, total_pop)
    pops = np.zeros(len(throughs), dtype=int)
    for i, through in enumerate(throughs):
        if through:
            pops[i] += 1
    # print(f'pop divided: {pops = }, {waiting_times = }')

    fpops = f(pops)
    if np.sum(fpops) > 0:
        # print(pops, throughs, waiting_times, f(pops))
        return args
    best_lower_pops = pops
    best_lower_diff = np.sum(fpops)

    while True:
        # print(f'Testing population of {pops}')
        # print(f'{waiting_times = }, achieved: {waiting_times + fpops}')

        # print(pops, num_dram_requests, waiting_times, fpops)
        for i in range(len(pops)):
            if fpops[i] == min(fpops):
                pops[i] += 1
        old_fpops = fpops
        fpops = f(pops)
        if max(pops) == MAX_POP_SIZE or np.sum(abs(old_fpops - fpops)) < 0.001:
            print('Warning: Waiting time could not be reached:')
            # print(f'Desired waiting times: {waiting_times}')
            # print(f'Differences: {fpops}')
            break
        if np.sum(fpops) > -0.5 * len(throughs):
            break
        best_lower_pops = np.copy(pops)
        best_lower_diff = np.sum(fpops)


    # print('The returned parameters have a difference of:')
    # print(f'{waiting_times = }, achieved: {waiting_times + fpops}')
    if np.abs(best_lower_diff) > np.abs(np.sum(fpops)):
        return args
    args[0] = best_lower_pops
    return args


if __name__ == '__main__':
    num_dram_requests = np.array([5, 5])
    waiting_times = np.array([65, 69])
    time = 100
    print(find_all_params(num_dram_requests / time, waiting_times))

