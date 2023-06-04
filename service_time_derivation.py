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
    def fun_root(*service_times):
        args_cpy = args.copy()
        args_cpy[4][core_ids] = service_times
        # print(args_cpy[4])
        # args[4][core_ids] = service_times
        mem_throughputs = mva(*args_cpy)[2][core_ids]
        # print('f')
        return np.abs(throughs - mem_throughputs)
    def fun_min(*service_times):
        return np.sum(fun_root(service_times))

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
    service_times = optimize.root(fun_root, x1).x
    # if (fun_min(*new_service_times) > TOL) or np.any(new_service_times < 0):
    #     # throughputs = mva(*args)[2]
    #     # args[4][core_ids] = x1
    #     # print(f'\nWarning: throughput could not be reached:')
    #     # print(f'{x1 = }')
    #     # print(f'{new_service_times = }')
    #     # print(f'Pop = {args[0]}')
    #     # print(f'desired: {throughs}, achieved: {throughputs}\n')
    #     print(f'too high, {new_service_times = }, {fun_min(*new_service_times) = }')
    #     return 'too high', x1
    # args[4][core_ids] = new_service_times

    # print(f'The difference is {fun_min(new_service_times)}')

    neg_service_times = [i for i in range(len(core_ids)) if service_times[i] < 0]
    # print(f'ffffff{neg_service_times = }')
    return neg_service_times, service_times

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
        args_cpy = args.copy()
        args_cpy[0] = pops
        neg_service_times, service_times = find_cores_service_times(args_cpy, range(R), throughs)
        # if error == 'too high':
        #     print('Something went wrong')
        #     exit(0)
        # args[4] = new_service_times
        args_cpy[4][:-1] = service_times
        return neg_service_times, service_times, mva(*args_cpy)[1][-1] - waiting_times

    args = model(R)
    args[3][:-1] = np.ones(R, dtype=int) # Capacities of cores

    # total_pop = max(CAP_MEM, R)
    # pops = divide_proportional(waiting_times, total_pop)
    pops = np.zeros(len(throughs), dtype=int)
    for i, through in enumerate(throughs):
        if through:
            pops[i] += 1
    # print(f'pop divided: {pops = }, {waiting_times = }')

    neg_service_times, service_times, diff = f(pops)
    if not neg_service_times and np.sum(diff) > 0:
        # print(pops, throughs, waiting_times, f(pops))
        args[0] = pops
        args[4][:-1] = service_times
        return args

    best_lower_pops = np.copy(pops)
    best_lower_diff = np.sum(diff)
    best_lower_service_times = np.copy(service_times)
    best_lower_neg_service_times = np.copy(neg_service_times)

    while True:
        # print(f'Testing population of {pops}')
        # print(f'{waiting_times = }, achieved: {waiting_times + fpops}')

        if neg_service_times:
            for core in neg_service_times:
                pops[core] += 1
        else:
            for i in range(len(pops)):
                if diff[i] == min(diff):
                    pops[i] += 1
        old_diff = np.copy(diff)
        neg_service_times, service_times, diff = f(pops)
        # print(f'{error = }, {new_service_times = }')
        # print('ggggggggggggg', neg_service_times, service_times)


        # print(f'{pops = }, {throughs = }, {neg_service_times = }\n{old_diff = }, {diff = }')
        if not neg_service_times \
            and (np.sum(abs(old_diff - diff)) < 0.2 or max(pops) == MAX_POP_SIZE) \
            and (np.sum(pops) >= CAP_MEM + 2):
            # print('Warning: Waiting time could not be reached:')
            # print(f'Desired waiting times: {waiting_times}')
            # print(f'Differences: {diff}')
            # print(f'Old difference: {old_diff}')
            # print(f'{np.sum(abs(old_diff - diff)) = }')
            # print(f'{neg_service_times = }')
            break
        if not neg_service_times and np.sum(diff) > -0.5 * len(throughs):
            break
        best_lower_pops = np.copy(pops)
        best_lower_diff = np.sum(diff)
        best_lower_service_times = np.copy(service_times)
        best_lower_neg_service_times = np.copy(neg_service_times)

    # print('The returned parameters have a difference of:')
    # print(f'{waiting_times = }, achieved: {waiting_times + fpops}')
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

