from mva import mva
import numpy as np
from scipy import optimize
from constants import *


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
    def fun(*new_service_times):
        args[4][core_ids] = new_service_times
        mem_throughputs = mva(*args)[2][core_ids]
        return np.sum(np.abs(num_dram_requests - mem_throughputs * time))

    # Initial guess for service time is given by:
    # service time of memory * capacity of core / capacity of memory.
    x0 = [args[4][-1] * args[3][core0_id] / args[3][-1] for core0_id in core_ids]

    new_service_times = optimize.root(fun, x0).x
    args[4][core_ids] = new_service_times
    return args[4] if (fun(*new_service_times) < TOL) and new_service_times > 0 else None








