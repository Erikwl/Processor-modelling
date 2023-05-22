from mva import mva
import numpy as np
import sys
import json
from scipy import optimize
import matplotlib.pyplot as plt
from time import time as current_time

TOL = 1

# def subs_service_times(cur, core_ids, replacements):
#     for id, repl in zip(core_ids, replacements):
#         cur
    # print(cur, core_ids, new)
    # service_times = []
    # for core_id in range(len(cur)):
    #     if core_id in core_ids:
    #         service_times.append(new[core_ids.index(core_id)])
    #     else:
    #         service_times.append(cur[core_id])
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
    x0 = [args[4][-1] * args[3][core_id] / args[3][-1] for core_id in core_ids]

    new_service_times = optimize.root(fun, x0).x
    # print(new_service_times, fun(*new_service_times))
    # service_times = np.array([mem_service_time, *core_service_times])
    # mem_throughputs = mva(*args, service_times)[2]
    # error = np.sum(np.abs(mem_throughputs * time - num_dram_requests))
    # error = fun(*core_service_times) > TOL
    # return core_service_times, error
    args[4][core_ids] = new_service_times
    return args[4] if (fun(*new_service_times) < TOL) and new_service_times > 0 else None


def read_json(filename):
    try:
        file = open(filename)
    except Exception as e:
        print(e)
        print('Please run: python3 main.py <model filename>')
        exit(0)

    data = json.load(file)
    pops_vector = np.array(data["population vector"], dtype='int')
    refs = np.array(data["reference stations"], dtype='int')
    visits = np.array(data["visits"], dtype="float")
    caps = np.array(data["capacities"], dtype="int")
    service_times = np.array(data["service times"], dtype="float")
    return [pops_vector, refs, visits, caps, service_times]

def time_execution(args):
    start = current_time()
    mva(*args)
    print(f'The mva algo took {(current_time() - start) * 1000:.4f} ms')

def plot_1core_waiting_times(args, core_id, num_dram_requests, time):
    pops_lst = range(1, 46, 1)
    caps_lst = range(1, 41, 10)

    fig = plt.figure(figsize=(10,6), dpi=150)

    for cap in caps_lst:
        x_vals = []
        y_vals = []
        y2_vals = []
        for pop in pops_lst:
            args[0][core_id] = pop
            args[3][core_id] = cap
            new = find_cores_service_times(args, [core_id], [num_dram_requests], time)
            if new is None:
                continue
            args[4] = new
            # print(pop, cap, new)
            wait = mva(*args)[1][-1][core_id]
            x_vals.append(pop)
            y_vals.append(wait)

        plt.plot(x_vals, y_vals, label=f'Capacity = {cap}')
        # plt.plot(x_vals, y2_vals, label=f'nrs, C_{{{core_id}}} = {cap}')
        # plt.show()


    # ax = plt.axes(projection='3d')
    # print(x_vals, y_vals, z_vals)
    # ax.scatter3D(x_vals, y_vals, z_vals, 'gray')
    # ax.set_xlabel(rf'$N_{core_id}$')
    # ax.set_ylabel(rf'$C_{core_id}$')
    # ax.set_zlabel(rf'$w_{{{core_id},{core_id}}}$')
    plt.xlabel('population size')
    plt.ylabel('DRAM waiting time')
    plt.legend()
    # plt.title(f'num DRAM requests = {num_dram_requests}, time = {time}')
    plt.savefig(f'pictures/1core_graph_{num_dram_requests}_{time}')

def verify_model():
    args = read_json('models/1core_model.json')
    perf_measures = mva(*args)
    # print(perf_measures)
    for i, name in enumerate(['numbers', 'waits', 'throughputs', 'utils', 'probs']):
        print(f'{name}: {perf_measures[i]}')



if __name__ == '__main__':
    if len(sys.argv) == 5:
        args = read_json(sys.argv[1])
        core_id = int(sys.argv[2])
        num_dram_requests = int(sys.argv[3])
        time = int(sys.argv[4])
        plot_1core_waiting_times(args, core_id, num_dram_requests, time)
    elif sys.argv[1] == 'verification':
        verify_model()
    elif sys.argv[1] == 'timing':
        args = read_json('models/timing_model.json')
        time_execution()
