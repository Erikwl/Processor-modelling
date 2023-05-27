from benchmark_params import *

def benchmark_sim_verification(file_nr):
    data_dict = determine_params_of_benchmark(file_nr)
    stepsize = 1000

    print(data_dict.keys())
    cores = data_dict['cores']
    split_intervals = data_dict['split_intervals']
    throughputs = data_dict['throughputs']
    split_throughputs = data_dict['split_throughputs']
    avg_latency = data_dict['split_avg_latency']
    split_avg_latency = data_dict['split_avg_latency']
    pops_lst = data_dict['pops_lst']
    caps_lst = data_dict['caps_lst']
    service_times_lst = data_dict['service_times_lst']

    args = model(len(cores))

    cur_time = {core : 0 for core in cores}
    start = split_intervals[0]
    for i, end in enumerate(split_intervals[1:]):
        args[0] = np.array(pops_lst[i], dtype=int)
        args[3][:-1] = np.array(caps_lst[i], dtype=int)
        args[4][:-1] = service_times_lst[i]
        _, waits, throughs, _, _ = mva(*args)
        # print(cur_time)
        # for j, core in enumerate(cores):
        #     total_requests = split_throughputs[core][i] * stepsize * (end - start)
        #     print(core, total_requests)
        #     if throughs[j] > 0:

        #         cur_time[core] += total_requests / throughs[j]
        for j, core in enumerate(cores):
            print(f'{core = } on interval {start}-{end}')
            print(f'{throughs[j] = }')
            print(f'{split_throughputs[core][i] = }')
            print(f'{waits[-1][j] = }')
            print(f'{split_avg_latency[core][i] = }\n')
        start = end

def parallel_benchmarks_sim(file_nrs):
    data_dict = {}
    for file_nr in file_nrs:
        data_dict[file_nr] = determine_params_of_benchmark(file_nr)




if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    file_nr = DATA_FILES[benchmark][1]
    benchmark_sim_verification(file_nr)
