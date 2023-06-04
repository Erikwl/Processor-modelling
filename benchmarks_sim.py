from benchmark_params import *

def benchmark_sim_verification(file_nr, n_clusters):
    data_dict = benchmark_params(file_nr, n_clusters)
    cores = data_dict['cores']
    pops_lst = data_dict['pops_lst']
    caps_lst = data_dict['caps_lst']
    service_times_lst = data_dict['service_times_lst']

    data_dict = split_throughputs_latency(file_nr, n_clusters)
    split_intervals = data_dict['split_intervals']
    split_throughputs = data_dict['split_throughputs']
    split_avg_latency = data_dict['split_avg_latency']

    args = model(len(cores))

    # cur_time = {core : 0 for core in cores}
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

def parallel_benchmarks_sim(file_nrs, n_clusters):
    data_dict = {}
    cluster_centers = {}
    for file_nr in file_nrs:
        data_dict[file_nr] = split_throughputs_latency(file_nr, n_clusters)
        for core in data_dict[file_nr]['cores']:
            cluster_centers[file_nr] = data_dict[file_nr]['kmeans'][core].cluster_centers_

    

if __name__ == '__main__':
    benchmarks = ['parsec-bodytrack', 'parsec-blackscholes']
    parallellisms = [2, 3]
    file_nrs = []
    for benchmark, parallellism in zip(benchmarks, parallellisms):
        file_nrs.extend([DATA_FILES[benchmark][1]] * parallellism)
    n_clusters = 6
    # parallel_benchmarks_sim(file_nrs, n_clusters)
