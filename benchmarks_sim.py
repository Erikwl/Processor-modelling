import matplotlib.pyplot as plt
import numpy as np
from constants import *
from benchmark_params import *
parallellisms = np.arange(1, 4)


def benchmark_preprocessing(file_nr, n_clusters, stepsize=None):
    if stepsize == None:
        stepsize = STEPSIZE
    data_filename = f'data/benchmark_preprocessing_{file_nr}_{n_clusters}_{stepsize}_{START_TIME}-{END_TIME}.npy'

    if os.path.exists(data_filename):
        print(f'Preprocessing of benchmark has already been done.')
        return np.load(data_filename, allow_pickle=True)[()]

    other_data_name = 'avg_count'
    combine_cores = True

    # used_cores = np.zeros(len(file_nrs))

    data_dict = split_throughputs(file_nr, n_clusters, combine_cores, other_data_name, stepsize=stepsize)
    end_time = data_dict['end_time']
    timestamps = np.append(data_dict['timestamps'], [end_time])
    permutated_indices = data_dict['permutated_indices']
    split_intervals = data_dict['split_intervals']
    predicted_clusters = data_dict['predicted_clusters'].flatten()
    model_throughputs_lst = data_dict['split_model_throughputs']
    pops_lst = data_dict['pops_lst']
    caps_lst = data_dict['caps_lst']
    service_times_lst = data_dict['service_times_lst']

    model_throughputs = np.zeros(len(timestamps))
    pops = np.zeros(len(timestamps))
    caps = np.zeros(len(timestamps))
    service_times = np.zeros(len(timestamps))

    low = 0
    for j, high in enumerate(split_intervals[1:]):
        # print(f, pops, pops_lst)
        pops[permutated_indices[low:high]] = pops_lst[0][j]
        caps[permutated_indices[low:high]] = caps_lst[0][j]
        service_times[permutated_indices[low:high]] = service_times_lst[0][j]
        model_throughputs[permutated_indices[low:high]] = model_throughputs_lst[0][j]
        low = high

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'predicted_clusters' : predicted_clusters,
                 'model_throughputs' : model_throughputs,
                 'pops' : pops,
                 'caps' : caps,
                 'service_times' : service_times}
    np.save(data_filename, data_dict)

    print('Parallell benchmark preprocessing has been retrieved.')
    return data_dict


def parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=None):
    timestamps = []
    end_times = []
    predicted_clusters = []
    model_throughputs = []
    pops = []
    caps = []
    service_times = []

    for file_nr in file_nrs:
        data_dict = benchmark_preprocessing(file_nr, n_clusters, stepsize=stepsize)
        end_times.append(data_dict['end_time'])
        timestamps.append(data_dict['timestamps'])
        predicted_clusters.append(data_dict['predicted_clusters'])
        model_throughputs.append(data_dict['model_throughputs'])
        pops.append(data_dict['pops'])
        caps.append(data_dict['caps'])
        service_times.append(data_dict['service_times'])

    mva_cache = {}
    real_time = 0

    benchmark_response_times = {}
    executing_benchmarks = list(range(len(file_nrs)))
    current_benchmark_time_indices = np.zeros(len(file_nrs), dtype=int)
    current_predicted_clusters = np.array([predicted_clusters[f][0] for f in range(len(file_nrs))], dtype=np.int8)

    remaining_benchmark_times = np.array([timestamps[f][1] for f in range(len(file_nrs))])

    args = model(len(executing_benchmarks))
    args[0] = np.array([pops[f][current_benchmark_time_indices[f]] for f in executing_benchmarks], dtype=int)
    args[3][:-1] = np.array([caps[f][current_benchmark_time_indices[f]] for f in executing_benchmarks])
    args[4][:-1] = np.array([service_times[f][current_benchmark_time_indices[f]] for f in executing_benchmarks])
    throughputs = mva(*args)[2]
    mva_cache[tuple(current_predicted_clusters)] = throughputs

    remaining_nr_requests = np.multiply(remaining_benchmark_times,
                                        [model_throughputs[f][0] for f in executing_benchmarks])

    i = 0
    while len(executing_benchmarks) != 0:
        if tuple(current_predicted_clusters) in mva_cache:
            throughputs = mva_cache[tuple(current_predicted_clusters)]
        else:
            # print(current_benchmark_time_indices)
            args[0][executing_benchmarks] = np.array([pops[f][current_benchmark_time_indices[f]] for f in executing_benchmarks], dtype=int)
            args[3][executing_benchmarks] = np.array([caps[f][current_benchmark_time_indices[f]] for f in executing_benchmarks])
            args[4][executing_benchmarks] = np.array([service_times[f][current_benchmark_time_indices[f]] for f in executing_benchmarks])
            throughputs = mva(*args)[2]
            mva_cache[tuple(current_predicted_clusters)] = throughputs

        required_times = np.divide(remaining_nr_requests, throughputs,
                                    out=np.copy(remaining_benchmark_times),
                                    where=throughputs!=0)
        min_required_time = np.min(required_times[executing_benchmarks])
        real_time += min_required_time

        for f in executing_benchmarks:
            required_time = required_times[f]
            current_benchmark_time_index = current_benchmark_time_indices[f]
            if required_time == min_required_time:
                cur_time = timestamps[f][current_benchmark_time_index + 1]

                # Check if benchmark has finished
                if cur_time == end_times[f]:
                    executing_benchmarks.remove(f)
                    current_predicted_clusters[f] = -1
                    args[0][f] = 0
                    benchmark_response_times[f] = real_time
                else:
                    next_time = timestamps[f][current_benchmark_time_index + 2]
                    remaining_nr_requests[f] = (next_time - cur_time) * model_throughputs[f][current_benchmark_time_index + 1]
                    remaining_benchmark_times[f] = next_time - cur_time
                    current_predicted_clusters[f] = predicted_clusters[f][current_benchmark_time_indices[f] + 1]
                    current_benchmark_time_indices[f] += 1
            else:
                remaining_nr_requests[f] -= min_required_time * throughputs[f]
                remaining_benchmark_times[f] -= min_required_time * throughputs[f] / model_throughputs[f][current_benchmark_time_index]

        # if np.abs(real_time - 24153569.38919499) < 5000:
        #     print(f'after loop:\n{min_required_time = }\n{remaining_nr_requests = }\n{remaining_benchmark_times = }\n')

    return benchmark_response_times
    # print(real_time, benchmark_response_times)

def one_benchmark_parallellism_accuracy(benchmark, n_clusters, stepsize=None):
    modelled_response_times = []
    real_response_times = []
    for parallellism in parallellisms:
        file_nrs = DATA_FILES[benchmark][1][:parallellism]
        file_nr = DATA_FILES[benchmark][parallellism][0]
        data = retrieve_data(file_nr, combine_cores=True)
        real_response_times.append(data[-1,0] + data[-1,2])

        modelled_response_times.append(max(parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=stepsize).values()))

    return real_response_times, modelled_response_times

def plot_different_n_clusters(benchmark, n_clusters_lst, stepsize):
    reals = []
    models = []
    for n_clusters in n_clusters_lst:
        print(f'{n_clusters = }')
        real, model = one_benchmark_parallellism_accuracy(benchmark, n_clusters, stepsize=stepsize)
        reals.append(real)
        models.append(model)

    fig = plt.figure(figsize=(5,5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(parallellisms, reals[0], label='real')
    for n_clusters, model in zip(n_clusters_lst, models):
        ax.plot(parallellisms, model, label=f' n_clusters = {n_clusters}')
    ax.set_xlabel('parallellism')
    ax.set_ylabel('execution time (ns)')
    ax.legend()
    fig.savefig(f'pictures/one_benchmark_parallellism/one_benchmark_parallellism_{benchmark}_{n_clusters_lst}_{STEPSIZE}_{START_TIME}-{int(END_TIME)}')

def plot_different_stepsizes(benchmark, n_clusters, stepsizes):
    reals = []
    models = []
    n_clusters = 8
    for stepsize in stepsizes:
        print(f'{stepsize = }')
        real, model = one_benchmark_parallellism_accuracy(benchmark, n_clusters, stepsize=stepsize)
        reals.append(real)
        models.append(model)
    fig = plt.figure(figsize=(5,5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(parallellisms, reals[0], label='real')
    for stepsize, model in zip(stepsizes, models):
        ax.plot(parallellisms, model, label=f'stepsize = {stepsize}')
    ax.set_xlabel('parallellism')
    ax.set_ylabel('execution time (ns)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'pictures/one_benchmark_parallellism/one_benchmark_parallellism_{benchmark}_{n_clusters}_{stepsizes}_{START_TIME}-{int(END_TIME)}')


if __name__ == '__main__':
    benchmarks = ['parsec-bodytrack', 'parsec-streamcluster']
    n_clusters = 8
    stepsize = 100
    n_clusters_lst = [4, 6, 8, 10, 12]
    stepsizes = [100, 200, 500, 1000, 5000, 10000]
    for benchmark in benchmarks:
        plot_different_stepsizes(benchmark, n_clusters, stepsizes)
        plot_different_n_clusters(benchmark, n_clusters_lst, stepsize)

    # plt.plot([1,2,3], [4,5,6], '.', label='1')
    # plt.plot([1,2,3], [2,1,2], '.', label='2')
    # plt.legend()
    # plt.show()

