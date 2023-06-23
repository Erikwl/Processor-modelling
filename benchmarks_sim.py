import matplotlib.pyplot as plt
import numpy as np
from constants import *
from benchmark_params import *
import multiprocessing

PLOT_PARALLELLISM = [1, 2, 3]
COMPARE_PARALLELLISMS = [1, 2]


# def benchmark_preprocessing(file_nr, n_clusters, stepsize=None):
#     if stepsize == None:
#         stepsize = STEPSIZE
#     data_filename = f'data/benchmark_preprocessing_{file_nr}_{n_clusters}_{stepsize}_{START_TIME}-{END_TIME}.npy'

#     if os.path.exists(data_filename):
#         print(f'Preprocessing of benchmark has already been done.')
#         return np.load(data_filename, allow_pickle=True)[()]

#     other_data_name = 'avg_count'
#     combine_cores = True

#     # used_cores = np.zeros(len(file_nrs))

#     data_dict = split_throughputs(file_nr, n_clusters, combine_cores, other_data_name, stepsize=stepsize)
#     end_time = data_dict['end_time']
#     timestamps = np.append(data_dict['timestamps'], [end_time])
#     permutated_indices = data_dict['permutated_indices']
#     split_intervals = data_dict['split_intervals']
#     predicted_clusters = data_dict['predicted_clusters'].flatten()
#     model_throughputs_lst = data_dict['split_model_throughputs']
#     pops_lst = data_dict['pops_lst']
#     # caps_lst = data_dict['caps_lst']
#     service_times_lst = data_dict['service_times_lst']

#     # model_throughputs = np.zeros(len(timestamps))
#     # pops = np.zeros(len(pops_lst))
#     # caps = np.zeros(len(timestamps))
#     # service_times = np.zeros(len(timestamps))

#     low = 0
#     for j, high in enumerate(split_intervals[1:]):
#         # print(f, pops, pops_lst)
#         pops[permutated_indices[low:high]] = pops_lst[0][j]
#         # caps[permutated_indices[low:high]] = caps_lst[0][j]
#         service_times[permutated_indices[low:high]] = service_times_lst[0][j]
#         model_throughputs[permutated_indices[low:high]] = model_throughputs_lst[0][j]
#         low = high

#     data_dict = {'end_time' : end_time,
#                  'timestamps' : timestamps,
#                  'predicted_clusters' : predicted_clusters,
#                  'model_throughputs' : model_throughputs,
#                  'pops' : pops,
#                 #  'caps' : caps,
#                  'service_times' : service_times}
#     np.save(data_filename, data_dict)

#     print('Parallell benchmark preprocessing has been retrieved.')
#     return data_dict


def combined_split_throughputs_data(file_nr, n_clusters, stepsize):
    data_filename = f'data/combined_split_throughputs_{file_nr}_{n_clusters}_{stepsize}.npy'

    if os.path.exists(data_filename):
        print('Combined split throughputs data has already been calculated.')
        return np.load(data_filename, allow_pickle=True)[()]

    other_data_name = 'avg_count'
    combine_cores = True
    data_dict = split_throughputs(file_nr, n_clusters, combine_cores, other_data_name, stepsize=stepsize)
    end_time = data_dict['end_time']
    predicted_clusters = data_dict['predicted_clusters']
    timestamps = data_dict['timestamps']
    combined_indices = [0]
    predicted_clusters[-1] = 1
    cur_cluster = predicted_clusters[0]
    for i, predicted_cluster in enumerate(predicted_clusters):
        if predicted_cluster == cur_cluster:
            continue
        combined_indices.append(i)
        cur_cluster = predicted_cluster

    timestamps = np.append(timestamps[combined_indices], [end_time])
    predicted_clusters = predicted_clusters[combined_indices]
    split_model_throughputs = data_dict['split_model_throughputs'][0]

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'predicted_clusters' : predicted_clusters,
                 'split_model_throughputs' : split_model_throughputs,
                 'pops_lst' : data_dict['pops_lst'][0],
                 'service_times_lst' : data_dict['service_times_lst'][0]}

    np.save(data_filename, data_dict)
    print('Combined split throughputs data has been retrieved.')
    return data_dict

def parallel_benchmarks_sim(file_nrs, n_clusters, stepsize):
    data_filename = f'data/parallel_benchmarks_sim-{file_nrs}_{n_clusters}_{stepsize}.npy'

    if os.path.exists(data_filename):
        print('Parallel benchmarks simulation data has already been done.')
        return np.load(data_filename, allow_pickle=True)[()]

    timestamps = []
    end_times = []
    predicted_clusters = []
    model_throughputs_lst = []
    pops_lst = []
    # caps = []
    service_times_lst = []

    for file_nr in file_nrs:
        data_dict = combined_split_throughputs_data(file_nr, n_clusters, stepsize=stepsize)
        end_times.append(data_dict['end_time'])
        timestamps.append(data_dict['timestamps'])
        predicted_clusters.append(data_dict['predicted_clusters'])
        model_throughputs_lst.append(data_dict['split_model_throughputs'])
        pops_lst.append(data_dict['pops_lst'])
        # caps.append(data_dict['caps'])
        service_times_lst.append(data_dict['service_times_lst'])

    mva_cache = {}
    real_time = 0

    benchmark_response_times = {}
    executing_benchmarks = list(range(len(file_nrs)))
    current_benchmark_time_indices = np.zeros(len(file_nrs), dtype=int)
    current_predicted_clusters = np.array([predicted_clusters[f][0] for f in range(len(file_nrs))], dtype=np.int8)

    remaining_benchmark_times = np.array([timestamps[f][1] for f in range(len(file_nrs))])

    args = model(len(executing_benchmarks))
    args[0] = np.array([pops_lst[f][cluster] for f, cluster in zip(executing_benchmarks, current_predicted_clusters)], dtype=int)
    # args[3][:-1] = np.array([caps[f][current_benchmark_time_indices[f]] for f in executing_benchmarks])
    args[4][:-1] = np.array([service_times_lst[f][cluster] for f, cluster in zip(executing_benchmarks, current_predicted_clusters)])
    throughputs = mva(*args)[2]
    mva_cache[tuple(current_predicted_clusters)] = throughputs

    remaining_nr_requests = np.multiply(remaining_benchmark_times,
                                        [model_throughputs_lst[f][cluster] for f, cluster in zip(executing_benchmarks, current_predicted_clusters)])

    i = 0
    while len(executing_benchmarks) != 0:
        if tuple(current_predicted_clusters) in mva_cache:
            throughputs = mva_cache[tuple(current_predicted_clusters)]
        else:
            # print(current_benchmark_time_indices)
            args[0][executing_benchmarks] = np.array([pops_lst[f][current_predicted_clusters[f]] for f in executing_benchmarks], dtype=int)
            # args[3][executing_benchmarks] = np.array([caps[f][current_benchmark_time_indices[f]] for f in executing_benchmarks])
            args[4][executing_benchmarks] = np.array([service_times_lst[f][current_predicted_clusters[f]] for f in executing_benchmarks])
            throughputs = mva(*args)[2]
            mva_cache[tuple(current_predicted_clusters)] = throughputs

        required_times = np.divide(remaining_nr_requests, throughputs,
                                    out=np.copy(remaining_benchmark_times),
                                    where=throughputs!=0)
        min_required_time = np.min(required_times[executing_benchmarks])
        real_time += min_required_time

        for f in executing_benchmarks:
            required_time = required_times[f]
            # current_benchmark_time_index = current_benchmark_time_indices[f]
            if required_time == min_required_time:
                current_benchmark_time_indices[f] += 1
                cur_time = timestamps[f][current_benchmark_time_indices[f]]

                # Check if benchmark has finished
                if cur_time == end_times[f]:
                    executing_benchmarks.remove(f)
                    current_predicted_clusters[f] = -1
                    args[0][f] = 0
                    benchmark_response_times[f] = real_time
                else:
                    current_predicted_clusters[f] = predicted_clusters[f][current_benchmark_time_indices[f]]
                    # if current_benchmark_time_index >= 9596880:
                    #     print(current_benchmark_time_indices, cur_time, end_times)
                    next_time = timestamps[f][current_benchmark_time_indices[f] + 1]
                    remaining_nr_requests[f] = (next_time - cur_time) * model_throughputs_lst[f][current_predicted_clusters[f]]
                    remaining_benchmark_times[f] = next_time - cur_time
            else:
                # if real_time < 10000:
                #     print(f)
                remaining_nr_requests[f] -= min_required_time * throughputs[f]
                remaining_benchmark_times[f] -= min_required_time * throughputs[f] / model_throughputs_lst[f][current_predicted_clusters[f]]
        # if real_time < 10000:
        #     print(real_time, required_times, current_predicted_clusters)
        # if np.abs(real_time - 24153569.38919499) < 5000:
        #     print(f'after loop:\n{min_required_time = }\n{remaining_nr_requests = }\n{remaining_benchmark_times = }\n')

    data_dict = {'benchmark_response_times' : benchmark_response_times}
    np.save(data_filename, data_dict)

    print('Parallel benchmarks simulation data has been saved.')

    return data_dict
    # print(real_time, benchmark_response_times)

def one_benchmark_parallellism_accuracy(benchmark, n_clusters, stepsize):
    modelled_response_times = []
    real_response_times = []
    for parallellism in PLOT_PARALLELLISM:
        file_nrs = DATA_FILES[benchmark][1][:parallellism]
        file_nr = DATA_FILES[benchmark][parallellism][0]
        end_time = retrieve_end_time(file_nr)['end_time']
        real_response_times.append(end_time)
        benchmark_response_times = parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=stepsize)['benchmark_response_times']
        print(benchmark_response_times)
        modelled_response_times.append(max(benchmark_response_times.values()))

    return real_response_times, modelled_response_times

def plot_different_n_clusters(benchmark, n_clusters_lst, stepsize):
    reals = []
    models = []
    for n_clusters in n_clusters_lst:
        print(f'{n_clusters = }')
        real, model = one_benchmark_parallellism_accuracy(benchmark, n_clusters, stepsize=stepsize)
        reals.append(real)
        models.append(model)

    fig = plt.figure(figsize=(4,3.5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(PLOT_PARALLELLISM, reals[0], s=100, marker=MARKERS[0], label='measured data')
    for n_clusters, model, marker in zip(n_clusters_lst, models, MARKERS[1:len(stepsizes) + 1]):
        ax.scatter(PLOT_PARALLELLISM, model, s=100, marker=marker, label=f' n_clusters = {n_clusters}')
    ax.set_xlabel('parallellism')
    ax.set_ylabel('execution time (ns)')
    fig.subplots_adjust(left=0.16, bottom=0.12, right=0.95, top=0.95)
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
    fig = plt.figure(figsize=(4,3.5), dpi=150)

    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(PLOT_PARALLELLISM, reals[0], label='real data')
    ax.scatter(PLOT_PARALLELLISM, reals[0], s=100, marker=MARKERS[0], label='measured data')

    # markers = ['']
    for stepsize, model, marker in zip(stepsizes, models, MARKERS[1:len(stepsizes) + 1]):
        # ax.plot(PLOT_PARALLELLISM, model, label=f'stepsize = {stepsize}')
        ax.scatter(PLOT_PARALLELLISM, model, s=100, marker=marker, label=f'stepsize = {stepsize} ns')
    # ax.set_xlabel('parallellism')
    ax.set_xlabel('parallellism')

    # ax.set_ylabel('execution time (ns)')
    ax.set_ylabel('execution time (ns)')
    ax.legend()
    fig.subplots_adjust(left=0.16, bottom=0.12, right=0.95, top=0.95)
    # fig.tight_layout()
    fig.savefig(f'pictures/one_benchmark_parallellism/one_benchmark_parallellism_{benchmark}_{n_clusters}_{stepsizes}_{START_TIME}-{int(END_TIME)}')

def plot_different_benchmarks(extra_benchmark):
    stepsize = 100
    n_clusters = 8

    # extra_benchmark = 'parsec-streamcluster'

    low_dram_intensity_benchmarks = ['parsec-blackscholes', 'parsec-fluidanimate', 'parsec-swaptions']
    high_dram_intensity_benchmarks = ['parsec-bodytrack', 'parsec-streamcluster', 'parsec-dedup']

    for benchmark in low_dram_intensity_benchmarks:
        print(f'{benchmark = }')
        for parallellism in COMPARE_PARALLELLISMS:
            used_cores = np.arange(NUM_CORES_USED[benchmark] * parallellism)

            real_file_nr = DATA_FILES[benchmark + ',' + extra_benchmark][(parallellism, 2)][0]
            real_end_time = max(retrieve_end_time(real_file_nr)['end_times'][i] for i in used_cores)

            file_nrs = [DATA_FILES[benchmark][1][i] for i in range(parallellism)]
            file_nrs.extend([DATA_FILES[extra_benchmark][1][0], DATA_FILES[extra_benchmark][1][1]])
            benchmark_response_times = parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=stepsize)['benchmark_response_times']
            model_end_time = max(benchmark_response_times[i] for i in range(parallellism))

            trivial_guess = retrieve_end_time(DATA_FILES[benchmark][1][0])['end_time']
            real_time_increase = real_end_time - trivial_guess
            model_time_increase = model_end_time - trivial_guess
            real_percentage_time_increase = real_time_increase / trivial_guess
            model_percentage_time_increase = model_time_increase / trivial_guess
            print(f'{parallellism} x {benchmark}, 2 x {extra_benchmark}')
            print(f'{real_end_time = }')
            print(f'{model_end_time = }')
            print(f'{trivial_guess = }')
            print(f'{real_time_increase = }')
            print(f'{model_time_increase = }')
            print(f'{real_percentage_time_increase = }')
            print(f'{model_percentage_time_increase = }\n')
        print('\n')

    print('\n\n\n-------------------------------------------------------\n\n\n')

    for benchmark in high_dram_intensity_benchmarks:
        print(f'{benchmark = }')
        # for parallellism in [2]:
        real_file_nr = DATA_FILES[benchmark][parallellism][0]
        real_end_time = retrieve_end_time(real_file_nr)['end_time']

        file_nrs = [DATA_FILES[benchmark][1][0],
                    DATA_FILES[benchmark][1][1]]

        if benchmark == 'parsec-dedup':
            benchmark_response_times = parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=200)['benchmark_response_times']
        else:
            benchmark_response_times = parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=stepsize)['benchmark_response_times']
        model_end_time = max(benchmark_response_times.values())

        trivial_guess = retrieve_end_time(DATA_FILES[benchmark][1][0])['end_time']
        real_time_increase = real_end_time - trivial_guess
        model_time_increase = model_end_time - trivial_guess
        real_percentage_time_increase = real_time_increase / trivial_guess
        model_percentage_time_increase = model_time_increase / trivial_guess

        print(f'{parallellism} x {benchmark}')
        print(f'{real_end_time = }')
        print(f'{model_end_time = }')
        print(f'{trivial_guess = }')
        print(f'{real_time_increase = }')
        print(f'{model_time_increase = }')
        print(f'{real_percentage_time_increase = }')
        print(f'{model_percentage_time_increase = }\n')
        print('\n')

    print('\n\n\n-------------------------------------------------------\n\n\n')

    used_cores = np.arange(NUM_CORES_USED['parsec-bodytrack'] * 2)

    real_file_nr = DATA_FILES['parsec-bodytrack,parsec-streamcluster'][(2, 2)][0]
    print(retrieve_end_time(real_file_nr)['end_times'])
    real_end_time = max(retrieve_end_time(real_file_nr)['end_times'][i] for i in used_cores)

    file_nrs = [DATA_FILES['parsec-bodytrack'][1][0],
                DATA_FILES['parsec-bodytrack'][1][1],
                DATA_FILES['parsec-streamcluster'][1][0],
                DATA_FILES['parsec-streamcluster'][1][1]]
    benchmark_response_times = parallel_benchmarks_sim(file_nrs, n_clusters, stepsize=stepsize)['benchmark_response_times']
    model_end_time = max(benchmark_response_times[i] for i in range(parallellism))

    trivial_guess = retrieve_end_time(DATA_FILES['parsec-bodytrack'][1][0])['end_time']
    real_time_increase = real_end_time - trivial_guess
    model_time_increase = model_end_time - trivial_guess
    real_percentage_time_increase = real_time_increase / trivial_guess
    model_percentage_time_increase = model_time_increase / trivial_guess

    print(f'2 x parsec-bodytrack, 2 x parsec-streamcluster')
    print(f'{real_end_time = }')
    print(f'{model_end_time = }')
    print(f'{trivial_guess = }')
    print(f'{real_time_increase = }')
    print(f'{model_time_increase = }')
    print(f'{real_percentage_time_increase = }')
    print(f'{model_percentage_time_increase = }\n')


if __name__ == '__main__':
    benchmarks = [
        'parsec-bodytrack',
        'parsec-streamcluster'
        ]

    n_clusters = 8
    stepsize = 1000
    n_clusters_lst = [4, 6, 8, 10, 12]
    stepsizes = [100, 200, 500, 1000, 10000]
    for benchmark in benchmarks:
        plot_different_stepsizes(benchmark, n_clusters, stepsizes)
        plot_different_n_clusters(benchmark, n_clusters_lst, stepsize)

    extra_benchmark = 'parsec-streamcluster'
    # plot_different_benchmarks(extra_benchmark=extra_benchmark)
    # print(retrieve_end_time(235))
    # print(retrieve_end_time(234))

    # data = retrieve_data(226, combine_cores=False)
    # print(data[:30])
    # print(retrieve_end_time(226))

    # for file_nr in [226, 227, 206, 207, 208, 213, 214, 215]:
    #     print(retrieve_end_time(file_nr=file_nr)['end_time'])

    # benchmarks = NUM_CORES_USED.keys()
    # for benchmark in benchmarks:
    #     file_nrs = DATA_FILES[benchmark][1]
    #     used_cores = NUM_CORES_USED[benchmark]
    #     print(f'{benchmark = }')

    #     fst = retrieve_end_time(file_nrs[0])['end_time']
    #     snd = retrieve_end_time(file_nrs[1])['end_time']
    #     print(abs((fst - snd) / snd) * 100)