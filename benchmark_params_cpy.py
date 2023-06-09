import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from service_time_derivation import *
import os

from constants import *
from data_retrieval import *

def permutated_timestamps(file_nr, n_clusters, combine_cores, other_data_name):
    data_dict = analyse_dram_data(file_nr, combine_cores)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    throughputs = data_dict['throughputs']
    other_data = data_dict[other_data_name]

    data_filename = f'data/perm_time_{combine_cores}_{other_data_name}_{file_nr}_{n_clusters}_{STEPSIZE}_{START_TIME}-{end_time}.npy'

    if os.path.exists(data_filename):
        print('Permutated timestamps have already been calculated.')
        return np.load(data_filename, allow_pickle=True)[()]

    scalers = {core : StandardScaler() for core in cores}
    # scaled_throughputs = {core : scaler.fit_transform(throughputs[core]) for core in cores}
    # scaled_avg_latency = {core : scaler.fit_transform(avg_latency[core]) for core in cores}

    kmeans = {}
    cluster_centers = {}
    predicted_clusters = {}
    for core in cores:
        data = list(zip(throughputs[core], other_data[core]))
        scaled_data = scalers[core].fit_transform(data)
        kmeans[core] = KMeans(n_clusters=min(len(np.unique(scaled_data)), n_clusters), n_init=10).fit(scaled_data)
        predicted_clusters[core] = kmeans[core].predict(scaled_data)
        cluster_centers[core] = scalers[core].inverse_transform(kmeans[core].cluster_centers_)

    print('Clusters have been created.')
    # for core in cores:
    #     print(f'{core = }, {kmeans[core].cluster_centers_}')

    split_intervals = [0, len(timestamps)]
    # split_fitted_throughputs = {core : [] for core in cores}
    permutated_indices = np.arange(len(timestamps))
    for core in cores:

        low = split_intervals[0]
        for high in list(split_intervals[1:]):
            cur_permutated_indices = permutated_indices[low:high]
            # scaled_through = scaled_throughputs[core][cur_permutated_indices]
            # scaled_lat = scaled_avg_latency[core][cur_permutated_indices]

            # cluster_indices = kmeans[core].predict(list(zip(scaled_through, scaled_lat)))
            cluster_indices = predicted_clusters[core][cur_permutated_indices]
            # cluster_indices[through == 0] = n_clusters

            zipped_cluster_indices = np.column_stack((cur_permutated_indices, cluster_indices))
            sorted_cluster_indices = np.array(sorted(zipped_cluster_indices, key=lambda x : x[1]))

            permutated_indices[low:high] = sorted_cluster_indices[:,0]

            count = Counter(sorted_cluster_indices[:,1])

            time_index = low
            interval_index = split_intervals.index(low)

            # print(f'{low = }, {high = }')
            # print(f'{core}: {split_fitted_throughputs = }')


            # first_cluster = True
            for cluster in range(n_clusters):
                if count[cluster] == 0:
                    continue

                # if first_cluster:
                #     first_cluster = False
                # else:
                #     for prev_core in cores[:core_index]:
                #         split_fitted_throughputs[prev_core].insert(interval_index,
                #                                             split_fitted_throughputs[prev_core][interval_index - 1])

                # split_fitted_throughputs[core].append(kmeans[core].cluster_centers_[cluster][0])
                # split_fitted_throughputs[core].append()
                time_index += count[cluster]
                if time_index in split_intervals:
                    continue

                split_intervals.insert(interval_index + 1, time_index)
                interval_index += 1

            # print(count)
            # print(core, low, high, split_intervals, '\n')
            low = high
        # print(f'end, {core}: {split_fitted_throughputs = }\n')

    # print(f'{split_intervals = }')

    print('Throughputs and latencies at every interval have been calculated.')

    data_dict = {'end_time': end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'kmeans' : kmeans,
                #  'scalers' : scalers,
                 'cluster_centers' : cluster_centers,
                 'predicted_clusters' : predicted_clusters,
                 'permutated_indices' : permutated_indices,
                 'split_intervals' : split_intervals}

    np.save(data_filename, data_dict)
    print('Permutated timestamps have been retrieved.')

    return data_dict

def split_throughputs(file_nr, n_clusters, combine_cores, other_data_name):
    data_dict = permutated_timestamps(file_nr, n_clusters, combine_cores, other_data_name)
    cores = data_dict['cores']
    timestamps = data_dict['timestamps']
    end_time = data_dict['end_time']
    cluster_centers = data_dict['cluster_centers']
    predicted_clusters = data_dict['predicted_clusters']
    permutated_indices = data_dict['permutated_indices']
    split_intervals = data_dict['split_intervals']

    data_filename = f'data/split_throughs_lats_{combine_cores}_{other_data_name}_{file_nr}_{n_clusters}_{STEPSIZE}_{START_TIME}-{end_time}.npy'

    if os.path.exists(data_filename):
        print(f'Split throughputs and {other_data_name} have already been calculated')
        return np.load(data_filename, allow_pickle=True)[()]

    # data_dict = analyse_dram_data(file_nr, combine_cores)
    # other_data = data_dict[other_data_name]

    # data_dict = benchmark_params(file_nr, n_clusters)
    # pops_lst = data_dict['pops_lst']
    # caps_lst = data_dict['caps_lst']
    # service_times_lst = data_dict['service_times_lst']

    split_fitted_throughputs = {core : np.zeros(len(split_intervals) - 1) for core in cores}
    split_fitted_other_data = {core : np.zeros(len(split_intervals) - 1) for core in cores}

    pops_lst = np.zeros((len(split_intervals) - 1, len(cores)))
    caps_lst = np.zeros((len(split_intervals) - 1, len(cores)))
    service_times_lst = np.zeros((len(split_intervals) - 1, len(cores)))

    split_model_throughputs = {core : np.zeros(len(split_intervals) - 1) for core in cores}
    split_model_other_data = {core : np.zeros(len(split_intervals) - 1) for core in cores}

    low = split_intervals[0]
    for i, high in enumerate(split_intervals[1:]):

        args = model(len(cores))

        # permutated_indices_part = permutated_indices[low:high]
        for core in cores:
            # print(permutated_indices_part)
            # print(predicted_clusters[core][permutated_indices_part])
            # print(predicted_clusters[core][permutated_indices[low]].transpose())
            split_fitted_throughputs[core][i], split_fitted_other_data[core][i] = \
                cluster_centers[core][predicted_clusters[core][permutated_indices[low]]].transpose()

            # avg_count_part = avg_count[core][permutated_indices_part]
            # avg_count_part = avg_count_part[avg_count_part != 0]
            # if len(avg_count_part):
            #     split_fitted_avg_count[core][i] = np.sum(avg_count_part) / len(avg_count_part)

            # avg_latency_part = avg_latency[core][permutated_indices_part]
            # avg_latency_part = avg_latency_part[avg_latency_part != 0]
            # if len(avg_latency_part):
            #     throughputs_part = throughputs[core][permutated_indices_part]
            #     cluster_index = kmeans[core].predict(throughputs_part.reshape(-1, 1))

            #     split_fitted_throughputs[core][i] = kmeans[core].cluster_centers_[cluster_index][0]
            #     split_fitted_avg_latency[core][i] = np.sum(avg_latency_part) / len(avg_latency_part)

        mem_throughputs = np.array([split_fitted_throughputs[core][i] for core in cores])
        other_execution_data = np.array([split_fitted_other_data[core][i] for core in cores])
        args[0] = np.array(pops_lst[i], dtype=int)
        pops, _, _, caps, service_times = find_all_params(mem_throughputs, other_execution_data, other_data_name)
        pops_lst[i] = pops
        caps_lst[i] = caps[:-1]
        service_times_lst[i] = service_times[:-1]
        for j, core in enumerate(cores):
            args[0] = pops
            args[3][:-1] = np.array(caps_lst[i], dtype=int)
            args[4][:-1] = service_times_lst[i]
            waits, throughs = mva(*args)[1:3]

            split_model_throughputs[core][i] = throughs[j]
            if other_data_name == 'avg_latency':
                split_model_other_data[core][i] = waits[-1][j]
            else:
                split_model_other_data[core][i] = throughs[j] * waits[-1][j]

            print(f'{core = } on interval {low}-{high}')
            print(f'{pops = }\n{caps = }\n{service_times = }')
            print(f'{throughs[j] = }')
            print(f'{split_fitted_throughputs[core][i] = }')
            print(f'{split_model_other_data[core][i] = }')
            print(f'{split_fitted_other_data[core][i] = }\n')

        low = high

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'permutated_indices' : permutated_indices,
                 'split_intervals' : split_intervals,
                 'split_fitted_throughputs' : split_fitted_throughputs,
                 'split_fitted_other_data' : split_fitted_other_data,
                 'pops_lst' : pops_lst,
                 'caps_lst' : caps_lst,
                 'service_times_lst' : service_times_lst,
                 'split_model_throughputs' : split_model_throughputs,
                 'split_model_other_data' : split_model_other_data}

    np.save(data_filename, data_dict)

    print('Split throughputs and latency have been retrieved.')

    return data_dict


def plot_split_throughputs(file_nr, n_clusters, combine_cores, other_data_name):
    data_dict = split_throughputs(file_nr, n_clusters, combine_cores, other_data_name)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    permutated_indices = data_dict['permutated_indices']
    split_intervals = data_dict['split_intervals']
    split_fitted_throughputs = data_dict['split_fitted_throughputs']
    split_fitted_other_data = data_dict['split_fitted_other_data']
    split_model_throughputs = data_dict['split_model_throughputs']
    split_model_other_data = data_dict['split_model_other_data']

    data_dict = analyse_dram_data(file_nr, combine_cores)
    throughputs = data_dict['throughputs']
    other_data = data_dict[other_data_name]


    ylabels = ['throughput', other_data_name]
    real_vals_lst = [throughputs, other_data]
    fitted_vals_lst = [split_fitted_throughputs, split_fitted_other_data]
    model_vals_lst = [split_model_throughputs, split_model_other_data]
    names = ['fitted_throughput', 'fitted_access_latency']

    subplot_nr = 0
    fig = plt.figure(figsize=(5,5), dpi=150)
    for ylabel, real_vals, fitted_vals, model_vals, name in zip(ylabels, real_vals_lst, fitted_vals_lst, model_vals_lst, names):

        for i, core in enumerate(cores):
            ax = fig.add_subplot(2, len(cores), i + 1 + subplot_nr)
            ax.vlines(x=timestamps, ymin=np.zeros(len(timestamps)), ymax=real_vals[core][permutated_indices], alpha=0.5)
            low = split_intervals[0]
            for fitted_y, model_y, high in zip(fitted_vals[core], model_vals[core], split_intervals[1:]):
                xmax = end_time if high == len(timestamps) else timestamps[high]
                labels = ['desired', 'model'] if low == split_intervals[0] else ['', '']
                ax.hlines(y=fitted_y, xmin=timestamps[low], xmax=xmax, linestyles='-', color='r', label=labels[0])
                ax.hlines(y=model_y, xmin=timestamps[low], xmax=xmax, linestyles='-', color='black', label=labels[1])
                low = high

            if subplot_nr == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('permuted time (ns)')
            if subplot_nr == 0:
                ax.set_title(f'Core {int(core)}')

            # ax.set_yscale('log')
            # ax.set_ylim([min(vals[vals != 0]), 2 * max(vals)])
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            ax.legend()
        subplot_nr += len(cores)

    fig.savefig(f'pictures/perm_time/perm_time_{combine_cores}_{other_data_name}_{file_nr}_{n_clusters}_{STEPSIZE}_{START_TIME}-{end_time}')

# def benchmark_params(file_nr, n_clusters):
#     data_dict = split_throughputs_latency(file_nr, n_clusters)
#     end_time = data_dict['end_time']
#     timestamps = data_dict['timestamps']
#     cores = data_dict['cores']
#     split_intervals = data_dict['split_intervals']
#     split_fitted_throughputs = data_dict['split_fitted_throughputs']
#     split_fitted_avg_latency = data_dict['split_fitted_avg_latency']

#     # data_filename = f'data/params_{file_nr}_{n_clusters}_{STEPSIZE}_{START_TIME}-{end_time}.npy'

#     # if os.path.exists(data_filename):
#     #     print('Params have already been calculated')
#     #     return np.load(data_filename, allow_pickle=True)[()]


#     # data_dict = analyse_dram_data(data, stepsize, start_time=start_time, end_time=end_time, file_nr=file_nr)
#     # avg_latency = data_dict['avg_latency']
#     # print('DRAM data has been retrieved.')

#     # data_dict = permutated_timestamps(data, stepsize, start_time, end_time, n_clusters, file_nr)

#     # cores = data_dict['cores']
#     # permutated_indices = data_dict['permutated_indices']
#     # split_intervals = data_dict['split_intervals']
#     # throughputs = data_dict['throughputs']
#     # split_fitted_throughputs = data_dict['split_fitted_throughputs']

#     # print('Timestamps permutation data has been retrieved.')

#     # # permutated_latency = {core : avg_latency[core][permutated_indices] for core in cores}
#     # split_fitted_avg_latency = {core : np.zeros(len(split_fitted_throughputs[core])) for core in cores}

#     # active_cores_lst = []
#     pops_lst = np.zeros((len(split_intervals) - 1, len(cores)))
#     caps_lst = np.zeros((len(split_intervals) - 1, len(cores)))
#     service_times_lst = np.zeros((len(split_intervals) - 1, len(cores)))

#     # for i, (through, lat) in enumerate(zip(throughputs, avg_latency)):
#     #     if through != 0 and lat == 0:
#     #         print(i, through, lat)

#     # print(split_intervals)
#     # start = split_intervals[0]

#     for i in range(len(split_intervals) - 1):
#         # active_cores = [core for core in cores if split_fitted_throughputs[core][i]]
#         # active_cores_lst.append(active_cores)
#         # print(start, end)

#         # for core in cores:
#         #     lat_part = avg_latency[core][permutated_indices[start:end]]
#         #     # print(len(permutated_indices[core]))
#         #     lat_part = lat_part[lat_part != 0]
#         #     if len(lat_part):
#         #         split_fitted_avg_latency[core][i] = np.sum(lat_part) / len(lat_part)
#         #     # if i == len(split_intervals) - 2:
#         #     #     print(start, end, lat_part, np.sum(lat_part) / len(lat_part))
#         #     # print(core, split_fitted_avg_latency[core][i])
#         #     # print(core, split_fitted_throughputs[core][i])
#         mem_throughputs = [split_fitted_throughputs[core][i] for core in cores]
#         waiting_times = [split_fitted_avg_latency[core][i] for core in cores]
#         # print(timestamps[start], timestamps[end], throughs, waiting_times)
#         pops, _, _, caps, service_times = find_all_params(mem_throughputs, waiting_times)
#         # print(np.shape(pops_lst[core]))
#         # print(mem_throughputs, waiting_times)
#         pops_lst[i] = pops
#         caps_lst[i] = caps[:-1]
#         service_times_lst[i] = service_times[:-1]

#         # start = end

#     # print(split_fitted_throughputs)
#     # print(split_fitted_avg_latency)

#     data_dict = {'end_time' : end_time,
#                  'timestamps' : timestamps,
#                  'cores' : cores,
#                  'pops_lst' : pops_lst,
#                  'caps_lst' : caps_lst,
#                  'service_times_lst' : service_times_lst}
#     # np.save(data_filename, data_dict)
#     print('Benchmarks parameters have been retrieved.')
#     return data_dict


if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    file_nr = DATA_FILES[benchmark][1]
    n_clusters = 15
    combine_cores = True
    # benchmark_params(file_nr, n_clusters)
    # STEPSIZE = 1_000
    # START_TIME = 0
    # END_TIME = 1_000_000
    for other_data_name in ['avg_latency', 'avg_count']:
        plot_split_throughputs(file_nr, n_clusters, combine_cores, other_data_name)
    # for num in range(1, 2):
    #     file_nr = DATA_FILES[benchmark][num]
    #     benchmark_params(file_nr)
