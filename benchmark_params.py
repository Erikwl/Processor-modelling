import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from service_time_derivation import *
import os

from constants import *
from data_retrieval import *

def permutated_timestamps(data, stepsize, start_time, end_time, n_clusters, file_nr, plot=False):
    timestamps = np.arange(start_time, end_time, stepsize)
    data_filename = f'data/time_perm_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{end_time}.npy'

    if os.path.exists(data_filename) and plot == False:
        print('Permutated timestamps have already been calculated.')
        return np.load(data_filename, allow_pickle=True)[()]

    data_dict = avg_dram_requests(data, stepsize, start_time, end_time, file_nr)
    cores = data_dict['cores']
    throughputs = data_dict['total_arrivals']

    print('Throughputs have been retrieved.')

    for core in cores:
        throughputs[core] /= stepsize

    # throughputs_ = {core : throughputs[core][throughputs[core] != 0] for core in cores}
    # normalized_throughput_ = {core : throughput_[core] / max(throughput_[core]) for core in cores}
    X = {core : throughputs[core].reshape(-1, 1) for core in cores}
    # for core in cores:
    #     print(np.shape(X[core]))
    kmeans = {core : KMeans(n_clusters=min(len(np.unique(X[core])), n_clusters),
                            init=np.linspace(0, 0.1, min(len(np.unique(X[core])), n_clusters)).reshape(-1,1),
                            n_init=1).fit(X[core]) for core in cores}
    # cluster_centers = {core : np.append([0], kmeans[core].cluster_centers_).reshape(-1,1) for core in cores}
    # # for core in cores:
    # #     print(cluster_centers[core])
    # kmeans = {core : KMeans(n_clusters=min(len(X[core]), n_clusters) + 1,
    #                         init=cluster_centers[core].reshape(-1, 1),
    #                         n_init=1).fit(cluster_centers[core]) for core in cores}
    # for core_index, core in enumerate(cores):
    #     print(kmeans[core].cluster_centers_)
    # begin = now()

    print('Clusters have been created.')
    for core in cores:
        print(f'{core = }, {kmeans[core].cluster_centers_}')

    split_intervals = [0, len(timestamps)]
    split_throughputs = {core : [] for core in cores}
    permutated_indices = np.arange(len(timestamps))
    for core_index, core in enumerate(cores):

        low = 0
        # print(f'{split_intervals[1:] = }')
        for high in list(split_intervals[1:]):
            cur_permutated_indices = permutated_indices[low:high]
            through = throughputs[core][cur_permutated_indices]
            # if all(through == 0):
            #     low = high
            #     continue

            cluster_indices = kmeans[core].predict(through.reshape(-1, 1))
            # cluster_indices[through == 0] = n_clusters

            zipped_cluster_indices = np.column_stack((cur_permutated_indices, cluster_indices))
            sorted_cluster_indices = np.array(sorted(zipped_cluster_indices, key=lambda x : x[1]))

            permutated_indices[low:high] = sorted_cluster_indices[:,0]

            count = Counter(sorted_cluster_indices[:,1])

            time_index = low
            interval_index = split_intervals.index(low)

            # print(f'{low = }, {high = }')
            # print(f'{core}: {split_throughputs = }')


            first_cluster = True
            for cluster in range(n_clusters):
                if count[cluster] == 0:
                    continue

                if first_cluster:
                    first_cluster = False
                else:
                    for prev_core in cores[:core_index]:
                        split_throughputs[prev_core].insert(interval_index,
                                                            split_throughputs[prev_core][interval_index - 1])

                split_throughputs[core].append(kmeans[core].cluster_centers_[cluster][0])
                # split_throughputs[core].append()
                time_index += count[cluster]
                if time_index in split_intervals:
                    continue

                split_intervals.insert(interval_index + 1, time_index)
                interval_index += 1

            # print(count)
            # print(core, low, high, split_intervals, '\n')
            low = high
        # print(f'end, {core}: {split_throughputs = }\n')

    # print(f'{split_intervals = }')

    print('Throughputs and latencies at every interval have been calculated.')


    if plot == True:
        fig = plt.figure(figsize=(8,8), dpi=150)

        for i, core in enumerate(cores):
            vals = throughputs[core][permutated_indices]
            ax = fig.add_subplot(len(cores), 1, i + 1)
            ax.vlines(x=timestamps, ymin=np.zeros(len(timestamps)), ymax=vals, alpha=0.5)
            low = split_intervals[0]
            for y, high in zip(split_throughputs[core], split_intervals[1:]):
                if high == len(timestamps):
                    ax.hlines(y=y, xmin=timestamps[low], xmax=end_time, linestyles='-', color='r')
                else:
                    ax.hlines(y=y, xmin=timestamps[low], xmax=timestamps[high], linestyles='-', color='r')
                low = high

            if i < len(cores) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('permuted time (ns)')
            if core == 0:
                ax.set_title(f'Stepsize: {stepsize}, End_time: {end_time}, Core {core}')
            else:
                ax.set_title(f'Core {core}')

            # ax.set_yscale('log')
            # ax.set_ylim([min(vals[vals != 0]), 2 * max(vals)])
            ax.set_ylabel('throughputs')
        fig.savefig(f'pictures/time_perm_throughputs_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{end_time}')

    data_dict = {'cores' : cores,
                 'permutated_indices' : permutated_indices,
                 'split_intervals' : split_intervals,
                 'throughputs' : throughputs,
                 'split_throughputs' : split_throughputs}

    np.save(data_filename, data_dict)
    print('Permutated timestamps file has been saved.')

    return data_dict

def determine_params_of_benchmark(file_nr, plot=False):
    data = retrieve_data(file_nr)
    print('File is retrieved')

    stepsize = 1_000
    start_time = 0
    # end_time = 1_000_000
    end_time = data[-1, 0] + data[-1, 2] + stepsize
    timestamps = np.arange(start_time, end_time, stepsize)
    n_clusters = 5
    data_filename = f'data/params_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{end_time}.npy'


    if os.path.exists(data_filename) and plot == False:
        print('Params have already been calculated')
        return np.load(data_filename, allow_pickle=True)[()]

    # stepsize = int(end_time / 1000)

    data_dict = avg_dram_requests(data, stepsize, start_time=start_time, end_time=end_time, file_nr=file_nr)
    avg_latency = data_dict['avg_latency']
    print('DRAM data has been retrieved.')

    data_dict = permutated_timestamps(data, stepsize, start_time, end_time, n_clusters, file_nr)

    cores = data_dict['cores']
    permutated_indices = data_dict['permutated_indices']
    split_intervals = data_dict['split_intervals']
    throughputs = data_dict['throughputs']
    split_throughputs = data_dict['split_throughputs']

    print('Timestamps permutation data has been retrieved.')

    # permutated_latency = {core : avg_latency[core][permutated_indices] for core in cores}
    split_avg_latency = {core : np.zeros(len(split_throughputs[core])) for core in cores}

    # active_cores_lst = []
    pops_lst = np.zeros((len(split_intervals) - 1, len(cores)))
    caps_lst = np.zeros((len(split_intervals) - 1, len(cores)))
    service_times_lst = np.zeros((len(split_intervals) - 1, len(cores)))

    # for i, (through, lat) in enumerate(zip(throughputs, avg_latency)):
    #     if through != 0 and lat == 0:
    #         print(i, through, lat)

    # print(split_intervals)
    start = split_intervals[0]
    for i, end in enumerate(split_intervals[1:]):
        # active_cores = [core for core in cores if split_throughputs[core][i]]
        # active_cores_lst.append(active_cores)
        # print(start, end)

        for core in cores:
            lat_part = avg_latency[core][permutated_indices[start:end]]
            # print(len(permutated_indices[core]))
            lat_part = lat_part[lat_part != 0]
            if len(lat_part):
                split_avg_latency[core][i] = np.sum(lat_part) / len(lat_part)
            # if i == len(split_intervals) - 2:
            #     print(start, end, lat_part, np.sum(lat_part) / len(lat_part))
            # print(core, split_avg_latency[core][i])
            # print(core, split_throughputs[core][i])
        throughs = [split_throughputs[core][i] for core in cores]
        waiting_times = [split_avg_latency[core][i] for core in cores]
        # print(timestamps[start], timestamps[end], throughs, waiting_times)
        pops, _, _, caps, service_times = find_all_params(throughs, waiting_times)
        # print(np.shape(pops_lst[core]))
        print(throughs, waiting_times)
        pops_lst[i] = pops
        caps_lst[i] = caps[:-1]
        service_times_lst[i] = service_times[:-1]
        start = end

    print(split_throughputs)
    print(split_avg_latency)

    print('Parameters have been determined.')
    if plot == True:
        fig = plt.figure(figsize=(8,8), dpi=150)

        for i, core in enumerate(cores):
            vals = avg_latency[core][permutated_indices]
            ax = fig.add_subplot(len(cores), 1, i + 1)
            ax.vlines(x=timestamps, ymin=np.zeros(len(timestamps)), ymax=vals, alpha=0.5)
            low = split_intervals[0]
            for y, high in zip(split_avg_latency[core], split_intervals[1:]):
                if high == len(timestamps):
                    ax.hlines(y=y, xmin=timestamps[low], xmax=end_time, linestyles='-', color='r')
                else:
                    ax.hlines(y=y, xmin=timestamps[low], xmax=timestamps[high], linestyles='-', color='r')
                low = high

            if i < len(cores) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('permuted time (ns)')
            if core == 0:
                ax.set_title(f'Stepsize: {stepsize}, End_time: {end_time}, Core {core}')
            else:
                ax.set_title(f'Core {core}')

            # ax.set_yscale('log')
            # ax.set_ylim([min(vals[vals != 0]), 2 * max(vals)])
            ax.set_ylabel('Access latency')
        fig.savefig(f'pictures/time_perm_avg_latency_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{end_time}')


    data_dict = {'cores' : cores,
                 'permutated_indices' : permutated_indices,
                 'split_intervals' : split_intervals,
                 'throughputs' : throughputs,
                 'split_throughputs' : split_throughputs,
                 'avg_latency' : avg_latency,
                 'split_avg_latency' : split_avg_latency,
                 'pops_lst' : pops_lst,
                 'caps_lst' : caps_lst,
                 'service_times_lst' : service_times_lst}
    np.save(data_filename, data_dict)
    print('Benchmarks parameters have been saved.')
    return data_dict

if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    file_nr = DATA_FILES[benchmark][1]
    determine_params_of_benchmark(file_nr)
    # for num in range(1, 2):
    #     file_nr = DATA_FILES[benchmark][num]
    #     determine_params_of_benchmark(file_nr)
