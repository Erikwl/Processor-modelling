from parameter_derivation import find_cores_service_times
from data_retrieval import avg_dram_requests, retrieve_data, retrieve_throughputs
import numpy as np
from constants import ONE_GHZ_DATA_FILE_NUMBER
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from time import time as now

LATENCY_TOL = 1

def model(n):
    pop_vector = np.zeros(n)
    refs = np.ones(n) * n
    visits = np.eye(n + 1, n)
    visits[-1] = np.ones(n)
    caps = np.zeros(n + 1)
    caps[-1] = 5
    service_times = np.zeros(n + 1)
    return [pop_vector, refs, visits, caps, service_times]

def create_step_func(data, stepsize, n_clusters):
    start_time = 0
    end_time = 1_00_000_000
    timestamps, throughput = retrieve_throughputs(data, stepsize, start_time, end_time)
    cores = np.unique(data[:,1])

    # for core in cores:
    #     print(np.count_nonzero(throughput[core]))

    for core in cores:
        throughput[core] /= stepsize

    throughput_ = {core : throughput[core][throughput[core] != 0] for core in cores}
    # normalized_throughput_ = {core : throughput_[core] / max(throughput_[core]) for core in cores}
    X = {core : throughput_[core].reshape(-1, 1) for core in cores}
    # for core in cores:
    #     print(np.shape(X[core]))
    kmeans = {core : KMeans(n_clusters=min(len(X[core]), n_clusters), n_init=10).fit(X[core]) for core in cores}
    cluster_centers = {core : np.append(kmeans[core].cluster_centers_, [0]).reshape(-1,1) for core in cores}
    # for core in cores:
    #     print(cluster_centers[core])
    kmeans = {core : KMeans(n_clusters=min(len(X[core]), n_clusters) + 1,
                            init=cluster_centers[core].reshape(-1, 1),
                            n_init=1).fit(cluster_centers[core]) for core in cores}
    # for core_index, core in enumerate(cores):
    #     print(kmeans[core].cluster_centers_)
    # begin = now()

    split_intervals = [0, len(timestamps)]
    split_throughput = {core : [] for core in cores}
    permuted_indices = np.arange(len(timestamps))
    for core_index, core in enumerate(cores):

        low = 0
        # print(f'{split_intervals[1:] = }')
        for high in list(split_intervals[1:]):
            cur_permuted_indices = permuted_indices[low:high]
            through = throughput[core][cur_permuted_indices]
            # if all(through == 0):
            #     low = high
            #     continue

            cluster_indices = kmeans[core].predict(through.reshape(-1, 1))
            # cluster_indices[through == 0] = n_clusters

            zipped_cluster_indices = np.column_stack((cur_permuted_indices, cluster_indices))
            sorted_cluster_indices = np.array(sorted(zipped_cluster_indices, key=lambda x : x[1]))

            permuted_indices[low:high] = sorted_cluster_indices[:,0]

            count = Counter(sorted_cluster_indices[:,1])

            time_index = low
            interval_index = split_intervals.index(low)

            # print(f'{low = }, {high = }')
            # print(f'{core}: {split_throughput = }')


            first_cluster = True
            for cluster in range(n_clusters + 1):
                if count[cluster] == 0:
                    continue

                if first_cluster:
                    first_cluster = False
                else:
                    for prev_core in cores[:core_index]:
                        split_throughput[prev_core].insert(interval_index,
                                                           split_throughput[prev_core][interval_index - 1])

                split_throughput[core].append(kmeans[core].cluster_centers_[cluster][0])
                time_index += count[cluster]
                if time_index in split_intervals:
                    continue

                split_intervals.insert(interval_index + 1, time_index)
                interval_index += 1

            # print(count)
            # print(core, low, high, split_intervals, '\n')
            low = high
        # print(f'end, {core}: {split_throughput = }\n')

    # print(f'{split_intervals = }')

    fig = plt.figure(figsize=(8,8), dpi=150)

    for i, core in enumerate(cores):
        vals = throughput[core][permuted_indices]
        ax = fig.add_subplot(len(cores), 1, i + 1)
        ax.vlines(x=timestamps, ymin=np.zeros(len(timestamps)), ymax=vals, alpha=0.5)
        low = split_intervals[0]
        for y, high in zip(split_throughput[core], split_intervals[1:]):
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
            ax.set_title(f'Stepsize: {stepsize}, Core {core}')
        else:
            ax.set_title(f'Core {core}')

        # ax.set_yscale('log')
        # ax.set_ylim([min(vals[vals != 0]), 2 * max(vals)])
        ax.set_ylabel('throughput')
    plt.savefig(f'pictures/time_permutation_{stepsize}_{start_time}-{end_time}')

if __name__ == '__main__':
    data = retrieve_data(ONE_GHZ_DATA_FILE_NUMBER)
    stepsize = 1_000_00
    n_clusters = 6
    create_step_func(data, stepsize, n_clusters)
