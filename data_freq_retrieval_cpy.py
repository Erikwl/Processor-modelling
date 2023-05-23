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
    end_time = 1_000_000_000
    # end_time = data[-1,0] + data[-1,2]
    # data1, t, arrivals, avg_counts_for_arrivals, avg_counts, avg_latency = avg_dram_requests(data, stepsize, start_time=start_time, end_time=end_time)
    t, throughput = retrieve_throughputs(data, stepsize, start_time, end_time)
    cores = np.unique(data[:,1])
    # for core in cores:
    #     throughput[core] /= stepsize
    #     print(max(throughput[core]))

    # arrivals_ = {core : arrivals[core][arrivals[core] != 0] for core in cores}


    # nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)
    # time_intervals = {core : [0] for core in cores}

    # for core in cores:
    #     plt.scatter(avg_latency[core], throughput[core])
    #     plt.show()

    # avg_latency_ = {core : avg_latency[core][avg_latency[core] != 0] for core in cores}
    # throughput_ = {core : throughput[core][throughput[core] != 0] for core in cores}
    # normalized_latency = {core : avg_latency_[core] / max(avg_latency_[core]) for core in cores}

    # X = {core : list(zip(normalized_latency[core], normalized_throughput[core])) for core in cores}
    # X_ = {core : list(zip(normalized_latency[core], normalized_throughput[core])) for core in cores}
    # remaining = {core : }

    # for core in cores:
    #     kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(X_[core])
    #     cur = n_clusters if X[core][0][0] == 0 else kmeans.predict([X[core][0]])

    #     for i, (lat, through) in enumerate(X[core][1:], start=1):
    #         if lat == 0:
    #             next = n_clusters
    #         else:
    #             next = kmeans.predict([[lat, through]])
    #         if next != cur:
    #             time_intervals[core].append(i)
    #         cur = next
    #     print(time_intervals[core])

    #     centers = kmeans.cluster_centers_ * [max(avg_latency_[core]), max(throughput_[core])]
    #     # print(kmeans.predict(X[core]))
    #     np.set_printoptions(suppress=True)
    throughput_ = {core : throughput[core][throughput[core] != 0] for core in cores}
    normalized_throughput_ = {core : throughput_[core] / max(throughput_[core]) for core in cores}
    X = {core : normalized_throughput_[core].reshape(-1, 1) for core in cores}
    kmeans = {core : KMeans(n_clusters=n_clusters, n_init=10).fit(X[core]) for core in cores}

    begin = now()

    # accumulation = {core : {i : [] for i in range(n_clusters + 1)} for core in cores}
    split_intervals = [0, len(timestamps)]
    # splits = []
    permuted_indices = np.arange(len(timestamps))
    for core in cores:
        # split_throughput = np.split(throughput[core], splits)
        # for split_through in split_throughput:
        #     cluster_indices = np.array(kmeans[core].predict([x]) if x else n_clusters for x in split_through)

        low = 0
        for high in split_intervals[1:]:
            print(begin - now())
            begin = now()
            cur_permuted_indices = permuted_indices[low:high]
            # print(cur_permuted_indices)
            cluster_indices = list(kmeans[core].predict([[x]]) if x else [n_clusters]
                                   for x in throughput[core][cur_permuted_indices])
            cluster_indices = kme
            # print(cluster_indices, cur_permuted_indices)
            print(begin - now())
            begin = now()
            cluster_indices = list(itertools.chain(*cluster_indices))
            print(begin - now())
            begin = now()
            zipped_cluster_indices = np.column_stack((cur_permuted_indices, cluster_indices))
            sorted_cluster_indices = np.array(sorted(zipped_cluster_indices, key=lambda x : x[1]))
            # print(sorted_cluster_indices)
            print(begin - now())
            begin = now()
            permuted_indices[low:high] = sorted_cluster_indices[:,0]

            count = Counter(sorted_cluster_indices[:,1])
            print(begin - now(), '\n')
            begin = now()
            time_index = low
            interval_index = split_intervals.index(low)
            for cluster in range(n_clusters):
                if count[cluster] == 0:
                    continue
                time_index += count[cluster]
                split_intervals.insert(interval_index + 1, time_index)
                interval_index += 1

            low = high
            print(split_intervals)
        # print(kmeans[core].cluster_centers_)
        # print(cluster_indices)
        # print(throughput[core][permuted_indices])
    for core in cores
    plt.scatter(timestamps, throughput[core][permuted_indices])
    plt.show()

    
        # print(permuted_indices)
    # for core in cores:
    #     plt.scatter(timestamps, throughput[core][permuted_indices])
    #     plt.show()

    #     print(time_intervals[core])
    #     centers = kmeans.cluster_centers_
    #     # print(kmeans.predict(X[core]))
    #     print(np.round(centers, 6))



if __name__ == '__main__':
    data = retrieve_data(ONE_GHZ_DATA_FILE_NUMBER)
    stepsize = 1_000
    n_clusters = 6
    create_step_func(data, stepsize, n_clusters)
