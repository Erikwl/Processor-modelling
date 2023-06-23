import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import os

from service_time_derivation import *
from data_retrieval import *

def permutated_timestamps(file_nr, n_clusters, combine_cores, other_data_name, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data_filename = f'data/perm_time_{combine_cores}_{other_data_name}_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{end_time}.npy'

    if os.path.exists(data_filename):
        print('DRAM data has already been permutated.')
        return np.load(data_filename, allow_pickle=True)[()]

    data_dict = analyse_dram_data(file_nr, combine_cores=combine_cores, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    throughputs = data_dict['throughputs']
    other_data = data_dict[other_data_name]


    throughputs_data = np.array([throughputs[core] for core in cores])
    other_data = np.array([other_data[core] for core in cores])
    data = np.append(throughputs_data, other_data, axis=0).transpose()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=min(len(np.unique(scaled_data)), n_clusters), n_init=20).fit(scaled_data)
    predicted_clusters = kmeans.predict(scaled_data)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print('Clusters have been created.')

    split_intervals = [0]
    enumerated_predicted_clusters = np.array(list(enumerate(predicted_clusters)))
    sorted_cluster_indices = np.array(sorted(enumerated_predicted_clusters, key=lambda x : x[1]))

    permutated_indices = sorted_cluster_indices[:,0]

    count = Counter(sorted_cluster_indices[:,1])

    time_index = 0
    interval_index = 0

    for cluster in range(n_clusters):
        if count[cluster] == 0:
            continue

        time_index += count[cluster]
        if time_index in split_intervals:
            continue

        split_intervals.insert(interval_index + 1, time_index)
        interval_index += 1


    print('Throughputs and latencies at every interval have been calculated.')

    data_dict = {'end_time': end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'kmeans' : kmeans,
                 'cluster_centers' : cluster_centers,
                 'predicted_clusters' : predicted_clusters,
                 'permutated_indices' : permutated_indices,
                 'split_intervals' : split_intervals}

    np.save(data_filename, data_dict)
    print('Permutated timestamps have been retrieved.')

    return data_dict

def split_throughputs(file_nr, n_clusters, combine_cores, other_data_name, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data_filename = f'data/split_throughs_{combine_cores}_{other_data_name}_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{end_time}.npy'

    if os.path.exists(data_filename):
        print(f'Split throughputs and {other_data_name} have already been calculated')
        return np.load(data_filename, allow_pickle=True)[()]
    data_dict = permutated_timestamps(file_nr, n_clusters, combine_cores, other_data_name, stepsize=stepsize, start_time=start_time, end_time=end_time)
    cores = data_dict['cores']
    timestamps = data_dict['timestamps']
    end_time = data_dict['end_time']
    cluster_centers = data_dict['cluster_centers']
    predicted_clusters = data_dict['predicted_clusters']
    permutated_indices = data_dict['permutated_indices']
    split_intervals = data_dict['split_intervals']


    split_fitted_throughputs = np.zeros((len(split_intervals) - 1, len(cores)))
    split_fitted_other_data = np.zeros((len(split_intervals) - 1, len(cores)))

    pops_lst = np.zeros((len(split_intervals) - 1, len(cores)))
    # caps_lst = np.zeros((len(split_intervals) - 1, len(cores)))
    service_times_lst = np.zeros((len(split_intervals) - 1, len(cores)))

    split_model_throughputs = np.zeros((len(split_intervals) - 1, len(cores)))
    split_model_other_data = np.zeros((len(split_intervals) - 1, len(cores)))

    args = model(len(cores))
    # args[3][:-1] = np.ones(len(cores), dtype=int)
    low = split_intervals[0]
    for i, high in enumerate(split_intervals[1:]):
        split_fitted_throughputs[i] = cluster_centers[i][:len(cores)]
        split_fitted_other_data[i] = cluster_centers[i][len(cores):]
        pops, _, _, _, service_times = find_all_params(split_fitted_throughputs[i],
                                                          split_fitted_other_data[i],
                                                          other_data_name)
        pops_lst[i] = pops
        # caps_lst[i] = caps[:-1]
        service_times_lst[i] = service_times[:-1]
        args[0] = pops
        args[4][:-1] = service_times_lst[i]
        waits, throughs = mva(*args)[1:3]
        for j, core in enumerate(cores):

            split_model_throughputs[i][j] = throughs[j]
            if other_data_name == 'avg_latency':
                split_model_other_data[i][j] = waits[-1][j]
            else:
                split_model_other_data[i][j] = throughs[j] * waits[-1][j]

            # print(f'{core = } on interval {low}-{high}')
            # print(f'{pops = }\n{caps = }\n{service_times = }')
            # print(f'{throughs[j] = }')
            # print(f'{split_fitted_throughputs[i][j] = }')
            # print(f'{split_model_other_data[i][j] = }')
            # print(f'{split_fitted_other_data[i][j] = }\n')

        low = high

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'permutated_indices' : permutated_indices,
                 'split_intervals' : split_intervals,
                 'predicted_clusters' : predicted_clusters,
                 'split_fitted_throughputs' : split_fitted_throughputs.transpose(),
                 'split_fitted_other_data' : split_fitted_other_data.transpose(),
                 'pops_lst' : pops_lst.transpose(),
                #  'caps_lst' : caps_lst.transpose(),
                 'service_times_lst' : service_times_lst.transpose(),
                 'split_model_throughputs' : split_model_throughputs.transpose(),
                 'split_model_other_data' : split_model_other_data.transpose()}

    np.save(data_filename, data_dict)

    print('Split throughputs and latency have been retrieved.')

    return data_dict


def plot_split_throughputs(file_nr, n_clusters, combine_cores, other_data_name, only_throughput=False, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data_dict = split_throughputs(file_nr, n_clusters, combine_cores, other_data_name, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    permutated_indices = data_dict['permutated_indices']
    split_intervals = data_dict['split_intervals']
    split_fitted_throughputs = data_dict['split_fitted_throughputs']
    split_fitted_other_data = data_dict['split_fitted_other_data']
    split_model_throughputs = data_dict['split_model_throughputs']
    split_model_other_data = data_dict['split_model_other_data']

    data_dict = analyse_dram_data(file_nr, combine_cores=combine_cores, stepsize=stepsize, start_time=start_time, end_time=end_time)
    throughputs = data_dict['throughputs']
    other_data = data_dict[other_data_name]

    ylabels = ['throughput', other_data_name]
    real_vals_lst = [throughputs, other_data]
    fitted_vals_lst = [split_fitted_throughputs, split_fitted_other_data]
    model_vals_lst = [split_model_throughputs, split_model_other_data]

    if only_throughput:
        real_vals_lst = [throughputs]
        fitted_vals_lst = [split_fitted_throughputs]
        model_vals_lst = [split_model_throughputs]

    subplot_nr = 0
    fig = plt.figure(figsize=(len(cores) * 5,5.5), dpi=150)
    for ylabel, real_vals, fitted_vals, model_vals in zip(ylabels, real_vals_lst, fitted_vals_lst, model_vals_lst):

        for i, core in enumerate(cores):
            if only_throughput:
                ax = fig.add_subplot(1, len(cores), i + 1 + subplot_nr)
            else:
                ax = fig.add_subplot(2, len(cores), i + 1 + subplot_nr)

            ax.vlines(x=timestamps, ymin=np.zeros(len(timestamps)), ymax=real_vals[core][permutated_indices])
            low = split_intervals[0]
            for fitted_y, model_y, high in zip(fitted_vals[i], model_vals[i], split_intervals[1:]):
                xmax = end_time if high == len(timestamps) else timestamps[high]
                labels = ['desired', 'model'] if low == split_intervals[0] else ['', '']
                ax.hlines(y=fitted_y, xmin=timestamps[low], xmax=xmax, linestyles='-', color='r', label=labels[0])
                ax.hlines(y=model_y, xmin=timestamps[low], xmax=xmax, linestyles='-', color='black', label=labels[1])
                low = high


            if subplot_nr == 0 and not only_throughput:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('permutated time (ns)')
            # if subplot_nr == 0 and not combine_cores:
            #     ax.set_title(f'Core {int(core)}')
            # elif subplot_nr == 0:
            #     ax.set_title(f'Combined cores')

            if core == cores[0]:
                ax.set_ylabel(ylabel)
            if subplot_nr == 0 and core == cores[-1]:
                ax.legend()
        subplot_nr += len(cores)

    fig.tight_layout()
    fig.savefig(f'pictures/perm_time/perm_time_{combine_cores}_{other_data_name}_{file_nr}_{n_clusters}_{stepsize}_{start_time}-{int(end_time)}')

if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    file_nr = DATA_FILES[benchmark][1][0]
    # # avg count plot
    # n_clusters = 10
    # other_data_name = 'avg_count'
    # combine_cores = False
    # stepsize = 10000
    # start_time = 930_000_000
    # end_time = 931_000_000
    # plot_split_throughputs(file_nr, n_clusters, combine_cores=combine_cores, other_data_name=other_data_name,
    #                        stepsize=stepsize, start_time=start_time, end_time=end_time)

    # # avg latency plot
    # n_clusters = 10
    # other_data_name = 'avg_latency'
    # combine_cores = False
    # stepsize = 10000
    # start_time = 930_000_000
    # end_time = 931_000_000
    # plot_split_throughputs(file_nr, n_clusters, combine_cores=combine_cores, other_data_name=other_data_name,
    #                        stepsize=stepsize, start_time=start_time, end_time=end_time)


    # # The cores separate
    # n_clusters = 6
    # other_data_name = 'avg_count'
    # combine_cores = False
    # stepsize = 1000
    # start_time = 927_500_000
    # end_time = 927_800_000
    # plot_split_throughputs(file_nr, n_clusters, combine_cores=combine_cores, other_data_name=other_data_name,
    #                        stepsize=stepsize, start_time=start_time, end_time=end_time, only_throughput=True)

    # # The cores combined
    # n_clusters = 6
    # other_data_name = 'avg_count'
    # combine_cores = True
    # stepsize = 1000
    # start_time = 927_500_000
    # end_time = 927_800_000
    # plot_split_throughputs(file_nr, n_clusters, combine_cores=combine_cores, other_data_name=other_data_name,
    #                        stepsize=stepsize, start_time=start_time, end_time=end_time, only_throughput=True)

    # Difference in stepsizes
    n_clusters = 8
    other_data_name = 'avg_count'
    combine_cores = True
    stepsize = 100
    start_time = 0
    end_time = 100_000
    plot_split_throughputs(file_nr, n_clusters, combine_cores=combine_cores, other_data_name=other_data_name,
                           stepsize=stepsize, start_time=start_time, end_time=end_time, only_throughput=False)
    stepsize = 1000
    plot_split_throughputs(file_nr, n_clusters, combine_cores=combine_cores, other_data_name=other_data_name,
                           stepsize=stepsize, start_time=start_time, end_time=end_time, only_throughput=False)

