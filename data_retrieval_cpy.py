from constants import *
import numpy as np
import matplotlib.pyplot as plt
import os


def retrieve_data(nr):
    filename = ACCESS_DATA_PATH + 'dram_access_data_raw' + str(nr) + '.csv'

    data_filename = f'data/data_{nr}.npy'
    if os.path.exists(data_filename):
        print('File has already been retrieved.')
        return np.load(data_filename, allow_pickle=True)[()]

    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    data[:,2] /= 1000 # Convert picoseconds to nanoseconds.

    np.save(data_filename, data)
    print('File has been saved.')
    return data

def retrieve_throughputs(file_nr):
    data = retrieve_data(file_nr)
    if END_TIME == -1:
        end_time = data[-1,0] + data[-1,2] + STEPSIZE # Last dram access + latency
    else:
        end_time = END_TIME

    data_filename = f'data/throughputs_{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.npy'
    if os.path.exists(data_filename):
        print('Throughputs have already been calculated.')
        return np.load(data_filename, allow_pickle=True)[()]

    filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - START_TIME) / STEPSIZE))
    timestamps = np.arange(START_TIME, end_time, STEPSIZE)

    throughputs = {core : np.zeros([nr_bins]) for core in cores}

    for t, core_id, latency in data:
        index = int((t - START_TIME) / STEPSIZE)
        # throughputs[core_id][index] += 1
        execution_begin = t + latency - SERVICE_TIME_MEM
        start = execution_begin - timestamps[index]
        end = start + SERVICE_TIME_MEM
        # print(t, latency, execution_begin, throughputs)
        while end:
            if index >= nr_bins:
                break
            if end >= STEPSIZE:
                throughputs[core_id][index] += (STEPSIZE - start) / SERVICE_TIME_MEM
                end -= STEPSIZE
                start = 0
                index += 1
            else:
                throughputs[core_id][index] += (end - start) / SERVICE_TIME_MEM
                break

    for core in cores:
        throughputs[core] /= STEPSIZE
    data_dict = {'cores' : cores,
                 'timestamps' : timestamps,
                 'throughputs' : throughputs}

    np.save(data_filename, data_dict)
    print('Throughputs have been saved.')
    return data_dict

def analyse_dram_data(file_nr):
    data = retrieve_data(file_nr)
    if END_TIME == -1:
        end_time = data[-1,0] + data[-1,2] + STEPSIZE # Last dram access + latency
    else:
        end_time = END_TIME
    data_filename = f'data/dram_data_{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.npy'

    if os.path.exists(data_filename):
        print('DRAM data has already been calculated.')
        return np.load(data_filename, allow_pickle=True)[()]

    filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)

    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - START_TIME) / STEPSIZE))
    timestamps = np.arange(START_TIME, end_time, STEPSIZE)

    total_arrivals = {core : np.zeros([nr_bins]) for core in cores}
    # total_counts_for_arrivals = {core : np.zeros([nr_bins]) for core in cores}
    avg_count = {core : np.zeros([nr_bins]) for core in cores}
    avg_latency = {core : np.zeros([nr_bins]) for core in cores}
    # total_latency_of_arrivals = {core : np.zeros([nr_bins]) for core in cores}

    for t, core_id, latency in data:
        index = int((t - START_TIME) / STEPSIZE)
        start = t - timestamps[index]
        end = start + latency
        total_arrivals[core_id][index] += 1
        # total_latency_of_arrivals[core_id][index] += latency
        # total_counts_for_arrivals[core_id][index] += avg_count[core_id][index]
        temp_index = index
        while end:
            if temp_index >= nr_bins:
                break
            if end >= STEPSIZE:
                avg_count[core_id][temp_index] += (STEPSIZE - start) / STEPSIZE
                end -= STEPSIZE
                start = 0
                temp_index += 1
            else:
                avg_count[core_id][temp_index] += (end - start) / STEPSIZE
                break
    # avg_latency_for_arrivals = {core : np.divide(total_latency_of_arrivals[core], total_arrivals[core],
    #                                 out=np.zeros_like(total_arrivals[core]),
    #                                 where=total_arrivals[core]!=0)
                #    for core in cores}
    # avg_counts_for_arrivals = {core : np.divide(total_counts_for_arrivals[core], total_arrivals[core],
    #                                             out=np.zeros_like(total_arrivals[core]),
    #                                             where=total_arrivals[core]!=0) for core in cores}
    avg_arrivals = {core : total_arrivals[core] / STEPSIZE for core in cores}
    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'avg_arrivals' : avg_arrivals,
                #  'avg_counts_for_arrivals' : avg_counts_for_arrivals,
                 'avg_count' : avg_count,
                #  'avg_latency' : avg_latency_for_arrivals
                }
    np.save(data_filename, data_dict)
    print('Analysed DRAM data has been retrieved.')
    return data_dict

def dram_through_lat_count(file_nr):
    data_dict = retrieve_throughputs(file_nr)
    cores = data_dict['cores']
    throughputs = data_dict['throughputs']
    throughputs = np.sum([throughputs[core] for core in cores], axis=0)

    data_dict = analyse_dram_data(file_nr)



def plot_time_plot(file_nr):
    def plot_bar_chart(subplot_num, vals, xlabel='',
                       ylabel='', title=False, logscale=False, ticks=False, ymin=1E-1):
        for index, core in enumerate(cores):
            ax = fig.add_subplot(2, len(cores), index + 1 + subplot_num * len(cores))
            ax.vlines(x=timestamps / 1_000_000, ymin=np.zeros(len(timestamps)), ymax=vals[core])

            if logscale:
                ax.set_yscale('log')
                ax.set_ylim([ymin, 2 * max(vals[core])])
            if not ticks:
                ax.set_xticklabels([])
            if xlabel:
                ax.set_xlabel(xlabel)
            if title:
                ax.set_title(f'Core {core}')
            if core == cores[0]:
                ax.set_ylabel(ylabel)

    fig = plt.figure(figsize=(12,8), dpi=150)
    data_dict = analyse_dram_data(file_nr)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    avg_arrivals = data_dict['avg_arrivals']
    # avg_count = data_dict['avg_count']
    avg_latency = data_dict['avg_latency']

    # plot_bar_chart(0, avg_count, ylabel='average number of requests',
    #                 title=True)
    plot_bar_chart(0, {core : avg_arrivals[core] for core in cores}, ylabel='number of arrivals per ns')
    plot_bar_chart(1, avg_latency, xlabel='time (ms)', ylabel='average access latency', ticks=True)

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    fig.tight_layout()
    fig.savefig(f'pictures/time_plot{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.png')

def plot_correlation(file_nr):
    # data[:,1] = 0
    data_dict = analyse_dram_data(file_nr)
    end_time = data_dict['end_time']
    # timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    avg_latency = data_dict['avg_latency']
    avg_count_for_arrivals = data_dict['avg_count_for_arrivals']
    # avg_count_for_arrivals = avg_count_for_arrivals[0]
    # latency = avg_latency[0]
    # arrivals = arrivals[0]
    # for i in range(len(avg_count_for_arrivals)):
    #     if avg_count_for_arrivals[i]>= 580 and avg_count_for_arrivals[i] <= 590:
    #         print(avg_count_for_arrivals[i], avg_latency[i], arrivals[i], timestamps[i])

    for i in range(len(avg_count_for_arrivals)):
        if avg_latency[i] == 0:
            avg_count_for_arrivals[i] = 0

    for core in cores:
        plt.figure(figsize=(12,8), dpi=150)
        plt.xlabel(f'average number of DRAM requests within {STEPSIZE} ns.')
        plt.ylabel('average access latency (ns)')
        plt.scatter(avg_count_for_arrivals[core], avg_latency[core], s=0.3)
        plt.tight_layout()
    plt.savefig(f'pictures/correlation_{STEPSIZE}__{START_TIME}-{end_time}.png')

# def plot_different_correlations(data, STEPSIZEs, START_TIME, end_time):
#     for STEPSIZE in STEPSIZEs:
#         plot_correlation(data, STEPSIZE, START_TIME, end_time)

def plot_DRAM_throughputs(file_nr):
    data_dict = retrieve_throughputs(file_nr)
    cores = data_dict['cores']
    timestamps = data_dict['timestamps']
    throughputs = data_dict['throughputs']
    fig = plt.figure(figsize=(8,8), dpi=150)
    combined_throughputs = np.sum([throughputs[core] for core in cores], axis=0)
    plt.plot(timestamps, combined_throughputs)
    plt.show()

if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    num = 1
    file_nr = DATA_FILES[benchmark][num]
    # data = retrieve_data(file_nr)
    START_TIMEs = [21_100_000, 0]
    end_times = [21_200_000, 1_000_000_000]
    STEPSIZEs = [53, 1_000_000]
    # plot_DRAM_throughputs(file_nr)
    plot_time_plot(file_nr)






