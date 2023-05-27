from constants import *
import numpy as np
import matplotlib.pyplot as plt
import os


def retrieve_data(nr):
    filename = ACCESS_DATA_PATH + 'dram_access_data_raw' + str(nr) + '.csv'
    return np.loadtxt(filename, dtype=np.uintc, skiprows=1, delimiter=',')

def retrieve_throughputs(data, stepsize, start_time, end_time, file_nr):
    data_filename = f'data/throughputs_{file_nr}_{stepsize}_{start_time}-{end_time}.npy'
    if os.path.exists(data_filename):
        return np.load(data_filename, allow_pickle=True)[()]

    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    throughputs = {core : np.zeros([nr_bins]) for core in cores}

    for t, core_id, latency in data:
        index = int((t - start_time) / stepsize)
        throughputs[core_id][index] += 1
        # execution_begin = t + latency - SERVICE_TIME_MEM
        # start = execution_begin - timestamps[index]
        # end = start + SERVICE_TIME_MEM
        # print(t, latency, execution_begin, throughputs)
        # while end:
        #     if index >= nr_bins:
        #         break
        #     if end >= stepsize:
        #         throughputs[core_id][index] += (stepsize - start) / SERVICE_TIME_MEM
        #         end -= stepsize
        #         start = 0
        #         index += 1
        #     else:
        #         throughputs[core_id][index] += (end - start) / SERVICE_TIME_MEM
        #         break

    data_dict = {'cores' : cores,
                 'throughputs' : throughputs}

    np.save(data_filename, data_dict)
    return data_dict

def avg_dram_requests(data, stepsize, start_time, end_time, file_nr):
    if end_time == -1:
        end_time = data[-1,0] + data[-1,2] + stepsize # Last dram access + latency

    data_filename = f'data/dram_data_{file_nr}_{stepsize}_{start_time}-{end_time}.npy'
    if os.path.exists(data_filename):
        print('DRAM data has already been calculated.')
        return np.load(data_filename, allow_pickle=True)[()]

    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    total_arrivals = {core : np.zeros([nr_bins]) for core in cores}
    total_counts_for_arrivals = {core : np.zeros([nr_bins]) for core in cores}
    avg_count = {core : np.zeros([nr_bins]) for core in cores}
    total_latency = {core : np.zeros([nr_bins]) for core in cores}

    for t, core_id, latency in data:
        index = int((t - start_time) / stepsize)
        start = t - timestamps[index]
        end = start + latency
        total_arrivals[core_id][index] += 1
        total_latency[core_id][index] += latency
        total_counts_for_arrivals[core_id][index] += avg_count[core_id][index]
        temp_index = index
        while end:
            if temp_index >= nr_bins:
                break
            if end >= stepsize:
                avg_count[core_id][temp_index] += (stepsize - start) / stepsize
                end -= stepsize
                start = 0
                temp_index += 1
            else:
                avg_count[core_id][temp_index] += (end - start) / stepsize
                break
    avg_latency = {i : np.divide(total_latency[i], total_arrivals[i], out=np.zeros_like(total_arrivals[i]), where=total_arrivals[i]!=0) for i in cores}
    avg_counts_for_arrivals = {i : np.divide(total_counts_for_arrivals[i], total_arrivals[i], out=np.zeros_like(total_arrivals[i]), where=total_arrivals[i]!=0) for i in cores}
    data_dict = {'cores' : cores,
                 'total_arrivals' : total_arrivals,
                 'avg_counts_for_arrivals' : avg_counts_for_arrivals,
                 'avg_count' : avg_count,
                 'avg_latency' : avg_latency}
    np.save(data_filename, data_dict)
    print('DRAM data file has been saved.')
    return data_dict


def plot_time_plots(data, stepsizes, start_times, end_times, file_nr, plot_total_arrivals=False):
    def plot_bar_chart(subplot_num, t, vals, xlabel='',
                       ylabel='', title=False, logscale=False, ticks=False, ymin=1E-1):
        for index, core in enumerate(cores):
            ax = fig.add_subplot(num_rows, len(cores), index + 1 + subplot_num * len(cores))
            ax.vlines(x=t, ymin=np.zeros(len(t)), ymax=vals[core])

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

    num_rows = 4 if plot_total_arrivals else 3

    for stepsize, start_time, end_time in zip(stepsizes, start_times, end_times):
        fig = plt.figure(figsize=(8,8), dpi=150)
        t, throughputs = retrieve_throughputs(data, stepsize, start_time, end_time, file_nr)
        data_dict = avg_dram_requests(data, stepsize, start_time, end_time, file_nr)
        total_arrivals = data_dict['total_arrivals']
        avg_counts = data_dict['avg_counts']
        avg_latency = data_dict['avg_latency']
        cores = np.unique(data[:,1])

        for core in cores:
            total_arrivals[core] /= stepsize
            throughputs[core] /= stepsize

        plot_bar_chart(0, t, avg_counts, ylabel='average number of requests',
                       title=True)
        plot_bar_chart(1, t, avg_latency, ylabel='average access latency', ymin=10)
        plot_bar_chart(2, t, throughputs, ylabel='throughput',
                       xlabel='time (ns)', ticks=True)
        if plot_total_arrivals:
            plot_bar_chart(3, t, total_arrivals, ylabel='number of arrivals')

        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
        fig.tight_layout()
        fig.savefig(f'pictures/{len(cores)}core_time_plot{stepsize}_{start_time}-{end_time}.png')

def plot_correlation(data, stepsize, start_time, end_time, file_nr):
    data[:,1] = 0
    data_dict = avg_dram_requests(data, stepsize, start_time, end_time, file_nr)
    timestamps = np.arange(start_time, end_time, stepsize)
    avg_count_for_arrivals = data_dict['avg_count_for_arrivals'][0]
    avg_latency = data_dict['avg_latency']
    # avg_count_for_arrivals = avg_count_for_arrivals[0]
    # latency = avg_latency[0]
    arrivals = arrivals[0]
    for i in range(len(avg_count_for_arrivals)):
        if avg_count_for_arrivals[i]>= 580 and avg_count_for_arrivals[i] <= 590:
            print(avg_count_for_arrivals[i], avg_latency[i], arrivals[i], timestamps[i])

    for i in range(len(avg_count_for_arrivals)):
        if avg_latency[i] == 0:
            avg_count_for_arrivals[i] = 0

    plt.figure(figsize=(8,5), dpi=150)
    plt.xlabel(f'average number of DRAM requests within {stepsize} ns.')
    plt.ylabel('average access latency (ns)')
    plt.scatter(avg_count_for_arrivals, latency, s=0.3)
    plt.tight_layout()
    plt.savefig(f'pictures/correlation_{stepsize}__{int(start_time/1000)}-{int(end_time/1000)}.png')

def plot_different_correlations(data, stepsizes, start_time, end_time):
    for stepsize in stepsizes:
        plot_correlation(data, stepsize, start_time, end_time)

if __name__ == '__main__':
    benchmark = 'parsec-blackscholes'
    num = 1
    file_nr = DATA_FILES[benchmark][num]
    data = retrieve_data(file_nr)
    start_times = [21_100_000, 0]
    end_times = [21_200_000, 1_000_000_000]
    stepsizes = [53, 1_000_000]
    plot_time_plots(data, stepsizes, start_times, end_times, file_nr, plot_total_arrivals=False)






