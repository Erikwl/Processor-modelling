from constants import ACCESS_DATA_PATH, NUMPY_FILES_PATH
import numpy as np
import matplotlib.pyplot as plt
from time import time as current_time
from scipy.optimize import curve_fit

data_filenumber = 106

def retrieve_data(nr):
    filename = ACCESS_DATA_PATH + 'dram_access_data_raw' + str(nr) + '.csv'
    return np.loadtxt(filename, dtype=np.uintc, skiprows=1, delimiter=',')

# def convert_to_np_file(nr):
#     np_filename = NUMPY_FILES_PATH + str(nr) + '.npy'
#     with open(np_filename, 'wb') as f:
#         np.save(f, retrieve_data_from_csv(nr))

# def retrieve_data_from_np(nr):
#     np_filename = NUMPY_FILES_PATH + str(nr) + '.npy'
#     with open(np_filename, 'rb') as f:
#         return np.load(f)

def retrieve_accumulated_data(data, stepsize, start_time=0, end_time=-1):
    filter = np.all([data[:,0] >= start_time, data[:,0] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    if end_time == -1:
        end_time = data[-1,0] # Last dram access
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    count_accum = {core : np.zeros([nr_bins]) for core in cores}
    latency_accum = {core : np.zeros([nr_bins]) for core in cores}
    for t, core_id, latency in data:
        index = int((t - start_time) / stepsize)
        count_accum[core_id][index] += 1
        latency_accum[core_id][index] += latency
    latency_avg = {i : np.divide(latency_accum[i], count_accum[i], out=np.zeros_like(latency_accum[i]), where=count_accum[i]!=0) for i in cores}
    return data, timestamps, count_accum, latency_avg


def avg_dram_requests_v2(data, stepsize, start_time=0, end_time=-1):
    if end_time == -1:
        end_time = data[-1,0] + data[-1,2] # Last dram access + latency
    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    count_accum = {core : np.zeros([nr_bins]) for core in cores}
    latency_accum = {core : np.zeros([nr_bins]) for core in cores}
    for t, core_id, latency in data:
        index = int((t - start_time) / stepsize)
        start = t - timestamps[index]
        cur = start + latency

        while cur:
            if index >= len(count_accum[cores[0]]):
                break
            if cur + start >= stepsize:
                count_accum[core_id][index] += (stepsize - start) / stepsize
                latency_accum[core_id][index] += (stepsize - start) / stepsize * latency
                cur -= (stepsize - start)
                start = 0
                index += 1
            else:
                count_accum[core_id][index] += (cur - start) / stepsize
                latency_accum[core_id][index] += (cur - start) / stepsize * latency
                break
    latency_avg = {i : np.divide(latency_accum[i], count_accum[i], out=np.zeros_like(latency_accum[i]), where=count_accum[i]!=0) for i in cores}
    return data, timestamps, count_accum, latency_avg

def avg_dram_requests(data, stepsize, start_time=0, end_time=-1):
    if end_time == -1:
        end_time = data[-1,0] + data[-1,2] # Last dram access + latency
    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    total_arrival_accum = {core : np.zeros([nr_bins]) for core in cores}
    count_accum = {core : np.zeros([nr_bins]) for core in cores}
    avg_count_accum = {core : np.zeros([nr_bins]) for core in cores}
    latency_accum = {core : np.zeros([nr_bins]) for core in cores}
    for t, core_id, latency in data:
        x = 2.1105E7
        if t >= x and t <= x + 5000:
            print(t, latency)
        index = int((t - start_time) / stepsize)
        start = t - timestamps[index]
        cur = start + latency
        total_arrival_accum[core_id][index] += 1
        latency_accum[core_id][index] += latency
        count_accum[core_id][index] += avg_count_accum[core_id][index]
        # print(t, core_id, latency, index, start, cur)
        while cur:
            if index >= len(count_accum[cores[0]]):
                break
            if cur + start >= stepsize:
                avg_count_accum[core_id][index] += (stepsize - start) / stepsize
                # latency_accum[core_id][index] += (stepsize - start) / stepsize * latency
                cur -= (stepsize - start)
                start = 0
                index += 1
            else:
                avg_count_accum[core_id][index] += (cur - start) / stepsize
                # latency_accum[core_id][index] += (cur - start) / stepsize * latency
                break
    latency_avg = {i : np.divide(latency_accum[i], total_arrival_accum[i], out=np.zeros_like(total_arrival_accum[i]), where=total_arrival_accum[i]!=0) for i in cores}
    count_accum = {i : np.divide(count_accum[i], total_arrival_accum[i], out=np.zeros_like(total_arrival_accum[i]), where=total_arrival_accum[i]!=0) for i in cores}
    return data, timestamps, total_arrival_accum, count_accum, avg_count_accum, latency_avg


def plot_bar_chart(times, values, xlabel, ylabel, title=None):
    plt.vlines(x=times, ymin=np.zeros(len(times)), ymax=values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

def plot_time_plots(data, stepsize, start_times, end_times):
    start = current_time()
    print(current_time() - start)

    for start_time, end_time in zip(start_times, end_times):
        data1, t, total_arrivals, count, avg_count, latency = avg_dram_requests(data, stepsize, start_time=start_time, end_time=end_time)
        cores = np.unique(data1[:,1])
        print(current_time() - start)
        print('val=',max(avg_count[1]) / max(latency[1]))

        fig = plt.figure(figsize=(10,6), dpi=150)

        for index, core in enumerate(cores):
            # plot_bar_chart(t, count[core], 'time', 'count', f'core {core}')
            ax = fig.add_subplot(int(f'2{len(cores)}{index + 1}'))
            ax.vlines(x=t, ymin=np.zeros(len(t)), ymax=avg_count[core])

            ax.set_yscale('log')
            ax.set_ylim([1E-1, 2 * max(avg_count[core])])
            ax.set_xticklabels([])

            ax.set_title(f'core {core}')
            if core == cores[0]:
                ax.set_ylabel('average number of DRAM requests')

        for index, core in enumerate(cores):
            # plot_bar_chart(t, latency[core], 'time', 'latency', f'core {core}')
            ax = fig.add_subplot(int(f'2{len(cores)}{index + 1 + len(cores)}'))
            ax.vlines(x=t, ymin=np.zeros(len(t)), ymax=latency[core])

            ax.set_yscale('log')
            ax.set_ylim([10, 2 * max(latency[core])])

            ax.set_xlabel('time (ns)')
            if core == cores[0]:
                ax.set_ylabel('average access latency (ns)')

        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
        plt.tight_layout()
        plt.savefig(f'pictures/{len(cores)}core_time_plot{stepsize}_{start_time}-{end_time}.png')
        plt.show()

def plot_correlation(data, stepsize, start_time, end_time):
    data[:,1] = 0
    data1, t, arrivals, count, avg_count, latency = avg_dram_requests(data, stepsize, start_time=start_time, end_time=end_time)
    # print(data1)
    # print(arrivals[0])
    # print(count[0])
    # print(latency[0])
    plt.figure(figsize=(8,5), dpi=150)
    count = count[0]
    latency = latency[0]
    arrivals = arrivals[0]
    for i in range(len(count)):
        if count[i]>= 580 and count[i] <= 590:
            print(count[i], latency[i], arrivals[i], t[i])

    for i in range(len(count)):
        if latency[i] == 0:
            count[i] = 0
    # cores = count.keys()
    # flattened_count = [item for core in cores for item in count[core]]
    # flattened_latency = [item for core in cores for item in latency[core]]
    fig = plt.figure(figsize=(10,6), dpi=150)
    plt.xlabel(f'average number of DRAM requests within {stepsize} ns.')
    plt.ylabel('average access latency (ns)')
    plt.scatter(count, latency, s=0.3)
    plt.tight_layout()
    plt.savefig(f'pictures/correlation_{stepsize}__{int(start_time/1000)}-{int(end_time/1000)}.png')

def plot_different_correlations(data, stepsizes, start_time, end_time):
    for stepsize in stepsizes:
        plot_correlation(data, stepsize, start_time, end_time)

if __name__ == '__main__':
    stepsizes = [100, 500, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000] # in nanoseconds
    data = retrieve_data(data_filenumber)
    start_times = [0]
    # end_time = data[-1,0] + data[-1,2]
    end_times = [1_000_000_000]
    stepsize = 1_000_000


    start_times = [21_105_000]
    end_times = [21_110_000]
    stepsize = 1_00
    # plot_different_correlations(data, stepsizes, start_time, end_time)
    plot_time_plots(data, stepsize, start_times, end_times)
    # plot_time_plots(data, stepsize)




