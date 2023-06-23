from constants import *
import numpy as np
import matplotlib.pyplot as plt
import os

def retrieve_data(file_nr, combine_cores):

    filename = ACCESS_DATA_PATH + 'dram_access_data_raw' + str(file_nr) + '.csv'

    # data_filename = f'data/data_{combine_cores}_{file_nr}.npy'
    # if os.path.exists(data_filename):
    #     # print('File has already been retrieved.')
    #     return np.load(data_filename, allow_pickle=True)[()]

    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    data[:,2] /= 1000 # Convert picoseconds to nanoseconds
    data[:,2] -= 45 # Subtract DRAM service time

    if combine_cores:
        data[:,1] = 0

    # np.save(data_filename, data)
    # print('File has been saved.')
    return data

def retrieve_end_time(file_nr):
    data_filename = f'data/end_time_{file_nr}.npy'
    if os.path.exists(data_filename):
        print('End time has been retrieved')
        return np.load(data_filename, allow_pickle=True)[()]
    data = retrieve_data(file_nr, combine_cores=False)
    cores = np.unique(data[:,1])
    end_times = {}
    for core in cores:
        final_request_index = np.where(data[:,1] == core)[0].max()
        end_times[core] = data[final_request_index,0] + data[final_request_index,2]

    end_time = data[-1,0] + data[-1,2]

    data_dict = {'end_time' : end_time,
                 'end_times' : end_times}
    np.save(data_filename, data_dict)

    print('End time has been retrieved.')

    return data_dict

def split_benchmark(file_nr, new_file_nrs):
    filename = ACCESS_DATA_PATH + 'dram_access_data_raw' + str(file_nr) + '.csv'

    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    data1 = data[data[:,0] <= 1_500_000_000]
    data2 = data[data[:,0] > 1_500_000_000]

    print(len(data), len(data1), len(data2))


    np.savetxt(ACCESS_DATA_PATH + 'dram_access_data_raw' + str(new_file_nrs[0]) + '.csv', data1, delimiter=',')
    np.savetxt(ACCESS_DATA_PATH + 'dram_access_data_raw' + str(new_file_nrs[1]) + '.csv', data2, delimiter=',')

def retrieve_num_reads_writes(file_nr):
    data = retrieve_data(file_nr, combine_cores=True)
    num_writes = np.count_nonzero(data[:,3])
    num_reads = len(data) - num_writes
    return num_reads, num_writes

def show_reads_writes(benchmarks):
    file_nrs = [DATA_FILES[benchmark][1][0] for benchmark in benchmarks]
    for file_nr, benchmark in zip(file_nrs, benchmarks):
        reads, writes = retrieve_num_reads_writes(file_nr)
        print(f'{benchmark = }')
        print(f'{reads = }')
        print(f'{writes = }')
        print(f'There are {reads / writes :.2f} more reads than writes.')

def analyse_dram_data(file_nr, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME, combine_cores=True, types=False):
    data_filename = f'data/analysed_data_{combine_cores}_{types}_{file_nr}_{stepsize}_{start_time}-{end_time}.npy'
    # print(file_nr, data_filename)
    if os.path.exists(data_filename):
        print('DRAM data has already been analysed.')
        return np.load(data_filename, allow_pickle=True)[()]

    data = retrieve_data(file_nr, combine_cores)
    if end_time == -1:
        end_time = data[-1,0] + data[-1,2]


    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    throughputs = {core : np.zeros([nr_bins]) for core in cores}
    avg_count = {core : np.zeros([nr_bins]) for core in cores}
    total_latency = {core : np.zeros([nr_bins]) for core in cores}

    if types:
        throughputs_type = {core : np.zeros([2, nr_bins]) for core in cores}
        avg_count_type = {core : np.zeros([2, nr_bins]) for core in cores}
        total_latency_type = {core : np.zeros([2, nr_bins]) for core in cores}


    total_count_latency = {core : np.zeros([nr_bins]) for core in cores}
    total_count_latency_type = {core : np.zeros([2, nr_bins]) for core in cores}

    for item in data:
        if len(item) == 4:
            t, core, latency, type = item
            type = int(type)
            index = int((t - start_time) / stepsize)
        else:
            t, core, latency = item
            index = int((t - start_time) / stepsize)
        t -= timestamps[index]

        total_latency[core][index] += latency
        total_count_latency[core][index] += 1
        if types:
            total_latency_type[core][type][index] += latency
            total_count_latency_type[core][type][index] += 1

        while latency:
            if index >= nr_bins:
                break
            if t + latency >= stepsize:
                avg_count[core][index] += (stepsize - t) / stepsize
                if types:
                    avg_count_type[core][type][index] += (stepsize - t) / stepsize
                if t + latency - SERVICE_TIME_MEM < stepsize:
                    throughputs[core][index] += (stepsize - (t + latency - SERVICE_TIME_MEM)) / SERVICE_TIME_MEM
                    if types:
                        throughputs_type[core][type][index] += (stepsize - (t + latency - SERVICE_TIME_MEM)) / SERVICE_TIME_MEM
                latency -= stepsize - t
                t = 0
                index += 1
            else:
                avg_count[core][index] += latency / stepsize
                throughputs[core][index] += latency / max(SERVICE_TIME_MEM, latency)

                if types:
                    avg_count_type[core][type][index] += latency / stepsize
                    throughputs_type[core][type][index] += latency / max(SERVICE_TIME_MEM, latency)
                break

    avg_latency = {core : np.divide(total_latency[core], total_count_latency[core],
                                    out=np.zeros(len(total_latency[core])),
                                    where=total_count_latency[core]!=0) for core in cores}
    if types:
        avg_latency_type = {core : np.divide(total_latency_type[core], total_count_latency_type[core],
                                    out=np.zeros_like(total_latency_type[core]),
                                    where=total_count_latency_type[core]!=0) for core in cores}

    for core in cores:
        throughputs[core] /= stepsize
        if types:
            throughputs_type[core] /= stepsize

    if types:
        data_dict = {'end_time' : end_time,
                    'timestamps' : timestamps,
                    'cores' : cores,
                    'throughputs' : throughputs,
                    'throughputs_type' : throughputs_type,
                    'avg_count' : avg_count,
                    'avg_count_type' : avg_count_type,
                    'avg_latency' : avg_latency,
                    'avg_latency_type' : avg_latency_type}
    else:
        data_dict = {'end_time' : end_time,
                    'timestamps' : timestamps,
                    'cores' : cores,
                    'throughputs' : throughputs,
                    'avg_count' : avg_count,
                    'avg_latency' : avg_latency}

    np.save(data_filename, data_dict)
    print('Analysed DRAM data has been saved.')
    return data_dict

def plot_time_plot(file_nr, combine_cores=True, only_throughput=False, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    def plot_bar_chart(subplot_num, vals, xlabel='',
                       ylabel='', title=False, logscale=False, ticks=False, ymin=1E-1):
        for index, core in enumerate(cores):
            if only_throughput:
                ax = fig.add_subplot(1, len(cores), index + 1 + subplot_num * len(cores))
            else:
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
                # title = 'Combined cores' if combine_cores else f'Core {int(core)}'
                ax.set_title(title)
            if core == cores[0]:
                ax.set_ylabel(ylabel)

    data_dict = analyse_dram_data(file_nr, combine_cores=combine_cores, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    # avg_count = data_dict['avg_count']
    throughputs = data_dict['throughputs']
    avg_latency = data_dict['avg_latency']
    if only_throughput:
        fig = plt.figure(figsize=(len(cores) * 4,3), dpi=150)
    else:
        fig = plt.figure(figsize=(len(cores) * 4,5), dpi=150)

    if only_throughput:
        plot_bar_chart(0, throughputs, xlabel='time (ms)', ylabel='throughput', ticks=True, title=True)
    else:
        plot_bar_chart(0, throughputs, ylabel='DRAM throughput', title=True)
        plot_bar_chart(1, avg_latency, xlabel='time (ms)', ylabel='DRAM access latency', ticks=True)

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    fig.tight_layout()
    fig.savefig(f'pictures/time_plot/time_plot_{file_nr}_{stepsize}_{start_time}-{end_time}.png')

def plot_littles_law(file_nr, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    # data[:,1] = 0
    data_dict = analyse_dram_data(file_nr, combine_cores=True, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    cores = data_dict['cores']
    throughputs = data_dict['throughputs'][cores[0]]
    avg_latency = data_dict['avg_latency'][cores[0]]
    avg_count = data_dict['avg_count'][cores[0]]
    diff = np.multiply(throughputs, avg_latency) - avg_count
    relative_diff = np.divide(diff, avg_count,
                                out=np.zeros(len(diff)),
                                where=avg_latency!=0)
    plt.figure(figsize=(4,4), dpi=150)
    plt.xlabel(r'$\overline{n}_M$')
    plt.ylabel('relative difference')
    plt.scatter(avg_count, relative_diff, s=1)
    plt.tight_layout()
    plt.savefig(f'pictures/littles_law/littles_law_{stepsize}__{start_time}-{end_time}.png')


def plot_littles_law_per_type(file_nr, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data_dict = analyse_dram_data(file_nr, types=True, combine_cores=True, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    cores = data_dict['cores']
    throughputs_type = data_dict['throughputs_type'][cores[0]]
    avg_latency_type = data_dict['avg_latency_type'][cores[0]]
    avg_count_type = data_dict['avg_count_type'][cores[0]]
    diff = np.multiply(throughputs_type, avg_latency_type) - avg_count_type
    relative_diff = np.divide(diff, avg_count_type,
                                out=np.zeros_like(diff),
                                where=avg_latency_type!=0)

    for type in [0, 1]:
        plt.figure(figsize=(4,4), dpi=150)
        plt.xlabel(r'$\overline{n}_M$')
        plt.ylabel(r'relative difference')
        plt.scatter(avg_count_type[type], relative_diff[type], s=1)
        plt.tight_layout()
        plt.savefig(f'pictures/littles_law/littles_law_{type}_{stepsize}__{start_time}-{end_time}.png')

def plot_arrival_times(file_nr, detailed=False, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data = retrieve_data(file_nr, combine_cores=True)
    end_time = data[-1,0] + data[-1,2] if end_time == -1 else end_time

    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    fig = plt.figure(figsize=(4,4), dpi=150)
    read_requests = np.array([x[0] for x in data if x[3] == 0])
    read_requests_order = [i for i in range(len(data)) if data[i,3] == 0]

    write_requests = np.array([x[0] for x in data if x[3] == 1])
    write_requests_order = [i for i in range(len(data)) if data[i,3] == 1]

    MARKERS = ['1', 'x']
    COLORS = ['black', 'red']
    ax = fig.add_subplot(1, 1, 1)
    if detailed == False:
        ax.scatter(read_requests_order, read_requests, color=COLORS[0], s=7, marker=MARKERS[0], label='read request')
        ax.scatter(write_requests_order, write_requests, color=COLORS[1], s=7, marker=MARKERS[1], label='write request')
    else:
        ax.scatter(read_requests_order, read_requests, marker=MARKERS[0], color=COLORS[0], s=20, label='read request')
        ax.scatter(write_requests_order, write_requests, marker=MARKERS[1], color=COLORS[1], s=15, label='write request')
        ax.vlines(x=range(len(data)), ymin=data[:,0], ymax=data[:,0] + data[:,2] - 9, color='black', label='time before service')
        ax.vlines(x=range(len(data)), ymin=data[:,0] + data[:,2] - 9, ymax=data[:,0] + data[:,2], color='blue', label='serviced by DRAM controller')
    ax.set_xlabel('Request arrival order in DRAM controller')
    ax.set_ylabel('time (ns)')
    ax.legend()
    fig.subplots_adjust(left=0.21, bottom=0.11, right=0.95, top=0.95)
    fig.savefig(f'pictures/arrival_times_plot/arrival_times_plot_{detailed}_{file_nr}_{stepsize}_{start_time}-{END_TIME}')

if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    num = 1
    file_nr = DATA_FILES[benchmark][num][0]

    benchmarks = ['parsec-bodytrack', 'parsec-streamcluster']
    # show_reads_writes(benchmarks)
    # plot_time_plot(DATA_FILES[benchmarks[1]][1][0], combine_cores=True, only_throughput=False, stepsize=1000, start_time=0, end_time=-1)

    # Little's law plots
    # stepsize = 1_000
    # start_time = 0
    # end_time = -1
    # plot_littles_law(file_nr, stepsize=stepsize, start_time=start_time, end_time=end_time)
    # plot_littles_law_per_type(file_nr, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # No priority
    stepsize = 1000
    start_time = 900_040_000
    end_time = 900_100_000
    plot_arrival_times(file_nr, detailed=False, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # Priority
    start_time = 932_533_200
    end_time = 932_600_000
    plot_arrival_times(file_nr, detailed=False, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # Zoomed in on priority
    start_time = 932_533_200
    end_time = 932_534_000
    plot_arrival_times(file_nr, detailed=True, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # stepsize = 1_000_000
    # start_time = 0
    # end_time = -1
    # plot_time_plot(file_nr, combine_cores=True)

    # stepsize = 1_000
    # start_time = 0
    # end_time = 1_000_000
    # plot_time_plot(file_nr, combine_cores=True, only_throughput=True, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # # one core time plot
    # combine_cores = True
    # stepsize = 1_000_000
    # start_time = 0
    # end_time = -1
    # plot_time_plot(file_nr, combine_cores=combine_cores, only_throughput=True,
    #                stepsize=stepsize, start_time=start_time, end_time=end_time)
