from constants import *
import numpy as np
import matplotlib.pyplot as plt
import os
from math import isnan

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

# def retrieve_throughputs(file_nr):
#     data = retrieve_data(file_nr)
#     if END_TIME == -1:
#         end_time = data[-1,0] + data[-1,2] + stepsize # Last dram access + latency
#     else:
#         end_time = END_TIME

#     data_filename = f'data/throughputs_{file_nr}_{stepsize}_{START_TIME}-{end_time}.npy'
#     if os.path.exists(data_filename):
#         print('Throughputs have already been calculated.')
#         return np.load(data_filename, allow_pickle=True)[()]

#     filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)
#     data = data[filter]
#     cores = np.unique(data[:,1])
#     nr_bins = int(np.ceil((end_time - START_TIME) / stepsize))
#     timestamps = np.arange(START_TIME, end_time, stepsize)

#     throughputs = {core : np.zeros([nr_bins]) for core in cores}

#     for t, core_id, latency in data:
#         # print(t, core_id, latency)
#         # throughputs[core_id][index] += 1
#         execution_begin = t + latency - SERVICE_TIME_MEM
#         index = int((execution_begin - START_TIME) / stepsize)
#         start = execution_begin - timestamps[index]
#         end = start + SERVICE_TIME_MEM
#         # print(t, latency, execution_begin, start, end, throughputs)
#         while end:
#             if index >= nr_bins:
#                 break
#             if end >= stepsize:
#                 throughputs[core_id][index] += (stepsize - start) / SERVICE_TIME_MEM
#                 # print(f'if{(stepsize - start) / SERVICE_TIME_MEM}')
#                 end -= stepsize
#                 start = 0
#                 index += 1
#             else:
#                 throughputs[core_id][index] += (end - start) / SERVICE_TIME_MEM
#                 # print(f'else:{(end - start) / SERVICE_TIME_MEM}')
#                 break
#     # print(throughputs)
#     print(f'throughput: {len(data) / stepsize}')
#     print(f'avg_count: {np.sum(data[:,2]) / stepsize}')
#     print(f'avg_latency: {np.sum(data[:,2]) / len(data)}')

#     for core in cores:
#         throughputs[core] /= stepsize


#     data_dict = {'end_time' : end_time,
#                  'timestamps' : timestamps,
#                  'cores' : cores,
#                  'timestamps' : timestamps,
#                  'throughputs' : throughputs}

#     np.save(data_filename, data_dict)
#     print('Throughputs have been saved.')
#     return data_dict

# def retrieve_counts_latencies(file_nr):
#     data = retrieve_data(file_nr)
#     if END_TIME == -1:
#         end_time = data[-1,0] + data[-1,2] + stepsize # Last dram access + latency
#     else:
#         end_time = END_TIME
#     data_filename = f'data/dram_data_{file_nr}_{stepsize}_{START_TIME}-{end_time}.npy'

#     if os.path.exists(data_filename):
#         print('DRAM data has already been calculated.')
#         return np.load(data_filename, allow_pickle=True)[()]

#     filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)

#     data = data[filter]
#     cores = np.unique(data[:,1])
#     nr_bins = int(np.ceil((end_time - START_TIME) / stepsize))
#     timestamps = np.arange(START_TIME, end_time, stepsize)

#     # total_arrivals = {core : np.zeros([nr_bins]) for core in cores}
#     avg_count = {core : np.zeros([nr_bins]) for core in cores}
#     total_latency = {core : np.zeros([nr_bins]) for core in cores}
#     total_count_latency = {core : np.zeros([nr_bins]) for core in cores}

#     for t, core_id, latency in data:
#         index = int((t - START_TIME) / stepsize)
#         start = t - timestamps[index]
#         end = start + latency
#         temp_index = index
#         # total_latency[core_id][index] += latency
#         while end:
#             if temp_index >= nr_bins:
#                 break
#             if end >= stepsize:
#                 avg_count[core_id][temp_index] += (stepsize - start) / stepsize
#                 total_latency[core_id][temp_index] += (stepsize - start) / stepsize * latency
#                 total_count_latency[core_id][temp_index] += (stepsize - start) / stepsize
#                 # avg_count[core_id][temp_index] += latency / stepsize
#                 # total_latency[core_id][temp_index] += latency
#                 # total_count_latency[core_id][temp_index] += 1
#                 end -= stepsize
#                 start = 0
#                 temp_index += 1
#             else:
#                 avg_count[core_id][temp_index] += (end - start) / stepsize
#                 total_latency[core_id][temp_index] += (end - start)
#                 total_count_latency[core_id][temp_index] += (end - start) / latency
#                 # avg_count[core_id][temp_index] += latency / stepsize
#                 # total_latency[core_id][temp_index] += latency
#                 # total_count_latency[core_id][temp_index] += 1
#                 break
#     avg_latency = {core : np.divide(total_latency[core], total_count_latency[core],
#                                     out=np.zeros_like(total_latency[core]),
#                                     where=total_count_latency[core]!=0) for core in cores}
#     # avg_counts_for_arrivals = {core : np.divide(total_counts_for_arrivals[core], total_arrivals[core],
#     #                                             out=np.zeros_like(total_arrivals[core]),
#     #                                             where=total_arrivals[core]!=0) for core in cores}
#     data_dict = {'end_time' : end_time,
#                  'timestamps' : timestamps,
#                  'cores' : cores,
#                  'avg_count' : avg_count,
#                  'avg_latency' : avg_latency}
#     np.save(data_filename, data_dict)
#     print('Analysed DRAM data has been retrieved.')
#     return data_dict










































def analyse_dram_data(file_nr, combine_cores, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data = retrieve_data(file_nr, combine_cores)
    if end_time == -1:
        end_time = data[-1,0] + data[-1,2]

    data_filename = f'data/analysed_data_{combine_cores}_{file_nr}_{stepsize}_{start_time}-{end_time}.npy'
    if os.path.exists(data_filename):
        print('DRAM data has already been analysed.')
        return np.load(data_filename, allow_pickle=True)[()]

    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - start_time) / stepsize))
    timestamps = np.arange(start_time, end_time, stepsize)

    throughputs = {core : np.zeros([nr_bins]) for core in cores}
    throughputs_type = {core : np.zeros([2, nr_bins]) for core in cores}

    avg_count = {core : np.zeros([nr_bins]) for core in cores}
    avg_count_type = {core : np.zeros([2, nr_bins]) for core in cores}

    total_latency = {core : np.zeros([nr_bins]) for core in cores}
    total_latency_type = {core : np.zeros([2, nr_bins]) for core in cores}

    total_count_latency = {core : np.zeros([nr_bins]) for core in cores}
    total_count_latency_type = {core : np.zeros([2, nr_bins]) for core in cores}

    for t, core, latency, type in data:
        type = int(type)
        # if t > 388260000 - 1000 and t < 388260000 + 2000:
        index = int((t - start_time) / stepsize)
        t -= timestamps[index]
        # start_latency = latency
        # queueing_time = latency - SERVICE_TIME_MEM
        # execution_begin = t + latency - SERVICE_TIME_MEM
        # execution_end = t + latency
        # index = int((execution_begin - start_time) / stepsize)
        # start = execution_begin - timestamps[index]
        # end = start + SERVICE_TIME_MEM

        total_latency[core][index] += latency
        total_latency_type[core][type][index] += latency

        total_count_latency[core][index] += 1
        total_count_latency_type[core][type][index] += 1

        while latency:
            if index >= nr_bins:
                break
            if t + latency >= stepsize:
                avg_count[core][index] += (stepsize - t) / stepsize
                avg_count_type[core][type][index] += (stepsize - t) / stepsize
                # total_latency[core][index] += stepsize - t
                # total_count_latency[core][index] += (stepsize - t) / start_latency
                if t + latency - SERVICE_TIME_MEM < stepsize:
                    throughputs[core][index] += (stepsize - (t + latency - SERVICE_TIME_MEM)) / SERVICE_TIME_MEM
                    throughputs_type[core][type][index] += (stepsize - (t + latency - SERVICE_TIME_MEM)) / SERVICE_TIME_MEM
                latency -= stepsize - t
                t = 0
                index += 1
            else:
                avg_count[core][index] += latency / stepsize
                avg_count_type[core][type][index] += latency / stepsize

                # total_latency[core][index] += latency
                # total_count_latency[core][index] += latency / start_latency
                throughputs[core][index] += latency / max(SERVICE_TIME_MEM, latency)
                throughputs_type[core][type][index] += latency / max(SERVICE_TIME_MEM, latency)
                break
    # print(f'throughput: {len(data) / stepsize}')
    # print(f'avg_count: {np.sum(data[:,2]) / stepsize}')
    # print(f'avg_latency: {np.sum(data[:,2]) / len(data)}')
    # print(avg_count)
    # print(avg_count_type)

    avg_latency = {core : np.divide(total_latency[core], total_count_latency[core],
                                    out=np.zeros(len(total_latency[core])),
                                    where=total_count_latency[core]!=0) for core in cores}
    avg_latency_type = {core : np.divide(total_latency_type[core], total_count_latency_type[core],
                                    out=np.zeros_like(total_latency_type[core]),
                                    where=total_count_latency_type[core]!=0) for core in cores}


    for core in cores:
        throughputs[core] /= stepsize
        throughputs_type[core] /= stepsize

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'timestamps' : timestamps,
                 'throughputs' : throughputs,
                 'throughputs_type' : throughputs_type,
                 'avg_count' : avg_count,
                 'avg_count_type' : avg_count_type,
                 'avg_latency' : avg_latency,
                 'avg_latency_type' : avg_latency_type}

    np.save(data_filename, data_dict)
    print('Analysed DRAM data has been saved.')
    return data_dict














# def combined_dram_data(file_nr, combine_cores):
#     data_dict = analyse_dram_data(file_nr, combine_cores)
#     end_time = data_dict['end_time']
#     timestamps = data_dict['timestamps']
#     cores = data_dict['cores']
#     throughputs = data_dict['throughputs']
#     avg_count = data_dict['avg_count']
#     avg_latency = data_dict['avg_latency']

#     for core in cores:
#         print(min(throughputs[core]), max(throughputs[core]))
#         print(min(avg_count[core]), max(avg_count[core]))
#         print(min(avg_latency[core]), max(avg_latency[core]))

#     throughputs = np.sum([throughputs[core] for core in cores], axis=0)
#     avg_count = np.sum([avg_count[core] for core in cores], axis=0)
#     # avg_latency = np.array([np.sum([avg_latency[core][i] for core in cores])
#     #                                  / np.count_nonzero([avg_latency[core][i] for core in cores])
#     #                                  for i in range(len(timestamps))])
#     # print(throughputs, throughputs)

#     avg_latency = []
#     for i in range(len(timestamps)):
#         lst = [avg_latency[core][i] for core in cores]
#         tot = sum(lst)
#         count = np.count_nonzero(lst)
#         if count > 0:
#             avg_latency.append(tot / count)
#         else:
#             avg_latency.append(0)

#     data_dict = {'end_time' : end_time,
#                  'timestamps' : timestamps,
#                  'cores' : cores,
#                  'throughputs' : throughputs,
#                  'avg_count' : avg_count,
#                  'avg_latency' : np.array(avg_latency)}

#     print('DRAM data has been combined.')
#     return data_dict


def plot_time_plot(file_nr, combine_cores, only_throughput=False, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
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
                title = 'Combined cores' if combine_cores else f'Core {int(core)}'
                ax.set_title(title)
            if core == cores[0]:
                ax.set_ylabel(ylabel)

    data_dict = analyse_dram_data(file_nr, combine_cores, stepsize=stepsize, start_time=start_time, end_time=end_time)
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

    # plot_bar_chart(0, avg_count, ylabel='average number of requests',
    #                 title=True)
    if only_throughput:
        plot_bar_chart(0, throughputs, xlabel='time (ms)', ylabel='throughput', ticks=True, title=True)
    else:
        plot_bar_chart(0, throughputs, ylabel='throughput', title=True)
        plot_bar_chart(1, avg_latency, xlabel='time (ms)', ylabel='access latency', ticks=True)

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    fig.tight_layout()
    fig.savefig(f'pictures/time_plot/time_plot_{file_nr}_{stepsize}_{start_time}-{end_time}.png')

def plot_littles_law(file_nr, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    # data[:,1] = 0
    data_dict = analyse_dram_data(file_nr, combine_cores=True, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    # timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    throughputs = data_dict['throughputs'][cores[0]]
    avg_latency = data_dict['avg_latency'][cores[0]]
    avg_count = data_dict['avg_count'][cores[0]]
    diff = np.multiply(throughputs, avg_latency) - avg_count
    # print(f'{throughputs = }')
    # print(f'{avg_count = }')
    # print(f'{avg_latency = }')
    # print(f'{diff = }')
    relative_diff = np.divide(diff, avg_count,
                                out=np.zeros(len(diff)),
                                where=avg_latency!=0)
    # print(f'{throughputs = }')
    # print(f'{avg_count = }')
    # print(f'{avg_latency = }')
    # print(f'{diff = }')
    # print(f'{diff_percentage = }')
    # avg_counts_for_arrivals = {core : np.divide(total_counts_for_arrivals[core], total_arrivals[core],
    #                                             out=np.zeros(len(total_arrivals[core])),
    #                                             where=total_arrivals[core]!=0) for core in cores}
    # avg_count_for_arrivals = avg_count_for_arrivals[0]
    # latency = avg_latency[0]
    # arrivals = arrivals[0]
    # for i in range(len(avg_count_for_arrivals)):
    #     if avg_count_for_arrivals[i]>= 580 and avg_count_for_arrivals[i] <= 590:
    #         print(avg_count_for_arrivals[i], avg_latency[i], arrivals[i], timestamps[i])

    # for i in range(len(avg_count_for_arrivals)):
    #     if avg_latency[i] == 0:
    #         avg_count_for_arrivals[i] = 0

    # for i, (t, through, lat, count, diff) in enumerate(zip(timestamps,
    #                                                        throughputs,
    #                                                        avg_latency,
    #                                                        avg_count,
    #                                                        diff_percentage)):
    #     if lat == 0 and through != 0:
    #         print(f'{t = }')
    #         print(f'{through = }')
    #         print(f'{lat = }')
    #         print(f'{count = }')
    #         print(f'{diff = }')

    # avg = np.mean(relative_diff)

    # x_values = avg_count
    # log_x_values = x_values[x_values != 0]
    # log_y_values = relative_diff[x_values != 0]
    # # y_values = relative_diff
    # num_bins = 20
    # x_min, x_max = min(log_x_values), max(log_x_values)
    # log_x_min = np.log10(x_min)
    # log_x_min = -2
    # log_x_max = np.log10(x_max)
    # log_bin_size = (log_x_max - log_x_min) / num_bins
    # avgs = []
    # centers = []


    # # Iterate over each bin
    # for i in range(num_bins):
    #     log_bin_start = log_x_min + i * log_bin_size
    #     log_bin_end = log_bin_start + log_bin_size

    #     bin_indices = np.where((np.log10(log_x_values) >= log_bin_start) & (np.log10(log_x_values) < log_bin_end))[0]
    #     # print(np.log10(log_x_values) >= log_bin_start)
    #     # print(np.log10(log_x_values) < log_bin_end)
    #     # print(len(bin_indices))
    #     bin_average = np.mean(log_y_values[bin_indices])
    #     # # print(bin_average)
    #     # if i == 19:
    #     #     print(x_values[bin_indices])
    #     #     print(log_y_values[bin_indices])
    #     #     print(bin_indices)

    #     # Store the bin average and center value
    #     avgs.append(bin_average)
    #     centers.append(10 ** (log_bin_start + log_bin_size / 2))

    plt.figure(figsize=(5,5), dpi=150)
    # plt.plot(centers, avgs, color='black', label='average')


    plt.xlabel(r'$\overline{n}_M$')
    plt.ylabel('relative difference')
            #    $\left|\frac{\overline{w}_M\overline{x}_M - \overline{n}_M}{\overline{n}_M}\right|$')
    # plt.hlines(y=[0], xmin=[min(avg_count)], xmax=[max(avg_count)], color='r', label='desired value')
    # plt.hlines(y=[avg], xmin=[min(avg_count)], xmax=[max(avg_count)], color='black', label='average')
    plt.scatter(avg_count, relative_diff, s=1)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'pictures/littles_law/littles_law_{stepsize}__{start_time}-{end_time}.png')


def plot_littles_law_per_type(file_nr, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data_dict = analyse_dram_data(file_nr, combine_cores=True, stepsize=stepsize, start_time=start_time, end_time=end_time)
    end_time = data_dict['end_time']
    # timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    throughputs_type = data_dict['throughputs_type'][cores[0]]
    avg_latency_type = data_dict['avg_latency_type'][cores[0]]
    avg_count_type = data_dict['avg_count_type'][cores[0]]
    diff = np.multiply(throughputs_type, avg_latency_type) - avg_count_type
    relative_diff = np.divide(diff, avg_count_type,
                                out=np.zeros_like(diff),
                                where=avg_latency_type!=0)

    for type in [0, 1]:
        plt.figure(figsize=(5,5), dpi=150)
        plt.xlabel(r'$\overline{n}_M$')
        plt.ylabel(r'relative difference')
                #    $\left|\frac{\overline{w}\overline{x} - \overline{n}}{\overline{n}}\right|$')
        # plt.hlines(y=[0], xmin=[min(avg_count_type[type])], xmax=[max(avg_count_type[type])], color='r', label='desired value')
        # plt.hlines(y=[avg], xmin=[min(avg_count)], xmax=[max(avg_count)], color='black', label='average')
        plt.scatter(avg_count_type[type], relative_diff[type], s=1)
        plt.tight_layout()
        plt.savefig(f'pictures/littles_law/littles_law_{type}_{stepsize}__{start_time}-{end_time}.png')


# def plot_different_correlations(data, stepsizes, start_time, end_time):
#     for stepsize in stepsizes:
#         plot_littles_law(data, stepsize, start_time, end_time)

# def plot_DRAM_throughputs(file_nr):
#     data_dict = retrieve_throughputs(file_nr)
#     cores = data_dict['cores']
#     timestamps = data_dict['timestamps']
#     throughputs = data_dict['throughputs']
#     fig = plt.figure(figsize=(8,8), dpi=150)
#     throughputs = np.sum([throughputs[core] for core in cores], axis=0)
#     plt.plot(timestamps, throughputs)
#     plt.show()

def plot_arrival_times(file_nr, detailed=False, stepsize=STEPSIZE, start_time=START_TIME, end_time=END_TIME):
    data = retrieve_data(file_nr, combine_cores=True)
    end_time = data[-1,0] + data[-1,2] if end_time == -1 else end_time

    filter = np.all([data[:,0] >= start_time, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    fig = plt.figure(figsize=(5,5), dpi=150)
    read_requests = np.array([x[0] for x in data if x[3] == 0])
    read_requests_order = [i for i in range(len(data)) if data[i,3] == 0]

    write_requests = np.array([x[0] for x in data if x[3] == 1])
    write_requests_order = [i for i in range(len(data)) if data[i,3] == 1]

    ax = fig.add_subplot(1, 1, 1)
    if detailed == False:
        ax.scatter(read_requests_order, read_requests, color='black', s=5, label='read request')
        ax.scatter(write_requests_order, write_requests, color='purple', s=5, label='write request')
    else:
        ax.scatter(read_requests_order, read_requests, marker='x', color='black', s=10, label='read request')
        ax.scatter(write_requests_order, write_requests, marker='o', color='purple', s=10, label='write request')
        ax.vlines(x=range(len(data)), ymin=data[:,0], ymax=data[:,0] + data[:,2] - 9, label='time between creation and service')
        ax.vlines(x=range(len(data)), ymin=data[:,0] + data[:,2] - 9, ymax=data[:,0] + data[:,2], color='red', label='serviced by DRAM controller')
    # plt.plot(range(len(data)), data[:,0])
    ax.set_xlabel('Request scheduling time by DRAM controller')
    ax.set_ylabel('time (ns)')
    ax.legend()
    fig.subplots_adjust(left=0.17, bottom=0.1, right=0.95, top=0.95)
    fig.savefig(f'pictures/arrival_times_plot/arrival_times_plot_{detailed}_{file_nr}_{stepsize}_{start_time}-{END_TIME}')

if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    num = 1
    file_nr = DATA_FILES[benchmark][num]

    # # Little's law plots
    stepsize = 1_000_00
    start_time = 0
    end_time = -1
    # plot_littles_law(file_nr, stepsize=stepsize, start_time=start_time, end_time=end_time)
    plot_littles_law_per_type(file_nr, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # # No priority
    # start_time = 900_040_000
    # end_time = 900_300_000
    # plot_arrival_times(file_nr, detailed=False, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # # Priority
    # start_time = 932_533_200
    # end_time = 932_600_000
    # plot_arrival_times(file_nr, detailed=False, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # # Zoomed in on priority
    # start_time = 932_533_200
    # end_time = 932_534_000
    # plot_arrival_times(file_nr, detailed=False, stepsize=stepsize, start_time=start_time, end_time=end_time)

    # stepsize = 1_000_000
    # start_time = 0
    # end_time = -1
    # plot_time_plot(file_nr, combine_cores=True)

    # # one core time plot
    # combine_cores = True
    # stepsize = 1_000_000
    # start_time = 0
    # end_time = -1
    # plot_time_plot(file_nr, combine_cores=combine_cores, only_throughput=True,
    #                stepsize=stepsize, start_time=start_time, end_time=end_time)


