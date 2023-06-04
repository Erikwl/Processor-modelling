from constants import *
import numpy as np
import matplotlib.pyplot as plt
import os
from math import isnan

def retrieve_data(nr):
    filename = ACCESS_DATA_PATH + 'dram_access_data_raw' + str(nr) + '.csv'

    # data_filename = f'data/data_{nr}.npy'
    # if os.path.exists(data_filename):
    #     # print('File has already been retrieved.')
    #     return np.load(data_filename, allow_pickle=True)[()]

    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    data[:,2] /= 1000 # Convert picoseconds to nanoseconds
    data[:,2] -= 45 # Subtract DRAM service time

    # np.save(data_filename, data)
    # print('File has been saved.')
    return data

# def retrieve_throughputs(file_nr):
#     data = retrieve_data(file_nr)
#     if END_TIME == -1:
#         end_time = data[-1,0] + data[-1,2] + STEPSIZE # Last dram access + latency
#     else:
#         end_time = END_TIME

#     data_filename = f'data/throughputs_{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.npy'
#     if os.path.exists(data_filename):
#         print('Throughputs have already been calculated.')
#         return np.load(data_filename, allow_pickle=True)[()]

#     filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)
#     data = data[filter]
#     cores = np.unique(data[:,1])
#     nr_bins = int(np.ceil((end_time - START_TIME) / STEPSIZE))
#     timestamps = np.arange(START_TIME, end_time, STEPSIZE)

#     throughputs = {core : np.zeros([nr_bins]) for core in cores}

#     for t, core_id, latency in data:
#         # print(t, core_id, latency)
#         # throughputs[core_id][index] += 1
#         execution_begin = t + latency - SERVICE_TIME_MEM
#         index = int((execution_begin - START_TIME) / STEPSIZE)
#         start = execution_begin - timestamps[index]
#         end = start + SERVICE_TIME_MEM
#         # print(t, latency, execution_begin, start, end, throughputs)
#         while end:
#             if index >= nr_bins:
#                 break
#             if end >= STEPSIZE:
#                 throughputs[core_id][index] += (STEPSIZE - start) / SERVICE_TIME_MEM
#                 # print(f'if{(STEPSIZE - start) / SERVICE_TIME_MEM}')
#                 end -= STEPSIZE
#                 start = 0
#                 index += 1
#             else:
#                 throughputs[core_id][index] += (end - start) / SERVICE_TIME_MEM
#                 # print(f'else:{(end - start) / SERVICE_TIME_MEM}')
#                 break
#     # print(throughputs)
#     print(f'throughput: {len(data) / STEPSIZE}')
#     print(f'avg_count: {np.sum(data[:,2]) / STEPSIZE}')
#     print(f'avg_latency: {np.sum(data[:,2]) / len(data)}')

#     for core in cores:
#         throughputs[core] /= STEPSIZE


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
#         end_time = data[-1,0] + data[-1,2] + STEPSIZE # Last dram access + latency
#     else:
#         end_time = END_TIME
#     data_filename = f'data/dram_data_{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.npy'

#     if os.path.exists(data_filename):
#         print('DRAM data has already been calculated.')
#         return np.load(data_filename, allow_pickle=True)[()]

#     filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)

#     data = data[filter]
#     cores = np.unique(data[:,1])
#     nr_bins = int(np.ceil((end_time - START_TIME) / STEPSIZE))
#     timestamps = np.arange(START_TIME, end_time, STEPSIZE)

#     # total_arrivals = {core : np.zeros([nr_bins]) for core in cores}
#     avg_count = {core : np.zeros([nr_bins]) for core in cores}
#     total_latency = {core : np.zeros([nr_bins]) for core in cores}
#     total_count_latency = {core : np.zeros([nr_bins]) for core in cores}

#     for t, core_id, latency in data:
#         index = int((t - START_TIME) / STEPSIZE)
#         start = t - timestamps[index]
#         end = start + latency
#         temp_index = index
#         # total_latency[core_id][index] += latency
#         while end:
#             if temp_index >= nr_bins:
#                 break
#             if end >= STEPSIZE:
#                 avg_count[core_id][temp_index] += (STEPSIZE - start) / STEPSIZE
#                 total_latency[core_id][temp_index] += (STEPSIZE - start) / STEPSIZE * latency
#                 total_count_latency[core_id][temp_index] += (STEPSIZE - start) / STEPSIZE
#                 # avg_count[core_id][temp_index] += latency / STEPSIZE
#                 # total_latency[core_id][temp_index] += latency
#                 # total_count_latency[core_id][temp_index] += 1
#                 end -= STEPSIZE
#                 start = 0
#                 temp_index += 1
#             else:
#                 avg_count[core_id][temp_index] += (end - start) / STEPSIZE
#                 total_latency[core_id][temp_index] += (end - start)
#                 total_count_latency[core_id][temp_index] += (end - start) / latency
#                 # avg_count[core_id][temp_index] += latency / STEPSIZE
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










































def analyse_dram_data(file_nr):
    data = retrieve_data(file_nr)
    if END_TIME == -1:
        end_time = data[-1,0] + data[-1,2] + STEPSIZE # Last dram access + latency
    else:
        end_time = END_TIME

    data_filename = f'data/analysed_data_{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.npy'
    if os.path.exists(data_filename):
        print('DRAM data has already been analysed.')
        return np.load(data_filename, allow_pickle=True)[()]

    filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    cores = np.unique(data[:,1])
    nr_bins = int(np.ceil((end_time - START_TIME) / STEPSIZE))
    timestamps = np.arange(START_TIME, end_time, STEPSIZE)

    throughputs = {core : np.zeros([nr_bins]) for core in cores}
    avg_count = {core : np.zeros([nr_bins]) for core in cores}
    total_latency = {core : np.zeros([nr_bins]) for core in cores}
    total_count_latency = {core : np.zeros([nr_bins]) for core in cores}

    for t, core, latency in data:
        # if t > 388260000 - 1000 and t < 388260000 + 2000:
        # print(t, core, latency)
        index = int((t - START_TIME) / STEPSIZE)
        t -= timestamps[index]
        # start_latency = latency
        # queueing_time = latency - SERVICE_TIME_MEM
        # execution_begin = t + latency - SERVICE_TIME_MEM
        # execution_end = t + latency
        # index = int((execution_begin - START_TIME) / STEPSIZE)
        # start = execution_begin - timestamps[index]
        # end = start + SERVICE_TIME_MEM

        total_latency[core][index] += latency
        total_count_latency[core][index] += 1


        while latency:
            if index >= nr_bins:
                break
            if t + latency >= STEPSIZE:
                avg_count[core][index] += (STEPSIZE - t) / STEPSIZE
                # total_latency[core][index] += STEPSIZE - t
                # total_count_latency[core][index] += (STEPSIZE - t) / start_latency
                if t + latency - SERVICE_TIME_MEM < STEPSIZE:
                    throughputs[core][index] += (STEPSIZE - (t + latency - SERVICE_TIME_MEM)) / SERVICE_TIME_MEM
                latency -= STEPSIZE - t
                t = 0
                index += 1
            else:
                avg_count[core][index] += latency / STEPSIZE
                # total_latency[core][index] += latency
                # total_count_latency[core][index] += latency / start_latency
                throughputs[core][index] += latency / max(SERVICE_TIME_MEM, latency)
                break
    # print(f'throughput: {len(data) / STEPSIZE}')
    # print(f'avg_count: {np.sum(data[:,2]) / STEPSIZE}')
    # print(f'avg_latency: {np.sum(data[:,2]) / len(data)}')

    avg_latency = {core : np.divide(total_latency[core], total_count_latency[core],
                                    out=np.zeros(len(total_latency[core])),
                                    where=total_count_latency[core]!=0) for core in cores}


    for core in cores:
        throughputs[core] /= STEPSIZE

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'timestamps' : timestamps,
                 'throughputs' : throughputs,
                 'avg_count' : avg_count,
                 'avg_latency' : avg_latency}

    np.save(data_filename, data_dict)
    print('Analysed DRAM data has been saved.')
    return data_dict














def combined_dram_data(file_nr):
    data_dict = analyse_dram_data(file_nr)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    throughputs = data_dict['throughputs']
    avg_count = data_dict['avg_count']
    avg_latency = data_dict['avg_latency']

    for core in cores:
        print(min(throughputs[core]), max(throughputs[core]))
        print(min(avg_count[core]), max(avg_count[core]))
        print(min(avg_latency[core]), max(avg_latency[core]))

    combined_throughputs = np.sum([throughputs[core] for core in cores], axis=0)
    combined_avg_count = np.sum([avg_count[core] for core in cores], axis=0)
    # combined_avg_latency = np.array([np.sum([avg_latency[core][i] for core in cores])
    #                                  / np.count_nonzero([avg_latency[core][i] for core in cores])
    #                                  for i in range(len(timestamps))])
    # print(throughputs, combined_throughputs)

    combined_avg_latency = []
    for i in range(len(timestamps)):
        lst = [avg_latency[core][i] for core in cores]
        tot = sum(lst)
        count = np.count_nonzero(lst)
        if count > 0:
            combined_avg_latency.append(tot / count)
        else:
            combined_avg_latency.append(0)

    data_dict = {'end_time' : end_time,
                 'timestamps' : timestamps,
                 'cores' : cores,
                 'combined_throughputs' : combined_throughputs,
                 'combined_avg_count' : combined_avg_count,
                 'combined_avg_latency' : np.array(combined_avg_latency)}

    print('DRAM data has been combined.')
    return data_dict


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
                ax.set_title(f'Core {int(core)}')
            if core == cores[0]:
                ax.set_ylabel(ylabel)

    fig = plt.figure(figsize=(5,5), dpi=150)
    data_dict = analyse_dram_data(file_nr)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    cores = data_dict['cores']
    # avg_count = data_dict['avg_count']
    throughputs = data_dict['throughputs']
    avg_latency = data_dict['avg_latency']

    # plot_bar_chart(0, avg_count, ylabel='average number of requests',
    #                 title=True)
    plot_bar_chart(0, throughputs, ylabel='throughput', title=True)
    plot_bar_chart(1, avg_latency, xlabel='time (ms)', ylabel='access latency', ticks=True)

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    fig.tight_layout()
    fig.savefig(f'pictures/time_plot{file_nr}_{STEPSIZE}_{START_TIME}-{end_time}.png')

def plot_correlation(file_nr):
    # data[:,1] = 0
    data_dict = combined_dram_data(file_nr)
    end_time = data_dict['end_time']
    timestamps = data_dict['timestamps']
    # cores = data_dict['cores']
    combined_throughputs = data_dict['combined_throughputs']
    combined_avg_latency = data_dict['combined_avg_latency']
    combined_avg_count = data_dict['combined_avg_count']
    diff = np.multiply(combined_throughputs, combined_avg_latency) - combined_avg_count
    # print(f'{combined_throughputs = }')
    # print(f'{combined_avg_count = }')
    # print(f'{combined_avg_latency = }')
    # print(f'{diff = }')
    relative_diff = np.abs(np.divide(diff, combined_avg_count,
                                out=np.zeros(len(diff)),
                                where=combined_avg_latency!=0))
    # print(f'{combined_throughputs = }')
    # print(f'{combined_avg_count = }')
    # print(f'{combined_avg_latency = }')
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
    #                                                        combined_throughputs,
    #                                                        combined_avg_latency,
    #                                                        combined_avg_count,
    #                                                        diff_percentage)):
    #     if lat == 0 and through != 0:
    #         print(f'{t = }')
    #         print(f'{through = }')
    #         print(f'{lat = }')
    #         print(f'{count = }')
    #         print(f'{diff = }')

    # avg = np.mean(relative_diff)

    x_values = combined_avg_count
    log_x_values = combined_avg_count[combined_avg_count != 0]
    log_y_values = relative_diff[combined_avg_count != 0]
    # y_values = relative_diff
    num_bins = 20
    x_min, x_max = min(log_x_values), max(log_x_values)
    log_x_min = np.log10(x_min)
    log_x_min = -2
    log_x_max = np.log10(x_max)
    log_bin_size = (log_x_max - log_x_min) / num_bins
    avgs = []
    centers = []


    # Iterate over each bin
    for i in range(num_bins):
        log_bin_start = log_x_min + i * log_bin_size
        log_bin_end = log_bin_start + log_bin_size

        print(log_bin_start, log_bin_end)

        bin_indices = np.where((np.log10(log_x_values) >= log_bin_start) & (np.log10(log_x_values) < log_bin_end))[0]
        # print(np.log10(log_x_values) >= log_bin_start)
        # print(np.log10(log_x_values) < log_bin_end)
        # print(len(bin_indices))
        bin_average = np.mean(log_y_values[bin_indices])
        # # print(bin_average)
        # if i == 19:
        #     print(x_values[bin_indices])
        #     print(log_y_values[bin_indices])
        #     print(bin_indices)

        # Store the bin average and center value
        avgs.append(bin_average)
        centers.append(10 ** (log_bin_start + log_bin_size / 2))

    print(centers)
    plt.figure(figsize=(5,5), dpi=150)
    plt.plot(centers, avgs, color='black', label='average')


    plt.xlabel(r'$\overline{n}_M$')
    plt.ylabel(r'$\left|\frac{\overline{w}_M\overline{x}_M - \overline{n}_M}{\overline{n}_M}\right|$')
    plt.hlines(y=[0], xmin=[min(combined_avg_count)], xmax=[max(combined_avg_count)], color='r', label='desired value')
    # plt.hlines(y=[avg], xmin=[min(combined_avg_count)], xmax=[max(combined_avg_count)], color='black', label='average')
    plt.scatter(combined_avg_count, relative_diff, s=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pictures/littles_law_{STEPSIZE}__{START_TIME}-{end_time}.png')

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

def plot_arrival_times(file_nr):
    data = retrieve_data(file_nr)
    if END_TIME == -1:
        end_time = data[-1,0] + data[-1,2] + STEPSIZE # Last dram access + latency
    else:
        end_time = END_TIME

    filter = np.all([data[:,0] >= START_TIME, data[:,0] + data[:,2] <= end_time], axis=0)
    data = data[filter]
    fig = plt.figure(figsize=(6,6), dpi=150)
    plt.scatter(range(len(data)), data[:,0], color='black', s=0.5, label='time of request creation by core 0')
    # plt.vlines(x=range(len(data)), ymin=data[:,0], ymax=data[:,0] + data[:,2] - 9, label='waiting time for DRAM controller')
    # plt.vlines(x=range(len(data)), ymin=data[:,0] + data[:,2] - 9, ymax=data[:,0] + data[:,2], color='red', label='serviced by DRAM controller')
    # plt.plot(range(len(data)), data[:,0])
    plt.xlabel('Arrival order in DRAM')
    plt.ylabel('time (ns)')
    plt.legend()
    plt.savefig(f'pictures/arrival_times_plot_{file_nr}_{STEPSIZE}_{START_TIME}-{END_TIME}')

if __name__ == '__main__':
    benchmark = 'parsec-bodytrack'
    num = 1
    file_nr = DATA_FILES[benchmark][num]
    # data = retrieve_data(file_nr)
    START_TIMEs = [21_100_000, 0]
    end_times = [21_200_000, 1_000_000_000]
    STEPSIZEs = [53, 1_000_000]
    # plot_DRAM_throughputs(file_nr)
    # for start in range(100_000_000, 1000_000_000, 100_000_000):
    START_TIME = 930_000_000
    # START_TIME = 0
    STEPSIZE = 1_000
    END_TIME = -1
    # END_TIME = 930_000_000
    # START_TIME = 935_124_500
    # END_TIME = 935_125_250
    # END_TIME = 935_180_000
    # plot_time_plot(file_nr)
    # plot_arrival_times(file_nr)
    plot_correlation(file_nr)
    # for stepsize in [500, 1000, 5000, 10000]:
        # STEPSIZE = stepsize
    # data = retrieve_data(file_nr)
    # print(data[:10])
    # combined_through_lat_count(file_nr)
    STEPSIZE = 1_00
    START_TIME = 0
    END_TIME = 1_000_00
    # plot_time_plot(file_nr)





