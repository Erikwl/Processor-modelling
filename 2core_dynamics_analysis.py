import numpy as np
import matplotlib.pyplot as plt
import sys

from service_time_derivation import find_cores_service_times
from mva import mva
from constants import *
from main import model

def plot_1_on_2_influence():
    args = model(2)
    core0_id = 0
    core1_id = 1

    core1_pops_lst = range(1, 2, 1)

    # for i in range(len(args)):
    #     print(type(args[i]))

    x_vals = {N1 : [] for N1 in core1_pops_lst}

    wait0_y_vals = {N1 : [] for N1 in core1_pops_lst}
    wait1_y_vals = {N1 : [] for N1 in core1_pops_lst}

    throughput0_y_vals = {N1 : [] for N1 in core1_pops_lst}
    throughput1_y_vals = {N1 : [] for N1 in core1_pops_lst}
    for N1 in core1_pops_lst:
        args[0][core1_id] = N1

        cap0_is_too_high = False
        for cap0 in range(1, 2):
            if cap0_is_too_high:
                break
            args[3][core0_id] = cap0

            best_lower_N0 = -1
            best_lower_wait = -1

            best_upper_N0 = -1
            best_upper_wait = -1

            # Find best value of N0
            for N0 in range(cap0, 7):
                print(f'{N1 = }, {cap0 = }, {N0 = }')
                args[0][core0_id] = N0

                new_service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS], TIME)
                if new_service_times is None:
                    continue
                args[4] = new_service_times

                wait = mva(*args)[1][-1]
                print(N0, wait)
                if wait[0] <= WAIT0:
                    best_lower_N0 = N0
                    best_lower_wait = wait[0]
                else:
                    best_upper_N0 = N0
                    best_upper_wait = wait[0]
                    break

            if min(abs(best_lower_wait - WAIT0), abs(best_upper_wait - WAIT0)) > TOL:
                continue
            if abs(best_lower_wait - WAIT0) < abs(best_upper_wait - WAIT0):
                N0 = best_lower_N0
            else:
                N0 = best_upper_N0
            args[0][core0_id] = N0
            new_service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS], TIME)
            args[4] = new_service_times

            throughput = mva(*args)[2]
            wait = mva(*args)[1][-1]

            x_vals[N1].append(cap0)
            wait0_y_vals[N1].append(wait[0])
            wait1_y_vals[N1].append(wait[1])

            throughput0_y_vals[N1].append(throughput[0])
            throughput1_y_vals[N1].append(throughput[1])

    fig = plt.figure(figsize=(8,8), dpi=150)

    y_vals_lst = [wait0_y_vals, wait1_y_vals, throughput0_y_vals, throughput1_y_vals]
    ylabels_lst = ['Waiting time of core 0 requests',
                   'Waiting time of core 1 requests',
                   'Throughput of core 0 requests',
                   'Throughput of core 1 requests']
    for i, (y_vals, ylabel) in enumerate(zip(y_vals_lst, ylabels_lst), start=1):
        ax = fig.add_subplot(int(f'22{i}'))
        for N1 in core1_pops_lst:
            ax.plot(x_vals[N1], y_vals[N1], label=f'N_1 = {N1}')
            ax.set_xlabel('Core 0 capacity')
            ax.set_ylabel(ylabel)

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    fig.tight_layout()
    plt.legend()
    plt.savefig(f'pictures/1_on_2_influence.png')

def plot_cap0_pop0_dynamics():
    args = model(2)
    pops0 = range(1, 46, 1)
    caps0 = range(1, 41, 10)
    core0_id = 0

    x_vals = {cap0 : [] for cap0 in caps0}

    service_time_y_vals = {cap0 : [] for cap0 in caps0}
    waiting_time_y_vals = {cap0 : [] for cap0 in caps0}
    # probs_x_vals = {(pop0, cap0) : }
    # probs = {(pop, cap) : []}

    for cap0 in caps0:
        args[3][core0_id] = cap0

        for pop0 in pops0:
            args[0][core0_id] = pop0

            new_service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS], TIME)
            if new_service_times is None:
                continue
            args[4] = new_service_times
            # print(pop, cap, new_service_times)
            wait = mva(*args)[1][-1]

            x_vals[cap0].append(pop0)
            service_time_y_vals[cap0].append(new_service_times[0])
            waiting_time_y_vals[cap0].append(wait[core0_id])

        # plt.plot(x_vals, y_vals, label=f'Capacity = {cap}')
    # if not service_time_plot:
        # plt.axhline(y = 60, color = 'r', linestyle = '-')
        # plt.plot(x_vals, y2_vals, label=f'nrs, C_{{{core0_id}}} = {cap}')
        # plt.show()time

    y_vals_lst = [service_time_y_vals, waiting_time_y_vals]
    titles = [f'Service time of core 0 to get a DRAM throughput of {THROUGHPUT:.2f}.',
              f'Waiting time when service time for a DRAM throughput of {THROUGHPUT:.2f}.']

    fig = plt.figure(figsize=(8,8), dpi=150)
    for i, (y_vals, title) in enumerate(zip(y_vals_lst, titles), start=1):
        ax = fig.add_subplot(int(f'21{i}'))
        for cap0 in caps0:
            ax.plot(x_vals[cap0], y_vals[cap0], label=f'Cap = {cap0}')
            ax.set_xlabel('Population size')
            ax.set_ylabel('time (ns)')
            ax.set_title(title)
            plt.legend()
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    fig.tight_layout()
    plt.savefig(f'pictures/cap0_pop0_dynamics')

    # ax = plt.axes(projection='3d')
    # print(x_vals, y_vals, z_vals)
    # ax.scatter3D(x_vals, y_vals, z_vals, 'gray')
    # ax.set_xlabel(rf'$N_{core0_id}$')
    # ax.set_ylabel(rf'$C_{core0_id}$')
    # ax.set_zlabel(rf'$w_{{{core0_id},{core0_id}}}$')
    # plt.xlabel('population size')
    # if service_time_plot:
    #     plt.ylabel('core service time')
    # else:
    #     plt.ylabel('avg number of requests in DRAM')
    # plt.legend()
    # plt.title('Observed throughput of 0.07 and waiting time of 60')
    # # plt.title(f'num DRAM requests = {num_dram_requests}, time = {time}')
    # if service_time_plot:
    #     plt.savefig(f'pictures/1core_service_time_graph_{num_dram_requests}_{time}')
    # else:
    #     plt.savefig(f'pictures/1core_graph_{num_dram_requests}_{time}')



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('please run python3 parameter_derivation.py <num> <*params>')
    num = int(sys.argv[1])

    # Plot the influence of cap0, pop0 on the service time of the core and waiting time in DRAM.
    if num == 0:
        plot_cap0_pop0_dynamics()

    # Plot the influence graph.
    elif num == 1:
        plot_1_on_2_influence()

