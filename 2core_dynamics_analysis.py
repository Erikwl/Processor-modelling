import numpy as np
import matplotlib.pyplot as plt
import sys

from service_time_derivation import *
from mva import mva
from constants import *
from main import *

def plot_0_on_1_influence():
    def f(N0):
        args[0][core0_id] = N0
        neg_service_times, service_times = find_cores_service_times(args, [core0_id], [THROUGHPUT0])
        if neg_service_times:
            return None
        args[4][core0_id] = service_times[0]
        return mva(*args)[1][-1][core0_id] - WAIT0


    args = model(2, test_model=True)
    core0_id = 0
    core1_id = 1

    pops1 = range(1, 5, 1)
    caps0 = range(1, 10, 1)

    # for i in range(len(args)):
    #     print(type(args[i]))

    x_vals = {N1 : [] for N1 in pops1}

    wait0_y_vals = {N1 : [] for N1 in pops1}
    wait1_y_vals = {N1 : [] for N1 in pops1}

    nrs1_y_vals = {N1 : [] for N1 in pops1}
    throughput1_y_vals = {N1 : [] for N1 in pops1}
    for N1 in pops1:
        args[0][core1_id] = N1

        cap0_is_too_high = False
        for cap0 in caps0:
            print(f'{N1 = }, {cap0 = }')
            if cap0_is_too_high:
                break
            args[3][core0_id] = cap0
            # print(args)
            N0 = 1
            fN0 = f(N0)
            while fN0 is None:
                N0 += 1
                fN0 = f(N0)

            if fN0 < 0:
                best_lower_diff = fN0

                while True:
                    print(N0, fN0)
                    # print(args)
                    N0 += 1
                    fN0 = f(N0)
                    if fN0 is None:
                        break
                    if fN0 > 0:
                        break
                    best_lower_diff = fN0

                if fN0 and np.abs(best_lower_diff) < np.abs(fN0):
                    N0 -= 1

            # # Find best value of N0
            # for N0 in range(cap0, 7):
            #     print(f'{N1 = }, {cap0 = }, {N0 = }')
            #     args[0][core0_id] = N0

            #     new_service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS], TIME)
            #     if new_service_times is None:
            #         continue
            #     args[4] = new_service_times

            #     wait = mva(*args)[1][-1]
            #     print(N0, wait)
            #     if wait[0] <= WAIT0:
            #         best_lower_N0 = N0
            #         best_lower_wait = wait[0]
            #     else:
            #         best_upper_N0 = N0
            #         best_upper_wait = wait[0]
            #         break

            # if min(abs(best_lower_wait - WAIT0), abs(best_upper_wait - WAIT0)) > TOL:
            #     continue
            # if abs(best_lower_wait - WAIT0) < abs(best_upper_wait - WAIT0):
            #     N0 = best_lower_N0
            # else:
            #     N0 = best_upper_N0


            args[0][core0_id] = N0
            neg_service_times, new_service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS / TIME])
            args[4][core0_id] = new_service_times[0]

            waits, throughputs = mva(*args)[1:3]
            nrs1 = throughputs[core1_id] * waits[-1,core1_id]

            # print(waits[2][0], throughputs[0])
            # args[4][1] = 0
            # waits2, throughputs2 = mva(*args)[1:3]
            # print(waits2[2][0], throughputs2[0])


            if fN0 < 10:
                x_vals[N1].append(cap0)
                wait0_y_vals[N1].append(waits[-1][core0_id])
                wait1_y_vals[N1].append(waits[-1][core1_id])

                nrs1_y_vals[N1].append(nrs1)
                throughput1_y_vals[N1].append(throughputs[1])
            else:
                break


    y_vals_lst = [wait0_y_vals, wait1_y_vals, nrs1_y_vals, throughput1_y_vals]
    ylabels = ['time (ns)', 'time (ns)', 'average number', 'throughput (per ns)']
    titles = ['Waiting time of core 0 requests',
              'Waiting time of core 1 requests',
              'Average number of core 1 requests',
              'Throughput of core 1 requests']
    ylabels = [r'$\overline{w}_{M,0}$',
              r'$\overline{w}_{M,1}$',
              r'$\overline{n}_{M,1}$',
              r'$\overline{x}_{M,1}$']
    # fig = plt.figure(figsize=(8,8), dpi=150)
    # for i, (y_vals, ylabel, title) in enumerate(zip(y_vals_lst, ylabels, titles), start=1):
    #     ax = fig.add_subplot(int(f'22{i}'))
    #     for N1 in pops1:
    #         ax.plot(x_vals[N1], y_vals[N1], label=f'N_1 = {N1}')
    #         ax.set_xlabel('Core 0 capacity')
    #         ax.set_title(title)
    #         ax.set_ylabel(ylabel)
    #         if i == 2:
    #             ax.legend()

    # fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    # fig.tight_layout()
    # # plt.legend()
    # fig.savefig(f'pictures/0_on_1_influence.png')

    for i, (y_vals, ylabel, title) in enumerate(zip(y_vals_lst, ylabels, titles), start=1):
        fig = plt.figure(figsize=(6,6), dpi=150)
        # ax = fig.add_subplot(int(f'22{i}'))
        for N1 in pops1:
            plt.plot(x_vals[N1], y_vals[N1], label=rf'$N_1 = {N1}$')
            plt.xlabel('Core 0 capacity')
            # plt.title(title)
            plt.ylabel(ylabel)
            if i == 2:
                plt.legend()

        # fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
        plt.tight_layout()
        # plt.legend()
        plt.savefig(f'pictures/0_on_1_influence_{i}.png')

def plot_cap0_pop0_dynamics():
    args = model(2, test_model=True)
    pops0 = range(1, 30, 1)
    caps0 = range(3, 22, 6)
    core0_id = 0
    print(args)
    low_cap0 = (3, 25)
    high_cap0 = (21, 25)

    low_pop0 = (9, 10)
    high_pop0 = (9, 25)

    x_vals = {cap0 : [] for cap0 in caps0}

    service_time_y_vals = {cap0 : [] for cap0 in caps0}
    waiting_time_y_vals = {cap0 : [] for cap0 in caps0}

    cap0_x_vals = {low_cap0 : [], high_cap0 : []}
    cap0_y_vals = {low_cap0 : [], high_cap0 : []}

    pop0_x_vals = {low_pop0 : [], high_pop0 : []}
    pop0_y_vals = {low_pop0 : [], high_pop0 : []}


    for cap0 in caps0:
        args[3][core0_id] = cap0

        for pop0 in pops0:
            print(f'{cap0 = }, {pop0 = }')
            args[0][core0_id] = pop0
            print(args)
            neg_service_times, service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS / TIME])
            print(service_times)
            if neg_service_times:
                continue
            args[4][core0_id] = service_times
            # print(pop, cap, new_service_times)
            wait = mva(*args)[1][-1]
            probs = mva(*args)[4][-1]

            x_vals[cap0].append(pop0)
            service_time_y_vals[cap0].append(service_times[0])
            waiting_time_y_vals[cap0].append(wait[core0_id])

            # print(probs)

            if (cap0, pop0) in [low_cap0, high_cap0]:
                print(sum(probs))
                cap0_x_vals[(cap0, pop0)] = np.arange(len(probs))
                cap0_y_vals[(cap0, pop0)] = probs
            elif (cap0, pop0) in [low_pop0, high_pop0]:
                print(sum(probs))
                pop0_x_vals[(cap0, pop0)] = np.arange(len(probs))
                pop0_y_vals[(cap0, pop0)] = probs

        # plt.plot(x_vals, y_vals, label=f'Capacity = {cap}')
    # if not service_time_plot:
        # plt.axhline(y = 60, color = 'r', linestyle = '-')
        # plt.plot(x_vals, y2_vals, label=f'nrs, C_{{{core0_id}}} = {cap}')
        # plt.show()time

    y_vals_lst = [service_time_y_vals, waiting_time_y_vals]
    titles = [f'required core service time',
              f'DRAM waiting time for core 0.']

    # fig = plt.figure(figsize=(8,8), dpi=150)
    # for i, (y_vals, title) in enumerate(zip(y_vals_lst, titles), start=1):
    #     ax = fig.add_subplot(int(f'21{i}'))
    #     ax.hlines(y=[WAIT0], xmin=[0], xmax=[pops0[-1]], linestyles='--', label='DRAM waiting time')
    #     for cap0 in caps0:
    #         ax.plot(x_vals[cap0], y_vals[cap0], label=rf'$C_0 = {cap0}$')
    #         ax.set_xlabel(r'$N_0$')
    #         ax.set_ylabel('time (ns)')
    #         ax.set_title(title)
    #         ax.legend()


    # fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    # fig.tight_layout()
    # fig.savefig(f'pictures/cap0_pop0_dynamics')

    y_labels = [r'$\left(\widetilde{\mu}_i(C_0, N_0)\right)^{-1}$',
                r'$\overline{w}_{M,0}$']

    for i, (y_vals, y_label, title) in enumerate(zip(y_vals_lst, y_labels, titles), start=1):
        fig = plt.figure(figsize=(4,4), dpi=150)
        # ax = fig.add_subplot(int(f'21{i}'))
        if i == 2:
            plt.hlines(y=[WAIT0], xmin=[0], xmax=[pops0[-1]], linestyles='--', label='desired DRAM waiting time')
        for cap0 in caps0:
            plt.plot(x_vals[cap0], y_vals[cap0], label=rf'$C_0 = {cap0}$')
            plt.xlabel(r'$N_0$')
            plt.ylabel(y_label)
            # plt.title(title)
            plt.legend()


        # fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
        fig.tight_layout()
        plt.savefig(f'pictures/cap0_pop0_dynamics_{i}')


    bar_width = 0.35
    differences = [[low_cap0, high_cap0],
                   [low_pop0, high_pop0]]
    x_vals_lst = [[cap0_x_vals[low_cap0], cap0_x_vals[high_cap0]],
                  [pop0_x_vals[low_pop0], pop0_x_vals[high_pop0]]]
    y_vals_lst = [[cap0_y_vals[low_cap0], cap0_y_vals[high_cap0]],
                  [pop0_y_vals[low_pop0], pop0_y_vals[high_pop0]]]
    titles = ['Different core 0 capacities',
              'Different core 0 population sizes']
    # fig = plt.figure(figsize=(8,8), dpi=150)
    # for i, (vec, x_vals, y_vals, title) in enumerate(zip(differences, x_vals_lst, y_vals_lst, titles), start=1):
    #     ax = fig.add_subplot(int(f'21{i}'))
    #     # print(probs_low_x_vals, probs_low_y_vals, probs_high_x_vals, probs_high_y_vals)
    #     ax.bar(np.array(x_vals[0]) - bar_width / 2, y_vals[0], width=bar_width, align='center', alpha=0.5, label=rf'$C_0 = {vec[0][0]}$, $N_0 = {vec[0][1]}$')
    #     ax.bar(np.array(x_vals[1]) + bar_width / 2, y_vals[1], width=bar_width, align='center', alpha=0.5, label=f'$C_0 = {vec[1][0]}$, $N_0 = {vec[1][1]}$')
    #     ax.vlines(6.5, [0], [max(max(y_vals[0]), max(y_vals[1]))], linestyles='--', label='DRAM capacity')
    #     ax.legend()
    #     ax.set_xlabel('Number of requests in DRAM')
    #     ax.set_ylabel('Probability')
    #     ax.set_title(title)

    # fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    # fig.tight_layout()
    # fig.savefig(f'pictures/cap0_pop0_dynamics_probs')

    for i, (vec, x_vals, y_vals, title) in enumerate(zip(differences, x_vals_lst, y_vals_lst, titles), start=1):
        fig = plt.figure(figsize=(8,3), dpi=150)
        # ax = fig.add_subplot(int(f'21{i}'))
        # print(probs_low_x_vals, probs_low_y_vals, probs_high_x_vals, probs_high_y_vals)
        plt.bar(np.array(x_vals[0]) - bar_width / 2, y_vals[0], width=bar_width, align='center', alpha=0.5, label=rf'$C_0 = {vec[0][0]}$, $N_0 = {vec[0][1]}$')
        plt.bar(np.array(x_vals[1]) + bar_width / 2, y_vals[1], width=bar_width, align='center', alpha=0.5, label=f'$C_0 = {vec[1][0]}$, $N_0 = {vec[1][1]}$')
        plt.vlines(6.5, [0], [max(max(y_vals[0]), max(y_vals[1]))], linestyles='--', label='DRAM capacity')
        plt.legend()
        plt.xlabel('Number of requests in DRAM')
        plt.ylabel('Probability')
        # plt.title(title)

        # fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
        plt.tight_layout()
        plt.savefig(f'pictures/cap0_pop0_dynamics_probs_{i}')



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('please run python3 parameter_derivation.py <num> <*params>')
    num = int(sys.argv[1])

    # Plot the influence of cap0, pop0 on the service time of the core and waiting time in DRAM.
    if num == 0:
        plot_cap0_pop0_dynamics()

    # Plot the influence graph.
    elif num == 1:
        plot_0_on_1_influence()

