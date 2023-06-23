import numpy as np
import matplotlib.pyplot as plt
import sys

from service_time_derivation import *
from mva_c_version import mva
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
    # pops1 = [2]
    caps0 = range(1, 10, 1)

    x_vals = {N1 : [] for N1 in pops1}

    wait0_y_vals = {N1 : [] for N1 in pops1}
    throughput0_y_vals = {N1 : [] for N1 in pops1}

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


            args[0][core0_id] = N0
            neg_service_times, new_service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS / TIME])
            args[4][core0_id] = new_service_times[0]

            waits, throughputs = mva(*args)[1:3]
            nrs1 = throughputs[core1_id] * waits[-1,core1_id]

            if fN0 < 10:
                x_vals[N1].append(cap0)
                wait0_y_vals[N1].append(waits[-1][core0_id])
                throughput0_y_vals[N1].append(throughputs[0])

                wait1_y_vals[N1].append(waits[-1][core1_id])
                nrs1_y_vals[N1].append(nrs1)
                throughput1_y_vals[N1].append(throughputs[1])
            else:
                break


    # y_vals_lst = [wait0_y_vals, wait1_y_vals, nrs1_y_vals, throughput1_y_vals]
    # ylabels = ['time (ns)', 'time (ns)', 'average number', 'throughput (per ns)']
    # titles = ['Waiting time of core 0 requests',
    #           'Waiting time of core 1 requests',
    #           'Average number of core 1 requests',
    #           'Throughput of core 1 requests']
    # ylabels = [r'$\overline{w}_{M,0}$',
    #           r'$\overline{w}_{M,1}$',
    #           r'$\overline{n}_{M,1}$',
    #           r'$\overline{x}_{M,1}$']
    # y_vals_lst = [wait0_y_vals, throughput1_y_vals, wait1_y_vals]

    ylabels = [r'$\overline{w}_{M,0}$', r'$\overline{x}_{M,0}$', r'$\overline{w}_{M,1}$', r'$\overline{x}_{M,1}$']
    # titles = ['Core 0', 'Core 1', 'Core 1']
    titles = ['Core 0', 'Core 0', 'Core 1', 'Core 1']
    # ylabels = ['DRAM doorvoer', 'DRAM wachttijd (ns)','DRAM doorvoer', 'DRAM wachttijd (ns)']
    y_vals_lst = [throughput0_y_vals, wait0_y_vals,
                  throughput1_y_vals, wait1_y_vals]

    for i, (y_vals, ylabel, title) in enumerate(zip(y_vals_lst, ylabels, titles), start=1):
        fig = plt.figure(figsize=(4,3), dpi=150)
        # if i == 1:
        #     ax = fig.add_subplot(int(f'21{i}'))
        # else:
        ax = fig.add_subplot(1, 1, 1)
        for N1 in pops1:
            ax.plot(x_vals[N1], y_vals[N1], label=rf'$N_1 = {N1}$')
            ax.set_xlabel(r'$C_0$')
            if i == 1:
                ax.set_ylim((0.05, 0.1))
            # ax.set_title(title)
            ax.set_ylabel(ylabel)
        if i == 2:
            ax.legend()

        fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
        # fig.tight_layout()
        # ax.legend()
        fig.savefig(f'pictures/0_on_1_influence_{i}.png')

    # plt.savefig(f'pictures/0_on_1_influence.png')

def plot_cap0_pop0_dynamics():
    args = model(2, test_model=True)
    pops0 = range(1, 30, 1)
    caps0 = range(3, 22, 6)
    core0_id = 0
    # print(args)
    low_cap0 = (3, 10)
    high_cap0 = (21, 10)

    low_pop0 = (9, 5)
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
            # print(f'{cap0 = }, {pop0 = }')
            args[0][core0_id] = pop0
            # print(args)
            neg_service_times, service_times = find_cores_service_times(args, [core0_id], [NUM0_DRAM_REQUESTS / TIME])
            # print(service_times)
            if neg_service_times:
                continue
            args[4][core0_id] = service_times
            # print(pop, cap, new_service_times)
            wait = mva(*args)[1][-1]
            probs = mva(*args, complete_probs=True)[4][-1]

            x_vals[cap0].append(pop0)
            service_time_y_vals[cap0].append(service_times[0])
            waiting_time_y_vals[cap0].append(wait[core0_id])

            # print(probs)

            if (cap0, pop0) in [low_cap0, high_cap0]:
                print(probs)
                print(sum(probs))
                cap0_x_vals[(cap0, pop0)] = np.arange(len(probs))
                cap0_y_vals[(cap0, pop0)] = probs
            elif (cap0, pop0) in [low_pop0, high_pop0]:
                # print(sum(probs))
                pop0_x_vals[(cap0, pop0)] = np.arange(len(probs))
                pop0_y_vals[(cap0, pop0)] = probs


    y_vals_lst = [service_time_y_vals, waiting_time_y_vals]
    titles = [f'required core service time',
              f'DRAM waiting time for core 0.']

    y_labels = [r'$\left(\widetilde{\mu}_0\right)^{-1}$',
                r'$\overline{w}_{M,0}$']
    # y_labels = ['servicetijd (ns)',
    #             'DRAM wachttijd (ns)']

    for i, (y_vals, y_label, title) in enumerate(zip(y_vals_lst, y_labels, titles), start=1):
        fig = plt.figure(figsize=(4,4), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        if i == 2:
            ax.hlines(y=[WAIT0], xmin=[0], xmax=[pops0[-1]], linestyles='--', label='measured DRAM access latency')
        for j, cap0 in enumerate(caps0):
            ax.plot(x_vals[cap0], y_vals[cap0], marker=MARKERS[j], label=rf'$C_0 = {cap0}$')
            ax.set_xlabel(r'$N_0$')
            ax.set_ylabel(y_label)
            # ax.title(title)
            ax.legend()


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

    for i, (vec, x_vals, y_vals, title) in enumerate(zip(differences, x_vals_lst, y_vals_lst, titles), start=1):
        fig = plt.figure(figsize=(8,3), dpi=150)
        plt.bar(np.array(x_vals[0]) - bar_width / 2, y_vals[0], color='black', fill=True, width=bar_width, align='center', label=rf'$C_0 = {vec[0][0]}$, $N_0 = {vec[0][1]}$')
        plt.bar(np.array(x_vals[1]) + bar_width / 2, y_vals[1], fill=False, width=bar_width, align='center', label=f'$C_0 = {vec[1][0]}$, $N_0 = {vec[1][1]}$')
        plt.vlines(CAP_MEM + 0.5, [0], [max(max(y_vals[0]), max(y_vals[1]))], colors='black', linestyles='--', label='DRAM controller capacity')
        plt.legend()
        plt.xlabel('Number of requests in DRAM')
        plt.ylabel('Probability')
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

