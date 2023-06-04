from data_retrieval import *
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from constants import *


file_nrs = np.arange(176, 196)
bus_bandwidths = np.arange(2, 40, 2)

datas = [[] for _ in range(4)]
for file_nr, bus_bandwidth in zip(file_nrs, bus_bandwidths):
    data = retrieve_data(file_nr)
    data = data[:,0]
    data = data[data < 100_000]
    print(file_nr, bus_bandwidth, sorted(Counter(data).items()))
    data = np.sort(np.unique(data))
    for i, elem in enumerate(data):
        datas[i].append(elem)

datas = np.array(datas)
print(datas[3]-datas[2])
print(np.multiply(datas[1]-datas[0], bus_bandwidths))

for i in [0,1,2,3]:
    plt.plot(bus_bandwidths, datas[i])
# plt.show()

filename = PATH + 'data/dram_read_data/dram_read_data131.csv'
file = open(filename, 'r')
s = 0
i = 1
for x in file:
    if i:
        i = 0
        continue
    # print(x)
    x = x[:-2].split(',')
    # print(x)
    s += sum(map(int, x[1].split(' ')))
print(s)
