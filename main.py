import numpy as np
import sys
import json
from time import time as current_time
from mva import mva

from constants import *

def read_json(filename):
    try:
        file = open(filename)
    except Exception as e:
        print(e)
        print('Please run: python3 main.py <model filename>')
        exit(0)

    data = json.load(file)
    pop_vector = np.array(data["population vector"], dtype='int')
    refs = np.array(data["reference stations"], dtype='int')
    visits = np.array(data["visits"], dtype="float")
    caps = np.array(data["capacities"], dtype="int")
    service_times = np.array(data["service times"], dtype="float")
    return [pop_vector, refs, visits, caps, service_times]

def model(n, test_model=False):
    pop_vector = np.zeros(n, dtype=int)
    refs = np.ones(n, dtype=int) * n
    visits = np.eye(n + 1, n)
    service_times = np.zeros(n + 1)
    caps = np.zeros(n + 1, dtype=int)

    visits[-1] = np.ones(n)
    caps[-1] = CAP_MEM
    service_times[-1] = SERVICE_TIME_MEM
    if n == 2 and test_model == True:
        caps[0:2] = [CAP0, CAP1]
        service_times[0:2] = [SERVICE_TIME0, SERVICE_TIME1]
        pop_vector[0:2] = [N0, N1]
    return [pop_vector, refs, visits, caps, service_times]

def time_execution(args):
    start = current_time()
    for _ in range(1000):
        mva(*args)
    print(f'The mva algo took {(current_time() - start):.4f} ms')

def verify_model():
    args = read_json('models/1core_model.json')
    perf_measures = mva(*args)
    # print(perf_measures)
    for i, name in enumerate(['numbers', 'waits', 'throughputs', 'utils', 'probs']):
        print(f'{name}: {perf_measures[i]}')

if __name__ == '__main__':
    num = int(sys.argv[1])
    # Verify the correctness of the model by comparing it to a known model.
    if num == 0:
        verify_model()
    # Time the model.
    elif num == 1:
        args = read_json('models/timing_model.json')
        time_execution(args)
