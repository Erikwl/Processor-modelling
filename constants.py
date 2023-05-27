import numpy as np

RESULTS_PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/CoMeT/results/'
PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/'
ACCESS_DATA_PATH = PATH + 'data/dram_access_data/'

DATA_FILES = {'parsec-blackscholes' : {1 : 109, 2 : 110, 3 : 111},
              'parsec-bodytrack'    : {1 : 112, 2 : 113, 3 : 114}}


BENCHMARKS = [
            'parsec-blackscholes',
            'parsec-bodytrack',
            'parsec-canneal',
            'parsec-dedup',
            'parsec-fluidanimate',
            'parsec-streamcluster',
            'parsec-swaptions',
            'parsec-x264',
            'splash2-barnes',
            'splash2-fmm',
            'splash2-ocean.cont',
            'splash2-ocean.ncont',
            'splash2-radiosity',
            'splash2-raytrace',
            'splash2-water.nsq',
            'splash2-water.sp',
            'splash2-cholesky',
            'splash2-fft',
            'splash2-lu.cont',
            'splash2-lu.ncont',
            'splash2-radix'
            ]

# Mem constants.
CAP_MEM = 6
SERVICE_TIME_MEM = 53

# Core1 constants
CAP1 = 2
SERVICE_TIME1 = 100
N1 = 3

# Core 1 waiting time and throughput (= num0_dram_requests / time).
WAIT0 = 65
NUM0_DRAM_REQUESTS = 9
TIME = 100
THROUGHPUT = NUM0_DRAM_REQUESTS / TIME

TOL = 0.5
COMPLETE_PROBS = False
MAX_POP_SIZE = 25

