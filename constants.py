RESULTS_PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/CoMeT/results/'
PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/'
ACCESS_DATA_PATH = PATH + 'data/dram_access_data/'

DATA_FILES = {'parsec-blackscholes' : {1 : 128, 2 : 129, 3 : 130},
              'parsec-bodytrack'    : {1 : 206, 2 : 126, 3 : 127},
            #   'parsec-blackscholes' : {1 : 109, 2 : 110, 3 : 111},
            #   'parsec-bodytrack'    : {1 : 112, 2 : 113, 3 : 114},
              'parsec-dedup' : {1 : 132},
              'parsec-fluidanimate' : {1 : 133},
              'parsec-streamcluster' : {1 : 134},
              'parsec-swaptions' : {1 : 135}}

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

# DRAM data constants
STEPSIZE = 1000
# STEPSIZE = 1_000_000
# START_TIME = 930_310_000
START_TIME = 932_533_200
# START_TIME = 0
# END_TIME = 930_320_000
# END_TIME = 932_534_000
END_TIME = 932_600_000
# END_TIME = 8_000
# END_TIME = -1

# Mem constants.
CAP_MEM = 1
SERVICE_TIME_MEM = 9.010

# Core1 constants
CAP1 = 2
SERVICE_TIME1 = 100
N1 = 3

# Core0 constants
CAP0 = 2
SERVICE_TIME0 = 100
N0 = 3

# Core 1 waiting time and throughput (= num0_dram_requests / time).
# Cap0_pop0
# WAIT0 = 75
# NUM0_DRAM_REQUESTS = 9

# 0_on_1_influence
WAIT0 = 25
NUM0_DRAM_REQUESTS = 7
TIME = 100
THROUGHPUT0 = NUM0_DRAM_REQUESTS / TIME

TOL = 0.5
COMPLETE_PROBS = False
MAX_POP_SIZE = 40

