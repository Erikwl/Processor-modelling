RESULTS_PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/CoMeT/results/'
PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/'
ACCESS_DATA_PATH = PATH + 'data/dram_access_data/'

# DATA_FILES = {'parsec-blackscholes' : {1 : 128, 2 : 129, 3 : 130},
#               'parsec-bodytrack'    : {1 : 206, 2 : 126, 3 : 127},
#             #   'parsec-blackscholes' : {1 : 109, 2 : 110, 3 : 111},
#             #   'parsec-bodytrack'    : {1 : 112, 2 : 113, 3 : 114},
#               'parsec-dedup' : {1 : 132},
#               'parsec-fluidanimate' : {1 : 133},
#               'parsec-streamcluster' : {1 : 134},
#               'parsec-swaptions' : {1 : 135}}

# DATA_FILES = {'parsec-blackscholes' : {1 : [128], 2 : [129], 3 : [130]},
#               'parsec-bodytrack' : {1 : [206, 207, 208], 2: [209, 211], 3 : [210]},
#               'parsec-dedup' : {1 : [132]},
#               'parsec-fluidanimate' : {1 : [133]},
#               'parsec-streamcluster' : {1 : [213, 214, 215], 2 : [216], 3 : [217]},
#               'parsec-swaptions' : {1 : [135]},
#               'parsec-bodytrack,parsec-streamcluster' : {(2,2) : 236}}

DATA_FILES = {'parsec-blackscholes' : {1 : [226, 227]},
              'parsec-blackscholes,parsec-bodytrack' : {(1, 2) : [223], (2, 2) : [225]},
              'parsec-blackscholes,parsec-streamcluster' : {(1, 2) : [241], (2, 2) : [242]},

              'parsec-fluidanimate' : {1 : [234, 235]},
              'parsec-fluidanimate,parsec-bodytrack' : {(1, 2) : [232], (2, 2) : [233]},
              'parsec-fluidanimate,parsec-streamcluster' : {(1, 2) : [245], (2, 2) : [246]},

              'parsec-swaptions' : {1 : [231, 230]},
              'parsec-swaptions,parsec-bodytrack' : {(1, 2) : [228], (2, 2) : [229]},
              'parsec-swaptions,parsec-streamcluster' : {(1, 2) : [243], (2, 2) : [244]},

              'parsec-dedup' : {1 : [132, 247], 2 : [248]},

              'parsec-bodytrack' : {1 : [206, 207, 208], 2: [209, 211], 3 : [210]},

              'parsec-streamcluster' : {1 : [213, 214, 215], 2 : [216], 3 : [217]},

              'parsec-bodytrack,parsec-streamcluster' : {(2, 2) : [236]}}

NUM_CORES_USED = {'parsec-blackscholes' : 2,
                  'parsec-bodytrack' : 3,
                  'parsec-dedup' : 4,
                  'parsec-fluidanimate' : 2,
                  'parsec-streamcluster' : 2,
                  'parsec-swaptions' : 2}

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

MARKERS = ['.', '1', '*', 'x', 'v', 'd']

# DRAM data constants
STEPSIZE = 1000
START_TIME = 0
END_TIME = -1


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

