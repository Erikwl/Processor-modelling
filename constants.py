RESULTS_PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/CoMeT/results/'
PATH = '/home/erik/Documents/jaar3-bach/bachelorthesis/code/'
ACCESS_DATA_PATH = PATH + 'data/dram_access_data/'

NUMPY_FILES_PATH = PATH + 'tests/numpy_files/'
ONE_GHZ_DATA_FILE_NUMBER = 106


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

MEM_CAPACITY = 6
SERVICE_TIME = 45
PROCESSING_TIME = 8
MEM_TIME = SERVICE_TIME + PROCESSING_TIME

