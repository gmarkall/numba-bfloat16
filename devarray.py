from numba import cuda
from prototype import bfloat16

x = cuda.device_array(1, dtype=bfloat16)
