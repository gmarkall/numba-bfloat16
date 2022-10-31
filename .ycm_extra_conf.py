import os
import sys

from pathlib import Path


CUDA_INCLUDE_DIR = '/usr/local/cuda/include'

flags = [
    '--cuda-gpu-arch=sm_50',
    f'-I{CUDA_INCLUDE_DIR}',
]


def Settings(**kwargs):
    return {'flags': flags}


if __name__ == '__main__':
    print(Settings())
