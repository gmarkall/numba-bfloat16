NVCC = /usr/local/cuda-11.8/bin/nvcc
NVCCFLAGS = -gencode arch=compute_80,code=sm_80

all:
	$(NVCC) $(NVCCFLAGS) -rdc true -ptx functions.cu 
