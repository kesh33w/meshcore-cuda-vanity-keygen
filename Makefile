NVCC ?= nvcc
CUDA_ARCH ?= sm_89

.PHONY: all clean test

all: meshcore_cuda_vanity

meshcore_cuda_vanity: cuda_vanity.cu vendor/cuda-ed25519/common.cu vendor/cuda-ed25519/fe.cu vendor/cuda-ed25519/ge.cu vendor/cuda-ed25519/sha512.cu
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -ccbin g++-13 -Ivendor/cuda-ed25519 cuda_vanity.cu -o $@

test: meshcore_cuda_vanity
	./meshcore_cuda_vanity --prefix a

clean:
	$(RM) meshcore_cuda_vanity
