NVCC ?= nvcc
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '/^[0-9][0-9]*\.[0-9][0-9]*$$/{s/\.//;s/^/sm_/;p;q;}')
CUDA_ARCH ?= $(if $(DETECTED_ARCH),$(DETECTED_ARCH),sm_89)
HOST_CXX ?= $(shell command -v g++-13 2>/dev/null || command -v g++)
CUDA_MAX_REGISTERS ?= 128
CUDA_ATTEMPTS_PER_THREAD ?= 512

.PHONY: all clean test test-cpu test-gpu

all: meshcore_cuda_vanity

meshcore_cuda_vanity: cuda_vanity.cu vendor/cuda-ed25519/common.cu vendor/cuda-ed25519/fe.cu vendor/cuda-ed25519/ge.cu vendor/cuda-ed25519/sha512.cu
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -ccbin $(HOST_CXX) --maxrregcount=$(CUDA_MAX_REGISTERS) \
		-DMC_ATTEMPTS_PER_THREAD=$(CUDA_ATTEMPTS_PER_THREAD) -Ivendor/cuda-ed25519 cuda_vanity.cu -o $@

test: test-cpu

test-cpu:
	python3 -m unittest discover -s tests -v

test-gpu: meshcore_cuda_vanity
	RUN_CUDA_TESTS=1 python3 -m unittest discover -s tests -v

clean:
	$(RM) meshcore_cuda_vanity
