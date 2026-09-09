NVCC ?= nvcc
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '/^[0-9][0-9]*\.[0-9][0-9]*$$/{s/\.//;s/^/sm_/;p;q;}')
CUDA_ARCH ?= $(if $(DETECTED_ARCH),$(DETECTED_ARCH),sm_89)
HOST_CXX ?= $(shell command -v g++-13 2>/dev/null || command -v g++)
CUDA_MAX_REGISTERS ?= 128
CUDA_ATTEMPTS_PER_THREAD ?= 4096
CUDA_THREADS ?= 128
CUDA_BLOCKS_PER_SM ?= 16

.PHONY: all clean check test test-cpu test-gpu install-smoke

all: meshcore_cuda_vanity

meshcore_cuda_vanity: cuda_vanity.cu vendor/cuda-ed25519/common.cu vendor/cuda-ed25519/fe.cu vendor/cuda-ed25519/ge.cu vendor/cuda-ed25519/sha512.cu
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -ccbin $(HOST_CXX) --maxrregcount=$(CUDA_MAX_REGISTERS) \
		-DMC_ATTEMPTS_PER_THREAD=$(CUDA_ATTEMPTS_PER_THREAD) -DMC_THREADS=$(CUDA_THREADS) \
		-DMC_BLOCKS_PER_SM=$(CUDA_BLOCKS_PER_SM) \
		-Ivendor/cuda-ed25519 cuda_vanity.cu -o $@

test: test-cpu

test-cpu:
	python3 -m unittest discover -s tests -v

test-gpu: meshcore_cuda_vanity
	RUN_CUDA_TESTS=1 python3 -m unittest discover -s tests -v

check:
	python3 -m py_compile meshcore_vanity.py tests/test_keygen.py
	bash -n install.sh uninstall.sh meshcore-vanity-keygen publish_to_github.sh

install-smoke: meshcore_cuda_vanity
	tmp_dir=$$(mktemp -d); trap 'rm -rf "$$tmp_dir"' EXIT; \
	./install.sh --prefix "$$tmp_dir/prefix" --skip-packages --skip-build --no-desktop; \
	"$$tmp_dir/prefix/bin/meshcore-vanity-keygen" --version; \
	"$$tmp_dir/prefix/bin/meshcore-vanity-keygen" --self-test; \
	test -f "$$tmp_dir/prefix/share/icons/hicolor/scalable/apps/meshcore-vanity-keygen.svg"; \
	test -f "$$tmp_dir/prefix/lib/meshcore-vanity-keygen/assets/meshcore-vanity-keygen.png"

clean:
	$(RM) meshcore_cuda_vanity
