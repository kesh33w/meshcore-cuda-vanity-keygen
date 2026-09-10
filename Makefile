NVCC ?= nvcc
HOST_CXX ?= $(shell command -v g++-13 2>/dev/null || command -v g++)
NVCCFLAGS ?= -O3
CUDA_MAX_REGISTERS ?= 128
CUDA_ATTEMPTS_PER_THREAD ?= 4096
CUDA_THREADS ?= 128
CUDA_BLOCKS_PER_SM ?= 16
CUDA_MIN_MAJOR := 11
CUDA_MIN_MINOR := 8

# When CUDA_ARCH is explicitly supplied, retain the historical single-arch
# nvcc interface (notably, CI uses CUDA_ARCH=sm_75 without a GPU). Otherwise,
# compile SASS for every distinct visible compute capability and retain PTX for
# the newest one so the binary can still JIT on a compatible newer GPU.
DETECTED_CUDA_ARCHES := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
	sed -n '/^[0-9][0-9]*\.[0-9][0-9]*$$/{s/\.//;s/^/sm_/;p;}' | sort -t_ -k2,2n -u)
ifeq ($(origin CUDA_ARCH),undefined)
CUDA_ARCH_MODE := auto
CUDA_ARCHES := $(if $(strip $(DETECTED_CUDA_ARCHES)),$(DETECTED_CUDA_ARCHES),sm_89)
CUDA_HIGHEST_ARCH := $(lastword $(CUDA_ARCHES))
cuda_compute = compute_$(patsubst sm_%,%,$(1))
CUDA_ARCH_FLAGS := $(foreach arch,$(CUDA_ARCHES),-gencode arch=$(call cuda_compute,$(arch)),code=$(arch)) \
	-gencode arch=$(call cuda_compute,$(CUDA_HIGHEST_ARCH)),code=$(call cuda_compute,$(CUDA_HIGHEST_ARCH))
else
CUDA_ARCH_MODE := explicit
CUDA_ARCHES := $(strip $(CUDA_ARCH))
ifeq ($(CUDA_ARCHES),)
$(error CUDA_ARCH must not be empty when explicitly set)
endif
CUDA_ARCH_FLAGS := -arch=$(CUDA_ARCHES)
endif

empty :=
space := $(empty) $(empty)
comma := ,
CUDA_ARCH_LIST := $(subst $(space),$(comma),$(strip $(CUDA_ARCHES)))
CUDA_DEPFILE := meshcore_cuda_vanity.d
CUDA_VENDOR_DEPS := $(wildcard vendor/cuda-ed25519/*.cu vendor/cuda-ed25519/*.h)
CUDA_RARE_HEADER := generated/rare_rules_default.cuh
CUDA_RARE_INPUTS := rare_rules.json rare_rules.py tools/generate_rare_rules.py
CUDA_GENERATED_DEPS := $(CUDA_RARE_HEADER) $(wildcard rare_rules_generated.h generated/rare_rules_generated.h)

# The source/configuration hash is both a make prerequisite and a non-secret
# build fingerprint reported by --probe. A changed tuning, compiler, arch, or
# engine source therefore cannot accidentally reuse an old executable.
CUDA_SOURCE_ID := $(shell sha256sum Makefile cuda_vanity.cu $(CUDA_VENDOR_DEPS) $(CUDA_RARE_INPUTS) 2>/dev/null | sha256sum | cut -c1-16)
CUDA_BUILD_CONFIG := source=$(CUDA_SOURCE_ID)|nvcc=$(NVCC)|host=$(HOST_CXX)|flags=$(NVCCFLAGS)|arch-mode=$(CUDA_ARCH_MODE)|arch-flags=$(CUDA_ARCH_FLAGS)|registers=$(CUDA_MAX_REGISTERS)|attempts=$(CUDA_ATTEMPTS_PER_THREAD)|threads=$(CUDA_THREADS)|blocks-per-sm=$(CUDA_BLOCKS_PER_SM)
CUDA_BUILD_ID := $(shell printf '%s' '$(CUDA_BUILD_CONFIG)' | sha256sum | cut -c1-16)
CUDA_CONFIG_STAMP := .meshcore_cuda_vanity.build-$(CUDA_BUILD_ID)
CUDA_CONFIG_STAMP_GLOB := .meshcore_cuda_vanity.build-*

.DELETE_ON_ERROR:
.PHONY: all clean check check-nvcc test test-cpu test-gpu install-smoke

all: meshcore_cuda_vanity

meshcore_cuda_vanity: Makefile cuda_vanity.cu $(CUDA_VENDOR_DEPS) $(CUDA_GENERATED_DEPS) $(CUDA_CONFIG_STAMP) | check-nvcc
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) -ccbin $(HOST_CXX) --maxrregcount=$(CUDA_MAX_REGISTERS) \
		-DMC_ATTEMPTS_PER_THREAD=$(CUDA_ATTEMPTS_PER_THREAD) -DMC_THREADS=$(CUDA_THREADS) \
		-DMC_BLOCKS_PER_SM=$(CUDA_BLOCKS_PER_SM) -DMC_MAX_REGISTERS=$(CUDA_MAX_REGISTERS) \
		-DMC_BUILD_ARCHES=\"$(CUDA_ARCH_LIST)\" -DMC_BUILD_FINGERPRINT=\"$(CUDA_BUILD_ID)\" \
		-MMD -MP -MF $(CUDA_DEPFILE) -MT $@ \
		-Ivendor/cuda-ed25519 cuda_vanity.cu -o $@

check-nvcc:
	@set -- $$($(NVCC) --version 2>/dev/null | \
		sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | tail -n 1); \
	if [ "$$#" -ne 2 ]; then \
		echo "Could not determine the nvcc version. NVIDIA CUDA toolkit $(CUDA_MIN_MAJOR).$(CUDA_MIN_MINOR) or newer is required." >&2; \
		exit 2; \
	fi; \
	if [ "$$1" -lt "$(CUDA_MIN_MAJOR)" ] || { [ "$$1" -eq "$(CUDA_MIN_MAJOR)" ] && [ "$$2" -lt "$(CUDA_MIN_MINOR)" ]; }; then \
		echo "nvcc $$1.$$2 is too old. NVIDIA CUDA toolkit $(CUDA_MIN_MAJOR).$(CUDA_MIN_MINOR) or newer is required." >&2; \
		exit 2; \
	fi

$(CUDA_CONFIG_STAMP):
	@$(RM) $(filter-out $@,$(wildcard $(CUDA_CONFIG_STAMP_GLOB)))
	@touch $@

$(CUDA_RARE_HEADER): $(CUDA_RARE_INPUTS)
	python3 tools/generate_rare_rules.py --input rare_rules.json --output $@
	@touch $@

test: test-cpu

test-cpu:
	python3 -m unittest discover -s tests -v

test-gpu: meshcore_cuda_vanity
	./meshcore_cuda_vanity --internal-test-lane-isolation
	RUN_CUDA_TESTS=1 python3 -m unittest discover -s tests -v

check:
	python3 tools/generate_rare_rules.py --check
	python3 -m py_compile meshcore_vanity.py meshcore_key_audit.py rare_rules.py \
		tools/generate_rare_rules.py $(wildcard tests/test_*.py)
	bash -n install.sh uninstall.sh meshcore-vanity-keygen meshcore-key-audit publish_to_github.sh

install-smoke: meshcore_cuda_vanity
	tmp_dir=$$(mktemp -d); trap 'rm -rf "$$tmp_dir"' EXIT; \
	./install.sh --prefix "$$tmp_dir/prefix" --skip-packages --skip-build --no-desktop; \
	"$$tmp_dir/prefix/bin/meshcore-vanity-keygen" --version; \
	"$$tmp_dir/prefix/bin/meshcore-vanity-keygen" --self-test; \
	"$$tmp_dir/prefix/bin/meshcore-key-audit" --help >/dev/null; \
	test -f "$$tmp_dir/prefix/lib/meshcore-vanity-keygen/rare_rules.py"; \
	test -f "$$tmp_dir/prefix/lib/meshcore-vanity-keygen/rare_rules.json"; \
	test -f "$$tmp_dir/prefix/share/icons/hicolor/scalable/apps/meshcore-vanity-keygen.svg"; \
	test -f "$$tmp_dir/prefix/lib/meshcore-vanity-keygen/assets/meshcore-vanity-keygen.png"

clean:
	$(RM) meshcore_cuda_vanity $(CUDA_DEPFILE) $(CUDA_CONFIG_STAMP_GLOB)

-include $(CUDA_DEPFILE)
