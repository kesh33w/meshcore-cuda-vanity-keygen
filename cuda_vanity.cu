// MeshCore CUDA vanity engine. Derived from vikulin/ed25519-gpu-vanity
// (GPL-3.0); see LICENSE and vendor/cuda-ed25519/license.txt.
#include <cuda_runtime.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "vendor/cuda-ed25519/fixedint.h"
#include "vendor/cuda-ed25519/sha512.h"
#include "vendor/cuda-ed25519/ge.h"
#include "vendor/cuda-ed25519/sha512.cu"
#include "vendor/cuda-ed25519/fe.cu"
#include "vendor/cuda-ed25519/ge.cu"

namespace {
constexpr int kMaxPattern = 60;
#ifndef MC_THREADS
#define MC_THREADS 256
#endif
constexpr int kThreads = MC_THREADS;
#ifndef MC_BLOCKS_PER_SM
#define MC_BLOCKS_PER_SM 8
#endif
constexpr int kBlocksPerSm = MC_BLOCKS_PER_SM;
#ifndef MC_ATTEMPTS_PER_THREAD
#define MC_ATTEMPTS_PER_THREAD 256
#endif
constexpr int kAttemptsPerThread = MC_ATTEMPTS_PER_THREAD;
constexpr int kResultCheckInterval = 32;
constexpr int kWatchCapacity = 64;
static_assert(kThreads > 0 && kThreads <= 1024 && kThreads % 32 == 0,
              "MC_THREADS must be a positive warp multiple no greater than 1024");
static_assert(kAttemptsPerThread > 0 && kAttemptsPerThread % 32 == 0,
              "MC_ATTEMPTS_PER_THREAD must be a positive multiple of 32");

__constant__ char gpu_prefix[kMaxPattern + 1];
__constant__ char gpu_suffix[kMaxPattern + 1];
__constant__ char gpu_contains[kMaxPattern + 1];
__constant__ int gpu_prefix_len;
__constant__ int gpu_suffix_len;
__constant__ int gpu_contains_len;
__constant__ int gpu_collect_only;
__constant__ char gpu_watch_words[3][11] = {
    "f00df00d00", "1337133713", "fadefade00"
};
__constant__ char gpu_pi_prefix[11] = "3141592653";

struct DeviceResult {
    int found;
    unsigned char private_key[64];
    unsigned char public_key[32];
};

struct WatchResult {
    int rule;
    unsigned char private_key[64];
    unsigned char public_key[32];
};

struct WatchBatch {
    int count;
    WatchResult results[kWatchCapacity];
};

void cuda_check(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(2);
    }
}

__device__ char hex_at(const unsigned char *key, int index) {
    const unsigned char nibble = (index & 1) ? (key[index / 2] & 15) : (key[index / 2] >> 4);
    return nibble < 10 ? static_cast<char>('0' + nibble) : static_cast<char>('a' + nibble - 10);
}

__device__ bool equal_at(const unsigned char *key, const char *pattern, int length, int offset) {
    for (int i = 0; i < length; ++i)
        if (hex_at(key, offset + i) != pattern[i]) return false;
    return true;
}

__device__ bool key_matches(const unsigned char *key) {
    if (gpu_prefix_len && !equal_at(key, gpu_prefix, gpu_prefix_len, 0)) return false;
    if (gpu_suffix_len && !equal_at(key, gpu_suffix, gpu_suffix_len, 64 - gpu_suffix_len)) return false;
    if (gpu_contains_len) {
        bool located = false;
        for (int offset = 0; offset <= 64 - gpu_contains_len; ++offset) {
            if (equal_at(key, gpu_contains, gpu_contains_len, offset)) {
                located = true;
                break;
            }
        }
        if (!located) return false;
    }
    return key[0] != 0 && key[0] != 255;
}

__device__ int interesting_rule(const unsigned char *key) {
    bool bookend = true, mirror = true;
    for (int i = 0; i < 10; ++i) {
        if (hex_at(key, i) != hex_at(key, 54 + i)) bookend = false;
        if (hex_at(key, i) != hex_at(key, 63 - i)) mirror = false;
    }
    if (bookend) return 0;
    if (mirror) return 1;
    bool repeated_prefix = true;
    for (int i = 1; i < 10; ++i)
        if (hex_at(key, i) != hex_at(key, 0)) repeated_prefix = false;
    if (repeated_prefix) return 2;
    for (int word = 0; word < 3; ++word) {
        if (equal_at(key, gpu_watch_words[word], 10, 0)) return 3 + word;
    }
    if (equal_at(key, gpu_pi_prefix, 10, 0)) return 6;
    return -1;
}

__device__ void increment_seed(unsigned char *seed, unsigned long long amount) {
    unsigned int carry = 0;
    for (int i = 0; i < 32; ++i) {
        unsigned int add = static_cast<unsigned int>(amount & 255ULL) + carry;
        unsigned int sum = static_cast<unsigned int>(seed[i]) + add;
        seed[i] = static_cast<unsigned char>(sum);
        carry = sum >> 8;
        amount >>= 8;
        if (!amount && !carry) break;
    }
}

__device__ bool inspect_candidate(const unsigned char *private_key,
                                  const unsigned char *public_key,
                                  DeviceResult *result,
                                  WatchBatch *watch_batch) {
    const bool meshcore_valid = public_key[0] != 0 && public_key[0] != 255;
    int rule = meshcore_valid ? interesting_rule(public_key) : -1;
    if (rule >= 0) {
        int slot = atomicAdd(&watch_batch->count, 1);
        if (slot < kWatchCapacity) {
            watch_batch->results[slot].rule = rule;
            for (int i = 0; i < 32; ++i) watch_batch->results[slot].public_key[i] = public_key[i];
            for (int i = 0; i < 64; ++i) watch_batch->results[slot].private_key[i] = private_key[i];
        }
    }
    if (!gpu_collect_only && key_matches(public_key)
            && atomicCAS(&result->found, 0, 1) == 0) {
        for (int i = 0; i < 32; ++i) result->public_key[i] = public_key[i];
        for (int i = 0; i < 64; ++i) result->private_key[i] = private_key[i];
        __threadfence_system();
        return true;
    }
    return false;
}

__global__ void scan_kernel_baseline(const unsigned char *base_seed, DeviceResult *result,
                                     WatchBatch *watch_batch) {
    unsigned char seed[32];
    unsigned char private_key[64];
    unsigned char public_key[32];
    ge_p3 point;
    const unsigned long long lane = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; ++i) seed[i] = base_seed[i];
    increment_seed(seed, lane * kAttemptsPerThread);

    for (int attempt = 0; attempt < kAttemptsPerThread; ++attempt) {
        if ((attempt % kResultCheckInterval) == 0 && result->found) return;
        sha512(seed, 32, private_key);
        private_key[0] &= 248;
        private_key[31] &= 63;
        private_key[31] |= 64;
        ge_scalarmult_base(&point, private_key);
        ge_p3_tobytes(public_key, &point);
        if (inspect_candidate(private_key, public_key, result, watch_batch)) return;
        increment_seed(seed, 1);
    }
}

__device__ void encode_with_inverse(unsigned char *encoded, const ge_p3 *point,
                                    const fe inverse) {
    fe x;
    fe y;
    fe_mul(x, point->X, inverse);
    fe_mul(y, point->Y, inverse);
    fe_tobytes(encoded, y);
    encoded[31] ^= fe_isnegative(x) << 7;
}

template<int BatchSize>
__device__ void encode_batch(unsigned char encoded[BatchSize][32],
                             const ge_p3 points[BatchSize]) {
    fe prefixes[BatchSize];
    fe running_inverse;
    fe inverse;
    fe_copy(prefixes[0], points[0].Z);
    for (int i = 1; i < BatchSize; ++i) fe_mul(prefixes[i], prefixes[i - 1], points[i].Z);
    fe_invert(running_inverse, prefixes[BatchSize - 1]);
    for (int i = BatchSize - 1; i > 0; --i) {
        fe_mul(inverse, running_inverse, prefixes[i - 1]);
        encode_with_inverse(encoded[i], &points[i], inverse);
        fe_mul(running_inverse, running_inverse, points[i].Z);
    }
    encode_with_inverse(encoded[0], &points[0], running_inverse);
}

__global__ void scan_kernel_optimized(const unsigned char *base_scalar,
                                      DeviceResult *result, WatchBatch *watch_batch) {
    unsigned char private_key[64]{};
    unsigned char public_keys[32][32];
    ge_p3 points[32];
    ge_p1p1 next;
    const unsigned long long lane = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; ++i) private_key[i] = base_scalar[i];
    increment_seed(private_key, lane * kAttemptsPerThread * 8ULL);
    ge_scalarmult_base(&points[0], private_key);

    for (int attempt = 0; attempt < kAttemptsPerThread; attempt += 32) {
        if (result->found) return;
        for (int i = 1; i < 32; ++i) {
            ge_madd(&next, &points[i - 1], &base[0][7]);
            ge_p1p1_to_p3(&points[i], &next);
        }
        encode_batch<32>(public_keys, points);
        for (int i = 0; i < 32; ++i) {
            if (inspect_candidate(private_key, public_keys[i], result, watch_batch)) return;
            increment_seed(private_key, 8);
        }
        ge_madd(&next, &points[31], &base[0][7]);
        ge_p1p1_to_p3(&points[0], &next);
    }
}

bool valid_hex(const std::string &value) {
    if (value.size() > kMaxPattern) return false;
    for (char ch : value)
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    return true;
}

bool feasible(const std::string &prefix, const std::string &suffix,
              const std::string &contains, std::string &error) {
    if (prefix.size() >= 2 && (prefix.substr(0, 2) == "00" || prefix.substr(0, 2) == "ff")) {
        error = "prefix begins with a byte MeshCore rejects (00 or ff)";
        return false;
    }
    std::array<char, 64> assigned{};
    auto place = [&assigned](const std::string &value, size_t offset) {
        for (size_t i = 0; i < value.size(); ++i)
            if (assigned[offset + i] && assigned[offset + i] != value[i]) return false;
        for (size_t i = 0; i < value.size(); ++i) assigned[offset + i] = value[i];
        return true;
    };
    if (!place(prefix, 0) || !place(suffix, 64 - suffix.size())) {
        error = "prefix and suffix conflict where they overlap";
        return false;
    }
    if (!contains.empty()) {
        bool possible = false;
        for (size_t offset = 0; offset + contains.size() <= 64; ++offset) {
            bool fits = true;
            for (size_t i = 0; i < contains.size(); ++i)
                if (assigned[offset + i] && assigned[offset + i] != contains[i]) fits = false;
            if (fits) { possible = true; break; }
        }
        if (!possible) {
            error = "substring conflicts with the prefix and suffix";
            return false;
        }
    }
    return true;
}

std::string hex(const unsigned char *bytes, size_t length) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string value(length * 2, '0');
    for (size_t i = 0; i < length; ++i) {
        value[i * 2] = digits[bytes[i] >> 4];
        value[i * 2 + 1] = digits[bytes[i] & 15];
    }
    return value;
}

void random_bytes(unsigned char *destination, size_t length) {
    int descriptor = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        std::perror("open /dev/urandom");
        std::exit(2);
    }
    size_t done = 0;
    while (done < length) {
        ssize_t count = read(descriptor, destination + done, length - done);
        if (count <= 0) {
            std::perror("read /dev/urandom");
            close(descriptor);
            std::exit(2);
        }
        done += static_cast<size_t>(count);
    }
    close(descriptor);
}

void usage(const char *program) {
    std::fprintf(stderr, "Usage: %s [--engine optimized|baseline] [--device N] [--collect-only] [--prefix HEX] [--suffix HEX] [--contains HEX]\n", program);
}
}  // namespace

int main(int argc, char **argv) {
    std::string prefix, suffix, contains;
    std::string engine = "optimized";
    int selected_device = 0;
    bool collect_only = false;
    for (int i = 1; i < argc;) {
        std::string option = argv[i++];
        if (option == "--collect-only") {
            collect_only = true;
            continue;
        }
        if (i >= argc) { usage(argv[0]); return 2; }
        std::string value = argv[i++];
        if (option == "--device") {
            char *end = nullptr;
            long parsed = std::strtol(value.c_str(), &end, 10);
            if (!end || *end || parsed < 0 || parsed > 1024) { usage(argv[0]); return 2; }
            selected_device = static_cast<int>(parsed);
            continue;
        }
        if (option == "--engine") {
            if (value == "incremental") value = "optimized";  // pre-batching compatibility alias
            if (value != "optimized" && value != "baseline") { usage(argv[0]); return 2; }
            engine = value;
            continue;
        }
        for (char &ch : value) if (ch >= 'A' && ch <= 'F') ch += 'a' - 'A';
        if (option == "--prefix") prefix = value;
        else if (option == "--suffix") suffix = value;
        else if (option == "--contains") contains = value;
        else { usage(argv[0]); return 2; }
    }
    if ((!collect_only && prefix.empty() && suffix.empty() && contains.empty()) ||
        (collect_only && (!prefix.empty() || !suffix.empty() || !contains.empty())) ||
        !valid_hex(prefix) || !valid_hex(suffix) || !valid_hex(contains)) {
        usage(argv[0]);
        return 2;
    }
    std::string feasibility_error;
    if (!feasible(prefix, suffix, contains, feasibility_error)) {
        std::fprintf(stderr, "%s\n", feasibility_error.c_str());
        return 2;
    }

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (!device_count) { std::fprintf(stderr, "No CUDA device found\n"); return 2; }
    if (selected_device >= device_count) {
        std::fprintf(stderr, "CUDA device %d is unavailable; detected %d device(s)\n",
                     selected_device, device_count);
        return 2;
    }
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, selected_device), "cudaGetDeviceProperties");
    cuda_check(cudaSetDevice(selected_device), "cudaSetDevice");
    cuda_check(cudaMemcpyToSymbol(gpu_prefix, prefix.c_str(), prefix.size() + 1), "copy prefix");
    cuda_check(cudaMemcpyToSymbol(gpu_suffix, suffix.c_str(), suffix.size() + 1), "copy suffix");
    cuda_check(cudaMemcpyToSymbol(gpu_contains, contains.c_str(), contains.size() + 1), "copy contains");
    int prefix_len = prefix.size(), suffix_len = suffix.size(), contains_len = contains.size();
    cuda_check(cudaMemcpyToSymbol(gpu_prefix_len, &prefix_len, sizeof(int)), "copy prefix length");
    cuda_check(cudaMemcpyToSymbol(gpu_suffix_len, &suffix_len, sizeof(int)), "copy suffix length");
    cuda_check(cudaMemcpyToSymbol(gpu_contains_len, &contains_len, sizeof(int)), "copy contains length");
    int collect_only_value = collect_only ? 1 : 0;
    cuda_check(cudaMemcpyToSymbol(gpu_collect_only, &collect_only_value, sizeof(int)),
               "copy collector mode");

    int blocks = properties.multiProcessorCount * kBlocksPerSm;
    unsigned char *device_seed = nullptr;
    DeviceResult *device_result = nullptr;
    WatchBatch *device_watch = nullptr;
    cuda_check(cudaMalloc(&device_seed, 32), "cudaMalloc seed");
    cuda_check(cudaMalloc(&device_result, sizeof(DeviceResult)), "cudaMalloc result");
    cuda_check(cudaMalloc(&device_watch, sizeof(WatchBatch)), "cudaMalloc watch results");
    unsigned long long attempts = 0;
    auto started = std::chrono::steady_clock::now();
    std::fprintf(stderr, "GPU %d: %s, engine %s, mode %s, %d blocks x %d threads\n",
                 selected_device, properties.name, engine.c_str(),
                 collect_only ? "collector" : "vanity", blocks, kThreads);

    for (;;) {
        std::array<unsigned char, 32> seed{};
        DeviceResult result{};
        WatchBatch watch{};
        random_bytes(seed.data(), seed.size());
        if (engine != "baseline") {
            seed[0] &= 248;
            seed[31] &= 31;  // retain ample headroom for the per-lane offsets
            seed[31] |= 64;
        }
        cuda_check(cudaMemcpy(device_seed, seed.data(), 32, cudaMemcpyHostToDevice), "copy seed");
        cuda_check(cudaMemset(device_result, 0, sizeof(DeviceResult)), "clear result");
        cuda_check(cudaMemset(device_watch, 0, sizeof(WatchBatch)), "clear watch results");
        if (engine == "optimized")
            scan_kernel_optimized<<<blocks, kThreads>>>(device_seed, device_result, device_watch);
        else
            scan_kernel_baseline<<<blocks, kThreads>>>(device_seed, device_result, device_watch);
        cuda_check(cudaGetLastError(), "launch scan kernel");
        cuda_check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost), "copy result");
        cuda_check(cudaMemcpy(&watch, device_watch, sizeof(watch), cudaMemcpyDeviceToHost), "copy watch results");
        if (watch.count > kWatchCapacity) {
            std::fprintf(stderr, "Rare-key batch overflow (%d > %d); refusing to silently drop matches\n",
                         watch.count, kWatchCapacity);
            cudaFree(device_result);
            cudaFree(device_seed);
            cudaFree(device_watch);
            return 2;
        }
        for (int i = 0; i < watch.count; ++i) {
            if (engine != "baseline") random_bytes(&watch.results[i].private_key[32], 32);
            std::fprintf(stderr, "WATCH %d %s %s\n", watch.results[i].rule,
                         hex(watch.results[i].public_key, 32).c_str(),
                         hex(watch.results[i].private_key, 64).c_str());
        }
        std::fflush(stderr);
        attempts += static_cast<unsigned long long>(blocks) * kThreads * kAttemptsPerThread;
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (result.found) {
            if (engine != "baseline") random_bytes(&result.private_key[32], 32);
            std::printf("{\"public_key\":\"%s\",\"private_key\":\"%s\",\"engine\":\"%s\",\"attempts\":%llu,\"elapsed_seconds\":%.6f}\n",
                        hex(result.public_key, 32).c_str(), hex(result.private_key, 64).c_str(),
                        engine.c_str(), attempts, elapsed);
            break;
        }
        std::fprintf(stderr, "PROGRESS %llu %.6f %.0f\n", attempts, elapsed, attempts / elapsed);
        std::fflush(stderr);
    }
    cudaFree(device_result);
    cudaFree(device_seed);
    cudaFree(device_watch);
    return 0;
}
