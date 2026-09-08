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
constexpr int kThreads = 256;
constexpr int kAttemptsPerThread = 256;
constexpr int kWatchRules = 18;

__constant__ char gpu_prefix[kMaxPattern + 1];
__constant__ char gpu_suffix[kMaxPattern + 1];
__constant__ char gpu_contains[kMaxPattern + 1];
__constant__ int gpu_prefix_len;
__constant__ int gpu_suffix_len;
__constant__ int gpu_contains_len;
__constant__ char gpu_watch_words[8][11] = {
    "cafecafe00", "beefbeef00", "deadbeef00", "facebabe00",
    "babecafe00", "f00df00d00", "1337133713", "fadefade00"
};

struct DeviceResult {
    int found;
    unsigned char seed[32];
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
    WatchResult results[kWatchRules];
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
    for (int word = 0; word < 8; ++word) {
        if (equal_at(key, gpu_watch_words[word], 10, 0)) return 2 + word;
        if (equal_at(key, gpu_watch_words[word], 10, 54)) return 10 + word;
    }
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

__global__ void scan_kernel(const unsigned char *base_seed, DeviceResult *result,
                            unsigned long long *watch_mask, WatchBatch *watch_batch) {
    unsigned char seed[32];
    unsigned char private_key[64];
    unsigned char public_key[32];
    ge_p3 point;
    const unsigned long long lane = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; ++i) seed[i] = base_seed[i];
    increment_seed(seed, lane * kAttemptsPerThread);

    for (int attempt = 0; attempt < kAttemptsPerThread && !result->found; ++attempt) {
        sha512(seed, 32, private_key);
        private_key[0] &= 248;
        private_key[31] &= 63;
        private_key[31] |= 64;
        ge_scalarmult_base(&point, private_key);
        ge_p3_tobytes(public_key, &point);
        const bool meshcore_valid = public_key[0] != 0 && public_key[0] != 255;
        int rule = meshcore_valid ? interesting_rule(public_key) : -1;
        if (rule >= 0) {
            unsigned long long bit = 1ULL << rule;
            if (!(atomicOr(watch_mask, bit) & bit)) {
                int slot = atomicAdd(&watch_batch->count, 1);
                if (slot < kWatchRules) {
                    watch_batch->results[slot].rule = rule;
                    for (int i = 0; i < 32; ++i) watch_batch->results[slot].public_key[i] = public_key[i];
                    for (int i = 0; i < 64; ++i) watch_batch->results[slot].private_key[i] = private_key[i];
                }
            }
        }
        if (key_matches(public_key) && atomicCAS(&result->found, 0, 1) == 0) {
            for (int i = 0; i < 32; ++i) {
                result->seed[i] = seed[i];
                result->public_key[i] = public_key[i];
            }
            for (int i = 0; i < 64; ++i) result->private_key[i] = private_key[i];
            __threadfence_system();
            return;
        }
        increment_seed(seed, 1);
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
    std::fprintf(stderr, "Usage: %s [--device N] [--prefix HEX] [--suffix HEX] [--contains HEX]\n", program);
}
}  // namespace

int main(int argc, char **argv) {
    std::string prefix, suffix, contains;
    int selected_device = 0;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) { usage(argv[0]); return 2; }
        std::string option = argv[i];
        std::string value = argv[++i];
        if (option == "--device") {
            char *end = nullptr;
            long parsed = std::strtol(value.c_str(), &end, 10);
            if (!end || *end || parsed < 0 || parsed > 1024) { usage(argv[0]); return 2; }
            selected_device = static_cast<int>(parsed);
            continue;
        }
        for (char &ch : value) if (ch >= 'A' && ch <= 'F') ch += 'a' - 'A';
        if (option == "--prefix") prefix = value;
        else if (option == "--suffix") suffix = value;
        else if (option == "--contains") contains = value;
        else { usage(argv[0]); return 2; }
    }
    if ((prefix.empty() && suffix.empty() && contains.empty()) ||
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

    int blocks = properties.multiProcessorCount * 8;
    unsigned char *device_seed = nullptr;
    DeviceResult *device_result = nullptr;
    WatchBatch *device_watch = nullptr;
    unsigned long long *device_watch_mask = nullptr;
    cuda_check(cudaMalloc(&device_seed, 32), "cudaMalloc seed");
    cuda_check(cudaMalloc(&device_result, sizeof(DeviceResult)), "cudaMalloc result");
    cuda_check(cudaMalloc(&device_watch, sizeof(WatchBatch)), "cudaMalloc watch results");
    cuda_check(cudaMalloc(&device_watch_mask, sizeof(unsigned long long)), "cudaMalloc watch mask");
    cuda_check(cudaMemset(device_watch_mask, 0, sizeof(unsigned long long)), "clear watch mask");
    unsigned long long attempts = 0;
    auto started = std::chrono::steady_clock::now();
    std::fprintf(stderr, "GPU %d: %s, %d blocks x %d threads\n",
                 selected_device, properties.name, blocks, kThreads);

    for (;;) {
        std::array<unsigned char, 32> seed{};
        DeviceResult result{};
        WatchBatch watch{};
        random_bytes(seed.data(), seed.size());
        cuda_check(cudaMemcpy(device_seed, seed.data(), 32, cudaMemcpyHostToDevice), "copy seed");
        cuda_check(cudaMemset(device_result, 0, sizeof(DeviceResult)), "clear result");
        cuda_check(cudaMemset(device_watch, 0, sizeof(WatchBatch)), "clear watch results");
        scan_kernel<<<blocks, kThreads>>>(device_seed, device_result, device_watch_mask, device_watch);
        cuda_check(cudaGetLastError(), "launch scan_kernel");
        cuda_check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost), "copy result");
        cuda_check(cudaMemcpy(&watch, device_watch, sizeof(watch), cudaMemcpyDeviceToHost), "copy watch results");
        for (int i = 0; i < watch.count && i < kWatchRules; ++i) {
            std::fprintf(stderr, "WATCH %d %s %s\n", watch.results[i].rule,
                         hex(watch.results[i].public_key, 32).c_str(),
                         hex(watch.results[i].private_key, 64).c_str());
        }
        std::fflush(stderr);
        attempts += static_cast<unsigned long long>(blocks) * kThreads * kAttemptsPerThread;
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (result.found) {
            std::printf("{\"public_key\":\"%s\",\"private_key\":\"%s\",\"seed\":\"%s\",\"attempts\":%llu,\"elapsed_seconds\":%.6f}\n",
                        hex(result.public_key, 32).c_str(), hex(result.private_key, 64).c_str(),
                        hex(result.seed, 32).c_str(), attempts, elapsed);
            break;
        }
        std::fprintf(stderr, "PROGRESS %llu %.6f %.0f\n", attempts, elapsed, attempts / elapsed);
        std::fflush(stderr);
    }
    cudaFree(device_result);
    cudaFree(device_seed);
    cudaFree(device_watch);
    cudaFree(device_watch_mask);
    return 0;
}
