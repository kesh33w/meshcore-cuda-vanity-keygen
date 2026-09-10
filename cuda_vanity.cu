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
constexpr int kLaneDomainSize = 24;
constexpr int kIsolationTestLanes = 4;
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
// A fixed-width tag makes the lane derivation a separate SHA-512 domain from
// ordinary MeshCore seed hashing.  Each launch supplies a fresh 256-bit batch
// seed and each GPU lane appends its unique 64-bit little-endian index.
#define MC_LANE_DOMAIN "MeshCore-CUDA-Lane-v1"
constexpr unsigned char kLaneDomain[kLaneDomainSize] = MC_LANE_DOMAIN;
__constant__ unsigned char gpu_lane_domain[kLaneDomainSize] = MC_LANE_DOMAIN;
#undef MC_LANE_DOMAIN

// Public, non-secret regression inputs. These immutable vectors are
// SHA-512(domain || bytes(0..31) || little_endian_u64(lane)), followed by the
// same Ed25519 clamp used by production. The internal test never prints them.
constexpr unsigned char kIsolationExpected[kIsolationTestLanes][64] = {
    {0x48, 0xe0, 0x0a, 0xd3, 0x2e, 0xe3, 0x8a, 0x42, 0x88, 0xe3, 0x7f, 0x4e, 0x21, 0xd9, 0xc3, 0x6c, 0xa0, 0x4f, 0xc7, 0x2b, 0xe9, 0xb1, 0x67, 0x2b, 0xa6, 0x4d, 0x8a, 0x40, 0xba, 0x90, 0x32, 0x55, 0x57, 0xeb, 0x1f, 0x6e, 0x70, 0x5e, 0xbc, 0xef, 0xf8, 0x79, 0x28, 0x12, 0x35, 0xf2, 0xdd, 0x58, 0x02, 0x6a, 0x13, 0xc5, 0x22, 0x59, 0xfe, 0x42, 0x3d, 0xc4, 0x62, 0x26, 0x22, 0xd3, 0xea, 0x4f},
    {0x20, 0x17, 0xce, 0xd0, 0xbc, 0xdb, 0x95, 0xcb, 0x71, 0x16, 0x70, 0x16, 0xc6, 0xf4, 0x23, 0xb0, 0xce, 0xfe, 0x0c, 0xe2, 0xa1, 0x63, 0x18, 0x5b, 0x71, 0xe7, 0x90, 0xa7, 0x4c, 0x80, 0xda, 0x79, 0xdc, 0x52, 0x0f, 0xbb, 0x74, 0x6f, 0x30, 0xf4, 0xb4, 0x58, 0xac, 0x52, 0xe1, 0xed, 0x4b, 0x6c, 0xb6, 0x9c, 0x9a, 0xa9, 0x9f, 0x1d, 0x3e, 0xf1, 0x6b, 0x55, 0x79, 0xe2, 0x99, 0x8e, 0xa9, 0x12},
    {0x18, 0xc2, 0x3b, 0xd9, 0x57, 0x0e, 0xe2, 0x97, 0xbb, 0x05, 0xe6, 0x3f, 0x14, 0x6f, 0x3b, 0xd5, 0xf7, 0xd5, 0xac, 0x58, 0x8a, 0x6c, 0x76, 0x79, 0xca, 0xc5, 0x65, 0x89, 0x09, 0xc2, 0xa6, 0x69, 0xfb, 0xfc, 0x8c, 0x1d, 0x82, 0xdb, 0x36, 0x05, 0xdf, 0xea, 0x98, 0x2e, 0xbf, 0xe3, 0x81, 0xc2, 0x7e, 0x23, 0xa7, 0x5c, 0x6b, 0xef, 0x5c, 0xe5, 0x52, 0x86, 0xee, 0x3a, 0x74, 0x65, 0x18, 0xb4},
    {0x68, 0xd6, 0x13, 0xe0, 0x90, 0xc2, 0xe8, 0xc0, 0x5a, 0xa7, 0xa3, 0x81, 0x62, 0x32, 0x23, 0x91, 0xd9, 0x70, 0x92, 0xbb, 0xfa, 0x17, 0xe4, 0xb4, 0xfa, 0x3d, 0xd7, 0xa2, 0x96, 0x4a, 0xd9, 0x79, 0x65, 0x3b, 0xe4, 0x46, 0x70, 0xbb, 0xc8, 0x67, 0x54, 0x88, 0x2a, 0xa9, 0xea, 0x77, 0xc4, 0x21, 0x9f, 0x70, 0x2e, 0xee, 0x9c, 0xe3, 0x88, 0x4a, 0xad, 0x46, 0x8e, 0x87, 0x1c, 0xd1, 0xe4, 0xdd},
};
__constant__ char gpu_watch_words[1][11] = {
    "1337133713"
};
__constant__ char gpu_pi_prefix[11] = "3141592653";

struct DeviceResult {
    int found;
    unsigned long long lane;
    unsigned char private_key[64];
    unsigned char public_key[32];
};

struct WatchResult {
    int rule;
    unsigned long long lane;
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
    for (int word = 0; word < 1; ++word) {
        if (equal_at(key, gpu_watch_words[word], 10, 0)) return 3 + word;
    }
    if (equal_at(key, gpu_pi_prefix, 10, 0)) return 4;
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

__device__ bool increment_scalar(unsigned char *scalar, unsigned long long amount) {
    increment_seed(scalar, amount);
    // Ed25519's expanded scalar must keep bit 255 clear.  Standard clamping
    // leaves a tiny theoretical chance that the bounded +8 walk reaches that
    // bit, so terminate that lane instead of wrapping or emitting a malformed
    // expanded key.
    return (scalar[31] & 128) == 0;
}

__device__ bool inspect_candidate(const unsigned char *private_key,
                                  const unsigned char *public_key,
                                  DeviceResult *result,
                                  WatchBatch *watch_batch,
                                  unsigned long long lane,
                                  bool retain_one_watch_per_lane,
                                  bool *lane_has_watch) {
    const bool meshcore_valid = public_key[0] != 0 && public_key[0] != 255;
    const bool target_match = meshcore_valid && !gpu_collect_only
                              && key_matches(public_key);

    // Claim the requested vanity result before considering incidental rules.
    // Thus a key satisfying both is emitted once as the requested result, not
    // duplicated as WATCH output.  A losing lane can stop because another lane
    // has already completed the search.
    if (target_match) {
        if (atomicCAS(&result->found, 0, 1) == 0) {
            result->lane = lane;
            for (int i = 0; i < 32; ++i) result->public_key[i] = public_key[i];
            for (int i = 0; i < 64; ++i) result->private_key[i] = private_key[i];
            __threadfence_system();
        }
        return true;
    }

    const int rule = meshcore_valid ? interesting_rule(public_key) : -1;
    if (rule >= 0 && (!retain_one_watch_per_lane || !*lane_has_watch)) {
        // Optimized candidates within one lane intentionally differ by +8.
        // Retaining at most one prevents persisted private keys from having a
        // small, known scalar relationship.  Mark it retained even on global
        // buffer overflow so this lane never contributes a second identity.
        *lane_has_watch = true;
        int slot = atomicAdd(&watch_batch->count, 1);
        if (slot < kWatchCapacity) {
            watch_batch->results[slot].rule = rule;
            watch_batch->results[slot].lane = lane;
            for (int i = 0; i < 32; ++i) watch_batch->results[slot].public_key[i] = public_key[i];
            for (int i = 0; i < 64; ++i) watch_batch->results[slot].private_key[i] = private_key[i];
        }
    }
    return false;
}

__global__ void scan_kernel_baseline(const unsigned char *base_seed, DeviceResult *result,
                                     WatchBatch *watch_batch) {
    unsigned char seed[32];
    unsigned char private_key[64];
    unsigned char public_key[32];
    ge_p3 point;
    bool lane_has_watch = false;
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
        if (inspect_candidate(private_key, public_key, result, watch_batch, lane,
                              false, &lane_has_watch)) return;
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

__global__ void derive_lane_private_keys(const unsigned char *batch_seed,
                                         unsigned char *lane_private_keys,
                                         unsigned long long lane_count) {
    unsigned char lane_input[kLaneDomainSize + 32 + 8];
    unsigned char private_key[64];
    const unsigned long long lane =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= lane_count) return;

    // Security boundary: derive a pseudorandom expanded Ed25519 secret for
    // every lane instead of offsetting every lane from one public scalar.
    // This separate kernel also keeps SHA-512's working set out of the much
    // larger point-walking kernel, avoiding register/stack pressure there.
    for (int i = 0; i < kLaneDomainSize; ++i) lane_input[i] = gpu_lane_domain[i];
    for (int i = 0; i < 32; ++i) lane_input[kLaneDomainSize + i] = batch_seed[i];
    for (int i = 0; i < 8; ++i)
        lane_input[kLaneDomainSize + 32 + i] =
            static_cast<unsigned char>(lane >> (8 * i));
    sha512(lane_input, sizeof(lane_input), private_key);

    // Standard Ed25519 clamping; unlike the old shared-range construction,
    // random hash bit 253 is preserved.
    private_key[0] &= 248;
    private_key[31] &= 63;
    private_key[31] |= 64;

    // Transposed storage lets adjacent lanes read/write adjacent bytes.
    for (int i = 0; i < 64; ++i)
        lane_private_keys[static_cast<unsigned long long>(i) * lane_count + lane] =
            private_key[i];
}

__global__ void scan_kernel_optimized(const unsigned char *lane_private_keys,
                                      unsigned long long lane_count,
                                      DeviceResult *result, WatchBatch *watch_batch) {
    unsigned char private_key[64]{};
    unsigned char public_keys[32][32];
    ge_p3 points[32];
    ge_p1p1 next;
    bool lane_has_watch = false;
    const unsigned long long lane = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= lane_count) return;
    for (int i = 0; i < 64; ++i)
        private_key[i] =
            lane_private_keys[static_cast<unsigned long long>(i) * lane_count + lane];
    ge_scalarmult_base(&points[0], private_key);

    for (int attempt = 0; attempt < kAttemptsPerThread; attempt += 32) {
        if (result->found) return;
        for (int i = 1; i < 32; ++i) {
            ge_madd(&next, &points[i - 1], &base[0][7]);
            ge_p1p1_to_p3(&points[i], &next);
        }
        encode_batch<32>(public_keys, points);
        for (int i = 0; i < 32; ++i) {
            if (inspect_candidate(private_key, public_keys[i], result, watch_batch, lane,
                                  true, &lane_has_watch)) return;
            if (!increment_scalar(private_key, 8)) return;
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

bool bounded_plus_eight_relation(const unsigned char *left,
                                 const unsigned char *right) {
    int order = 0;
    for (int i = 31; i >= 0 && order == 0; --i) {
        if (left[i] < right[i]) order = -1;
        if (left[i] > right[i]) order = 1;
    }
    const unsigned char *larger = order < 0 ? right : left;
    const unsigned char *smaller = order < 0 ? left : right;
    unsigned char difference[32]{};
    unsigned int borrow = 0;
    for (int i = 0; i < 32; ++i) {
        const unsigned int subtrahend = static_cast<unsigned int>(smaller[i]) + borrow;
        difference[i] = static_cast<unsigned char>(
            static_cast<unsigned int>(larger[i]) - subtrahend);
        borrow = static_cast<unsigned int>(larger[i]) < subtrahend;
    }
    for (int i = 8; i < 32; ++i)
        if (difference[i] != 0) return false;
    unsigned long long delta = 0;
    for (int i = 7; i >= 0; --i)
        delta = (delta << 8) | difference[i];
    const unsigned long long maximum =
        static_cast<unsigned long long>(kAttemptsPerThread - 1) * 8ULL;
    return delta <= maximum && (delta % 8ULL) == 0;
}

int run_lane_isolation_self_test(int selected_device) {
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (!device_count) {
        std::fprintf(stderr, "CUDA lane-isolation self-test: no CUDA device found\n");
        return 2;
    }
    if (selected_device >= device_count) {
        std::fprintf(stderr, "CUDA device %d is unavailable; detected %d device(s)\n",
                     selected_device, device_count);
        return 2;
    }
    cuda_check(cudaSetDevice(selected_device), "cudaSetDevice");

    std::array<unsigned char, 32> seed{};
    for (size_t i = 0; i < seed.size(); ++i)
        seed[i] = static_cast<unsigned char>(i);
    unsigned char *device_seed = nullptr;
    unsigned char *device_lanes = nullptr;
    std::array<unsigned char, kIsolationTestLanes * 64> gpu_lanes{};
    cuda_check(cudaMalloc(&device_seed, seed.size()), "self-test cudaMalloc seed");
    cuda_check(cudaMalloc(&device_lanes, gpu_lanes.size()), "self-test cudaMalloc lanes");
    cuda_check(cudaMemcpy(device_seed, seed.data(), seed.size(), cudaMemcpyHostToDevice),
               "self-test copy seed");
    derive_lane_private_keys<<<1, kIsolationTestLanes>>>(
        device_seed, device_lanes, kIsolationTestLanes);
    cuda_check(cudaGetLastError(), "self-test launch lane derivation kernel");
    cuda_check(cudaMemcpy(gpu_lanes.data(), device_lanes, gpu_lanes.size(),
                          cudaMemcpyDeviceToHost),
               "self-test copy lane results");
    cudaFree(device_lanes);
    cudaFree(device_seed);

    std::array<std::array<unsigned char, 64>, kIsolationTestLanes> derived{};
    for (int lane = 0; lane < kIsolationTestLanes; ++lane) {
        unsigned char input[kLaneDomainSize + 32 + 8]{};
        std::memcpy(input, kLaneDomain, kLaneDomainSize);
        std::memcpy(input + kLaneDomainSize, seed.data(), seed.size());
        for (int i = 0; i < 8; ++i)
            input[kLaneDomainSize + 32 + i] =
                static_cast<unsigned char>(static_cast<unsigned long long>(lane) >> (8 * i));
        if (sha512(input, sizeof(input), derived[lane].data()) != 0) {
            std::fprintf(stderr, "CUDA lane-isolation self-test: host SHA-512 failed\n");
            return 1;
        }
        derived[lane][0] &= 248;
        derived[lane][31] &= 63;
        derived[lane][31] |= 64;
        if (std::memcmp(derived[lane].data(), kIsolationExpected[lane], 64) != 0) {
            std::fprintf(stderr,
                         "CUDA lane-isolation self-test: fixed formula vector %d failed\n",
                         lane);
            return 1;
        }
        for (int byte = 0; byte < 64; ++byte) {
            if (gpu_lanes[static_cast<size_t>(byte) * kIsolationTestLanes + lane]
                    != derived[lane][byte]) {
                std::fprintf(stderr,
                             "CUDA lane-isolation self-test: GPU lane %d disagrees with formula\n",
                             lane);
                return 1;
            }
        }
    }
    for (int left = 0; left < kIsolationTestLanes; ++left) {
        for (int right = left + 1; right < kIsolationTestLanes; ++right) {
            if (bounded_plus_eight_relation(derived[left].data(), derived[right].data())) {
                std::fprintf(stderr,
                             "CUDA lane-isolation self-test: lanes %d and %d share a bounded walk\n",
                             left, right);
                return 1;
            }
        }
    }
    std::printf("CUDA lane-isolation self-test: PASS (%d fixed lanes)\n",
                kIsolationTestLanes);
    return 0;
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
    bool lane_isolation_self_test = false;
    for (int i = 1; i < argc;) {
        std::string option = argv[i++];
        if (option == "--collect-only") {
            collect_only = true;
            continue;
        }
        if (option == "--internal-test-lane-isolation") {
            lane_isolation_self_test = true;
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
    if (lane_isolation_self_test) {
        if (collect_only || !prefix.empty() || !suffix.empty() || !contains.empty()
                || engine != "optimized") {
            usage(argv[0]);
            return 2;
        }
        return run_lane_isolation_self_test(selected_device);
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
    const unsigned long long lane_count =
        static_cast<unsigned long long>(blocks) * kThreads;
    unsigned char *device_seed = nullptr;
    unsigned char *device_lane_private_keys = nullptr;
    DeviceResult *device_result = nullptr;
    WatchBatch *device_watch = nullptr;
    cuda_check(cudaMalloc(&device_seed, 32), "cudaMalloc seed");
    if (engine == "optimized")
        cuda_check(cudaMalloc(&device_lane_private_keys, lane_count * 64),
                   "cudaMalloc lane private keys");
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
        cuda_check(cudaMemcpy(device_seed, seed.data(), 32, cudaMemcpyHostToDevice), "copy seed");
        cuda_check(cudaMemset(device_result, 0, sizeof(DeviceResult)), "clear result");
        cuda_check(cudaMemset(device_watch, 0, sizeof(WatchBatch)), "clear watch results");
        if (engine == "optimized") {
            derive_lane_private_keys<<<blocks, kThreads>>>(
                device_seed, device_lane_private_keys, lane_count);
            cuda_check(cudaGetLastError(), "launch lane derivation kernel");
            scan_kernel_optimized<<<blocks, kThreads>>>(
                device_lane_private_keys, lane_count, device_result, device_watch);
        } else {
            scan_kernel_baseline<<<blocks, kThreads>>>(device_seed, device_result, device_watch);
        }
        cuda_check(cudaGetLastError(), "launch scan kernel");
        cuda_check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost), "copy result");
        cuda_check(cudaMemcpy(&watch, device_watch, sizeof(watch), cudaMemcpyDeviceToHost), "copy watch results");
        if (watch.count > kWatchCapacity) {
            std::fprintf(stderr, "Rare-key batch overflow (%d > %d); refusing to silently drop matches\n",
                         watch.count, kWatchCapacity);
            cudaFree(device_result);
            cudaFree(device_seed);
            cudaFree(device_lane_private_keys);
            cudaFree(device_watch);
            return 2;
        }
        for (int i = 0; i < watch.count; ++i) {
            // A lane that found the requested vanity result may already have
            // queued an incidental match earlier in its +8 walk.  Persisting
            // both would expose their small scalar relationship, so the final
            // result takes precedence.  Baseline candidates are independently
            // hashed per attempt and do not share that relationship.
            if (engine == "optimized" && result.found
                    && watch.results[i].lane == result.lane) continue;
            std::fprintf(stderr, "WATCH %d %s %s\n", watch.results[i].rule,
                         hex(watch.results[i].public_key, 32).c_str(),
                         hex(watch.results[i].private_key, 64).c_str());
        }
        std::fflush(stderr);
        attempts += static_cast<unsigned long long>(blocks) * kThreads * kAttemptsPerThread;
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (result.found) {
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
    cudaFree(device_lane_private_keys);
    cudaFree(device_watch);
    return 0;
}
