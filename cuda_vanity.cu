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
constexpr int kInteractiveBlocksPerSm = 4;
constexpr int kInteractiveOptimizedAttemptsPerThread = 2048;
constexpr int kInteractiveBaselineAttemptsPerThread = 32;
#ifndef MC_MAX_REGISTERS
#define MC_MAX_REGISTERS 0
#endif
constexpr int kMaxRegisters = MC_MAX_REGISTERS;
#ifndef MC_BUILD_ARCHES
#define MC_BUILD_ARCHES "manual"
#endif
#ifndef MC_BUILD_FINGERPRINT
#define MC_BUILD_FINGERPRINT "manual"
#endif
constexpr const char *kBuildArches = MC_BUILD_ARCHES;
constexpr const char *kBuildFingerprint = MC_BUILD_FINGERPRINT;
constexpr int kProbeSchemaVersion = 1;
constexpr int kRareRulesetSchemaVersion = 2;
constexpr int kRareRuleProtocolVersion = 2;
constexpr int kMaxRareRules = 32;
constexpr int kMaxRareRuleValueNibbles = 64;
constexpr int kMaxRareClassifierInputs = 256;
constexpr int kResultCheckInterval = 32;
constexpr int kWatchCapacity = 64;
constexpr int kLaneDomainSize = 24;
constexpr int kIsolationTestLanes = 4;
static_assert(kThreads > 0 && kThreads <= 1024 && kThreads % 32 == 0,
              "MC_THREADS must be a positive warp multiple no greater than 1024");
static_assert(kAttemptsPerThread > 0 && kAttemptsPerThread % 32 == 0,
              "MC_ATTEMPTS_PER_THREAD must be a positive multiple of 32");
static_assert(kInteractiveBlocksPerSm > 0,
              "interactive blocks per SM must be positive");
static_assert(kInteractiveOptimizedAttemptsPerThread > 0
                  && kInteractiveOptimizedAttemptsPerThread % 32 == 0,
              "interactive optimized attempts must be a positive multiple of 32");
static_assert(kInteractiveBaselineAttemptsPerThread > 0,
              "interactive baseline attempts must be positive");

__constant__ char gpu_prefix[kMaxPattern + 1];
__constant__ char gpu_suffix[kMaxPattern + 1];
__constant__ char gpu_contains[kMaxPattern + 1];
__constant__ int gpu_prefix_len;
__constant__ int gpu_suffix_len;
__constant__ int gpu_contains_len;
__constant__ int gpu_collect_only;

struct CudaRareRuleSpec {
    unsigned char kind;
    unsigned char threshold_nibbles;
    unsigned short excluded_mask;
    unsigned char value[kMaxRareRuleValueNibbles];
};
static_assert(sizeof(CudaRareRuleSpec) == 68,
              "CUDA rare-rule layout must remain stable");

struct HostRareRuleSpec {
    unsigned char kind;
    unsigned char threshold_nibbles;
    unsigned short excluded_mask;
    std::string value;
};

__constant__ CudaRareRuleSpec gpu_rare_rules[kMaxRareRules];
__constant__ int gpu_rare_rule_count;
__constant__ int gpu_use_generated_default_rules;
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

// This deliberately performs no key work. It verifies that the selected CUDA
// context can execute and synchronize a real kernel rather than merely being
// listed by the driver.
__global__ void readiness_probe_kernel(unsigned int *marker, unsigned int engine_marker) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *marker = 0x4d435052U ^ engine_marker;  // "MCPR"
}

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

__device__ __forceinline__ unsigned char rare_nibble_at(
        const unsigned char *key, int index) {
    return (index & 1) ? (key[index / 2] & 15) : (key[index / 2] >> 4);
}

__device__ __forceinline__ unsigned int rare_eight_nibble_window(
        const unsigned char *key, int offset) {
    const int byte = offset >> 1;
    if ((offset & 1) == 0) {
        return (static_cast<unsigned int>(key[byte]) << 24)
            | (static_cast<unsigned int>(key[byte + 1]) << 16)
            | (static_cast<unsigned int>(key[byte + 2]) << 8)
            | static_cast<unsigned int>(key[byte + 3]);
    }
    return (static_cast<unsigned int>(key[byte] & 15) << 28)
        | (static_cast<unsigned int>(key[byte + 1]) << 20)
        | (static_cast<unsigned int>(key[byte + 2]) << 12)
        | (static_cast<unsigned int>(key[byte + 3]) << 4)
        | (static_cast<unsigned int>(key[byte + 4]) >> 4);
}

// Keep the effectively-never-taken slow path outside the scan kernel. A call
// occurs only after eight nibbles already matched (roughly once per 2^32
// alignments), so inlining this variable-length comparison would add register
// pressure and instructions to every generated candidate for no steady-state
// benefit.
__device__ __noinline__ bool rare_bookend_remainder_matches(
        const unsigned char *key, int suffix_start, int length) {
    for (int index = 8; index < length; ++index) {
        if (rare_nibble_at(key, index)
                != rare_nibble_at(key, suffix_start + index)) return false;
    }
    return true;
}

__device__ __forceinline__ bool rare_bookend_minimum(
        const unsigned char *key, int minimum_length) {
    // A longer bookend starts earlier in the suffix, so widths are not nested:
    // matching width 12 does not imply matching width 10. Search every possible
    // suffix alignment with a rolling, collision-free eight-nibble window. The
    // safety floor guarantees minimum_length >= 9; consequently the slower
    // remainder comparison is reached only after a 32-bit exact match.
    const unsigned int prefix_window = rare_eight_nibble_window(key, 0);
    unsigned int suffix_window = rare_eight_nibble_window(key, 32);
    const int last_start = 32 - minimum_length;

    // Advancing two nibble positions consumes exactly one byte. Pairing the
    // even and odd windows halves the indexed key loads in this hot path.
    for (int start = 0; start <= last_start; start += 2) {
        const int even_length = 32 - start;
        if (prefix_window == suffix_window
                && rare_bookend_remainder_matches(
                    key, 32 + start, even_length)) return true;

        if (start == last_start) break;
        const unsigned char next_byte = key[20 + (start >> 1)];
        const unsigned int odd_window =
            (suffix_window << 4) | (static_cast<unsigned int>(next_byte) >> 4);
        const int odd_length = 31 - start;
        if (prefix_window == odd_window
                && rare_bookend_remainder_matches(
                    key, 33 + start, odd_length)) return true;

        suffix_window = (suffix_window << 8) | next_byte;
    }
    return false;
}

template<int MinimumLength>
__device__ __forceinline__ bool rare_static_bookend_minimum(
        const unsigned char *key) {
    return rare_bookend_minimum(key, MinimumLength);
}

template<int Length>
__device__ __forceinline__ bool rare_static_mirror(const unsigned char *key) {
    bool matches = true;
#pragma unroll
    for (int i = 0; i < Length; ++i)
        if (rare_nibble_at(key, i) != rare_nibble_at(key, 63 - i))
            matches = false;
    return matches;
}

template<int Length, unsigned int ExcludedMask>
__device__ __forceinline__ bool rare_static_repeat(const unsigned char *key) {
    const unsigned char first = rare_nibble_at(key, 0);
    bool matches = (ExcludedMask & (1U << first)) == 0;
#pragma unroll
    for (int i = 1; i < Length; ++i)
        if (rare_nibble_at(key, i) != first) matches = false;
    return matches;
}

template<int Index, unsigned int... Expected>
struct RareStaticLiteralMatcher;

template<int Index>
struct RareStaticLiteralMatcher<Index> {
    __device__ __forceinline__ static bool matches(const unsigned char *) {
        return true;
    }
};

template<int Index, unsigned int First, unsigned int... Remaining>
struct RareStaticLiteralMatcher<Index, First, Remaining...> {
    __device__ __forceinline__ static bool matches(const unsigned char *key) {
        return rare_nibble_at(key, Index) == First
            && RareStaticLiteralMatcher<Index + 1, Remaining...>::matches(key);
    }
};

template<unsigned int... Expected>
__device__ __forceinline__ bool rare_static_literal(const unsigned char *key) {
    return RareStaticLiteralMatcher<0, Expected...>::matches(key);
}

#include "generated/rare_rules_default.cuh"

static_assert(meshcore_rare_generated::kSchemaVersion
                  == kRareRulesetSchemaVersion,
              "generated rare-rule schema does not match CUDA engine");
static_assert(meshcore_rare_generated::kCudaRuleProtocolVersion
                  == kRareRuleProtocolVersion,
              "generated rare-rule protocol does not match CUDA engine");
static_assert(meshcore_rare_generated::kRuleCount > 0
                  && meshcore_rare_generated::kRuleCount <= kMaxRareRules,
              "generated default rules exceed CUDA capacity");

__device__ __forceinline__ bool rare_generic_rule_matches(
        const unsigned char *key, const CudaRareRuleSpec &rule) {
    if (rule.kind == 0) {
        return rare_bookend_minimum(key, rule.threshold_nibbles);
    }
    if (rule.kind == 1) {
        for (int i = 0; i < rule.threshold_nibbles; ++i)
            if (rare_nibble_at(key, i) != rare_nibble_at(key, 63 - i))
                return false;
        return true;
    }
    if (rule.kind == 2) {
        const unsigned char first = rare_nibble_at(key, 0);
        if (rule.excluded_mask & (1U << first)) return false;
        for (int i = 1; i < rule.threshold_nibbles; ++i)
            if (rare_nibble_at(key, i) != first) return false;
        return true;
    }
    if (rule.kind == 3 || rule.kind == 4) {
        for (int i = 0; i < rule.threshold_nibbles; ++i)
            if (rare_nibble_at(key, i) != rule.value[i]) return false;
        return true;
    }
    return false;
}

__device__ __forceinline__ int classify_generic_rules(
        const unsigned char *key) {
    for (int index = 0; index < gpu_rare_rule_count; ++index)
        if (rare_generic_rule_matches(key, gpu_rare_rules[index])) return index;
    return -1;
}

__device__ __forceinline__ int interesting_rule(const unsigned char *key) {
    if (gpu_use_generated_default_rules)
        return meshcore_rare_generated::classify_generated_default(key);
    return classify_generic_rules(key);
}

__global__ void rare_classifier_test_kernel(const unsigned char *keys, int count,
                                             int *selected_rules,
                                             int *default_mismatch) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned char *key = keys + static_cast<size_t>(index) * 32;
    const int generic = classify_generic_rules(key);
    int selected = generic;
    if (gpu_use_generated_default_rules) {
        selected = meshcore_rare_generated::classify_generated_default(key);
        if (selected != generic) atomicExch(default_mismatch, 1);
    }
    selected_rules[index] = selected;
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

template<int AttemptCount>
__global__ void scan_kernel_baseline(const unsigned char *base_seed, DeviceResult *result,
                                     WatchBatch *watch_batch) {
    static_assert(AttemptCount > 0, "baseline scan needs at least one attempt");
    unsigned char seed[32];
    unsigned char private_key[64];
    unsigned char public_key[32];
    ge_p3 point;
    bool lane_has_watch = false;
    const unsigned long long lane = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; ++i) seed[i] = base_seed[i];
    increment_seed(seed, lane * AttemptCount);

    for (int attempt = 0; attempt < AttemptCount; ++attempt) {
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

template<int AttemptCount>
__global__ void scan_kernel_optimized(const unsigned char *lane_private_keys,
                                      unsigned long long lane_count,
                                      DeviceResult *result, WatchBatch *watch_batch) {
    static_assert(AttemptCount > 0 && AttemptCount % 32 == 0,
                  "optimized scan attempts must be a positive multiple of 32");
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

    for (int attempt = 0; attempt < AttemptCount; attempt += 32) {
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

struct LaunchProfile {
    int blocks_per_sm;
    int attempts_per_thread;
};

cudaError_t select_launch_profile(const std::string &engine, bool interactive,
                                  LaunchProfile &profile) {
    profile = {kBlocksPerSm, kAttemptsPerThread};
    if (!interactive) return cudaSuccess;

    int resident_blocks = 0;
    cudaError_t status = engine == "optimized"
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &resident_blocks,
              scan_kernel_optimized<kInteractiveOptimizedAttemptsPerThread>,
              kThreads, 0)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &resident_blocks,
              scan_kernel_baseline<kInteractiveBaselineAttemptsPerThread>,
              kThreads, 0);
    if (status != cudaSuccess) return status;
    if (resident_blocks < 1) return cudaErrorInvalidConfiguration;
    const int interactive_limit = kInteractiveBlocksPerSm < kBlocksPerSm
        ? kInteractiveBlocksPerSm : kBlocksPerSm;
    profile.blocks_per_sm = resident_blocks < interactive_limit
        ? resident_blocks : interactive_limit;
    profile.attempts_per_thread = engine == "optimized"
        ? kInteractiveOptimizedAttemptsPerThread
        : kInteractiveBaselineAttemptsPerThread;
    return cudaSuccess;
}

bool valid_hex(const std::string &value) {
    if (value.size() > kMaxPattern) return false;
    for (char ch : value)
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    return true;
}

bool valid_lower_hex(const std::string &value, size_t maximum_length,
                     bool allow_empty = false) {
    if ((!allow_empty && value.empty()) || value.size() > maximum_length) return false;
    for (char ch : value)
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    return true;
}

bool parse_canonical_decimal(const std::string &value, int maximum, int &parsed) {
    if (value.empty() || (value.size() > 1 && value[0] == '0')) return false;
    int result = 0;
    for (char ch : value) {
        if (ch < '0' || ch > '9') return false;
        result = result * 10 + (ch - '0');
        if (result > maximum) return false;
    }
    parsed = result;
    return true;
}

bool parse_canonical_mask(const std::string &value, unsigned short &parsed) {
    if (value.size() != 4 || !valid_lower_hex(value, 4)) return false;
    unsigned int result = 0;
    for (char ch : value) {
        result <<= 4;
        result |= static_cast<unsigned int>(
            ch <= '9' ? ch - '0' : ch - 'a' + 10);
    }
    parsed = static_cast<unsigned short>(result);
    return true;
}

int mask_bit_count(unsigned short value) {
    int count = 0;
    for (int bit = 0; bit < 16; ++bit)
        if (value & (1U << bit)) ++count;
    return count;
}

bool parse_rare_rule_specification(const std::string &specification,
                                   HostRareRuleSpec &rule,
                                   std::string &error) {
    const size_t first = specification.find(':');
    const size_t second = first == std::string::npos
        ? std::string::npos : specification.find(':', first + 1);
    const size_t third = second == std::string::npos
        ? std::string::npos : specification.find(':', second + 1);
    if (first == std::string::npos || second == std::string::npos
            || third == std::string::npos
            || specification.find(':', third + 1) != std::string::npos) {
        error = "rule must contain exactly four colon-delimited fields";
        return false;
    }

    const std::string kind_field = specification.substr(0, first);
    const std::string length_field = specification.substr(first + 1, second - first - 1);
    const std::string mask_field = specification.substr(second + 1, third - second - 1);
    const std::string value = specification.substr(third + 1);
    int kind = 0;
    int length = 0;
    unsigned short mask = 0;
    if (kind_field.size() != 1 || !parse_canonical_decimal(kind_field, 4, kind)) {
        error = "rule kind must be one canonical digit from 0 through 4";
        return false;
    }
    if (!parse_canonical_decimal(length_field, 64, length) || length < 1) {
        error = "rule length must be a canonical integer from 1 through 64";
        return false;
    }
    if (!parse_canonical_mask(mask_field, mask)) {
        error = "rule mask must be exactly four lowercase hexadecimal digits";
        return false;
    }
    if (!valid_lower_hex(value, kMaxRareRuleValueNibbles, true)) {
        error = "rule value must contain at most 64 lowercase hexadecimal digits";
        return false;
    }

    if ((kind == 0 || kind == 1) && length > 32) {
        error = "bookend and mirror lengths must not exceed 32";
        return false;
    }
    if (kind == 0 || kind == 1) {
        if (mask != 0 || !value.empty()) {
            error = "bookend and mirror rules require zero mask and empty value";
            return false;
        }
    } else if (kind == 2) {
        if (!value.empty() || !(mask & 0x0001U) || !(mask & 0x8000U)
                || mask == 0xffffU) {
            error = "repeat rules require empty value and a non-total mask including 0 and f";
            return false;
        }
    } else if (kind == 3) {
        if (mask != 0 || value.empty() || static_cast<int>(value.size()) != length) {
            error = "literal rules require zero mask and value length equal to rule length";
            return false;
        }
        if (value.size() >= 2 && (value.substr(0, 2) == "00"
                                  || value.substr(0, 2) == "ff")) {
            error = "literal rule begins with a MeshCore-reserved byte";
            return false;
        }
    } else if (kind == 4) {
        if (mask != 0 || value.empty() || length > static_cast<int>(value.size())) {
            error = "sequence rules require zero mask and length within a non-empty value";
            return false;
        }
        if (length >= 2 && (value.substr(0, 2) == "00"
                            || value.substr(0, 2) == "ff")) {
            error = "sequence rule begins with a MeshCore-reserved byte";
            return false;
        }
    }

    const int alternatives = kind == 2 ? 16 - mask_bit_count(mask) : 1;
    long double probability = static_cast<long double>(alternatives);
    for (int nibble = 0; nibble < length; ++nibble) probability /= 16.0L;
    if (kind == 0) {
        // Bookend widths are differently aligned, non-nested events.  Use the
        // same conservative union bound as Python for protocol validation.
        long double combined_bookend_probability = 0.0L;
        for (int width = length; width <= 32; ++width) {
            combined_bookend_probability += probability;
            probability /= 16.0L;
        }
        probability = combined_bookend_probability;
    }
    constexpr long double kMaximumIndividualProbability =
        1.0L / 4294967296.0L;  // 2^-32
    if (probability > kMaximumIndividualProbability) {
        error = "rule is more frequent than the 32-bit safety floor";
        return false;
    }

    rule.kind = static_cast<unsigned char>(kind);
    rule.threshold_nibbles = static_cast<unsigned char>(length);
    rule.excluded_mask = mask;
    rule.value = value;
    return true;
}

long double rare_rule_probability(const HostRareRuleSpec &rule) {
    const int alternatives = rule.kind == 2
        ? 16 - mask_bit_count(rule.excluded_mask) : 1;
    long double probability = static_cast<long double>(alternatives);
    for (int nibble = 0; nibble < rule.threshold_nibbles; ++nibble)
        probability /= 16.0L;
    if (rule.kind == 0) {
        long double combined_bookend_probability = 0.0L;
        for (int width = rule.threshold_nibbles; width <= 32; ++width) {
            combined_bookend_probability += probability;
            probability /= 16.0L;
        }
        return combined_bookend_probability;
    }
    return probability;
}

bool validate_rare_rule_set(const std::vector<HostRareRuleSpec> &rules,
                            std::string &error) {
    if (rules.empty() || rules.size() > kMaxRareRules) {
        error = "ruleset must contain between 1 and 32 active rules";
        return false;
    }
    bool structural_seen[3]{};
    long double combined_probability = 0.0L;
    for (const HostRareRuleSpec &rule : rules) {
        if (rule.kind < 3) {
            if (structural_seen[rule.kind]) {
                error = "bookend, mirror, and repeat-prefix may each appear only once";
                return false;
            }
            structural_seen[rule.kind] = true;
        }
        combined_probability += rare_rule_probability(rule);
    }
    constexpr long double kMaximumCombinedProbability =
        1.0L / 268435456.0L;  // 2^-28
    if (combined_probability > kMaximumCombinedProbability) {
        error = "combined rules are more frequent than the 28-bit safety floor";
        return false;
    }
    return true;
}

std::vector<HostRareRuleSpec> generated_default_rules() {
    std::vector<HostRareRuleSpec> rules;
    rules.reserve(meshcore_rare_generated::kRuleCount);
    for (unsigned int index = 0; index < meshcore_rare_generated::kRuleCount; ++index) {
        const meshcore_rare_generated::RareRuleSpec &source =
            meshcore_rare_generated::kRules[index];
        rules.push_back({source.kind, source.minimum_nibbles,
                         source.excluded_mask, source.value});
    }
    return rules;
}

bool is_generated_default_ruleset(const std::string &fingerprint,
                                  const std::vector<HostRareRuleSpec> &rules) {
    if (fingerprint != meshcore_rare_generated::kRulesetFingerprint
            || rules.size() != meshcore_rare_generated::kRuleCount) return false;
    for (size_t index = 0; index < rules.size(); ++index) {
        const HostRareRuleSpec &actual = rules[index];
        const meshcore_rare_generated::RareRuleSpec &expected =
            meshcore_rare_generated::kRules[index];
        if (actual.kind != expected.kind
                || actual.threshold_nibbles != expected.minimum_nibbles
                || actual.excluded_mask != expected.excluded_mask
                || actual.value != expected.value) return false;
    }
    return true;
}

void copy_rare_rules_to_device(const std::vector<HostRareRuleSpec> &rules,
                               bool use_generated_default) {
    std::array<CudaRareRuleSpec, kMaxRareRules> encoded{};
    for (size_t index = 0; index < rules.size(); ++index) {
        encoded[index].kind = rules[index].kind;
        encoded[index].threshold_nibbles = rules[index].threshold_nibbles;
        encoded[index].excluded_mask = rules[index].excluded_mask;
        for (size_t nibble = 0; nibble < rules[index].value.size(); ++nibble) {
            const char ch = rules[index].value[nibble];
            encoded[index].value[nibble] = static_cast<unsigned char>(
                ch <= '9' ? ch - '0' : ch - 'a' + 10);
        }
    }
    const int count = static_cast<int>(rules.size());
    const int default_flag = use_generated_default ? 1 : 0;
    cuda_check(cudaMemcpyToSymbol(gpu_rare_rules, encoded.data(), sizeof(encoded)),
               "copy rare rules");
    cuda_check(cudaMemcpyToSymbol(gpu_rare_rule_count, &count, sizeof(count)),
               "copy rare rule count");
    cuda_check(cudaMemcpyToSymbol(gpu_use_generated_default_rules, &default_flag,
                                  sizeof(default_flag)),
               "copy rare rule mode");
}

bool decode_public_hex(const std::string &value,
                       std::array<unsigned char, 32> &decoded) {
    if (value.size() != 64 || !valid_lower_hex(value, 64)) return false;
    for (size_t index = 0; index < decoded.size(); ++index) {
        const char high = value[index * 2];
        const char low = value[index * 2 + 1];
        const unsigned int high_value = high <= '9' ? high - '0' : high - 'a' + 10;
        const unsigned int low_value = low <= '9' ? low - '0' : low - 'a' + 10;
        decoded[index] = static_cast<unsigned char>((high_value << 4) | low_value);
    }
    return true;
}

int run_rare_rule_parser_test(const std::vector<HostRareRuleSpec> &rules,
                              bool use_generated_default) {
    std::printf(
        "{\"protocol\":\"meshcore-cuda-rare-rules-v2\",\"rules\":%zu,"
        "\"default_fast_path\":%s}\n",
        rules.size(), use_generated_default ? "true" : "false");
    return 0;
}

int run_rare_classifier_test(
        int selected_device, const std::vector<HostRareRuleSpec> &rules,
        bool use_generated_default,
        const std::vector<std::array<unsigned char, 32>> &public_values) {
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "classifier cudaGetDeviceCount");
    if (!device_count || selected_device >= device_count) {
        std::fprintf(stderr, "CUDA device %d is unavailable for classifier test\n",
                     selected_device);
        return 2;
    }
    cuda_check(cudaSetDevice(selected_device), "classifier cudaSetDevice");
    copy_rare_rules_to_device(rules, use_generated_default);

    unsigned char *device_public = nullptr;
    int *device_rules = nullptr;
    int *device_mismatch = nullptr;
    const size_t public_bytes = public_values.size() * 32;
    cuda_check(cudaMalloc(&device_public, public_bytes), "classifier cudaMalloc public");
    cuda_check(cudaMalloc(&device_rules, public_values.size() * sizeof(int)),
               "classifier cudaMalloc results");
    cuda_check(cudaMalloc(&device_mismatch, sizeof(int)),
               "classifier cudaMalloc parity");
    cuda_check(cudaMemcpy(device_public, public_values.data(), public_bytes,
                          cudaMemcpyHostToDevice),
               "classifier copy public");
    cuda_check(cudaMemset(device_mismatch, 0, sizeof(int)),
               "classifier clear parity");
    const int threads = 128;
    const int blocks = (static_cast<int>(public_values.size()) + threads - 1) / threads;
    rare_classifier_test_kernel<<<blocks, threads>>>(
        device_public, static_cast<int>(public_values.size()),
        device_rules, device_mismatch);
    cuda_check(cudaGetLastError(), "classifier launch");
    std::vector<int> classified(public_values.size());
    int mismatch = 0;
    cuda_check(cudaMemcpy(classified.data(), device_rules,
                          classified.size() * sizeof(int), cudaMemcpyDeviceToHost),
               "classifier copy results");
    cuda_check(cudaMemcpy(&mismatch, device_mismatch, sizeof(mismatch),
                          cudaMemcpyDeviceToHost),
               "classifier copy parity");
    cudaFree(device_mismatch);
    cudaFree(device_rules);
    cudaFree(device_public);

    std::printf(
        "{\"protocol\":\"meshcore-cuda-rare-classifier-v2\",\"rules\":%zu,"
        "\"default_fast_path\":%s,\"default_generic_parity\":%s,\"indices\":[",
        rules.size(), use_generated_default ? "true" : "false",
        mismatch ? "false" : "true");
    for (size_t index = 0; index < classified.size(); ++index) {
        if (index) std::putchar(',');
        std::printf("%d", classified[index]);
    }
    std::printf("]}\n");
    return mismatch ? 1 : 0;
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

std::string json_escape(const std::string &value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size());
    for (unsigned char ch : value) {
        switch (ch) {
            case '\"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) {
                    escaped += "\\u00";
                    escaped.push_back(digits[ch >> 4]);
                    escaped.push_back(digits[ch & 15]);
                } else {
                    escaped.push_back(static_cast<char>(ch));
                }
        }
    }
    return escaped;
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

int emit_probe_failure(int selected_device, const std::string &engine,
                       const std::string &error) {
    std::printf(
        "{\"schema\":%d,\"protocol\":\"meshcore-cuda-probe-v2\","
        "\"rare_rule_protocol\":%d,\"default_ruleset_fingerprint\":\"%s\","
        "\"ready\":false,\"device\":%d,\"engine\":\"%s\","
        "\"build_fingerprint\":\"%s\",\"build_arches\":\"%s\","
        "\"threads\":%d,\"blocks_per_sm\":%d,\"attempts_per_thread\":%d,"
        "\"max_registers\":%d,\"error\":\"%s\"}\n",
        kProbeSchemaVersion, kRareRuleProtocolVersion,
        meshcore_rare_generated::kRulesetFingerprint,
        selected_device, engine.c_str(),
        kBuildFingerprint, kBuildArches, kThreads, kBlocksPerSm,
        kAttemptsPerThread, kMaxRegisters, json_escape(error).c_str());
    std::fflush(stdout);
    return 2;
}

int run_readiness_probe(int selected_device, const std::string &engine,
                        bool interactive) {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        return emit_probe_failure(
            selected_device, engine,
            std::string("cudaGetDeviceCount: ") + cudaGetErrorString(status));
    }
    if (!device_count)
        return emit_probe_failure(selected_device, engine, "no CUDA device found");
    if (selected_device >= device_count) {
        char message[128];
        std::snprintf(message, sizeof(message),
                      "CUDA device %d is unavailable; detected %d device(s)",
                      selected_device, device_count);
        return emit_probe_failure(selected_device, engine, message);
    }

    status = cudaSetDevice(selected_device);
    if (status != cudaSuccess) {
        return emit_probe_failure(
            selected_device, engine,
            std::string("cudaSetDevice: ") + cudaGetErrorString(status));
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, selected_device);
    if (status != cudaSuccess) {
        return emit_probe_failure(
            selected_device, engine,
            std::string("cudaGetDeviceProperties: ") + cudaGetErrorString(status));
    }
    char pci_bus_id[32]{};
    const cudaError_t pci_status = cudaDeviceGetPCIBusId(
        pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), selected_device);
    const std::string pci_json = pci_status == cudaSuccess
        ? std::string(",\"pci_bus_id\":\"") + json_escape(pci_bus_id) + "\""
        : std::string();
    if (pci_status != cudaSuccess) {
        // PCI identity enriches telemetry mapping but is not required for key
        // generation. Do not let an optional lookup poison later launch checks.
        (void)cudaGetLastError();
    }

    LaunchProfile profile{};
    status = select_launch_profile(engine, interactive, profile);
    if (status != cudaSuccess) {
        return emit_probe_failure(
            selected_device, engine,
            std::string("select launch profile: ") + cudaGetErrorString(status));
    }
    const int production_blocks =
        properties.multiProcessorCount * profile.blocks_per_sm;
    const unsigned long long production_lane_count =
        static_cast<unsigned long long>(production_blocks) * kThreads;
    if (production_blocks <= 0 || production_lane_count == 0) {
        return emit_probe_failure(
            selected_device, engine,
            "selected CUDA device reported an invalid launch geometry");
    }

    unsigned int *device_marker = nullptr;
    unsigned char *device_input = nullptr;
    unsigned char *device_lane_state = nullptr;
    DeviceResult *device_result = nullptr;
    WatchBatch *device_watch = nullptr;
    auto cleanup = [&]() {
        cudaError_t first_error = cudaSuccess;
        auto release = [&first_error](void *pointer) {
            if (!pointer) return;
            cudaError_t released = cudaFree(pointer);
            if (first_error == cudaSuccess && released != cudaSuccess)
                first_error = released;
        };
        release(device_watch);
        release(device_result);
        release(device_lane_state);
        release(device_input);
        release(device_marker);
        return first_error;
    };
    auto fail = [&](const char *operation, cudaError_t error) {
        const std::string detail =
            std::string(operation) + ": " + cudaGetErrorString(error);
        cleanup();
        return emit_probe_failure(selected_device, engine, detail);
    };

    status = cudaMalloc(&device_marker, sizeof(*device_marker));
    if (status != cudaSuccess) return fail("probe cudaMalloc marker", status);
    status = cudaMalloc(&device_input, 32);
    if (status != cudaSuccess) return fail("probe cudaMalloc input", status);
    status = cudaMalloc(&device_result, sizeof(DeviceResult));
    if (status != cudaSuccess) return fail("probe cudaMalloc result", status);
    status = cudaMalloc(&device_watch, sizeof(WatchBatch));
    if (status != cudaSuccess) return fail("probe cudaMalloc watch", status);
    if (engine == "optimized") {
        // Match the persistent state allocation used by a production search so
        // readiness also catches realistic device-memory pressure. Only the
        // first configured block is initialized and launched below.
        status = cudaMalloc(&device_lane_state, production_lane_count * 64);
        if (status != cudaSuccess) return fail("probe cudaMalloc lane state", status);
    }

    // A fixed public input keeps this readiness check deterministic and ensures
    // it never creates secret key material. Search output buffers are cleared,
    // never copied beyond their status counters, and discarded before return.
    std::array<unsigned char, 32> probe_input{};
    for (size_t index = 0; index < probe_input.size(); ++index)
        probe_input[index] = static_cast<unsigned char>(index);
    status = cudaMemcpy(device_input, probe_input.data(), probe_input.size(),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail("probe copy input", status);
    status = cudaMemset(device_result, 0, sizeof(DeviceResult));
    if (status != cudaSuccess) return fail("probe clear result", status);
    status = cudaMemset(device_watch, 0, sizeof(WatchBatch));
    if (status != cudaSuccess) return fail("probe clear watch", status);
    status = cudaMemset(device_marker, 0, sizeof(*device_marker));
    if (status != cudaSuccess) return fail("probe clear marker", status);

    const int zero = 0;
    const int one = 1;
    status = cudaMemcpyToSymbol(gpu_prefix_len, &zero, sizeof(zero));
    if (status != cudaSuccess) return fail("probe clear prefix", status);
    status = cudaMemcpyToSymbol(gpu_suffix_len, &zero, sizeof(zero));
    if (status != cudaSuccess) return fail("probe clear suffix", status);
    status = cudaMemcpyToSymbol(gpu_contains_len, &zero, sizeof(zero));
    if (status != cudaSuccess) return fail("probe clear substring", status);
    status = cudaMemcpyToSymbol(gpu_collect_only, &one, sizeof(one));
    if (status != cudaSuccess) return fail("probe set collector mode", status);
    status = cudaMemcpyToSymbol(gpu_rare_rule_count, &zero, sizeof(zero));
    if (status != cudaSuccess) return fail("probe clear rare rules", status);
    status = cudaMemcpyToSymbol(gpu_use_generated_default_rules, &zero, sizeof(zero));
    if (status != cudaSuccess) return fail("probe clear rare-rule mode", status);

    // Exercise one full configured block of the selected production kernel.
    // This validates its real block geometry and register feasibility while
    // remaining bounded to one block of the selected profile.
    if (engine == "optimized") {
        derive_lane_private_keys<<<1, kThreads>>>(
            device_input, device_lane_state, production_lane_count);
        status = cudaGetLastError();
        if (status != cudaSuccess) return fail("probe launch lane derivation", status);
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) return fail("probe synchronize lane derivation", status);
        if (interactive) {
            scan_kernel_optimized<kInteractiveOptimizedAttemptsPerThread>
                <<<1, kThreads>>>(device_lane_state, production_lane_count,
                                  device_result, device_watch);
        } else {
            scan_kernel_optimized<kAttemptsPerThread><<<1, kThreads>>>(
                device_lane_state, production_lane_count, device_result, device_watch);
        }
    } else {
        if (interactive) {
            scan_kernel_baseline<kInteractiveBaselineAttemptsPerThread>
                <<<1, kThreads>>>(device_input, device_result, device_watch);
        } else {
            scan_kernel_baseline<kAttemptsPerThread><<<1, kThreads>>>(
                device_input, device_result, device_watch);
        }
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail("probe launch selected scan engine", status);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return fail("probe synchronize selected scan engine", status);

    int found_status = -1;
    int watch_status = -1;
    status = cudaMemcpy(&found_status, device_result, sizeof(found_status),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return fail("probe copy result status", status);
    status = cudaMemcpy(&watch_status, device_watch, sizeof(watch_status),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return fail("probe copy watch status", status);
    if (found_status != 0 || watch_status != 0) {
        cleanup();
        return emit_probe_failure(
            selected_device, engine,
            "selected CUDA scan smoke check returned unexpected status");
    }

    const unsigned int engine_marker = engine == "optimized" ? 1U : 2U;
    const unsigned int expected = 0x4d435052U ^ engine_marker;
    readiness_probe_kernel<<<1, 1>>>(device_marker, engine_marker);
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail("probe launch completion marker", status);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return fail("probe synchronize completion marker", status);
    unsigned int observed = 0;
    status = cudaMemcpy(&observed, device_marker, sizeof(observed),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return fail("probe copy completion marker", status);
    if (observed != expected) {
        cleanup();
        return emit_probe_failure(selected_device, engine,
                                  "probe kernel returned an invalid marker");
    }
    cudaError_t cleanup_status = cleanup();
    if (cleanup_status != cudaSuccess) {
        return emit_probe_failure(
            selected_device, engine,
            std::string("probe cudaFree: ") + cudaGetErrorString(cleanup_status));
    }

    std::printf(
        "{\"schema\":%d,\"protocol\":\"meshcore-cuda-probe-v2\","
        "\"rare_rule_protocol\":%d,\"default_ruleset_fingerprint\":\"%s\","
        "\"ready\":true,\"device\":%d,\"device_name\":\"%s\"%s,"
        "\"compute_capability\":\"%d.%d\",\"engine\":\"%s\","
        "\"build_fingerprint\":\"%s\",\"build_arches\":\"%s\","
        "\"threads\":%d,\"blocks_per_sm\":%d,\"attempts_per_thread\":%d,"
        "\"max_registers\":%d}\n",
        kProbeSchemaVersion, kRareRuleProtocolVersion,
        meshcore_rare_generated::kRulesetFingerprint, selected_device,
        json_escape(properties.name).c_str(), pci_json.c_str(),
        properties.major, properties.minor,
        engine.c_str(), kBuildFingerprint, kBuildArches, kThreads,
        profile.blocks_per_sm, profile.attempts_per_thread, kMaxRegisters);
    std::fflush(stdout);
    return 0;
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
    std::fprintf(stderr,
                 "Usage: %s [--engine optimized|baseline] [--device N] [--interactive] [--collect-only] [--prefix HEX] [--suffix HEX] [--contains HEX]\n"
                 "          [--rare-rules-v2 FINGERPRINT [--rare-rule-v2 SPEC]...]\n"
                 "       %s --probe [--device N] [--engine optimized|baseline] [--interactive]\n",
                 program, program);
}

int rare_protocol_error(const char *program, const std::string &error) {
    std::fprintf(stderr, "Invalid rare-rule protocol: %s\n", error.c_str());
    usage(program);
    return 2;
}
}  // namespace

int main(int argc, char **argv) {
    std::string prefix, suffix, contains;
    std::string engine = "optimized";
    int selected_device = 0;
    bool collect_only = false;
    bool interactive = false;
    bool lane_isolation_self_test = false;
    bool rare_parser_self_test = false;
    bool probe = false;
    bool option_error = false;
    bool device_option_seen = false;
    bool engine_option_seen = false;
    bool incremental_engine_alias = false;
    bool rare_protocol_seen = false;
    bool rare_block_closed = false;
    std::string rare_fingerprint;
    std::vector<HostRareRuleSpec> rare_rules;
    std::vector<std::array<unsigned char, 32>> classifier_values;
    for (int i = 1; i < argc;) {
        std::string option = argv[i++];
        if (option == "--rare-rules-v1" || option == "--rare-rule-v1") {
            return rare_protocol_error(
                argv[0],
                "protocol v1 is unsupported; use --rare-rules-v2 and --rare-rule-v2");
        }
        if (option == "--rare-rules-v2") {
            if (i >= argc) return rare_protocol_error(argv[0], "missing fingerprint");
            const std::string value = argv[i++];
            if (rare_protocol_seen || rare_block_closed)
                return rare_protocol_error(argv[0], "duplicate or non-contiguous header");
            if (value.size() != 64 || !valid_lower_hex(value, 64))
                return rare_protocol_error(
                    argv[0], "fingerprint must be 64 lowercase hexadecimal digits");
            rare_protocol_seen = true;
            rare_fingerprint = value;
            continue;
        }
        if (option == "--rare-rule-v2") {
            if (i >= argc) return rare_protocol_error(argv[0], "missing rule specification");
            const std::string value = argv[i++];
            if (!rare_protocol_seen || rare_block_closed)
                return rare_protocol_error(
                    argv[0], "rules must immediately follow one protocol header");
            if (rare_rules.size() >= kMaxRareRules)
                return rare_protocol_error(argv[0], "more than 32 active rules");
            HostRareRuleSpec parsed{};
            std::string error;
            if (!parse_rare_rule_specification(value, parsed, error))
                return rare_protocol_error(argv[0], error);
            rare_rules.push_back(parsed);
            continue;
        }
        if (rare_protocol_seen) rare_block_closed = true;
        if (option == "--collect-only") {
            if (collect_only) option_error = true;
            collect_only = true;
            continue;
        }
        if (option == "--interactive") {
            if (interactive) option_error = true;
            interactive = true;
            continue;
        }
        if (option == "--internal-test-lane-isolation") {
            if (lane_isolation_self_test) option_error = true;
            lane_isolation_self_test = true;
            continue;
        }
        if (option == "--internal-test-rare-rules") {
            if (rare_parser_self_test) option_error = true;
            rare_parser_self_test = true;
            continue;
        }
        if (option == "--probe") {
            if (probe) option_error = true;
            probe = true;
            continue;
        }
        if (i >= argc) { usage(argv[0]); return 2; }
        std::string value = argv[i++];
        if (option == "--internal-test-rare-classifier") {
            if (classifier_values.size() >= kMaxRareClassifierInputs)
                return rare_protocol_error(argv[0], "more than 256 classifier inputs");
            std::array<unsigned char, 32> decoded{};
            if (!decode_public_hex(value, decoded))
                return rare_protocol_error(
                    argv[0], "classifier input must be 64 lowercase hexadecimal digits");
            classifier_values.push_back(decoded);
            continue;
        }
        if (option == "--device") {
            if (device_option_seen) option_error = true;
            device_option_seen = true;
            int parsed = 0;
            if (!parse_canonical_decimal(value, 1024, parsed)) {
                usage(argv[0]);
                return 2;
            }
            selected_device = parsed;
            continue;
        }
        if (option == "--engine") {
            if (engine_option_seen) option_error = true;
            engine_option_seen = true;
            if (value == "incremental") {
                incremental_engine_alias = true;
                value = "optimized";  // pre-batching compatibility alias
            }
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

    if (rare_protocol_seen) {
        std::string error;
        if (!validate_rare_rule_set(rare_rules, error))
            return rare_protocol_error(argv[0], error);
    } else {
        // Preserve the native engine's historical direct-use behavior. Python
        // always supplies an explicit frozen ruleset, while an omitted block
        // safely selects the rules compiled into this executable.
        rare_fingerprint = meshcore_rare_generated::kRulesetFingerprint;
        rare_rules = generated_default_rules();
    }
    const bool use_generated_default =
        is_generated_default_ruleset(rare_fingerprint, rare_rules);

    if (probe) {
        if (option_error || collect_only || lane_isolation_self_test
                || rare_parser_self_test || !classifier_values.empty()
                || rare_protocol_seen
                || !prefix.empty() || !suffix.empty() || !contains.empty()
                || incremental_engine_alias) {
            usage(argv[0]);
            return 2;
        }
        return run_readiness_probe(selected_device, engine, interactive);
    }
    if (lane_isolation_self_test) {
        if (option_error || collect_only || interactive || rare_parser_self_test
                || !classifier_values.empty() || rare_protocol_seen
                || !prefix.empty() || !suffix.empty() || !contains.empty()
                || engine != "optimized" || engine_option_seen) {
            usage(argv[0]);
            return 2;
        }
        return run_lane_isolation_self_test(selected_device);
    }
    if (rare_parser_self_test) {
        if (option_error || collect_only || interactive || !classifier_values.empty()
                || device_option_seen || engine_option_seen
                || !prefix.empty() || !suffix.empty() || !contains.empty()) {
            usage(argv[0]);
            return 2;
        }
        return run_rare_rule_parser_test(rare_rules, use_generated_default);
    }
    if (!classifier_values.empty()) {
        if (option_error || collect_only || interactive || engine_option_seen
                || !prefix.empty() || !suffix.empty() || !contains.empty()) {
            usage(argv[0]);
            return 2;
        }
        return run_rare_classifier_test(
            selected_device, rare_rules, use_generated_default, classifier_values);
    }
    if (option_error) {
        usage(argv[0]);
        return 2;
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
    copy_rare_rules_to_device(rare_rules, use_generated_default);
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

    LaunchProfile profile{};
    cuda_check(select_launch_profile(engine, interactive, profile),
               "select launch profile");
    int blocks = properties.multiProcessorCount * profile.blocks_per_sm;
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
    std::fprintf(stderr,
                 "GPU %d: %s, engine %s, mode %s, scheduling %s, "
                 "%d blocks x %d threads x %d attempts\n",
                 selected_device, properties.name, engine.c_str(),
                 collect_only ? "collector" : "vanity",
                 interactive ? "interactive" : "throughput",
                 blocks, kThreads, profile.attempts_per_thread);

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
            if (interactive) {
                scan_kernel_optimized<kInteractiveOptimizedAttemptsPerThread>
                    <<<blocks, kThreads>>>(device_lane_private_keys, lane_count,
                                           device_result, device_watch);
            } else {
                scan_kernel_optimized<kAttemptsPerThread><<<blocks, kThreads>>>(
                    device_lane_private_keys, lane_count, device_result, device_watch);
            }
        } else {
            if (interactive) {
                scan_kernel_baseline<kInteractiveBaselineAttemptsPerThread>
                    <<<blocks, kThreads>>>(device_seed, device_result, device_watch);
            } else {
                scan_kernel_baseline<kAttemptsPerThread><<<blocks, kThreads>>>(
                    device_seed, device_result, device_watch);
            }
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
        attempts += static_cast<unsigned long long>(blocks) * kThreads
                    * profile.attempts_per_thread;
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
