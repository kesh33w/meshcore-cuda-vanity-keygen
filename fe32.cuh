// Part of the MeshCore vanity key generator (see LICENSE).
// GF(2^255 - 19) arithmetic in eight 32-bit limbs (little-endian), using PTX
// carry chains. Values are "weakly reduced": any 256-bit integer, congruent to
// the represented element. Only fe32_tobytes / fe32_isnegative fully reduce.
#pragma once
#include <cstdint>

struct fe32 { uint32_t v[8]; };

#define FE32_INLINE __device__ __forceinline__

// ------------------------------------------------------------------ helpers

FE32_INLINE uint32_t add_cc(uint32_t a, uint32_t b) { uint32_t r; asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
FE32_INLINE uint32_t addc_cc(uint32_t a, uint32_t b) { uint32_t r; asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
FE32_INLINE uint32_t addc(uint32_t a, uint32_t b) { uint32_t r; asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
FE32_INLINE uint32_t sub_cc(uint32_t a, uint32_t b) { uint32_t r; asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
FE32_INLINE uint32_t subc_cc(uint32_t a, uint32_t b) { uint32_t r; asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
FE32_INLINE uint32_t subc(uint32_t a, uint32_t b) { uint32_t r; asm volatile("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
FE32_INLINE uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
FE32_INLINE uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
FE32_INLINE uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
FE32_INLINE uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
FE32_INLINE uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }

// Fold a 256-bit value plus an extra top word: (top * 2^256 + lo) mod-ish p,
// using 2^256 = 38 (mod p). Result fits in 256 bits.
FE32_INLINE void fe32_fold(uint32_t r[8], uint32_t top) {
    // top < 2^32; top*38 < 2^38: add as a two-word value.
    const uint64_t t = (uint64_t)top * 38u;
    r[0] = add_cc(r[0], (uint32_t)t);
    r[1] = addc_cc(r[1], (uint32_t)(t >> 32));
    r[2] = addc_cc(r[2], 0);
    r[3] = addc_cc(r[3], 0);
    r[4] = addc_cc(r[4], 0);
    r[5] = addc_cc(r[5], 0);
    r[6] = addc_cc(r[6], 0);
    r[7] = addc_cc(r[7], 0);
    const uint32_t c = addc(0, 0);
    // If that wrapped, the value is now below t < 2^38, so r[1] < 2^6 and
    // r[2..7] are zero: adding 38 can carry into r[1] but no further.
    r[0] = add_cc(r[0], c * 38u);
    r[1] = addc(r[1], 0);
}

// ---------------------------------------------------------------- arithmetic

FE32_INLINE void fe32_add(fe32 &h, const fe32 &f, const fe32 &g) {
    uint32_t r[8];
    r[0] = add_cc(f.v[0], g.v[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) r[i] = addc_cc(f.v[i], g.v[i]);
    const uint32_t c = addc(0, 0);
    fe32_fold(r, c);
#pragma unroll
    for (int i = 0; i < 8; ++i) h.v[i] = r[i];
}

FE32_INLINE void fe32_sub(fe32 &h, const fe32 &f, const fe32 &g) {
    // r = f - g mod 2^256; a borrow means we added 2^256 = 38, so take 38 off.
    uint32_t r[8];
    r[0] = sub_cc(f.v[0], g.v[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) r[i] = subc_cc(f.v[i], g.v[i]);
    uint32_t b = subc(0, 0) & 38u;
    r[0] = sub_cc(r[0], b);
#pragma unroll
    for (int i = 1; i < 8; ++i) r[i] = subc_cc(r[i], 0);
    b = subc(0, 0) & 38u;
    // A second borrow leaves r near 2^256; subtracting 38 cannot borrow.
    r[0] -= b;
#pragma unroll
    for (int i = 0; i < 8; ++i) h.v[i] = r[i];
}

// 8x8 -> 16 limb schoolbook product, row by row with lo/hi carry chains.
FE32_INLINE void fe32_mul_wide(uint32_t r[16], const uint32_t a[8], const uint32_t b[8]) {
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t bi = b[i];
        r[i] = madlo_cc(a[0], bi, r[i]);
#pragma unroll
        for (int j = 1; j < 8; ++j) r[i + j] = madloc_cc(a[j], bi, r[i + j]);
        r[i + 8] = addc(r[i + 8], 0);
        r[i + 1] = madhi_cc(a[0], bi, r[i + 1]);
#pragma unroll
        for (int j = 1; j < 7; ++j) r[i + j + 1] = madhic_cc(a[j], bi, r[i + j + 1]);
        r[i + 8] = madhic(a[7], bi, r[i + 8]);
    }
}

// Reduce a 512-bit product to 256 bits (weakly) via hi * 38 + lo.
FE32_INLINE void fe32_reduce_wide(fe32 &h, const uint32_t w[16]) {
    uint32_t r[8];
    r[0] = madlo_cc(w[8], 38u, w[0]);
#pragma unroll
    for (int j = 1; j < 8; ++j) r[j] = madloc_cc(w[8 + j], 38u, w[j]);
    uint32_t top = addc(0, 0);
    r[1] = madhi_cc(w[8], 38u, r[1]);
#pragma unroll
    for (int j = 1; j < 7; ++j) r[j + 1] = madhic_cc(w[8 + j], 38u, r[j + 1]);
    top = madhic(w[15], 38u, top);
    fe32_fold(r, top);
#pragma unroll
    for (int i = 0; i < 8; ++i) h.v[i] = r[i];
}

FE32_INLINE void fe32_mul(fe32 &h, const fe32 &f, const fe32 &g) {
    uint32_t w[16];
    fe32_mul_wide(w, f.v, g.v);
    fe32_reduce_wide(h, w);
}

// Squaring: off-diagonal products once, doubled, plus the diagonal.
FE32_INLINE void fe32_sq(fe32 &h, const fe32 &f) {
    const uint32_t *a = f.v;
    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = 0;
    // Off-diagonal: sum_{i<j} a_i a_j 2^{32(i+j)}.
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        const uint32_t ai = a[i];
        r[2 * i + 1] = madlo_cc(a[i + 1], ai, r[2 * i + 1]);
#pragma unroll
        for (int j = i + 2; j < 8; ++j) r[i + j] = madloc_cc(a[j], ai, r[i + j]);
        r[i + 8] = addc(r[i + 8], 0);
        r[2 * i + 2] = madhi_cc(a[i + 1], ai, r[2 * i + 2]);
#pragma unroll
        for (int j = i + 2; j < 8; ++j) r[i + j + 1] = madhic_cc(a[j], ai, r[i + j + 1]);
        if (i + 9 < 16) r[i + 9] = addc(r[i + 9], 0);
    }
    // Double.
    r[15] = (r[15] << 1) | (r[14] >> 31);
#pragma unroll
    for (int i = 14; i > 0; --i) r[i] = (r[i] << 1) | (r[i - 1] >> 31);
    r[0] <<= 1;
    // Diagonal.
    r[0] = madlo_cc(a[0], a[0], r[0]);
    r[1] = madhic_cc(a[0], a[0], r[1]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        r[2 * i] = madloc_cc(a[i], a[i], r[2 * i]);
        r[2 * i + 1] = madhic_cc(a[i], a[i], r[2 * i + 1]);
    }
    fe32_reduce_wide(h, r);
}

// Full reduction to [0, p).
FE32_INLINE void fe32_canonical(uint32_t r[8], const fe32 &f) {
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = f.v[i];
    // Fold bit 255: v = hi*2^255 + lo = hi*19 + lo.
    uint32_t hi = r[7] >> 31;
    r[7] &= 0x7fffffffu;
    r[0] = add_cc(r[0], hi * 19u);
#pragma unroll
    for (int i = 1; i < 8; ++i) r[i] = addc_cc(r[i], 0);
    // Now v < 2^255 + 19. Subtract p if v >= p: compute v + 19 and test bit 255.
    uint32_t t[8];
    t[0] = add_cc(r[0], 19u);
#pragma unroll
    for (int i = 1; i < 8; ++i) t[i] = addc_cc(r[i], 0);
    const uint32_t ge = t[7] >> 31;           // v + 19 >= 2^255  <=>  v >= p
    t[7] &= 0x7fffffffu;                      // (v + 19) - 2^255 = v - p
    const uint32_t mask = 0u - ge;
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = (t[i] & mask) | (r[i] & ~mask);
}

FE32_INLINE void fe32_tobytes(unsigned char *s, const fe32 &f) {
    uint32_t r[8];
    fe32_canonical(r, f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[4 * i + 0] = (unsigned char)r[i];
        s[4 * i + 1] = (unsigned char)(r[i] >> 8);
        s[4 * i + 2] = (unsigned char)(r[i] >> 16);
        s[4 * i + 3] = (unsigned char)(r[i] >> 24);
    }
}

FE32_INLINE int fe32_isnegative(const fe32 &f) {
    uint32_t r[8];
    fe32_canonical(r, f);
    return r[0] & 1;
}

FE32_INLINE void fe32_frombytes(fe32 &h, const unsigned char *s) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
        h.v[i] = (uint32_t)s[4 * i] | ((uint32_t)s[4 * i + 1] << 8)
               | ((uint32_t)s[4 * i + 2] << 16) | ((uint32_t)s[4 * i + 3] << 24);
}

FE32_INLINE void fe32_set_small(fe32 &h, uint32_t value) {
    h.v[0] = value;
#pragma unroll
    for (int i = 1; i < 8; ++i) h.v[i] = 0;
}

FE32_INLINE void fe32_sqn(fe32 &h, const fe32 &f, int n) {
    fe32_sq(h, f);
    for (int i = 1; i < n; ++i) fe32_sq(h, h);
}

// z^(p-2), same addition chain as ref10.
__device__ __noinline__ void fe32_invert(fe32 &out, const fe32 &z) {
    fe32 t0, t1, t2, t3;
    fe32_sq(t0, z);
    fe32_sqn(t1, t0, 2);
    fe32_mul(t1, z, t1);
    fe32_mul(t0, t0, t1);
    fe32_sq(t2, t0);
    fe32_mul(t1, t1, t2);
    fe32_sqn(t2, t1, 5);
    fe32_mul(t1, t2, t1);
    fe32_sqn(t2, t1, 10);
    fe32_mul(t2, t2, t1);
    fe32_sqn(t3, t2, 20);
    fe32_mul(t2, t3, t2);
    fe32_sqn(t2, t2, 10);
    fe32_mul(t1, t2, t1);
    fe32_sqn(t2, t1, 50);
    fe32_mul(t2, t2, t1);
    fe32_sqn(t3, t2, 100);
    fe32_mul(t2, t3, t2);
    fe32_sqn(t2, t2, 50);
    fe32_mul(t1, t2, t1);
    fe32_sqn(t1, t1, 5);
    fe32_mul(out, t1, t0);
}
