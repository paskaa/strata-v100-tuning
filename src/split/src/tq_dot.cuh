// src/split/src/tq_dot.cuh - Gyro-L pair-trellis (TQP5/6/7, ggml ids 147-149) decode dots.
//
// The split engine's decode path (job_rows / k_moe_down via vdot<type>) needs, per type:
// qk/qi traits, a vdr, and a vec-dot.  This header provides them for the three pair-trellis
// types, following the fork's own recipe (third_party/ggml/ggml-common-tq.h block structs,
// src/kernels/cuda/tq_lut_i8.cuh int8 codebook, the state walk of tqp_states8_cheap and the
// LUT hash of tq_load_tqp in src/kernels/cuda/iq_kernels.cu):
//   state t of step t = 16 stream bits from bit (63-t)*K, LSB first, circular over 8*K bytes;
//   pair index x = (s & 0x7FFF) * 0x9E3779B1u; weights = lut[x>>21], lut[(x>>10)&2047] as int8;
//   value ~= TQ_LUT_I8_SCALE * int8 * block_scale.
//
// vdot convention (matches tq_apply there): 8 calls per 128-value block, iqs = 0,4,...,28;
// call p covers weights [16p,16p+16) and reads 4 int32 of y (16 int8).
#pragma once

#include "../../../third_party/ggml/ggml-common-tq.h"   // block_tqp5/6/7 (agention fork layout)
#include "../../kernels/cuda/tq_lut_i8.cuh"                 // int8 codebook + TQ_LUT_I8_SCALE

// ggml traits the split decode path reads (moe.cuh job_rows / unit_dot / moe_g):
// qk = values per block, qi/vdr = lanes sharing a block (8), vdr = 4 (iqs stride).
// The primary template lives in frozen ggml common.cuh at GLOBAL scope, so these
// specializations must too (explicit specializations belong to the primary's namespace).
template <> struct ggml_cuda_type_traits<(ggml_type)147> {
    static constexpr int qk = 128, qr = 1, qi = 32;
};
template <> struct ggml_cuda_type_traits<(ggml_type)148> {
    static constexpr int qk = 128, qr = 1, qi = 32;
};
template <> struct ggml_cuda_type_traits<(ggml_type)149> {
    static constexpr int qk = 128, qr = 1, qi = 32;
};

namespace fnx {
namespace tq {

// K bits and block bytes per pair-trellis type (ggml-common-tq.h static_asserts the sizes)
template <ggml_type type> constexpr int tqp_k();
template <> constexpr int tqp_k<(ggml_type)147>() { return 5; }
template <> constexpr int tqp_k<(ggml_type)148>() { return 6; }
template <> constexpr int tqp_k<(ggml_type)149>() { return 7; }
template <ggml_type type> constexpr size_t tqp_block_bytes();
template <> constexpr size_t tqp_block_bytes<(ggml_type)147>() { return sizeof(block_tqp5); }
template <> constexpr size_t tqp_block_bytes<(ggml_type)148>() { return sizeof(block_tqp6); }
template <> constexpr size_t tqp_block_bytes<(ggml_type)149>() { return sizeof(block_tqp7); }
template <ggml_type type> constexpr bool is_tqp() {
    return type == (ggml_type)147 || type == (ggml_type)148 || type == (ggml_type)149;
}

// 8 trellis states for one lane (16 weights): steps [8*lane, 8*lane+8)
template <int K>
__device__ __forceinline__ void tqp_states8(const uint8_t* __restrict__ qs, int lane, uint32_t s[8]) {
    constexpr int NB = 8 * K;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t off = (uint32_t)(63 - (8 * lane + i)) * (uint32_t)K;
        const uint32_t i0 = off >> 3;
        const uint32_t i1 = i0 + 1 < (uint32_t)NB ? i0 + 1 : 0;
        const uint32_t i2 = i1 + 1 < (uint32_t)NB ? i1 + 1 : 0;
        const uint32_t w = (uint32_t)qs[i0] | ((uint32_t)qs[i1] << 8) | ((uint32_t)qs[i2] << 16);
        s[i] = (w >> (off & 7)) & 0xFFFFu;
    }
}

struct TqLane {
    int q[4];   // 16 int8 weights (8 pairs) as 4 dp4a words
    float d;    // block scale
};

// one lane (16 weights) of a TQP block: states -> LUT pairs -> dp4a words
template <int K>
__device__ __forceinline__ TqLane tqp_load_lane(const uint8_t* __restrict__ qs, uint16_t dh, int lane) {
    uint32_t s[8];
    tqp_states8<K>(qs, lane, s);
    TqLane r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {   // two steps (2+2 weights) per int
        const uint32_t x0 = (s[2 * i] & 0x7FFFu) * 0x9E3779B1u;
        const uint32_t x1 = (s[2 * i + 1] & 0x7FFFu) * 0x9E3779B1u;
        r.q[i] = (int)((uint32_t)tq_lut_i8[x0 >> 21] | ((uint32_t)tq_lut_i8[x1 >> 21] << 16));
        (void)x1;
    }
    r.d = __half2float(__ushort_as_half(dh));
    return r;
}

}  // namespace tq

namespace tq {

// gate|up dot: one vdot call = 16 weights (lane = iqs/4) against 4 int32 of y.
// y points at the call's 4 q8_1 blocks already (kby = kbx*4), as tq_apply does.
template <ggml_type type>
__device__ __forceinline__ float vec_dot_tqp(const void* __restrict__ vb, const block_q8_1* __restrict__ y,
                                             int kbx, int iqs) {
    constexpr int K = tqp_k<type>();
    const uint8_t* qs;
    uint16_t dh;
    if constexpr (type == (ggml_type)147) {
        const auto* b = (const block_tqp5*)vb + kbx;
        qs = b->qs;
        dh = b->d;
    } else if constexpr (type == (ggml_type)148) {
        const auto* b = (const block_tqp6*)vb + kbx;
        qs = b->qs;
        dh = b->d;
    } else {
        const auto* b = (const block_tqp7*)vb + kbx;
        qs = b->qs;
        dh = b->d;
    }
    const int lane = iqs / 4;
    TqLane r = tqp_load_lane<K>(qs, dh, lane);
    const block_q8_1* b8 = y + lane / 2;
    const int* q8 = (const int*)b8->qs + 4 * (lane & 1);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) sumi = ggml_cuda_dp4a(r.q[i], q8[i], sumi);
    return r.d * (__low2float(b8->ds) * TQ_LUT_I8_SCALE) * (float)sumi;
}

// down dot: 32-wide unit u of a row against ONE q8_1 block (down's hq[j][u]).
// The unit is quarter q = u%4 of block blk = u/4: parts 2q,2q+1 (32 values); y ints [0,8).
template <ggml_type type>
__device__ __forceinline__ float down_dot_tqp(const char* __restrict__ wrow, int blk, int q,
                                              const block_q8_1* __restrict__ y) {
    constexpr int K = tqp_k<type>();
    constexpr size_t BB = tqp_block_bytes<type>();
    const uint8_t* qs;
    uint16_t dh;
    if constexpr (type == (ggml_type)147) {
        const auto* b = (const block_tqp5*)(wrow + (size_t)blk * BB);
        qs = b->qs;
        dh = b->d;
    } else if constexpr (type == (ggml_type)148) {
        const auto* b = (const block_tqp6*)(wrow + (size_t)blk * BB);
        qs = b->qs;
        dh = b->d;
    } else {
        const auto* b = (const block_tqp7*)(wrow + (size_t)blk * BB);
        qs = b->qs;
        dh = b->d;
    }
    const float yd = __low2float(y->ds);
    const int* yq = (const int*)y->qs;
    float s = 0.0f;
    for (int k = 0; k < 2; ++k) {
        TqLane r = tqp_load_lane<K>(qs, dh, 2 * q + k);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) sumi = ggml_cuda_dp4a(r.q[i], yq[4 * k + i], sumi);
        s += r.d * yd * TQ_LUT_I8_SCALE * (float)sumi;
    }
    return s;
}

}  // namespace tq
}  // namespace fnx
