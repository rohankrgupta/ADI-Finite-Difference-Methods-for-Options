// -----------------------------------------------------------------------------
// Partitioned Thomas solves of (I - c*A2) X = Y - c*A2X along v and of
// (I - c*A1) X = R along s.
//
// With the factors precomputed, Thomas is two first-order linear recurrences,
// forward d_k = (r_k - a_k d_{k-1}) * p_k and back x_k = d_k - c*_k x_{k+1}.
// A line is cut into chunks, one per thread. For each recurrence a thread runs
// its chunk from zero; a scan of the chunk maps y_out = Z + M*y_in then gives
// every chunk its true incoming value, and a fix-up adds that value times the
// chunk's gain (d_k = z_k + G_k d_in). The gains depend only on the matrix and
// are precomputed per factor set. Only the incoming values are computed
// differently from Thomas, so results agree with it to rounding.
//
// Along v (A2 is the same for every row): one block per row, the row in shared
// memory (read and written coalesced; the threads read it with an odd stride, free
// of bank conflicts), the factors chunk-interleaved so their loads are coalesced
// too, and the scan's slope products precomputed: 6 FP64 operations per node.
// Warps per row: the fewest that keep about kTargetWarps warps resident per SM
// (CUDA occupancy calculator), without chunks shorter than kMinChunk; a row too
// long for shared memory is solved with Thomas.
//
// Along s (A1 differs from column to column): a block holds a group of
// neighbouring columns and all their chunks; a warp reads `group` columns, so
// loads are coalesced in whole sectors. A chunk's values stay in registers
// (chunk length is a template parameter), and the scan runs in shuffles within a
// warp and through shared memory across warps. It moves the same bytes as Thomas
// plus a scan, so it is used only when the columns cannot give every SM a warp;
// otherwise Thomas (tridiag.cu) already streams near the bandwidth limit.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "kernels/kernels.cuh"

#include <algorithm>

#ifdef ADI_PARTITIONED

namespace adi {

namespace {

constexpr int kWarp = 32;
constexpr int kScanSteps = 5;  // log2(kWarp)
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kTargetWarps = 32;  // resident warps per SM that hide the shared-memory and FP64 latency
constexpr int kMinChunk = 4;      // shorter chunks spend more on the scan than they save

// Where the precomputed arrays keep their values (seq 0: the Dirichlet row, 1: every row >= 1).
__host__ __device__ inline std::size_t element_index(int seq, int t, int q, int L, int C) {
    return (static_cast<std::size_t>(seq) * L + t) * C + q;
}
__host__ __device__ inline int window_index(int seq, int s, int q, int C) { return (seq * kScanSteps + s) * C + q; }

std::size_t row_shared_bytes(int L, int parts) {
    return (static_cast<std::size_t>(kWarp) * parts * L + 2 * static_cast<std::size_t>(parts)) * sizeof(real);
}

struct ChunkOut {
    real *a, *pivot, *fwd_gain, *cstar, *back_gain, *fwd_window, *back_window, *fwd_prefix, *back_suffix;
};

// One block per sequence, one thread per chunk. The default build stores reciprocal pivots, so a forward
// step's gain is -a_j * p_j.
__global__ void build_v_chunks_kernel(OperatorView op, real c, const real* v_cstar, const real* v_pivot, int L,
                                      ChunkOut out) {
    extern __shared__ real slope[];  // [0, C): forward slope of each chunk; [C, 2C): back slope
    const int q = threadIdx.x;
    const int C = blockDim.x;
    const int seq = blockIdx.x;
    const int cols = op.cols;

    // Zero a at j = 0 and c* at j = cols-1: d_0 = (r_0 - 0*d)*p_0 = r_0*p_0 and x = d - 0*x = d exactly, so the
    // sweeps need no special first and last element. Elsewhere a = -c * lower as the Thomas solve forms it.
    real g = 1;  // an empty chunk is the identity map
    for (int t = 0; t < L; ++t) {
        const int j = q * L + t;
        const std::size_t k = element_index(seq, t, q, L, C);
        if (j >= cols) {
            out.a[k] = out.pivot[k] = out.fwd_gain[k] = out.cstar[k] = out.back_gain[k] = 0;
            continue;
        }
        const real aj = (j == 0) ? real(0) : -c * (seq == 0 ? real(0) : op.a2_lower[j]);
        const real pj = v_pivot[seq * cols + j];
        out.a[k] = aj;
        out.pivot[k] = pj;
        out.cstar[k] = (j == cols - 1) ? real(0) : v_cstar[seq * cols + j];
        g *= -aj * pj;
        out.fwd_gain[k] = g;  // d_j per unit of d before the chunk
    }
    slope[q] = g;
    real h = 1;
    for (int t = L - 1; t >= 0; --t) {
        const int j = q * L + t;
        if (j >= cols) continue;
        const std::size_t k = element_index(seq, t, q, L, C);
        h *= -out.cstar[k];
        out.back_gain[k] = h;  // x_j per unit of x after the chunk
    }
    slope[C + q] = h;
    __syncthreads();

    // Products of chunk slopes the scans use, all within the chunk's warp.
    const int lane = q % kWarp;
    const int first = q - lane, last = first + kWarp - 1;
    for (int s = 0; s < kScanSteps; ++s) {
        const int span = 1 << s;
        real wf = 0, wb = 0;  // a window that leaves the warp is never used
        if (lane >= span - 1) {
            wf = 1;
            for (int u = q - span + 1; u <= q; ++u) wf *= slope[u];
        }
        if (lane + span - 1 < kWarp) {
            wb = 1;
            for (int u = q; u < q + span; ++u) wb *= slope[C + u];
        }
        out.fwd_window[window_index(seq, s, q, C)] = wf;
        out.back_window[window_index(seq, s, q, C)] = wb;
    }
    real pf = 1, sb = 1;
    for (int u = first; u <= q; ++u) pf *= slope[u];
    for (int u = q; u <= last; ++u) sb *= slope[C + u];
    out.fwd_prefix[seq * C + q] = pf;
    out.back_suffix[seq * C + q] = sb;
}

__global__ void solve_v_partition_kernel(int cols, VChunks ch, real c, const real* __restrict__ Y,
                                         const real* __restrict__ A2X, real* __restrict__ X) {
    extern __shared__ real smem[];
    const int L = ch.L;
    const int C = kWarp * ch.parts;
    real* line = smem;                        // the row: C*L values
    real* fwd_total = smem + C * L;           // per warp: d at its last chunk's end, from zero into the warp
    real* back_total = fwd_total + ch.parts;  // per warp: x at its first chunk's start, from zero after it

    const int q = threadIdx.x;  // chunk
    const int lane = q % kWarp;
    const int warp = q / kWarp;
    const int i = blockIdx.x;
    const std::size_t row = static_cast<std::size_t>(i) * cols;
    const int seq = (i == 0) ? 0 : 1;

    // Right-hand side into shared memory, coalesced.
    for (int j = q; j < cols; j += C) line[j] = A2X ? ld(Y + row + j) - c * ld(A2X + row + j) : ld(Y + row + j);
    __syncthreads();

    const int j0 = q * L;
    const int n = max(0, min(L, cols - j0));  // elements in this chunk
    real* v = line + j0;
    const std::size_t e = element_index(seq, 0, q, L, C);  // element t of this chunk at e + t*C
    const real* a = ch.a + e;
    const real* p = ch.pivot + e;
    const real* gain_f = ch.fwd_gain + e;
    const real* cs = ch.cstar + e;
    const real* gain_b = ch.back_gain + e;

    // Forward sweep of the chunk from zero: z, kept in place of r.
    real z = 0;
#pragma unroll 4
    for (int t = 0; t < n; ++t) {
        z = (v[t] - ld(a + t * C) * z) * ld(p + t * C);
        v[t] = z;
    }
    // Scan: d at the end of chunk q - 1, the value the chunk starts from.
#pragma unroll
    for (int s = 0; s < kScanSteps; ++s) {
        const real before = __shfl_up_sync(kFullMask, z, 1 << s);
        if (lane >= (1 << s)) z += ld(ch.fwd_window + window_index(seq, s, q, C)) * before;
    }
    real d_in = __shfl_up_sync(kFullMask, z, 1);
    if (lane == 0) d_in = 0;
    if (ch.parts > 1) {  // add what comes into the warp, times the product of slopes up to chunk q - 1
        if (lane == kWarp - 1) fwd_total[warp] = z;
        __syncthreads();
        real into_warp = 0;
        for (int w = 0; w < warp; ++w) {
            into_warp = fwd_total[w] + ld(ch.fwd_prefix + seq * C + w * kWarp + kWarp - 1) * into_warp;
        }
        d_in += (lane == 0 ? real(1) : ld(ch.fwd_prefix + seq * C + q - 1)) * into_warp;
    }

    // Back sweep of the chunk from zero, on the corrected d = z + G*d_in.
    real x = 0;
#pragma unroll 4
    for (int t = n - 1; t >= 0; --t) {
        const real d = v[t] + ld(gain_f + t * C) * d_in;
        x = d - ld(cs + t * C) * x;
        v[t] = x;
    }
    // Scan from the last chunk down: x at the start of chunk q + 1.
#pragma unroll
    for (int s = 0; s < kScanSteps; ++s) {
        const real after = __shfl_down_sync(kFullMask, x, 1 << s);
        if (lane + (1 << s) < kWarp) x += ld(ch.back_window + window_index(seq, s, q, C)) * after;
    }
    real x_in = __shfl_down_sync(kFullMask, x, 1);
    if (lane == kWarp - 1) x_in = 0;
    if (ch.parts > 1) {
        if (lane == 0) back_total[warp] = x;
        __syncthreads();
        real into_warp = 0;
        for (int w = ch.parts - 1; w > warp; --w) {
            into_warp = back_total[w] + ld(ch.back_suffix + seq * C + w * kWarp) * into_warp;
        }
        x_in += (lane == kWarp - 1 ? real(1) : ld(ch.back_suffix + seq * C + q + 1)) * into_warp;
    }

    // Fix-up: x = z' + H*x_in.
#pragma unroll 4
    for (int t = 0; t < n; ++t) v[t] += ld(gain_b + t * C) * x_in;
    __syncthreads();

    for (int j = q; j < cols; j += C) X[row + j] = line[j];
}

}  // namespace

VChunkBuffers build_v_chunks(const OperatorView& op, real c, const real* v_cstar, const real* v_pivot,
                             cudaStream_t stream) {
    int device = 0, sms = 0, max_smem = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CUDA_CHECK(cudaFuncSetAttribute(solve_v_partition_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));

    // The fewest warps per row that reach kTargetWarps resident warps per SM, counting only the rows there are;
    // otherwise the most resident warps, without chunks shorter than kMinChunk.
    VChunkBuffers b;  // L = 0: the row does not fit, solve with Thomas
    int best_warps = 0;
    const int rows_per_sm = (op.rows + sms - 1) / sms;
    for (int parts = 1; parts <= kWarp; ++parts) {
        int L = (op.cols + kWarp * parts - 1) / (kWarp * parts);
        if (L % 2 == 0) ++L;  // odd: the threads' shared-memory accesses fall in distinct banks
        if (parts > 1 && L < kMinChunk) break;
        const std::size_t smem = row_shared_bytes(L, parts);
        if (smem > static_cast<std::size_t>(max_smem)) continue;
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, solve_v_partition_kernel, kWarp * parts,
                                                                 smem));
        const int warps = std::min(blocks, rows_per_sm) * parts;
        if (warps > best_warps) {
            best_warps = warps;
            b.L = L;
            b.parts = parts;
        }
        if (warps >= kTargetWarps) break;
    }
    if (b.L == 0) return b;

    const int C = kWarp * b.parts;
    const std::size_t n = 2 * static_cast<std::size_t>(b.L) * C;
    const std::size_t w = 2 * static_cast<std::size_t>(kScanSteps) * C;
    b.a = DeviceBuffer<real>(n);
    b.pivot = DeviceBuffer<real>(n);
    b.fwd_gain = DeviceBuffer<real>(n);
    b.cstar = DeviceBuffer<real>(n);
    b.back_gain = DeviceBuffer<real>(n);
    b.fwd_window = DeviceBuffer<real>(w);
    b.back_window = DeviceBuffer<real>(w);
    b.fwd_prefix = DeviceBuffer<real>(2 * static_cast<std::size_t>(C));
    b.back_suffix = DeviceBuffer<real>(2 * static_cast<std::size_t>(C));
    const ChunkOut out{b.a.data(),     b.pivot.data(),     b.fwd_gain.data(),    b.cstar.data(),
                       b.back_gain.data(), b.fwd_window.data(), b.back_window.data(), b.fwd_prefix.data(),
                       b.back_suffix.data()};
    build_v_chunks_kernel<<<2, C, 2 * C * sizeof(real), stream>>>(op, c, v_cstar, v_pivot, b.L, out);
    CUDA_CHECK_LAST();
    return b;
}

void launch_solve_v_partition(const OperatorView& op, const VChunks& ch, real c, const real* Y, const real* A2X,
                              real* X, cudaStream_t stream) {
    solve_v_partition_kernel<<<op.rows, kWarp * ch.parts, row_shared_bytes(ch.L, ch.parts), stream>>>(
        op.cols, ch, c, Y, A2X, X);
    CUDA_CHECK_LAST();
}

// ----------------------------------------------------------------------------- along s

namespace {

constexpr int kRowChunks[] = {16, 8, 4};         // rows per chunk the kernel is compiled for, longest first
constexpr int kColumnGroups[] = {32, 16, 8, 4};  // columns per block, widest first (4 doubles: one sector)

// Thread (column c, chunk g) of a warp: lane = g*group + c, so a warp reads `group` neighbouring columns of
// 32/group chunks, and the chunks of one column are `group` lanes apart.
struct SLayout {
    int group, per_warp, warps;  // columns per block, chunks per warp, warps per block
};

// Scan of the chunk maps y_out = Z + M*y_in down each column of the block (chunks in order within a warp,
// then warp after warp); returns the value entering this thread's chunk. The warp totals go through shared
// memory: warp w scans the totals of columns w, w + warps, ...
__device__ real scan_down(real Z, real M, int lane, int warp, int c, int g, SLayout s, real* tot_z, real* tot_m,
                          real* enter) {
    for (int off = s.group; off < kWarp; off <<= 1) {
        const real z = __shfl_up_sync(kFullMask, Z, off), m = __shfl_up_sync(kFullMask, M, off);
        if (lane >= off) {
            Z += M * z;
            M *= m;
        }
    }
    real ez = __shfl_up_sync(kFullMask, Z, s.group), em = __shfl_up_sync(kFullMask, M, s.group);
    if (g == 0) {
        ez = 0;
        em = 1;
    }
    if (s.warps == 1) return ez;
    if (g == s.per_warp - 1) {
        tot_z[warp * s.group + c] = Z;
        tot_m[warp * s.group + c] = M;
    }
    __syncthreads();
    for (int cc = warp; cc < s.group; cc += s.warps) {
        real z = lane < s.warps ? tot_z[lane * s.group + cc] : real(0);
        real m = lane < s.warps ? tot_m[lane * s.group + cc] : real(1);
        for (int off = 1; off < kWarp; off <<= 1) {
            const real zb = __shfl_up_sync(kFullMask, z, off), mb = __shfl_up_sync(kFullMask, m, off);
            if (lane >= off) {
                z += m * zb;
                m *= mb;
            }
        }
        const real before = __shfl_up_sync(kFullMask, z, 1);
        if (lane < s.warps) enter[lane * s.group + cc] = (lane == 0) ? real(0) : before;
    }
    __syncthreads();
    return ez + em * enter[warp * s.group + c];
}

// The same up each column: the value entering this thread's chunk from the chunk after it.
__device__ real scan_up(real Z, real M, int lane, int warp, int c, int g, SLayout s, real* tot_z, real* tot_m,
                        real* enter) {
    for (int off = s.group; off < kWarp; off <<= 1) {
        const real z = __shfl_down_sync(kFullMask, Z, off), m = __shfl_down_sync(kFullMask, M, off);
        if (lane + off < kWarp) {
            Z += M * z;
            M *= m;
        }
    }
    real ez = __shfl_down_sync(kFullMask, Z, s.group), em = __shfl_down_sync(kFullMask, M, s.group);
    if (g == s.per_warp - 1) {
        ez = 0;
        em = 1;
    }
    if (s.warps == 1) return ez;
    if (g == 0) {
        tot_z[warp * s.group + c] = Z;
        tot_m[warp * s.group + c] = M;
    }
    __syncthreads();
    for (int cc = warp; cc < s.group; cc += s.warps) {
        real z = lane < s.warps ? tot_z[lane * s.group + cc] : real(0);
        real m = lane < s.warps ? tot_m[lane * s.group + cc] : real(1);
        for (int off = 1; off < kWarp; off <<= 1) {
            const real za = __shfl_down_sync(kFullMask, z, off), ma = __shfl_down_sync(kFullMask, m, off);
            if (lane + off < kWarp) {
                z += m * za;
                m *= ma;
            }
        }
        const real after = __shfl_down_sync(kFullMask, z, 1);  // lane 31 gets its own z: not used
        if (lane < s.warps) enter[lane * s.group + cc] = after;
    }
    __syncthreads();
    return (warp == s.warps - 1) ? ez : ez + em * enter[warp * s.group + c];  // nothing after the last warp
}

// (I - c*A1) X = R: a block solves `group` columns, one thread per chunk of L rows. Only the chunk's z stays
// in registers between the phases; everything else is loaded when it is used, coalesced across the columns.
template <int L>
__global__ void solve_s_partition_kernel(int rows, int cols, SChunks ch, const real* __restrict__ R,
                                         real* __restrict__ X) {
    extern __shared__ real smem[];
    const SLayout s{ch.group, kWarp / ch.group, ch.warps};
    real* tot_z = smem;
    real* tot_m = smem + s.warps * s.group;
    real* enter = tot_m + s.warps * s.group;

    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    const int c = lane % s.group, g = lane / s.group;
    const int j = blockIdx.x * s.group + c;
    const int i0 = (warp * s.per_warp + g) * L;
    const int n = (j < cols) ? max(0, min(L, rows - i0)) : 0;  // rows of this chunk
    const std::size_t base = static_cast<std::size_t>(i0) * cols + j;
    const auto at = [&](int u) { return base + static_cast<std::size_t>(u) * cols; };

    // Forward sweep of the chunk from zero (a row past the end leaves z at 0, which nothing reads).
    real z[L];
    {
        real r[L], a[L], p[L];
#pragma unroll
        for (int u = 0; u < L; ++u) {
            const bool ok = u < n;
            r[u] = ok ? ld(R + at(u)) : real(0);
            a[u] = ok ? ld(ch.a + at(u)) : real(0);
            p[u] = ok ? ld(ch.pivot + at(u)) : real(0);
        }
        real acc = 0;
#pragma unroll
        for (int u = 0; u < L; ++u) {
            acc = (r[u] - a[u] * acc) * p[u];
            z[u] = acc;
        }
    }
    const real slope_down = (n == L) ? ld(ch.fwd_gain + at(L - 1)) : real(0);  // a short chunk is the last
    const real d_in = scan_down(z[L - 1], slope_down, lane, warp, c, g, s, tot_z, tot_m, enter);

    // Fix-up d = z + G*d_in, and the back sweep of the chunk from zero.
    {
        real gain[L], cs[L];
#pragma unroll
        for (int u = 0; u < L; ++u) {
            const bool ok = u < n;
            gain[u] = ok ? ld(ch.fwd_gain + at(u)) : real(0);
            cs[u] = ok ? ld(ch.cstar + at(u)) : real(0);
        }
        real x = 0;
#pragma unroll
        for (int u = L - 1; u >= 0; --u) {
            x = (z[u] + gain[u] * d_in) - cs[u] * x;
            z[u] = x;
        }
    }
    const real slope_up = (n > 0) ? ld(ch.back_gain + at(0)) : real(0);
    const real x_in = scan_up(z[0], slope_up, lane, warp, c, g, s, tot_z, tot_m, enter);

    // Fix-up x = z' + H*x_in, stored once.
#pragma unroll
    for (int u = 0; u < L; ++u) {
        if (u < n) X[at(u)] = z[u] + ld(ch.back_gain + at(u)) * x_in;
    }
}

// One thread per (chunk, column): -c * lower, and the running products of the chunk's step slopes (with
// reciprocal pivots, a forward step's slope is -a_i * p_i).
__global__ void build_s_chunks_kernel(OperatorView op, real c, const real* s_cstar, const real* s_pivot, int L,
                                      real* a, real* fwd_gain, real* back_gain) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= op.cols) return;
    const int i0 = blockIdx.y * L;
    const int n = min(L, op.rows - i0);
    real g = 1;
    for (int u = 0; u < n; ++u) {
        const std::size_t k = static_cast<std::size_t>(i0 + u) * op.cols + j;
        const real aj = -c * op.a1_lower[k];  // as the Thomas solve forms it
        a[k] = aj;
        g *= -aj * s_pivot[k];
        fwd_gain[k] = g;
    }
    real h = 1;
    for (int u = n - 1; u >= 0; --u) {
        const std::size_t k = static_cast<std::size_t>(i0 + u) * op.cols + j;
        h *= -s_cstar[k];
        back_gain[k] = h;
    }
}

std::size_t s_shared_bytes(const SLayout& s) { return 3 * static_cast<std::size_t>(s.warps) * s.group * sizeof(real); }

template <int L>
int s_blocks_per_sm(const SLayout& s) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, solve_s_partition_kernel<L>, kWarp * s.warps,
                                                             s_shared_bytes(s)));
    return blocks;
}

int s_blocks_per_sm(int L, const SLayout& s) {
    switch (L) {
        case 16: return s_blocks_per_sm<16>(s);
        case 8: return s_blocks_per_sm<8>(s);
        default: return s_blocks_per_sm<4>(s);
    }
}

}  // namespace

SChunkBuffers build_s_chunks(const OperatorView& op, real c, const real* s_cstar, const real* s_pivot,
                             cudaStream_t stream) {
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

    // Thomas (one thread per column, loads issued ahead) streams close to the bandwidth limit once every SM has
    // a warp of columns, and the partitioned solve moves the same bytes plus a scan: partition only when the
    // columns cannot fill the SMs.
    SChunkBuffers b;  // L = 0: solve with Thomas
    if ((op.cols + kWarp - 1) / kWarp >= sms) return b;

    // A block holds whole columns, so all chunks of a column must fit in one block. Of the layouts that fit,
    // take the one that spreads over the most SMs (FP64 units, load/store units and L1 are per SM), then the
    // one with the most resident warps per SM up to kTargetWarps; ties go to longer chunks (less scan work)
    // and wider column groups (fewer memory transactions).
    int best_sms = 0, best_warps = 0;
    for (const int L : kRowChunks) {
        const int chunks = (op.rows + L - 1) / L;
        for (const int group : kColumnGroups) {
            const int per_warp = kWarp / group;
            const SLayout s{group, per_warp, (chunks + per_warp - 1) / per_warp};
            if (s.warps > kWarp) continue;  // more than 1024 threads
            const int fit = s_blocks_per_sm(L, s);
            if (fit == 0) continue;
            const int blocks = (op.cols + group - 1) / group;
            const int used = std::min(blocks, sms);
            const int warps = std::min(std::min(fit, (blocks + sms - 1) / sms) * s.warps, kTargetWarps);
            if (used > best_sms || (used == best_sms && warps > best_warps)) {
                best_sms = used;
                best_warps = warps;
                b.L = L;
                b.group = group;
                b.warps = s.warps;
            }
        }
    }
    if (b.L == 0) return b;

    const std::size_t n = static_cast<std::size_t>(op.rows) * op.cols;
    b.a = DeviceBuffer<real>(n);
    b.fwd_gain = DeviceBuffer<real>(n);
    b.back_gain = DeviceBuffer<real>(n);
    const dim3 grid((op.cols + 127) / 128, (op.rows + b.L - 1) / b.L);
    build_s_chunks_kernel<<<grid, 128, 0, stream>>>(op, c, s_cstar, s_pivot, b.L, b.a.data(), b.fwd_gain.data(),
                                                   b.back_gain.data());
    CUDA_CHECK_LAST();
    return b;
}

void launch_solve_s_partition(const OperatorView& op, const SChunks& ch, const real* R, real* X,
                              cudaStream_t stream) {
    const SLayout s{ch.group, kWarp / ch.group, ch.warps};
    const int blocks = (op.cols + ch.group - 1) / ch.group;
    const std::size_t smem = s_shared_bytes(s);
    switch (ch.L) {
        case 16:
            solve_s_partition_kernel<16><<<blocks, kWarp * s.warps, smem, stream>>>(op.rows, op.cols, ch, R, X);
            break;
        case 8:
            solve_s_partition_kernel<8><<<blocks, kWarp * s.warps, smem, stream>>>(op.rows, op.cols, ch, R, X);
            break;
        default:
            solve_s_partition_kernel<4><<<blocks, kWarp * s.warps, smem, stream>>>(op.rows, op.cols, ch, R, X);
            break;
    }
    CUDA_CHECK_LAST();
}

}  // namespace adi

#endif  // ADI_PARTITIONED
