// -----------------------------------------------------------------------------
// Partitioned solve of (I - c*A2) X = Y - c*A2X along v: one block per grid row.
//
// With the factors precomputed, Thomas is two first-order linear recurrences,
// forward d_j = (r_j - a_j d_{j-1}) * p_j and back x_j = d_j - c*_j x_{j+1}.
// The row is cut into chunks of L consecutive j, one per thread. For each
// recurrence a thread runs its chunk from zero; a scan over the chunks then
// gives every chunk its true incoming value, and a fix-up adds that value times
// the chunk's gain (d_k = z_k + G_k d_in). Everything that depends only on the
// matrix (the gains, the chunk slopes and the scan's window products) is
// precomputed per factor set, so a solve costs 6 FP64 operations per node.
//
// The scan is a Kogge-Stone scan over the 32 chunks of a warp (shuffles), then a
// short sequential pass over the warps of the row (shared memory). The row lives
// in shared memory: read and written coalesced, read by the threads with a
// stride of L (odd, so free of bank conflicts). The factors are the same for
// every row >= 1 and are stored chunk-interleaved, so their loads are coalesced.
//
// Warps per row: the fewest that keep about kTargetWarps warps resident per SM
// (CUDA occupancy calculator; shared memory usually limits the rows per SM),
// without chunks shorter than kMinChunk. A row too long for shared memory is
// solved with Thomas instead (tridiag.cu).
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "kernels/kernels.cuh"

#include <algorithm>

#ifdef ADI_PARTITION_V

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

}  // namespace adi

#endif  // ADI_PARTITION_V
