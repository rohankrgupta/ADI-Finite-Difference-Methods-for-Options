// -----------------------------------------------------------------------------
// K1, the explicit stage shared by every ADI scheme, and the probe kernel
// used by the single-kernel fixture checks.
//
// Thread mapping: one thread per node, 32x8 blocks with x along j (contiguous
// in memory), so a warp reads 32 neighbouring values of each array. Stencil
// neighbours are read through the read-only cache; no shared memory in v1.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "kernels/kernels.cuh"

namespace adi {

namespace {

constexpr int kBlockX = 32;  // along j
constexpr int kBlockY = 8;   // along i

dim3 node_grid(const OperatorView& op) {
    return dim3((op.cols + kBlockX - 1) / kBlockX, (op.rows + kBlockY - 1) / kBlockY);
}

// Python: _predict (Y0 = U + dtau*AU, then += dtau*lam) and the first line of _implicit_corrections.
__global__ void explicit_stage_kernel(OperatorView op, ExplicitStageArgs a) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= op.rows || j >= op.cols) return;
    const int k = i * op.cols + j;

    const real a0u = apply_A0_at(op, a.U, i, j);
    const real a1u = apply_A1_at(op, a.U, i, j);
    const real a2u = apply_A2_at(op, a.U, i, j);
    const real au = (a0u + a1u) + a2u;
    real y0 = ld(a.U + k) + a.dtau * au;
    if (a.lam) y0 += a.dtau * ld(a.lam + k);

    a.R1[k] = (i == 0) ? a.boundary : y0 - a.c * a1u;
    a.A2U[k] = a2u;
    if (a.A1U) a.A1U[k] = a1u;
    if (a.Y0) a.Y0[k] = y0;
    if (a.A0U) a.A0U[k] = a0u;
    if (a.AU) a.AU[k] = au;
}

__global__ void apply_probe_kernel(OperatorView op, const real* X, real* A0X, real* A1X, real* A2X) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= op.rows || j >= op.cols) return;
    const int k = i * op.cols + j;
    A0X[k] = apply_A0_at(op, X, i, j);
    A1X[k] = apply_A1_at(op, X, i, j);
    A2X[k] = apply_A2_at(op, X, i, j);
}

}  // namespace

void launch_explicit_stage(const OperatorView& op, const ExplicitStageArgs& args, cudaStream_t stream) {
    explicit_stage_kernel<<<node_grid(op), dim3(kBlockX, kBlockY), 0, stream>>>(op, args);
    CUDA_CHECK_LAST();
}

void launch_apply_probe(const OperatorView& op, const real* X, real* A0X, real* A1X, real* A2X,
                        cudaStream_t stream) {
    apply_probe_kernel<<<node_grid(op), dim3(kBlockX, kBlockY), 0, stream>>>(op, X, A0X, A1X, A2X);
    CUDA_CHECK_LAST();
}

}  // namespace adi
