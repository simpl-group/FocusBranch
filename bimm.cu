#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define MAX_THREADS_PER_BLOCK 256

__global__ void bimm64_kernel(
    const uint64_t* __restrict__ A,
    const uint64_t* __restrict__ B,
    uint8_t* __restrict__ C,
    const int na, const int nb, const int nv
) {
    const int ia = blockIdx.y * blockDim.y + threadIdx.y;
    const int ib = blockIdx.x * blockDim.x + threadIdx.x;

    if (ia < na && ib < nb) {
        int sum = 0;
        const uint64_t* Ai = A + ia * nv;
        const uint64_t* Bi = B + ib * nv;

        for (int k = 0; k < nv; k += 1) {
            sum += __popcll(Ai[k] & Bi[k]);
        }
        C[ia * nb + ib] = sum;
    }
}

void bimm64_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv
) {
    dim3 threads(16, 16);
    dim3 blocks((nb + 15) / 16, (na + 15) / 16);

    bimm64_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint64_t*>(A.data_ptr()),
        reinterpret_cast<const uint64_t*>(B.data_ptr()),
        C.data_ptr<uint8_t>(),
        na, nb, nv
    );
}

__global__ void superset_popc_kernel(
    const uint64_t* __restrict__ A,
    const uint64_t* __restrict__ B,
    uint8_t* __restrict__ C,
    const int na, const int nb, const int nv, const int nb_div8
) {
    const int ia = blockIdx.y * blockDim.y + threadIdx.y;
    const int ib = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x % 8;
    const int flags_id = threadIdx.x / 8 + threadIdx.y * 2;
    __shared__ alignas(8) uint8_t block_flags[MAX_THREADS_PER_BLOCK];
    uint8_t* local_flags = &block_flags[flags_id * 8];
    local_flags[local_id] = 1;

    if (ia < na && ib < nb) {
        const uint64_t* Ai = A + ia * nv;
        const uint64_t* Bi = B + ib * nv;

        for (int k = 0; k < nv; k += 1) {
            if (~Ai[k] & Bi[k]) {
                local_flags[local_id] = 0;
                break; // Early exit if not a superset
            }
        }
    } else {
        local_flags[local_id] = 0;
    }
    __syncthreads();
    if (ia < na && ib < nb && local_id == 0) {
        const int global_id = ia * nb_div8 + ib / 8;
        C[global_id] = __popcll(*reinterpret_cast<uint64_t*>(local_flags));
    }
}
void superset_popc_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv, const int nb_div8
) {
    dim3 threads(16, 16);
    dim3 blocks((nb + 15) / 16, (na + 15) / 16);

    superset_popc_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint64_t*>(A.data_ptr()),
        reinterpret_cast<const uint64_t*>(B.data_ptr()),
        C.data_ptr<uint8_t>(),
        na, nb, nv, nb_div8
    );
}

__global__ void bitmask2comb_kernel(
    const uint64_t* __restrict__ bitmasks,
    u_int64_t* __restrict__ combs,
    const int num_rows, const int nv, const int popc
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        const uint64_t* bitmask = bitmasks + row * nv;
        int count = 0;
        for (int i = 0; i < nv; i++) {
            uint64_t bits = bitmask[i];
            // While bits is not zero, we have at least one set bit.
            while (bits) {
                // __ffsll returns 1-based position of least-significant set bit.
                combs[row * popc + count] = i * 64 + __ffsll(bits) - 1;
                count++;
                // Remove the lowest set bit. Smart.
                bits &= bits - 1;
            }
        }
    }
}
void bitmask2comb_cuda(
    at::Tensor bitmasks, at::Tensor combs,
    const int num_rows, const int nv, const int popc
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);

    bitmask2comb_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint64_t*>(bitmasks.data_ptr()),
        reinterpret_cast<uint64_t*>(combs.data_ptr()),
        num_rows, nv, popc
    );
}

__global__ void comb2bitmask_kernel(
    const uint64_t* __restrict__ combs,
    uint64_t* __restrict__ bitmasks,
    const int num_rows, const int nv, const int popc
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        uint64_t* bitmask = bitmasks + row * nv;
        const uint64_t* comb = combs + row * popc;
        for (int i = 0; i < popc; i++) {
            int idx = comb[i] / 64;
            uint64_t mask = 1ULL << (comb[i] % 64);
            bitmask[idx] |= mask;
        }
    }
}
void comb2bitmask_cuda(
    at::Tensor combs, at::Tensor bitmasks,
    const int num_rows, const int nv, const int popc
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);

    comb2bitmask_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint64_t*>(combs.data_ptr()),
        reinterpret_cast<uint64_t*>(bitmasks.data_ptr()),
        num_rows, nv, popc
    );
}
