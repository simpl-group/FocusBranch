#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define MAX_THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 8

template <typename scalar_t>
__global__ void wipe_selected_kernel(
    scalar_t* __restrict__ W,
    const long* __restrict__ selected_idx,
    const int batch_size,
    const int num_rows,
    const int num_cols,
    const int rows_per_thread,
    const int num_selected
) {
    const int b = blockIdx.x;
    const long* sib = selected_idx + b * num_selected;
    scalar_t* Wb = W + b * num_rows * num_cols;
    int r = threadIdx.x * rows_per_thread;
    for (int i = 0; i < rows_per_thread && r < num_rows; i++, r++) {
        scalar_t* Wb_r = Wb + r * num_cols;
        for (int j = 0; j < num_selected; j++) {
            Wb_r[sib[j]] = static_cast<scalar_t>(0);
        }
    }
}
void wipe_selected_cuda(
    at::Tensor& WX_diff,
    const at::Tensor& selected_idx,
    const int batch_size,
    const int num_rows,
    const int num_cols,
    const int num_selected
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks(batch_size); // each block handles one batch element
    const int rows_per_thread = (num_rows + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    AT_DISPATCH_ALL_TYPES(
        WX_diff.scalar_type(),
        "wipe_selected_cuda",
        ([&] {
            wipe_selected_kernel<scalar_t><<<blocks, threads>>>(
                WX_diff.data_ptr<scalar_t>(),
                selected_idx.data_ptr<long>(),
                batch_size,
                num_rows,
                num_cols,
                rows_per_thread,
                num_selected
            );
        })
    );
}

template<typename scalar_t, int K> // k > 0
__global__ void selected_top_k_sum_warped_kernel(
    scalar_t* __restrict__ x,      // [num_rows, nv] assume all(x>0)
    const long* __restrict__ selected_idx,  // [batch,num_selected]
    scalar_t* __restrict__ out,    // [num_rows]
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    const int row = (blockDim.x * blockIdx.x + threadIdx.x) >> 5;
    if (row >= num_rows) return;

    const int lane      = threadIdx.x & 31;
    const unsigned mask = 0xffffffff; // full‑warp mask
    scalar_t* x_row     = x + size_t(row) * nv;
    scalar_t* o         = out + row;
    const long* S       = selected_idx + int(row / rows_per_batch) * num_selected;

    scalar_t sum = static_cast<scalar_t>(0);
    if (lane < num_selected) {
        const long col = S[lane];// in L2 → register
        sum            = x_row[col];
        x_row[col]     = static_cast<scalar_t>(0); // mark as used
    }
    #pragma unroll
    for (int offset = (num_selected+1)>>1; offset; offset >>= 1)
        sum += __shfl_down_sync(mask, sum, offset);

    scalar_t local[K] = {static_cast<scalar_t>(0)};
    #pragma unroll
    for (int col = lane; col < nv; col += 32) {
        scalar_t v = x_row[col];
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scalar_t temp = local[i];
            bool swap = v > temp;
            local[i] = swap ? v : temp;
            v        = swap ? temp : v;
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        scalar_t peer[K];
        #pragma unroll
        for (int i = 0; i < K; ++i)
            peer[i] = __shfl_down_sync(mask, local[i], offset);
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scalar_t v = peer[i];
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                scalar_t temp = local[j];
                bool swap   = v > temp;
                local[j]    = swap ? v : temp;
                v           = swap ? temp : v;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) sum += local[i];

    if (lane == 0) *o = sum;
}
template<int K>
void selected_top_k_sum_warped_cuda(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    AT_DISPATCH_ALL_TYPES(
        x.scalar_type(),
        "selected_top_k_sum_warped_cuda",
        ([&] {
            selected_top_k_sum_warped_kernel<scalar_t, K><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                selected_idx.data_ptr<long>(),
                out.data_ptr<scalar_t>(),
                num_rows,
                nv,
                rows_per_batch,
                num_selected
            );
        })
    );
}

template<typename scalar_t, int K> //k > 0
__global__ void selected_top_k_sum_single_kernel(
    scalar_t* __restrict__ x,      // [num_rows, nv] assume all(x>0)
    const long* __restrict__ selected_idx,  // [batch,num_selected]
    scalar_t* __restrict__ out,    // [num_rows]
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    scalar_t* x_row     = x + size_t(row) * nv;
    scalar_t* o         = out + row;
    const long* S       = selected_idx + int(row / rows_per_batch) * num_selected;

    scalar_t sum = static_cast<scalar_t>(0);
    #pragma unroll
    for (int i = 0; i < num_selected; ++i) {
        scalar_t* v = x_row + S[i];
        sum += *v;
        *v = static_cast<scalar_t>(0); // mark as used
    }
    scalar_t buf[K] = {static_cast<scalar_t>(0)};
    #pragma unroll
    for (int col = 0; col < nv; ++col) {
        scalar_t v = x_row[col];
        // if (v <= buf[K-1]) continue; // early skip
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scalar_t temp = buf[i];
            bool swap = v > temp;
            buf[i] = swap ? v : temp;
            v      = swap ? temp : v;
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) sum += buf[i];

    *o = sum;
}
template<int K>
void selected_top_k_sum_single_cuda(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(),
        "selected_top_k_sum_single_cuda",
        ([&] {
            selected_top_k_sum_single_kernel<scalar_t, K><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                selected_idx.data_ptr<long>(),
                out.data_ptr<scalar_t>(),
                num_rows,
                nv,
                rows_per_batch,
                num_selected
            );
        })
    );
}

template<typename scalar_t, int K> // k > 0
__global__ void selected_bottom_k_sum_warped_kernel(
    scalar_t* __restrict__ x,      // [num_rows, nv] assume all(x < 0)
    const long* __restrict__ selected_idx,  // [batch,num_selected]
    scalar_t* __restrict__ out,    // [num_rows]
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    const int row = (blockDim.x * blockIdx.x + threadIdx.x) >> 5;
    if (row >= num_rows) return;

    const int lane      = threadIdx.x & 31;
    const unsigned mask = 0xffffffff; // full‑warp mask
    scalar_t* x_row     = x + size_t(row) * nv;
    scalar_t* o         = out + row;
    const long* S       = selected_idx + int(row / rows_per_batch) * num_selected;

    scalar_t sum = static_cast<scalar_t>(0);
    if (lane < num_selected) {
        const long col = S[lane];// in L2 → register
        sum            = x_row[col];
        x_row[col]     = static_cast<scalar_t>(0); // mark as used
    }
    #pragma unroll
    for (int offset = (num_selected+1)>>1; offset; offset >>= 1)
        sum += __shfl_down_sync(mask, sum, offset);

    scalar_t local[K] = {static_cast<scalar_t>(0)};
    #pragma unroll
    for (int col = lane; col < nv; col += 32) {
        scalar_t v = x_row[col];
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scalar_t temp = local[i];
            bool swap = v < temp;
            local[i]  = swap ? v : temp;
            v         = swap ? temp : v;
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        scalar_t peer[K];
        #pragma unroll
        for (int i = 0; i < K; ++i)
            peer[i] = __shfl_down_sync(mask, local[i], offset);
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scalar_t v = peer[i];
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                scalar_t temp = local[j];
                bool swap   = v < temp;
                local[j]    = swap ? v : temp;
                v           = swap ? temp : v;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) sum += local[i];

    if (lane == 0) *o = sum;
}
template<int K>
void selected_bottom_k_sum_warped_cuda(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    AT_DISPATCH_ALL_TYPES(
        x.scalar_type(),
        "selected_bottom_k_sum_warped_cuda",
        ([&] {
            selected_bottom_k_sum_warped_kernel<scalar_t, K><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                selected_idx.data_ptr<long>(),
                out.data_ptr<scalar_t>(),
                num_rows,
                nv,
                rows_per_batch,
                num_selected
            );
        })
    );
}

template<typename scalar_t, int K> //k > 0
__global__ void selected_bottom_k_sum_single_kernel(
    scalar_t* __restrict__ x,      // [num_rows, nv] assume all(x<0)
    const long* __restrict__ selected_idx,  // [batch,num_selected]
    scalar_t* __restrict__ out,    // [num_rows]
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    scalar_t* x_row     = x + size_t(row) * nv;
    scalar_t* o         = out + row;
    const long* S       = selected_idx + int(row / rows_per_batch) * num_selected;

    scalar_t sum = static_cast<scalar_t>(0);
    #pragma unroll
    for (int i = 0; i < num_selected; ++i) {
        scalar_t* v = x_row + S[i];
        sum += *v;
        *v = static_cast<scalar_t>(0); // mark as used
    }
    scalar_t buf[K] = {static_cast<scalar_t>(0)};
    #pragma unroll
    for (int col = 0; col < nv; ++col) {
        scalar_t v = x_row[col];
        // if (v >= buf[K-1]) continue; // early skip
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scalar_t temp = buf[i];
            bool swap = v < temp;
            buf[i] = swap ? v : temp;
            v      = swap ? temp : v;
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) sum += buf[i];

    *o = sum;
}
template<int K>
void selected_bottom_k_sum_single_cuda(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(),
        "selected_bottom_k_sum_single_cuda",
        ([&] {
            selected_bottom_k_sum_single_kernel<scalar_t, K><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                selected_idx.data_ptr<long>(),
                out.data_ptr<scalar_t>(),
                num_rows,
                nv,
                rows_per_batch,
                num_selected
            );
        })
    );
}

//k=0
template <typename scalar_t>
__global__ void selected_sum_kernel(
    const scalar_t* __restrict__ x,      // [num_rows, nv]
    const long*     __restrict__ selected_idx,  // [batch,num_selected]
    scalar_t*       __restrict__ out,    // [num_rows]
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows)  return;

    const scalar_t* x_row = x + size_t(row) * nv;
    scalar_t* o = out + row;

    const int batch_idx = row / rows_per_batch;
    const long* S = selected_idx + batch_idx * num_selected;

    scalar_t s_sum = static_cast<scalar_t>(0);
    #pragma unroll
    for (int i = 0; i < num_selected; ++i)  s_sum += x_row[S[i]];

    *o = s_sum;
}
void selected_sum_cuda(
    const at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks((num_rows + threads.x - 1) / threads.x);

    AT_DISPATCH_ALL_TYPES(
        x.scalar_type(),
        "selected_sum_cuda",
        ([&] {
            selected_sum_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                selected_idx.data_ptr<long>(),
                out.data_ptr<scalar_t>(),
                num_rows,
                nv,
                rows_per_batch,
                num_selected
            );
        })
    );
}

template void selected_top_k_sum_warped_cuda<1>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<2>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<3>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<4>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<5>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<6>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<7>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<8>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<9>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_warped_cuda<10>(at::Tensor&,const at::Tensor&, at::Tensor&, const int, const int, const int, const int);

template void selected_top_k_sum_single_cuda<1>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<2>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<3>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<4>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<5>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<6>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<7>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<8>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<9>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_top_k_sum_single_cuda<10>(at::Tensor&,const at::Tensor&, at::Tensor&, const int, const int, const int, const int);

template void selected_bottom_k_sum_warped_cuda<1>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<2>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<3>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<4>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<5>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<6>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<7>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<8>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<9>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_warped_cuda<10>(at::Tensor&,const at::Tensor&, at::Tensor&, const int, const int, const int, const int);

template void selected_bottom_k_sum_single_cuda<1>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<2>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<3>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<4>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<5>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<6>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<7>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<8>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<9>(at::Tensor&, const at::Tensor&, at::Tensor&, const int, const int, const int, const int);
template void selected_bottom_k_sum_single_cuda<10>(at::Tensor&,const at::Tensor&, at::Tensor&, const int, const int, const int, const int);