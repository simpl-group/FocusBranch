#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#include <map>

// key: {num_selected, k}
std::map<std::pair<int, int>, int> min_nv_to_warp = {
#include "min_nv_to_warp_data.txt"
};

void wipe_selected_cuda(
    at::Tensor& WX_diff,
    const at::Tensor& selected_idx,
    const int batch_size,
    const int num_rows,
    const int num_cols,
    const int num_selected
);
at::Tensor& wipe_selected_laucher(at::Tensor& WX_diff, const at::Tensor& selected_idx) {
    CHECK_CUDA(WX_diff);
    CHECK_CUDA(selected_idx);

    const int batch_size   = WX_diff.size(0);
    const int num_rows     = WX_diff.size(1);
    const int num_cols     = WX_diff.size(2);
    const int num_selected = selected_idx.size(1);

    wipe_selected_cuda(WX_diff, selected_idx, batch_size, num_rows, num_cols, num_selected);

    return WX_diff;
}

void selected_sum_cuda(
    const at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
);
template<int K>
void selected_top_k_sum_warped_cuda(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
);
void __selected_top_k_sum_warped_cuda(
    int k,
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    switch (k) {
        case 0: selected_sum_cuda(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 1: selected_top_k_sum_warped_cuda<1>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 2: selected_top_k_sum_warped_cuda<2>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 3: selected_top_k_sum_warped_cuda<3>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 4: selected_top_k_sum_warped_cuda<4>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 5: selected_top_k_sum_warped_cuda<5>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 6: selected_top_k_sum_warped_cuda<6>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 7: selected_top_k_sum_warped_cuda<7>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 8: selected_top_k_sum_warped_cuda<8>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 9: selected_top_k_sum_warped_cuda<9>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 10:selected_top_k_sum_warped_cuda<10>(x,selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        default:
            throw std::invalid_argument("Unsupported value of k");
    }
}
at::Tensor selected_top_k_sum_warped_laucher(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    const int k
) {
    CHECK_CUDA(x);
    CHECK_CUDA(selected_idx);
    TORCH_CHECK(selected_idx.dtype() == torch::kLong, "selected_idx must be of type long");

    const int rows_per_batch = x.size(1);
    const int nv = x.size(2);
    const int num_rows = x.size(0) * rows_per_batch;
    const int num_selected = selected_idx.size(1);

    auto out = torch::zeros({x.size(0), x.size(1), 1}, torch::TensorOptions().dtype(x.dtype()).device(torch::kCUDA));

    __selected_top_k_sum_warped_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);

    return out;
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
);
void __selected_top_k_sum_single_cuda(
    int k,
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    switch (k) {
        case 0: selected_sum_cuda(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 1: selected_top_k_sum_single_cuda<1>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 2: selected_top_k_sum_single_cuda<2>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 3: selected_top_k_sum_single_cuda<3>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 4: selected_top_k_sum_single_cuda<4>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 5: selected_top_k_sum_single_cuda<5>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 6: selected_top_k_sum_single_cuda<6>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 7: selected_top_k_sum_single_cuda<7>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 8: selected_top_k_sum_single_cuda<8>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 9: selected_top_k_sum_single_cuda<9>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 10:selected_top_k_sum_single_cuda<10>(x,selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        default:
            throw std::invalid_argument("Unsupported value of k");
    }
}
at::Tensor selected_top_k_sum_single_laucher(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    const int k
) {
    CHECK_CUDA(x);
    CHECK_CUDA(selected_idx);
    TORCH_CHECK(selected_idx.dtype() == torch::kLong, "selected_idx must be of type long");

    const int rows_per_batch = x.size(1);
    const int nv = x.size(2);
    const int num_rows = x.size(0) * rows_per_batch;
    const int num_selected = selected_idx.size(1);

    auto out = torch::zeros({x.size(0), x.size(1), 1}, torch::TensorOptions().dtype(x.dtype()).device(torch::kCUDA));

    __selected_top_k_sum_single_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);

    return out;
}

at::Tensor selected_top_k_sum_laucher(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    const int k
) {
    CHECK_CUDA(x);
    CHECK_CUDA(selected_idx);
    TORCH_CHECK(selected_idx.dtype() == torch::kLong, "selected_idx must be of type long");

    const int rows_per_batch = x.size(1);
    const int nv = x.size(2);
    const int num_rows = x.size(0) * rows_per_batch;
    const int num_selected = selected_idx.size(1);

    auto out = torch::zeros({x.size(0), x.size(1), 1}, torch::TensorOptions().dtype(x.dtype()).device(torch::kCUDA));

    if (min_nv_to_warp.find({num_selected, k}) != min_nv_to_warp.end() && nv < min_nv_to_warp.at({num_selected, k}))
        __selected_top_k_sum_single_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);
    else  // Use warped version
        __selected_top_k_sum_warped_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);

    return out;
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
);
void __selected_bottom_k_sum_warped_cuda(
    int k,
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    switch (k) {
        case 0: selected_sum_cuda(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 1: selected_bottom_k_sum_warped_cuda<1>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 2: selected_bottom_k_sum_warped_cuda<2>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 3: selected_bottom_k_sum_warped_cuda<3>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 4: selected_bottom_k_sum_warped_cuda<4>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 5: selected_bottom_k_sum_warped_cuda<5>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 6: selected_bottom_k_sum_warped_cuda<6>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 7: selected_bottom_k_sum_warped_cuda<7>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 8: selected_bottom_k_sum_warped_cuda<8>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 9: selected_bottom_k_sum_warped_cuda<9>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 10:selected_bottom_k_sum_warped_cuda<10>(x,selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        default:
            throw std::invalid_argument("Unsupported value of k");
    }
}
at::Tensor selected_bottom_k_sum_warped_laucher(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    const int k
) {
    CHECK_CUDA(x);
    CHECK_CUDA(selected_idx);
    TORCH_CHECK(selected_idx.dtype() == torch::kLong, "selected_idx must be of type long");

    const int rows_per_batch = x.size(1);
    const int nv = x.size(2);
    const int num_rows = x.size(0) * rows_per_batch;
    const int num_selected = selected_idx.size(1);

    auto out = torch::zeros({x.size(0), x.size(1), 1}, torch::TensorOptions().dtype(x.dtype()).device(torch::kCUDA));

    __selected_bottom_k_sum_warped_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);

    return out;
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
);
void __selected_bottom_k_sum_single_cuda(
    int k,
    at::Tensor& x,
    const at::Tensor& selected_idx,
    at::Tensor& out,
    const int num_rows,
    const int nv,
    const int rows_per_batch,
    const int num_selected
) {
    switch (k) {
        case 0: selected_sum_cuda(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 1: selected_bottom_k_sum_single_cuda<1>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 2: selected_bottom_k_sum_single_cuda<2>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 3: selected_bottom_k_sum_single_cuda<3>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 4: selected_bottom_k_sum_single_cuda<4>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 5: selected_bottom_k_sum_single_cuda<5>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 6: selected_bottom_k_sum_single_cuda<6>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 7: selected_bottom_k_sum_single_cuda<7>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 8: selected_bottom_k_sum_single_cuda<8>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 9: selected_bottom_k_sum_single_cuda<9>(x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        case 10:selected_bottom_k_sum_single_cuda<10>(x,selected_idx, out, num_rows, nv, rows_per_batch, num_selected); break;
        default:
            throw std::invalid_argument("Unsupported value of k");
    }
}
at::Tensor selected_bottom_k_sum_single_laucher(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    const int k
) {
    CHECK_CUDA(x);
    CHECK_CUDA(selected_idx);
    TORCH_CHECK(selected_idx.dtype() == torch::kLong, "selected_idx must be of type long");

    const int rows_per_batch = x.size(1);
    const int nv = x.size(2);
    const int num_rows = x.size(0) * rows_per_batch;
    const int num_selected = selected_idx.size(1);

    auto out = torch::zeros({x.size(0), x.size(1), 1}, torch::TensorOptions().dtype(x.dtype()).device(torch::kCUDA));

    __selected_bottom_k_sum_single_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);

    return out;
}


at::Tensor selected_bottom_k_sum_laucher(
    at::Tensor& x,
    const at::Tensor& selected_idx,
    const int k
) {
    CHECK_CUDA(x);
    CHECK_CUDA(selected_idx);
    TORCH_CHECK(selected_idx.dtype() == torch::kLong, "selected_idx must be of type long");

    const int rows_per_batch = x.size(1);
    const int nv = x.size(2);
    const int num_rows = x.size(0) * rows_per_batch;
    const int num_selected = selected_idx.size(1);

    auto out = torch::zeros({x.size(0), x.size(1), 1}, torch::TensorOptions().dtype(x.dtype()).device(torch::kCUDA));

    if (min_nv_to_warp.find({num_selected, k}) != min_nv_to_warp.end() && nv < min_nv_to_warp.at({num_selected, k}))
        __selected_bottom_k_sum_single_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);
    else
        __selected_bottom_k_sum_warped_cuda(k, x, selected_idx, out, num_rows, nv, rows_per_batch, num_selected);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wipe_selected", &wipe_selected_laucher, "Wipe Selected Elements (CUDA)");
    m.def("selected_top_k_sum", &selected_top_k_sum_laucher, "Sum the elements from selected indices and the Top K elements from the rest (CUDA). Auto-decides between single and warped implementation based on the number of selected indicess, k and the number of elements per row.");
    m.def("selected_top_k_sum_warped", &selected_top_k_sum_warped_laucher, "Sum the elements from selected indices and the Top K elements from the rest (CUDA). Uses warp (32) threads per row.");
    m.def("selected_top_k_sum_single", &selected_top_k_sum_single_laucher, "Sum the elements from selected indices and the Top K elements from the rest (CUDA). Uses single thread per row.");
    m.def("selected_bottom_k_sum", &selected_bottom_k_sum_laucher, "Sum the elements from selected indices and the Bottom K elements from the rest (CUDA). Auto-decides between single and warped implementation based on the number of selected indicess, k and the number of elements per row.");
    m.def("selected_bottom_k_sum_warped", &selected_bottom_k_sum_warped_laucher, "Sum the elements from selected indices and the Bottom K elements from the rest (CUDA). Uses warp threads per row.");
    m.def("selected_bottom_k_sum_single", &selected_bottom_k_sum_single_laucher, "Sum the elements from selected indices and the Bottom K elements from the rest (CUDA). Uses single thread per row.");
}
