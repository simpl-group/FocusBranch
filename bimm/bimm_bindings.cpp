#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void bimm64_cuda(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const int na,
    const int nb,
    const int nv
);
at::Tensor bimm64_laucher(const at::Tensor& A, const at::Tensor& B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    auto C = torch::zeros({na, nb}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    bimm64_cuda(A, B, C, na, nb, A.size(1));
    return C;
}

void superset_popc_cuda(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const int na,
    const int nb,
    const int nv,
    const int nb_div8
);
at::Tensor superset_sum_laucher(const at::Tensor& A, const at::Tensor& B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    const int nb_div8 = (nb + 7) / 8;
    auto C = torch::zeros({na, nb_div8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    superset_popc_cuda(A, B, C, na, nb, A.size(1), nb_div8);
    return C.sum(1);
}

void comb2bitmask_cuda(
    const at::Tensor& combs,
    at::Tensor& bitmasks,
    const int num_rows,
    const int nv,
    const int popc
);
at::Tensor comb2bitmask_laucher(const at::Tensor& combs, const int nv) {
    CHECK_CUDA(combs);
    const int num_rows = combs.size(0);
    const int popc = combs.size(1);
    auto bitmasks = torch::zeros({num_rows, nv}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    comb2bitmask_cuda(combs, bitmasks, num_rows, nv, popc);
    return bitmasks;
}

void bitmask2comb_cuda(
    const at::Tensor& bitmasks,
    at::Tensor& combs,
    const int num_rows,
    const int nv,
    const int popc
);
at::Tensor bitmask2comb_laucher(const at::Tensor& bitmasks, const int popc) {
    CHECK_CUDA(bitmasks);
    const int num_rows = bitmasks.size(0);
    auto combs = torch::zeros({num_rows, popc}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    bitmask2comb_cuda(bitmasks, combs, num_rows, bitmasks.size(1), popc);
    return combs;
}

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bimm64", &bimm64_laucher, "Binary Matmul 64 (CUDA)");
    m.def("superset_sum", &superset_sum_laucher, "Superset Sum (CUDA)");
    m.def("comb2bitmask", &comb2bitmask_laucher, "Combination to Bitmask (CUDA)");
    m.def("bitmask2comb", &bitmask2comb_laucher, "Bitmask to Combination (CUDA)");
    m.def("wipe_selected", &wipe_selected_laucher, "Wipe Selected Elements (CUDA)");
}
