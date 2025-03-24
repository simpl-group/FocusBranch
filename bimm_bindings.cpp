#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void bimm64_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv
);
torch::Tensor bimm64_laucher(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    auto C = torch::zeros({na, nb}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    bimm64_cuda(A, B, C, na, nb, A.size(1));
    return C;
}

void bimm64_eq_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv, const int cmp
);
torch::Tensor bimm64_eq_laucher(torch::Tensor A, torch::Tensor B, const int cmp) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    auto C = torch::zeros({na, nb}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    bimm64_eq_cuda(A, B, C, na, nb, A.size(1), cmp);
    return C;  //.sum(1) here improves 8%
}

void bimm64_le_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv, const int cmp
);
torch::Tensor bimm64_le_laucher(torch::Tensor A, torch::Tensor B, const int cmp) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    auto C = torch::zeros({na, nb}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    bimm64_le_cuda(A, B, C, na, nb, A.size(1), cmp);
    return C;  //.any(1)
}

void bimm64_eq_popc_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv, const int cmp, const int nb_div8
);
torch::Tensor bimm64_eq_sum_laucher(torch::Tensor A, torch::Tensor B, const int cmp) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    const int nb_div8 = (nb + 7) / 8;
    auto C = torch::zeros({na, nb_div8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    bimm64_eq_popc_cuda(A, B, C, na, nb, A.size(1), cmp, nb_div8);
    return C.sum(1);
}

void bimm64_le_popc_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    const int na, const int nb, const int nv, const int cmp, const int nb_div8
);
torch::Tensor bimm64_le_any_laucher(torch::Tensor A, torch::Tensor B, const int cmp) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    const int na = A.size(0);
    const int nb = B.size(0);
    const int nb_div8 = (nb + 7) / 8;
    auto C = torch::zeros({na, nb_div8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    bimm64_le_popc_cuda(A, B, C, na, nb, A.size(1), cmp, nb_div8);
    return C.any(1);
}

void comb2bitmask_cuda(
    at::Tensor combs, at::Tensor bitmasks,
    const int num_rows, const int nv, const int popc
);
torch::Tensor comb2bitmask_laucher(torch::Tensor combs, const int nv) {
    CHECK_CUDA(combs);
    const int num_rows = combs.size(0);
    const int popc = combs.size(1);
    auto bitmasks = torch::zeros({num_rows, nv}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    comb2bitmask_cuda(combs, bitmasks, num_rows, nv, popc);
    return bitmasks;
}

void bitmask2comb_cuda(
    at::Tensor bitmasks, at::Tensor combs,
    const int num_rows, const int nv, const int popc
);
torch::Tensor bitmask2comb_laucher(torch::Tensor bitmasks, const int popc) {
    CHECK_CUDA(bitmasks);
    const int num_rows = bitmasks.size(0);
    auto combs = torch::zeros({num_rows, popc}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    bitmask2comb_cuda(bitmasks, combs, num_rows, bitmasks.size(1), popc);
    return combs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bimm64", &bimm64_laucher, "Binary Matmul 64 (CUDA)");
    m.def("bimm64_eq", &bimm64_eq_laucher, "Binary Matmul 64 Equal (CUDA)");
    m.def("bimm64_le", &bimm64_le_laucher, "Binary Matmul 64 Less Equal (CUDA)");
    m.def("bimm64_eq_sum", &bimm64_eq_sum_laucher, "Binary Matmul 64 Equal Sum (CUDA)");
    m.def("bimm64_le_any", &bimm64_le_any_laucher, "Binary Matmul 64 Less Equal Any (CUDA)");
    m.def("comb2bitmask", &comb2bitmask_laucher, "Combination to Bitmask (CUDA)");
    m.def("bitmask2comb", &bitmask2comb_laucher, "Bitmask to Combination (CUDA)");
}