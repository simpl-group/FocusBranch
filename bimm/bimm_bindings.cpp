#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

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
    if (na == 0 || nb == 0) {
        throw std::runtime_error("A and B must be non-empty");
    }
    if (A.size(1) != B.size(1)) {
        throw std::runtime_error("A and B must have the same number of columns");
    }
    const int nb_div8 = (nb + 7) / 8;
    auto C = torch::zeros({na, nb_div8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    superset_popc_cuda(A, B, C, na, nb, A.size(1), nb_div8);
    return C.sum(1);
}

void combs2bitmasks_cuda(
    const at::Tensor& combs,
    at::Tensor& bitmasks,
    const int num_rows,
    const int nv,
    const int popc
);
at::Tensor combs2bitmasks_laucher(const at::Tensor& combs, const int nv) {
    CHECK_CUDA(combs);
    const int num_rows = combs.size(0);
    if (num_rows == 0) {
        return torch::empty({0, nv}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    }
    const int popc = combs.size(1);
    auto bitmasks = torch::zeros({num_rows, nv}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    combs2bitmasks_cuda(combs, bitmasks, num_rows, nv, popc);
    return bitmasks;
}

void bitmasks2combs_cuda(
    const at::Tensor& bitmasks,
    at::Tensor& combs,
    const int num_rows,
    const int nv,
    const int popc
);
at::Tensor bitmasks2combs_laucher(const at::Tensor& bitmasks, const int popc) {
    CHECK_CUDA(bitmasks);
    const int num_rows = bitmasks.size(0);
    if (num_rows == 0) {
        return torch::empty({0, popc}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    }
    auto combs = torch::zeros({num_rows, popc}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    bitmasks2combs_cuda(bitmasks, combs, num_rows, bitmasks.size(1), popc);
    return combs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("superset_sum", &superset_sum_laucher, "Superset Sum (CUDA)");
    m.def("combs2bitmasks", &combs2bitmasks_laucher, "Combination to Bitmask (CUDA)");
    m.def("bitmasks2combs", &bitmasks2combs_laucher, "Bitmask to Combination (CUDA)");
}
