from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

assert torch.cuda.is_available()

arch_set = set()
num_devices = torch.cuda.device_count()
for i in range(num_devices):
    capability = torch.cuda.get_device_capability(i)
    arch = f"{capability[0]}.{capability[1]}"
    arch_set.add(arch)

os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arch_set)

bimm_extension = CUDAExtension(
    name='bimm.bimm',
    sources=[
        'bimm/bimm_bindings.cpp',
        'bimm/bimm.cu',
    ],
)

topksum_extension = CUDAExtension(
    name='topksum.topksum',
    sources=[
        'topksum/topksum_bindings.cpp',
        'topksum/topksum.cu',
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math"],
    },
)

setup(
    name='L0-Robustness-Verifier',
    version='0.1',
    packages=['bimm', 'topksum'],
    ext_modules=[bimm_extension, topksum_extension],
    cmdclass={'build_ext': BuildExtension}
)
