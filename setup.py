from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bimm',
    ext_modules=[
        CUDAExtension('bimm', [
            'bimm/bimm_bindings.cpp',
            'bimm/bimm.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)
