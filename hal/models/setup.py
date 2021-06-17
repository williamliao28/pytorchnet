from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nl_module',
    ext_modules=[
        CUDAExtension('nl_module_cuda', [
            'nl_wrapper.cpp',
            'nl_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })