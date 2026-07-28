from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mymuladd",
    packages=["mymuladd"],
    ext_modules=[
        CUDAExtension(
            name="mymuladd._C",
            sources=[
                "csrc/binding.cpp",
                "csrc/mymuladd_cpu.cpp",
                "csrc/mymuladd_cuda.cu",
                "csrc/mymuladd_backward_cpu.cpp",
                "csrc/mymuladd_backward_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2"],
                "nvcc": ["-O3", "-Xcompiler", "/Zc:preprocessor"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
