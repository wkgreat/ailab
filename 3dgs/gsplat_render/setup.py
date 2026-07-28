import os
import subprocess

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")


def require_cuda_home():
    if not CUDA_HOME:
        raise RuntimeError("CUDA_HOME or CUDA_PATH must be set to build gsplatrender")
    return CUDA_HOME


class BuildCudaExtension(build_ext):
    def build_extension(self, ext):
        cuda_home = require_cuda_home()
        self.mkpath(self.build_temp)

        cuda_obj = os.path.join(self.build_temp, "gsplatrender_cuda.obj")
        nvcc = os.path.join(cuda_home, "bin", "nvcc.exe")
        cuda_src = os.path.join(ROOT, "csrc", "gsplatrender.cu")

        include_args = [
            "-I" + os.path.join(ROOT, "csrc"),
            "-I" + np.get_include(),
            "-I" + os.path.join(cuda_home, "include"),
        ]

        subprocess.check_call(
            [
                nvcc,
                "-c",
                cuda_src,
                "-o",
                cuda_obj,
                "-O3",
                "-Xcompiler",
                "/Zc:preprocessor",
                *include_args,
            ]
        )

        ext.extra_objects = [*getattr(ext, "extra_objects", []), cuda_obj]
        super().build_extension(ext)

setup(
    name="gsplatrender",
    packages=["gsplatrender"],
    ext_modules=[
        Extension(
            name="gsplatrender._C",
            sources=["csrc/gsplatrender_binding.cpp"],
            include_dirs=[np.get_include(), os.path.join(ROOT, "csrc")],
            library_dirs=[os.path.join(require_cuda_home(), "lib", "x64")],
            libraries=["cudart"],
            extra_compile_args=["/O2"],
        ),
    ],
    cmdclass={"build_ext": BuildCudaExtension},
)
