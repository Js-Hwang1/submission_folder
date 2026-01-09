"""
CircuitKV: Current-Flow Betweenness for KV Cache Eviction in LLMs.

Build with: pip install -e .
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get absolute path to this directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_extensions():
    """Build the CUDA extension."""

    # Source files (relative paths from setup.py directory)
    sources = [
        "csrc/pybind.cpp",
        "csrc/circuit_manager.cpp",
        "csrc/kernels/graph_update.cu",
        "csrc/kernels/circuit_walker.cu",
        "csrc/kernels/spectral_power.cu",
        "csrc/kernels/landmark_walker.cu",
        "csrc/kernels/landmark_absorbing_walker.cu",  # v0.5.0: Landmark Absorbing
        "csrc/kernels/influence_walker.cu",  # v1.0.0: Causal Influence (VALIDATED BY PoC5)
    ]

    # Include directories (absolute paths required for compiler)
    include_dirs = [
        os.path.join(THIS_DIR, "csrc", "include"),
        os.path.join(THIS_DIR, "csrc"),
    ]

    # Compiler flags
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",
            "-Wall",
            "-Wextra",
        ],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-lineinfo",  # Debug info without perf hit
            "--ptxas-options=-v",  # Show register usage
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
            "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
            "-gencode=arch=compute_90,code=sm_90",  # H100
        ],
    }

    extension = CUDAExtension(
        name="circuit_kv._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )

    return [extension]


setup(
    name="circuit_kv",
    version="1.0.5",  # Fix: positional opportunity normalization
    author="CircuitKV Authors",
    description="Current-Flow Betweenness for KV Cache Eviction in LLMs",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
        ],
    },
)
