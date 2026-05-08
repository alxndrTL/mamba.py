"""
Setup script for Metal PScan extension.

Usage:
    pip install -e .  # Install mambapy
    python setup_metal.py install  # Install Metal extension
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

# Metal pscan extension
ext_modules = [
    CppExtension(
        name='metal_pscan._C',
        sources=['metal_pscan/csrc/metal_pscan.mm'],
        extra_compile_args={
            'cxx': ['-std=c++17', '-O3'],
        },
        extra_link_args=[
            '-framework', 'Metal',
            '-framework', 'Foundation',
        ],
    )
]

setup(
    name='metal_pscan',
    version='0.1.0',
    packages=['metal_pscan'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.10',
    install_requires=['torch>=2.0'],
)
