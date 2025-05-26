# -*- coding: utf-8 -*-

import io
import os
import os.path as osp
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package meta-data.
NAME = 'mesh_mesh_intersection'
DESCRIPTION = 'PyTorch module for Mesh-Mesh intersection detection'
URL = ''
EMAIL = 'vassilis.choutas@tuebingen.mpg.de'
AUTHOR = 'Vassilis Choutas'
REQUIRES_PYTHON = '>=3.8.0'  # Updated minimum Python version
VERSION = '0.2.0'

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

mesh_mesh_intersect_src_files = [
    'src/mesh_mesh_intersect.cpp', 
    'src/mesh_mesh_intersect_cuda_op.cu'
]

# Modern include paths
mesh_mesh_intersect_include_dirs = [
    osp.abspath('include'),
] + torch.utils.cpp_extension.include_paths()

# Add CUDA samples include if available
cuda_samples_inc = os.environ.get('CUDA_SAMPLES_INC', '')
if cuda_samples_inc and os.path.exists(cuda_samples_inc):
    mesh_mesh_intersect_include_dirs.append(osp.abspath(osp.expandvars(cuda_samples_inc)))

# Modern CUDA compilation flags
nvcc_flags = [
    '-DPRINT_TIMINGS=0', 
    '-DDEBUG_PRINT=0',
    '-DERROR_CHECKING=1',
    '-DCOLLISION_ORDERING=1',
    '--expt-relaxed-constexpr',  # For modern CUDA
    '--expt-extended-lambda',    # For modern CUDA
]

# Handle GPU architectures
cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
if cuda_arch_list:
    # Use environment-specified architectures
    for arch in cuda_arch_list.split(';'):
        arch = arch.strip()
        if arch:
            arch_code = arch.replace('.', '')
            nvcc_flags.extend([
                '-gencode', f'arch=compute_{arch_code},code=sm_{arch_code}'
            ])
else:
    # Default to modern GPU architectures
    default_archs = ['7.5', '8.0', '8.6', '8.9']  # Covers most modern GPUs
    for arch in default_archs:
        arch_code = arch.replace('.', '')
        nvcc_flags.extend([
            '-gencode', f'arch=compute_{arch_code},code=sm_{arch_code}'
        ])

# C++ compilation flags for modern standards
cxx_flags = [
    '-std=c++14',  # Modern C++ standard
]

mesh_mesh_intersect_extra_compile_args = {
    'nvcc': nvcc_flags,
    'cxx': cxx_flags
}

mesh_mesh_intersect_extension = CUDAExtension(
    'mesh_mesh_intersect_cuda', 
    mesh_mesh_intersect_src_files,
    include_dirs=mesh_mesh_intersect_include_dirs,
    extra_compile_args=mesh_mesh_intersect_extra_compile_args
)

render_reqs = ['pyrender>=0.1.23', 'trimesh>=2.37.6', 'shapely']

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    ext_modules=[mesh_mesh_intersect_extension],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Environment :: Console",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        'torch>=1.8.0',  # Updated minimum PyTorch version
        'numpy>=1.19.0',
    ],
    extras_require={
        'render': render_reqs,
        'all': render_reqs
    },
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}  # Disable ninja for compatibility
)