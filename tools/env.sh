# CUDALab environment definition.
# Source this file (or rely on PYTHON/CUDA_HOME defaults) instead of depending
# on an interactive shell's .bashrc state.
#
#   . /root/code/cuda/tools/env.sh
export CUDA_HOME=/usr/local/cuda
# conda env bin dir first (python, ninja), then CUDA toolkit bin
export PATH=/root/miniconda3/envs/pytorch/bin:/usr/local/cuda/bin:$PATH
export NVCC=/usr/local/cuda/bin/nvcc
export PYTHON=/root/miniconda3/envs/pytorch/bin/python
export TORCH_CUDA_ARCH_LIST=7.5
export CUDALAB_HOME=/root/code/cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Do not let PyTorch rebuild extensions on every run
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}
