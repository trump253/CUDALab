# CUDALab 环境定义。
# 请 source 本文件（或依赖 PYTHON/CUDA_HOME 默认值），不要依赖交互式
# shell 的 .bashrc 状态。
#
#   . /root/code/cuda/tools/env.sh
export CUDA_HOME=/usr/local/cuda
# conda 环境 bin 目录在前（python、ninja），其后是 CUDA 工具链 bin
export PATH=/root/miniconda3/envs/pytorch/bin:/usr/local/cuda/bin:$PATH
export NVCC=/usr/local/cuda/bin/nvcc
export PYTHON=/root/miniconda3/envs/pytorch/bin/python
export TORCH_CUDA_ARCH_LIST=7.5
export CUDALAB_HOME=/root/code/cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 避免 PyTorch 每次运行都重建扩展
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}
