#!/bin/bash

# Create conda environment
conda create -n mslr_arabic python=3.10 -y
conda activate mslr_arabic

# Install PyTorch with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core packages
conda install -c conda-forge \
    pandas \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    tqdm \
    pyyaml \
    opencv \
    jupyterlab \
    ipykernel \
    numba \
    nvtx \
    -y

# Install additional packages
pip install \
    mediapipe==0.10.9 \
    wandb==0.17.0 \
    tensorboard==2.16 \
    einops==0.8.0 \
    transformers==4.40 \
    pickle-mixin==1.0.2 \
    nvidia-cudnn-cu12==8.9.6.50 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-nvrtc-cu12==12.1.105

# Register environment with Jupyter
python -m ipykernel install --user --name mslr_arabic --display-name "MSLR Project"

echo "Environment setup complete!"