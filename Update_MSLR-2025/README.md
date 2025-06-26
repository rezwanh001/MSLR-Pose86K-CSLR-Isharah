### Project Structure
``` bash
MSLR-Pose86K-CSLR-Isharah/
├── configs/
│   └── default.yaml
├── data/
│   ├── public_si_dat/
│   │   └── pose_data_isharah1000_hands_lips_body_May12.pkl
│   └── annotations_v2/
│       └── SI/
│           └── pose_data_isharah1000_SI_test.pkl
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── outputs/
│   ├── models/
│   └── predictions/
├── scripts/
│   └── run_pipeline.sh
├── requirements.txt
├── setup_env.sh
├── requirements.txt
├── environment.yml
└── README.md
```

### Environment setup:
```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate mslr_arabic

# Verify CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')"

```
```bash



conda create -n ArbMamba python=3.10 -y
conda activate ArbMamba

# Install PyTorch 2.1.0 (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install speechbrain
pip install speechbrain

# Install the exact matching mamba-ssm wheel
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Test
python -c 'import torch; import mamba_ssm; import speechbrain; print("All OK!")'


pip install "numpy<2"

```