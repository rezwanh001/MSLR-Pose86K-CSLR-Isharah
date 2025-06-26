import os
import torch

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")

def load_checkpoint(filename, device):
    if os.path.exists(filename):
        return torch.load(filename, map_location=device)
    else:
        print(f"No checkpoint found at {filename}")
        return None

def setup_environment(cfg):
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.GPU_IDS))
    print(f"Using GPUs: {cfg.GPU_IDS}")