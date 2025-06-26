import torch
from torch.utils.data import DataLoader
from .dataset import PoseDataset, collate_fn
from .model import SignLanguageRecognizer, AdvancedSignLanguageRecognizer, SignLanguageConformer
# from .mamba_model import ArabicMamba
from .mamba_transformer import MambaSignLanguageRecognizer
from .mamba_transformer import SOTA_CSLR

import os
import sys
if os.environ.get("CONDA_DEFAULT_ENV") == "ArbMamba":
    from .mamba_model import ArabicMamba
else:
    print("Not importing mamba_ssm (not in ArbMamba environment).", file=sys.stderr)

from tqdm import tqdm
import os
import zipfile
from torch.cuda.amp import autocast
from configs import SIConfig, USConfig

def get_config(mode, model_type=None):
    if mode == 'SI':
        cfg = SIConfig()
    elif mode == 'US':
        cfg = USConfig()
    else:
        raise ValueError("Invalid mode. Choose either 'SI' or 'US'.")

    # Dynamically set model save path if model_type is provided
    if model_type is not None:
        cfg.MODEL_SAVE_PATH = f'/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/models/{cfg.MODE}/{cfg.MODE}_{model_type}_best_model.pth'
        # cfg.MODEL_SAVE_PATH = f'/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/models/{cfg.MODE}/{cfg.MODE}_{model_type}_checkpoint.pth'
        print(f"Using model save path: {cfg.MODEL_SAVE_PATH}")
    return cfg

def load_checkpoint(filename, device):
    return torch.load(filename, map_location=device)

def remove_duplicates(gloss_seq):
    # Remove consecutive duplicates (CTC-style)
    result = []
    prev = None
    for g in gloss_seq:
        if g != prev:
            result.append(g)
        prev = g
    return result

def generate_predictions(split='test', output_path=None, mode='SI', model_type='SignLanguageConformer'):
    cfg = get_config(mode, model_type)
    # Load trained model
    checkpoint = load_checkpoint(cfg.MODEL_SAVE_PATH, cfg.DEVICE)
    vocab = checkpoint['vocab']
    idx_to_gloss = {idx: gloss for gloss, idx in vocab.items()}

    mamba_config = {
        'd_state': 12,
        'expand': 4,
        'd_conv': 4,
        'bidirectional': True
    }
    # Initialize model based on type
    if model_type == 'SignLanguageRecognizer':
        model = SignLanguageRecognizer(len(vocab), cfg)
    elif model_type == 'ArabicMamba':
        model = ArabicMamba(len(vocab), cfg, mamba_config)
    elif model_type == 'SignLanguageConformer':
        model = SignLanguageConformer(len(vocab), cfg)

    elif model_type == 'AdvancedSignLanguageRecognizer': ### For Mode = US (where we got high performance) # (WER) val = 55.0847  & test = 47.7756 (Winner: 2nd)
        model = AdvancedSignLanguageRecognizer(len(vocab), cfg)

    elif model_type == 'MambaSignLanguageRecognizer':
        model = MambaSignLanguageRecognizer(len(vocab), cfg)

    elif model_type == 'SOTA_CSLR': ### For Mode = SI (where we got high performance) # (WER) val = 7.3123  & test = 13.0652 (Winner: 4th)
        model = SOTA_CSLR(len(vocab), cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    if cfg.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPU_IDS)
    model.to(cfg.DEVICE)
    model.eval()

    # Select dataset and annotation path
    if split == 'dev':
        print("Using DEV set for inference.") 
        pose_path = cfg.DEV_POSE_PATH
        annotation_path = cfg.DEV_ANNOTATION_PATH
        default_output = cfg.DEV_PREDICTION_OUTPUT_PATH
    else:
        print("Using TEST set for inference.")
        pose_path = cfg.TEST_POSE_PATH
        annotation_path = cfg.TEST_ANNOTATION_PATH
        default_output = cfg.PREDICTION_OUTPUT_PATH

    # Prepare dataset and loader
    dataset = PoseDataset(
        pose_path,
        annotation_path,
        mode,
        split,
        vocab=vocab
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    predictions = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split.capitalize()} Inference"):
            if len(batch) == 2:
                ids, inputs = batch
            else:
                ids, inputs = batch[0], batch[1]
            inputs = inputs.to(cfg.DEVICE, non_blocking=True)
            with autocast(dtype=cfg.FP16_DTYPE, enabled=cfg.MIXED_PRECISION):
                outputs = model(inputs)
            pred_indices = torch.argmax(outputs, dim=2).cpu().numpy()
            for i, sample_id in enumerate(ids):
                gloss_seq = []
                prev_idx = -1
                for idx in pred_indices[i]:
                    if idx != prev_idx and idx != 0:  # 0 is blank
                        gloss_seq.append(idx_to_gloss.get(idx, "<unk>"))
                    prev_idx = idx
                gloss_seq = remove_duplicates(gloss_seq)
                predictions[sample_id] = " ".join(gloss_seq)

    output_file = output_path if output_path else default_output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("id,gloss\n")
        for sample_id, gloss in predictions.items():
            f.write(f"{sample_id},{gloss}\n")
    print(f"{split.capitalize()} predictions saved to {output_file}")

    # Determine output directory and filenames
    output_dir = os.path.dirname(output_path if output_path else default_output)
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{split}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    zip_path = os.path.join(output_dir, f"{split}.zip")

    # Save CSV
    with open(csv_path, 'w') as f:
        f.write("id,gloss\n")
        for sample_id, gloss in predictions.items():
            f.write(f"{sample_id},{gloss}\n")
    print(f"{split.capitalize()} predictions saved to {csv_path}")

    # Save ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=csv_filename)
    print(f"{split.capitalize()} predictions zipped to {zip_path}")