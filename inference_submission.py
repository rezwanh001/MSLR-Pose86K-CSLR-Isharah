import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import csv
import zipfile
from tqdm import tqdm
from torchvision import transforms
from utils.text_ctc_utils import convert_text_for_ctc
from utils.decode import Decode
from utils.datasetv2 import PoseDatasetV2

from models.transformer import SlowFastCSLR, PoseCSLRTransformer
from models.llm_based_model import SlowFastLLMCSLR, LLMEnhancedPoseCSLR, AdvancedSlowFastLLMCSLR
from models.stgcn_conformer import STGCNConformer, get_body_adjacency_matrix
from models.spatio_temporal_transformer import SpatioTemporalTransformer, get_body_adjacency_matrix

MODELS = {
    "base": PoseCSLRTransformer,
    "slowfast": SlowFastCSLR,
    "llm_PoseCSLRT": LLMEnhancedPoseCSLR,
    "llm_slowfast": SlowFastLLMCSLR,
    "llm_advslowfast": AdvancedSlowFastLLMCSLR,
    "stgcn_conformer": STGCNConformer,
    "st_transformer": SpatioTemporalTransformer 
}

def load_dev_data(dev_csv, mode):
    """Load dev dataset and vocabulary mappings."""
    train_csv = f"./annotations_v2/{mode}/train.txt"  # Needed for vocab
    train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc("isharah", train_csv, dev_csv)
    dataset_dev = PoseDatasetV2(
        dataset_name2="isharah",
        label_csv=dev_csv,
        split_type="dev",
        target_enc_df=dev_processed,
        augmentations=False,
        max_frames=512,
        mode=mode
    )
    dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False, num_workers=4)
    return dataloader_dev, vocab_map, inv_vocab_map, vocab_list

def load_model(model_name, input_dim, num_classes, device):
    """Initialize and load the best model checkpoint."""
    model_class = MODELS[model_name]
    # Special handling for graph-based models
    if model_name in ["stgcn_conformer", "st_transformer"]:
        adj_matrix = get_body_adjacency_matrix()
        model = model_class(
            input_dim=2,  # x, y coordinates
            num_classes=num_classes,
            adj_matrix=adj_matrix,
            embed_dim=256,
            num_heads=4 if model_name == "stgcn_conformer" else 8,
            num_layers=6 if model_name == "stgcn_conformer" else 4,
            spatial_channels=64 if model_name == "st_transformer" else None
        ).to(device)
    else:
        model = model_class(input_dim=input_dim, num_classes=num_classes).to(device)
    return model

def inference(model, dataloader, decoder, device):
    """Run inference on the dev set and collect predictions."""
    model.eval()
    ids = []
    predictions = []
    with torch.no_grad():
        for file, poses, _ in tqdm(dataloader, desc="Inference", ncols=100):
            poses = poses.to(device)
            logits = model(poses)
            vid_lgt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
            decoded_list = decoder.decode(logits, vid_lgt=vid_lgt, batch_first=True, probs=False)
            flat_preds = [gloss for pred in decoded_list for gloss, _ in pred]
            current_preds = ' '.join(flat_preds) or "[blank]"
            ids.append(file[0])  # file is a list with one element
            predictions.append(current_preds)
    return ids, predictions

def save_submission(ids, predictions, output_csv, output_zip):
    """Save predictions to dev.csv and create dev.zip."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'gloss'])
        for _id, pred in zip(ids, predictions):
            writer.writerow([_id, pred])
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname=os.path.basename(output_csv))
    
    print(f"[✔] Submission saved to: {output_zip}")

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Setup device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load dev dataset
    dev_csv = f"./annotations_v2/{args.mode}/dev.txt"
    dataloader_dev, vocab_map, _, vocab_list = load_dev_data(dev_csv, args.mode)
    
    # Initialize model
    input_dim = 86 * 2  # 86 keypoints * (x, y)
    num_classes = len(vocab_map)
    model = load_model(args.model, input_dim, num_classes, device)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(args.work_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Setup decoder
    decoder = Decode(vocab_map, len(vocab_list), 'beam')
    
    # Run inference
    ids, predictions = inference(model, dataloader_dev, decoder, device)
    
    # Save submission
    output_csv = os.path.join(args.output_dir, "dev.csv")
    output_zip = os.path.join(args.output_dir, "dev.zip")
    save_submission(ids, predictions, output_csv, output_zip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default="./work_dir/test", help='Directory containing best_model.pt')
    parser.add_argument('--mode', type=str, default="SI", help='Task mode: SI or US')
    parser.add_argument('--model', type=str, default="base", help='Model name: base, slowfast, llm, stgcn_conformer, st_transformer')
    parser.add_argument('--device', type=str, default="0", help='CUDA device ID')
    parser.add_argument('--output_dir', type=str, default="./submission/task-1", help='Directory to save dev.csv and dev.zip')
    args = parser.parse_args()
    
    main(args)

""" 
# Inference Submission Script for ISHARAH Challenge

#### Task-1 (SI)

python inference_submission.py \
    --work_dir ./work_dir/llm_advslowfast_SI \
    --mode SI \
    --model llm_advslowfast \
    --device 0 \
    --output_dir ./submission/task-1

#### Task-2 (US)

python inference_submission.py \
    --work_dir ./work_dir/llm_advslowfast_US \
    --mode US \
    --model llm_advslowfast \
    --device 1 \
    --output_dir ./submission/task-2
"""