import os
import random
from tqdm import tqdm
import numpy as np
import argparse
import shutil
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from utils.text_ctc_utils import * 
from utils.decode import Decode
from utils.metrics import wer_list
from torchvision import transforms
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

def set_rng_state(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_workdir(work_dir):
    if os.path.exists(work_dir):
        answer = input('Current dir exists, do you want to remove and refresh it?\n')
        if answer in ['yes', 'y', 'ok', '1']:
            shutil.rmtree(work_dir)
            os.makedirs(work_dir)
    else:
        os.makedirs(work_dir)

    if not os.path.exists(os.path.join(work_dir, "pred_outputs")):
        os.mkdir(os.path.join(work_dir, "pred_outputs"))

def train_epoch(model, dataloader, optimizer, loss_encoder, device):
    total_loss = 0
    current_lr = optimizer.param_groups[0]['lr']

    model.train()
    for i, (_, poses, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="train", ncols=100):
        optimizer.zero_grad()

        logits = model(poses.to(device))  
        log_probs_enc = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # Required for CTC Loss
        log_probs_enc = log_probs_enc - (torch.tensor([1.0], device=log_probs_enc.device) * 
                                        (torch.arange(log_probs_enc.shape[-1], device=log_probs_enc.device) == 0).float())

        input_lengths = torch.full((log_probs_enc.size(1),), log_probs_enc.size(0), dtype=torch.long)
        target_lengths = torch.full((log_probs_enc.size(1),), labels.size(1), dtype=torch.long)
        loss_enc = loss_encoder(log_probs_enc, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss = loss_enc.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), current_lr


def evaluate_model(model, dataloader, decoder_dec, device, inv_vocab_map, work_dir, epoch):
    preds = []
    gt_labels = []

    model.eval()
    predictions_file = f"{work_dir}/pred_outputs/predictions_epoch_{epoch+1}.txt"
    with open(predictions_file, "w") as pred_file:
        pred_file.write(f"Epoch {epoch+1} Predictions\n")
        pred_file.write("=" * 50 + "\n")

        with torch.no_grad():
            for i, (file, poses, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="valid", ncols=100):
                poses = poses.to(device)

                logits = model(poses)

                vid_lgt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(device)
                decoded_list = decoder_dec.decode(logits, vid_lgt=vid_lgt, batch_first=True, probs=False)
                flat_preds = [gloss for pred in decoded_list for gloss, _ in pred]  # Flatten list
                current_preds = ' '.join(flat_preds)  # Convert list to string

                preds.append(current_preds)
                ground_truth = ' '.join(invert_to_chars(labels, inv_vocab_map))
                gt_labels.append(ground_truth)

                pred_file.write(f"GT: {ground_truth}\nPred: {current_preds}\n\n")

    wer_results = wer_list(preds, gt_labels)
    
    return wer_results

def main(args):
    set_rng_state(42)
    make_workdir(args.work_dir)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") 
        
    # train_csv = os.path.join(args.data_dir, f"isharah1000/annotations/{args.mode}/train.txt")
    # dev_csv = os.path.join(args.data_dir, f"isharah1000/annotations/{args.mode}/dev.txt")
    
    train_csv = f"./annotations_v2/{args.mode}/train.txt"
    dev_csv = f"./annotations_v2/{args.mode}/dev.txt"

    train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc("isharah", train_csv, dev_csv)

    # dataset_train = PoseDatasetV2("isharah", train_csv , "train", train_processed , augmentations=True , transform=transforms.Compose([GaussianNoise()]))
    # dataset_dev = PoseDatasetV2("isharah", dev_csv , "dev", dev_processed, augmentations=False)

    dataset_train = PoseDatasetV2("isharah", train_csv, "train", train_processed, augmentations=True, transform=transforms.Compose([GaussianNoise()]), max_frames=512, mode=args.mode)
    dataset_dev = PoseDatasetV2("isharah", dev_csv, "dev", dev_processed, augmentations=False, max_frames=512, mode=args.mode)

    traindataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=10)
    devdataloader = DataLoader(dataset_dev, batch_size=1, shuffle=False, num_workers=10)
    
    model = MODELS[args.model](input_dim=86*2, num_classes=len(vocab_map)).to(device)

    # adj_matrix = get_body_adjacency_matrix()
    # model = MODELS[args.model](
    #     input_dim=2,  # x,y coordinates
    #     num_classes=len(vocab_map),
    #     adj_matrix=adj_matrix,
    #     embed_dim=256,
    #     num_heads=4,
    #     num_layers=6
    # ).to(device)

    # # ...
    # adj_matrix = get_body_adjacency_matrix()
    # model = MODELS[args.model](
    #     input_dim=2,  # x,y coordinates
    #     num_classes=len(vocab_map),
    #     adj_matrix=adj_matrix,
    #     spatial_channels=64,
    #     embed_dim=256,
    #     num_layers=4,
    #     num_heads=8
    # ).to(device)
    # # ...

    decoder_dec = Decode(vocab_map, len(vocab_list), 'beam')
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_encoder = nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')

    # model = MODELS[args.model](input_dim=84, num_classes=len(vocab_map)).to(device)
    # decoder_dec = Decode(vocab_map, len(vocab_list), 'beam')
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # loss_encoder = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

    log_file = f"{args.work_dir}/training_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    best_wer = float("inf") 
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\n\nEpoch [{epoch+1}/{args.num_epochs}]")
        train_loss, current_lr = train_epoch(model, traindataloader, optimizer, loss_encoder, device)
        dev_wer_results = evaluate_model(model, devdataloader, decoder_dec, device, inv_vocab_map, args.work_dir, epoch)
        scheduler.step(dev_wer_results['wer'])

        if dev_wer_results['wer'] < best_wer:
            best_wer = dev_wer_results['wer']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.work_dir}/best_model.pt")
        else:
            patience_counter += 1
        
        log_msg = (f"Train Loss: {train_loss / len(traindataloader):.4f} "
               f"- Dev WER: {dev_wer_results['wer']:.4f} - Best Dev WER: {best_wer:.4f} - Best epoch: {best_epoch+1} "
               f"- Learning Rate: {current_lr:.8f}")
        
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
            log_msg = (f"Early stopping triggered! No improvement for {patience} consecutive epochs. Best WER: {best_wer:.4f} - Best epoch: {best_epoch+1}" )
            
            with open(log_file, "a") as f:
                f.write(log_msg + "\n")

            break

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work_dir', dest='work_dir', default="./work_dir/test")
    parser.add_argument('--data_dir', dest='data_dir', default="/data/sharedData/Smartphone/")
    parser.add_argument('--mode', dest='mode', default="SI")
    parser.add_argument('--model', dest='model', default="base")
    parser.add_argument('--device', dest='device', default="0")
    parser.add_argument('--lr', dest='lr', default="0.00001")
    parser.add_argument('--num_epochs', dest='num_epochs', default="300")

    args=parser.parse_args()
    args.lr = float(args.lr)
    args.num_epochs = int(args.num_epochs)
    
    main(args)

'''

SI : Task-1 : python main.py --work_dir ./work_dir/base_SI --model base --mode SI --num_epochs 5

                python main.py --work_dir ./work_dir/llm_SI --model llm --mode SI --num_epochs 150 --device 0

                python main.py --work_dir ./work_dir/slowfast_SI --model slowfast --mode SI --num_epochs 100 --device 0

                python main.py --work_dir ./work_dir/llm_advslowfast_SI --model llm_advslowfast --mode SI --num_epochs 100 --device 0



US : Task-2 : python main.py --work_dir ./work_dir/base_US --model base --mode US --num_epochs 5

                python main.py --work_dir ./work_dir/llm_US --model llm --mode US --num_epochs 150 --device 1

                python main.py --work_dir ./work_dir/slowfast_US --model slowfast --mode US --num_epochs 100 --device 1

                python main.py --work_dir ./work_dir/llm_advslowfast_US --model llm_advslowfast --mode US --num_epochs 100 --device 1


''' 