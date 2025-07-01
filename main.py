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

from models.transformer import CSLRTransformer, CSLRWithLLaMA, CSLRWithOPT, MixtureCSLRLLM, SlowFastCSLR
from models.transformer import SlowFastCSLRWithLLM, CSLRDETR
from models.llm_based_model import AdvancedSlowFastLLMCSLR, LLMEnhancedPoseCSLR
from models.gcnn import GNNCSLRTransformer
from models.pretrain_model import PretrainedSlowFastCSLR
from models.new_model import SOTA_CSLR

MODELS = {
    "base": CSLRTransformer,
    "llama": CSLRWithLLaMA, 
    "opt": CSLRWithOPT,
    "mixllama": MixtureCSLRLLM, ## LLaMA-Former
    "slowfast": SlowFastCSLR,
    "slowfastllm": SlowFastCSLRWithLLM, ## LLaMA-SlowFast
    "llm_advslowfast": AdvancedSlowFastLLMCSLR, ## LLM-SlowFast
    "llm_PoseCSLRT": LLMEnhancedPoseCSLR,
    "gnncslr": GNNCSLRTransformer,
    "pretrainedslowfast": PretrainedSlowFastCSLR,
    "detr": CSLRDETR,
    "SOTA_CSLR": SOTA_CSLR
}


# Add create_edge_index function at the top of main.py
def create_edge_index():
    # [Previous edge_index definition from above]
    right_hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20]]
    right_hand_edges = torch.tensor(right_hand_edges, dtype=torch.long).t()

    left_hand_edges = [[i + 21, j + 21] for i, j in right_hand_edges.t().tolist()]
    left_hand_edges = torch.tensor(left_hand_edges, dtype=torch.long).t()

    lip_edges = [[i, i + 1] for i in range(42, 60)]
    lip_edges[-1][1] = 42  # Loop back
    lip_edges = torch.tensor(lip_edges, dtype=torch.long).t()

    body_edges = [[61, 62], [62, 63], [63, 64], [64, 65], [64, 66], [65, 67], [67, 68], [68, 69],
                    [66, 70], [70, 71], [71, 72], [64, 73], [73, 74], [74, 75], [64, 76], [76, 77],
                    [77, 78], [73, 79], [76, 80], [79, 81], [80, 82], [81, 83], [82, 84], [83, 85]]
    body_edges = torch.tensor(body_edges, dtype=torch.long).t()

    edge_index = torch.cat([right_hand_edges, left_hand_edges, lip_edges, body_edges], dim=1)
    return edge_index

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

    return total_loss, current_lr


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


    if getattr(args, "train_on_all", False):
        # Combine train and dev CSVs into one
        print("[✔] Combine train and dev CSVs into one!")
        train_csv = f"./annotations_v2/{args.mode}/train.txt"
        dev_csv = f"./annotations_v2/{args.mode}/dev.txt"
        combined_csv = f"./annotations_v2/{args.mode}/train_plus_dev.txt"

        # Create combined CSV if it doesn't exist
        if not os.path.exists(combined_csv):
            with open(combined_csv, "w") as outfile:
                for fname in [train_csv, dev_csv]:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)

        # Use combined CSV for training, skip validation
        train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc("isharah", combined_csv, dev_csv)
        dataset_train = PoseDatasetV2("isharah", combined_csv, "train", train_processed, augmentations=True, transform=transforms.Compose([GaussianNoise()]))
        traindataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=10)

        dataset_dev = PoseDatasetV2("isharah", dev_csv , "dev", dev_processed, augmentations=False)
        devdataloader = DataLoader(dataset_dev, batch_size=1, shuffle=False, num_workers=10)
    else:
        train_csv = f"./annotations_v2/{args.mode}/train.txt"
        dev_csv = f"./annotations_v2/{args.mode}/dev.txt"

        train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc("isharah", train_csv, dev_csv)

        dataset_train = PoseDatasetV2("isharah", train_csv , "train", train_processed , augmentations=True , transform=transforms.Compose([GaussianNoise()]))
        dataset_dev = PoseDatasetV2("isharah", dev_csv , "dev", dev_processed, augmentations=False)
        traindataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=10)
        devdataloader = DataLoader(dataset_dev, batch_size=1, shuffle=False, num_workers=10)

    # Define edge_index
    edge_index = create_edge_index()  # Use the function defined above
    
    if args.model in ["slowfast", "slowfastllm", "llm_advslowfast", "llm_PoseCSLRT", "pretrainedslowfast"]:
        # model = SlowFastCSLR(input_dim=172, ...)  # not 86!
        model = MODELS[args.model](
            input_dim=86*2, 
            num_classes=len(vocab_map) 
        ).to(device)

    elif args.model in ["SOTA_CSLR"]:
        model = MODELS[args.model](
            vocab_size=1000,
            HIDDEN_SIZE=512
        ).to(device)


    elif args.model in ["gnncslr"]:
        model = MODELS[args.model](
            input_dim=86,
            num_classes=len(vocab_map),
            edge_index=edge_index
        ).to(device)
    else:
        model = MODELS[args.model](
            input_dim=86, 
            num_classes=len(vocab_map)
        ).to(device)

    decoder_dec = Decode(vocab_map, len(vocab_list), 'beam', inv_vocab_map=inv_vocab_map)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(traindataloader),
        epochs=args.num_epochs,
        anneal_strategy='cos'
    )

    loss_encoder = nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')

    log_file = f"{args.work_dir}/training_log.txt"
    # if os.path.exists(log_file):
    #     os.remove(log_file)

    start_epoch = 0
    best_wer = float("inf") 
    best_epoch = 0
    patience = 100
    patience_counter = 0

    checkpoint_path = f"{args.work_dir}/best_model.pt"

    # if getattr(args, "resume", False) and os.path.exists(checkpoint_path):
    #     print(f"Resuming from checkpoint: {checkpoint_path}")
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     # Optionally, load optimizer state if you saved it
    #     # optimizer.load_state_dict(torch.load(f"{args.work_dir}/best_optimizer.pt", map_location=device))
    #     # Optionally, load other states (best_wer, best_epoch, patience_counter) from a file
    #     # If you saved them, load here and set start_epoch = best_epoch + 1

    if getattr(args, "resume", False) and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_wer = checkpoint.get('best_wer', float("inf"))
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        start_epoch = best_epoch + 1  # Optionally resume from next epoch

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("Model device:", next(model.parameters()).device)

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        print(f"\n\nEpoch [{epoch+1}/{start_epoch+args.num_epochs}]")
        train_loss, current_lr = train_epoch(model, traindataloader, optimizer, loss_encoder, device)
        dev_wer_results = evaluate_model(model, devdataloader, decoder_dec, device, inv_vocab_map, args.work_dir, epoch)
        scheduler.step(dev_wer_results['wer'])

        if dev_wer_results['wer'] < best_wer:
            best_wer = dev_wer_results['wer']
            best_epoch = epoch
            patience_counter = 0
            # torch.save(model.state_dict(), f"{args.work_dir}/best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wer': best_wer,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                # ... add more if needed
            }, f"{args.work_dir}/best_model.pt")
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
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume training from best_model.pt')
    parser.add_argument('--train_on_all', action='store_true', help='Train on train+dev for final model')

    args=parser.parse_args()
    args.lr = float(args.lr)
    args.num_epochs = int(args.num_epochs)
    
    main(args)

'''
python main.py --work_dir ./work_dir/base_SI --data_dir ./data --mode SI --model base --device 0 --lr 0.0001 --num_epochs 200



python main.py --work_dir ./work_dir/base_SI --data_dir ./data --mode SI --model base --device 0 --lr 0.0001 --num_epochs 200


python main.py --work_dir ./work_dir/llama_SI --data_dir ./data --mode SI --model llama --device 0 --lr 0.00001 --num_epochs 250


python main.py --work_dir ./work_dir/opt_SI --data_dir ./data --mode SI --model opt --device 0 --lr 0.00001 --num_epochs 150


python main.py --work_dir ./work_dir/mixllama_SI --data_dir ./data --mode SI --model mixllama --device 0 --lr 0.00001 --num_epochs 200 --resume


python main.py --work_dir ./work_dir/pretrainedslowfast_SI --data_dir ./data --mode SI --model pretrainedslowfast --device 0 --lr 0.00001 --num_epochs 150


================================
python main.py --work_dir ./work_dir/detr_SI --data_dir ./data --mode SI --model detr --device 0 --lr 0.00001 --num_epochs 25

ps -p 3313654 -o cmd

python main.py --work_dir ./work_dir/detr_US --data_dir ./data --mode US --model detr --device 1 --lr 0.00001 --num_epochs 25

================================


python main.py --work_dir ./work_dir/slowfastllm_SI --data_dir ./data --mode SI --model slowfastllm --device 0 --lr 0.00001 --num_epochs 100


python main.py --work_dir ./work_dir/llm_PoseCSLRT_SI --data_dir ./data --mode SI --model llm_PoseCSLRT --device 0 --lr 0.000003 --num_epochs 100


####

gnncslr

python main.py --work_dir ./work_dir/gnncslr_SI --data_dir ./data --mode SI --model gnncslr --device 0 --lr 0.000003 --num_epochs 100

------------------------------------

python main.py --work_dir ./work_dir/base_US --data_dir ./data --mode US --model base --device 1 --lr 0.0001 --num_epochs 200

python main.py --work_dir ./work_dir/llama_US --data_dir ./data --mode US --model llama --device 1 --lr 0.00001 --num_epochs 250

python main.py --work_dir ./work_dir/opt_US --data_dir ./data --mode US --model opt --device 1 --lr 0.00001 --num_epochs 150


python main.py --work_dir ./work_dir/mixllama_US --data_dir ./data --mode US --model mixllama --device 1 --lr 0.00001 --num_epochs 300


python main.py --work_dir ./work_dir/mixllama_US --data_dir ./data --mode US --model mixllama --device 1 --lr 0.000003 --num_epochs 100 --resume


python main.py --work_dir ./work_dir/slowfastllm_US --data_dir ./data --mode US --model slowfastllm --device 1 --lr 0.00001 --num_epochs 100

------------------------------------------------------------------------------------------------------

python main.py --work_dir ./work_dir/slowfast_US --data_dir ./data --mode US --model slowfast --device 1 --lr 0.000003 --num_epochs 150 --resume


#########====== llm_advslowfast

python main.py --work_dir ./work_dir/llm_advslowfast_SI --data_dir ./data --mode SI --model llm_advslowfast --device 0 --lr 0.000003 --num_epochs 100 --resume


python main.py --work_dir ./work_dir/llm_advslowfast_US --data_dir ./data --mode US --model llm_advslowfast --device 1 --lr 0.000003 --num_epochs 100 --resume


===============================================================================================================================
SOTA_CSLR


python main.py --work_dir ./work_dir/SOTA_CSLR_SI --data_dir ./data --mode SI --model SOTA_CSLR --device 0 --num_epochs 120



'''