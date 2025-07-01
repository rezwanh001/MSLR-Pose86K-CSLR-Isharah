import torch
torch.cuda.empty_cache()
import torch.nn as nn
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
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
import os
import numpy as np
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
    return cfg

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename):
    return torch.load(filename)

def train(mode='SI', model_type='SignLanguageRecognizer'):
    # Get configuration based on the mode
    cfg = get_config(mode, model_type)

    # Initialize datasets 
    train_set = PoseDataset(
        cfg.TRAIN_POSE_PATH,
        cfg.TRAIN_ANNOTATION_PATH,
        mode,
        'train',
        augment=True
    )
    dev_set = PoseDataset(
        cfg.DEV_POSE_PATH,
        cfg.DEV_ANNOTATION_PATH,
        mode,
        'dev',
        vocab=train_set.vocab
    )
    
    # Update vocab size in config
    cfg.VOCAB_SIZE = len(train_set.vocab)
    
    # Data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_set, 
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"Using {len(train_set)} training samples and {len(dev_set)} validation samples.")
    print(f"Vocabulary size: {cfg.VOCAB_SIZE}")
    # Model setup
    mamba_config = {
        'd_state': 12,
        'expand': 4,
        'd_conv': 4,
        'bidirectional': True
    }
    # Initialize model based on type
    if model_type == 'SignLanguageRecognizer': ## CNN-BiLSTM
        model = SignLanguageRecognizer(cfg.VOCAB_SIZE, cfg)
    elif model_type == 'ArabicMamba':
        model = ArabicMamba(cfg.VOCAB_SIZE, cfg, mamba_config)
    elif model_type == 'SignLanguageConformer': ## Sign-Conformer
        model = SignLanguageConformer(cfg.VOCAB_SIZE, cfg)

    elif model_type == 'AdvancedSignLanguageRecognizer': ### For Mode = US (where we got high performance) # (WER) val = 55.0847  & test = 47.7756 (Winner: 2nd)
        model = AdvancedSignLanguageRecognizer(cfg.VOCAB_SIZE, cfg)
        '''
        paper model name: ``Multi-Scale Fusion Transformer``
        '''

    elif model_type == 'MambaSignLanguageRecognizer':  ## Mamba-Sign
        model = MambaSignLanguageRecognizer(cfg.VOCAB_SIZE, cfg)

    elif model_type == 'SOTA_CSLR':  ### For Mode = SI (where we got high performance) # (WER) val = 7.3123  & test = 13.0652 (Winner: 4th)
        model = SOTA_CSLR(cfg.VOCAB_SIZE, cfg)
        '''
        paper model name: ``Signer-Invariant Conformer``
        '''
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Multi-GPU setup (before moving to device and optimizer initialization)
    # This is crucial: wrap with DataParallel *before* moving to device for optimizer setup
    if cfg.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # DataParallel expects the model to be on the default device or moved afterwards
        # For simplicity, we'll move the base model to the desired device first.
        # DataParallel will handle distributing to other GPUs.
        model.to(cfg.DEVICE)
        model = DataParallel(model, device_ids=cfg.GPU_IDS)
    else:
        model.to(cfg.DEVICE) # Move model to device if not using DataParallel or only one GPU

    print(f"Model type: {model_type}, Device: {cfg.DEVICE}")
    ######----------------------------------------------------------
    # if model_type in ['SignLanguageRecognizer', 'AdvancedSignLanguageRecognizer']: ## this is for US Mode
    # # if model_type in ['AdvancedSignLanguageRecognizer', 'ArabicMamba', 'MambaSignLanguageRecognizer']: ## this is for SI Mode
    #     optimizer = optim.AdamW(model.parameters(), lr=0.00004, weight_decay=1e-2)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='max', factor=0.5, patience=15
    #     )
    # else: 

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=cfg.EPOCHS,
        anneal_strategy='cos'
    )

    print(f"Optimizer: {type(optimizer).__name__}, Initial LR: {optimizer.param_groups[0]['lr']}, scheduler: {type(scheduler).__name__}")

    ######----------------------------------------------------------

    # # Optimizer selection
    # if args.optimizer == 'adamw':
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # elif args.optimizer == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # else:
    #     raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # # Scheduler selection
    # if args.scheduler == 'plateau':
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='max', factor=0.5, patience=15
    #     )
    # elif args.scheduler == 'onecycle':
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=0.001,
    #         steps_per_epoch=len(train_loader),
    #         epochs=cfg.EPOCHS,
    #         anneal_strategy='cos'
    #     )
    # else:
    #     raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # print(f"Optimizer: {type(optimizer).__name__}, Initial LR: {optimizer.param_groups[0]['lr']}, scheduler: {type(scheduler).__name__}")

    ######----------------------------------------------------------

    criterion = nn.CTCLoss(blank=0)  # 0 is our <pad> token index
    scaler = GradScaler(enabled=cfg.MIXED_PRECISION)

    start_epoch = 0
    best_accuracy = 0
    best_loss = float('inf')

    # Resume from checkpoint if exists
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        print(f"Found checkpoint at {cfg.MODEL_SAVE_PATH}, resuming training...")
        checkpoint = load_checkpoint(cfg.MODEL_SAVE_PATH)
        
        # If using DataParallel, load state_dict into the base model's module
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
            
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # IMPORTANT: Manually move optimizer state to the current device if they aren't already
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.DEVICE)

        best_accuracy = checkpoint.get('accuracy', 0)
        start_epoch = checkpoint.get('epoch', 1)
        print(f"Resumed from epoch {start_epoch}, best accuracy so far: {best_accuracy:.2f}%")

    # Training loop
    for epoch in range(start_epoch, start_epoch+cfg.EPOCHS): # Corrected range for clarity, assuming EPOCHS is total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{start_epoch+cfg.EPOCHS} - Current LR: {current_lr}")

        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # Training phase
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            ids, inputs, labels, label_lengths = batch
            inputs = inputs.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)
            label_lengths = label_lengths.to(cfg.DEVICE, non_blocking=True)
            
            with autocast(dtype=cfg.FP16_DTYPE, enabled=cfg.MIXED_PRECISION):
                # Forward pass in mixed precision
                outputs = model(inputs)
                
                # Prepare for CTC loss
                input_lengths = torch.full(
                    size=(inputs.size(0),), 
                    fill_value=outputs.size(1),
                    dtype=torch.long,
                    device=cfg.DEVICE
                )
                
                # Convert outputs to FP32 for CTC loss
                log_probs = torch.nn.functional.log_softmax(outputs.float(), dim=2)
                
                # Calculate loss in FP32
                loss = criterion(
                    log_probs.permute(1, 0, 2),  # (T, B, C)
                    labels,
                    input_lengths,
                    label_lengths
                ) / cfg.GRAD_ACCUM_STEPS
            
            # Backpropagation with gradient accumulation
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * cfg.GRAD_ACCUM_STEPS
        
        # Validation phase
        epoch_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, dev_loader, train_set.vocab, cfg)
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={val_accuracy:.2f}%')

        # Update scheduler
        scheduler.step(val_accuracy)

        # Save best model (accuracy improved)
        if val_accuracy >= best_accuracy:
            best_accuracy = max(best_accuracy, val_accuracy)
            best_loss = min(best_loss, epoch_loss)
            model_to_save = model.module if cfg.USE_MULTI_GPU else model
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vocab': train_set.vocab,
                'accuracy': val_accuracy,
                'loss': epoch_loss
            }, cfg.MODEL_SAVE_PATH)
            print(f"\033[92m Trained and saved new best model for {cfg.MODE} with accuracy: {val_accuracy:.2f}% and loss: {epoch_loss:.4f} at epoch {epoch+1}\033[0m")


            # ===== Save predictions for best epoch =====
            # Evaluate on dev set and save predictions
            predictions = []
            idx_to_gloss = {idx: gloss for gloss, idx in train_set.vocab.items()}
            model_to_eval = model.module if cfg.USE_MULTI_GPU else model
            model_to_eval.eval()
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc=f"Saving best predictions for epoch {epoch+1}"):
                    ids, inputs, labels, label_lengths = batch
                    inputs = inputs.to(cfg.DEVICE, non_blocking=True)
                    with autocast(dtype=cfg.FP16_DTYPE, enabled=cfg.MIXED_PRECISION):
                        outputs = model_to_eval(inputs)
                        log_probs = torch.nn.functional.log_softmax(outputs, dim=2).float()
                    predictions_batch = torch.argmax(log_probs, dim=2).cpu()
                    for i, sample_id in enumerate(ids):
                        pred_seq = []
                        prev_idx = -1
                        for idx in predictions_batch[i]:
                            if idx != prev_idx and idx != 0:
                                pred_seq.append(idx_to_gloss[idx.item()])
                            prev_idx = idx
                        predictions.append((sample_id, " ".join(pred_seq)))

            # Prepare output directory and filename
            pred_dir = os.path.join(
                os.path.dirname(cfg.PREDICTION_OUTPUT_PATH),  # e.g., outputs/predictions/SI
            )
            os.makedirs(pred_dir, exist_ok=True)
            pred_filename = f"best_pred_{mode}_{epoch+1}.txt"
            pred_path = os.path.join(pred_dir, pred_filename)
            with open(pred_path, "w") as f:
                f.write("id,gloss\n")
                for sample_id, gloss in predictions:
                    f.write(f"{sample_id},{gloss}\n")
            print(f"Best predictions for epoch {epoch+1} saved to {pred_path}")

        # Save checkpoint if loss or accuracy is less than or equal to previous
        if epoch_loss <= best_loss or val_accuracy > best_accuracy:
            model_to_save = model.module if cfg.USE_MULTI_GPU else model
            checkpoint_path = os.path.join(
                os.path.dirname(cfg.MODEL_SAVE_PATH),
                f"{mode}_{model_type}_checkpoint.pth"
            )
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vocab': train_set.vocab,
                'accuracy': val_accuracy,
                'loss': epoch_loss
            }, checkpoint_path)
            print(f"\033[93m [CheckPoint] Saved checkpoint for {mode} at epoch {epoch+1} (loss: {epoch_loss:.4f}, acc: {val_accuracy:.2f}%)\033[0m")

def evaluate(model, loader, vocab, cfg):
    model.eval()
    correct = total = 0
    idx_to_gloss = {idx: gloss for gloss, idx in vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            ids, inputs, labels, label_lengths = batch
            inputs = inputs.to(cfg.DEVICE, non_blocking=True)
            labels = labels.cpu()
            
            with autocast(dtype=cfg.FP16_DTYPE, enabled=cfg.MIXED_PRECISION):
                outputs = model(inputs)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2).float()
            
            # Greedy decoding
            predictions = torch.argmax(log_probs, dim=2).cpu()
            
            # Convert to gloss sequences
            for i in range(len(predictions)):
                pred_seq = []
                prev_idx = -1
                for idx in predictions[i]:
                    if idx != prev_idx and idx != 0:  # 0 is blank
                        pred_seq.append(idx_to_gloss[idx.item()])
                    prev_idx = idx
                
                # Get target glosses
                target_glosses = [idx_to_gloss[t.item()] for t in labels[i] if t.item() != 0]
                
                # Compare 
                if pred_seq == target_glosses:
                    correct += 1
                total += 1

    return 100 * correct / total if total > 0 else 0

