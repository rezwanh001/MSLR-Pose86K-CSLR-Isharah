import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import random
from torch.nn.utils.rnn import pad_sequence
from configs import SIConfig, USConfig

class PoseDataset(Dataset):
    def __init__(self, pose_path, annotation_path, mode, split_type, vocab=None, augment=False):
        self.split_type = split_type
        self.augment = augment
        
        self.mode = mode
        if mode == 'SI':
            cfg = SIConfig()
        elif mode == 'US':
            cfg = USConfig()
        else:
            raise ValueError("Invalid mode. Choose either 'SI' or 'US'.") 
        
        self.max_frames = cfg.MAX_FRAMES

        # Load pose data
        with open(pose_path, 'rb') as f:
            self.pose_dict = pickle.load(f)
        self.pose_dict = {str(k): v for k, v in self.pose_dict.items()}  # Ensure keys are strings

        if split_type == 'test' or annotation_path is None:
            # For test, use all pose_dict keys as IDs
            self.ids = list(self.pose_dict.keys())
            self.annotations = None
        else:
            # Load annotations
            self.annotations = pd.read_csv(annotation_path, delimiter='|')
            self.ids = [str(i) for i in self.annotations['id'] if str(i) in self.pose_dict]

        print(f"Loaded {len(self.ids)} valid {split_type} samples.")
        print(f"First 5 {split_type} IDs: {self.ids[:5]}")
        print(f"First 5 pose_dict keys: {list(self.pose_dict.keys())[:5]}")
        
        # Build vocabulary with special tokens
        if vocab is None and split_type != 'test':
            self.vocab = self._build_vocab()
            # Add special tokens
            self.vocab['<unk>'] = len(self.vocab) + 1  # Unknown token
            self.vocab['<pad>'] = 0  # Padding token
        else:
            self.vocab = vocab

    def _build_vocab(self):
        glosses = set()
        for gloss in self.annotations['gloss']:
            if pd.notna(gloss):  # Handle NaN values
                glosses.update(gloss.split())
        return {g: i+1 for i, g in enumerate(sorted(glosses))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        pose_data = self.pose_dict[sample_id]['keypoints']
        
        # Process pose data
        processed_pose = self._process_pose(pose_data)

        # Ensure we return a 3D tensor (T, J*2, 2) -> (T, J*2*2)
        processed_pose = processed_pose.reshape(processed_pose.shape[0], -1)  # Flatten spatial dimensions
        
        # Get gloss labels with unknown handling
        if self.split_type != 'test':
            gloss = self.annotations[self.annotations['id'] == sample_id]['gloss'].values[0]
            if pd.isna(gloss):  # Handle missing glosses
                gloss = ""
            label = [self.vocab.get(g, self.vocab['<unk>']) for g in gloss.split()]
            return sample_id, processed_pose, torch.tensor(label)
        return sample_id, processed_pose

    def _process_pose(self, pose):
        # Select keypoints: [right_hand, left_hand, lips, body]
        right_hand = pose[:, 0:21, :2]
        left_hand = pose[:, 21:42, :2]
        lips = pose[:, 42:42+19, :2]
        body = pose[:, 42+19:, :2]
        
        # Normalization
        def normalize(joints):
            if np.sum(joints) == 0: return joints
            joints -= joints[0]
            max_val = np.max(np.abs(joints))
            return joints / max_val if max_val > 0 else joints
        
        right_hand = np.array([normalize(f) for f in right_hand])
        left_hand = np.array([normalize(f) for f in left_hand])
        lips = np.array([normalize(f) for f in lips])
        body = np.array([normalize(f) for f in body])
        
        # Concatenate features
        concatenated = np.concatenate(
            [right_hand, left_hand, lips, body], 
            axis=1
        )
        
        # Temporal cropping/padding
        T = concatenated.shape[0]
        if T > self.max_frames:
            if self.augment:
                start = np.random.randint(0, T - self.max_frames)
            else:
                start = (T - self.max_frames) // 2
            concatenated = concatenated[start:start+self.max_frames]
        elif T < self.max_frames:
            pad_len = self.max_frames - T
            concatenated = np.pad(concatenated, ((0, pad_len), (0, 0), (0, 0)))
        
        return torch.tensor(concatenated).float()

def collate_fn(batch):
    # If batch items have 3 elements, it's train/val mode
    if len(batch[0]) == 3:
        ids, poses, labels = zip(*batch)
        poses_padded = torch.stack([p.float() for p in poses])
        label_lengths = torch.tensor([len(l) for l in labels])
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
        return ids, poses_padded, labels_padded, label_lengths
    else:
        # Inference/test mode: (id, pose)
        ids, poses = zip(*batch)
        poses_padded = torch.stack([p.float() for p in poses])
        return ids, poses_padded