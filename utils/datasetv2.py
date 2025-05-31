# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import glob
# import pandas as pd
# import math
# import random
# from random import randrange
# import cv2
# import pickle

# lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 291]
# lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
# lips = sorted(set(lipsUpperOuter + lipsLowerOuter))  # Remove duplicates and sort
# NUM_LIPS = 19  # after deduplication
        
# class PoseDatasetV2(Dataset):
#     def rotate(self, origin, point, angle):
#         """ Rotates a point around an origin. """
#         ox, oy = origin
#         px, py = point
#         qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
#         qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
#         return qx, qy

#     def augment_jitter(self, keypoints, std_dev=0.01):
#         """ Adds Gaussian noise to keypoints. """
#         noise = np.random.normal(loc=0, scale=std_dev, size=keypoints.shape)
#         return keypoints + noise

#     def augment_time_warp(self, pose_data, max_shift=2):
#         """ Randomly shifts frames to simulate varying signing speed. """
#         T = pose_data.shape[0]
#         new_data = np.zeros_like(pose_data)
#         for i in range(T):
#             shift = np.random.randint(-max_shift, max_shift + 1)
#             new_idx = np.clip(i + shift, 0, T - 1)
#             new_data[i] = pose_data[new_idx]
#         return new_data

#     def augment_dropout(self, keypoints, drop_prob=0.1):
#         """ Randomly drops some keypoints. """
#         mask = np.random.rand(*keypoints.shape[:1]) > drop_prob
#         keypoints *= mask[:, np.newaxis]
#         return keypoints

#     def augment_scale(self, keypoints, scale_range=(0.8, 1.2)):
#         """ Randomly scales keypoints."""
#         scale = np.random.uniform(*scale_range)
#         return keypoints * [scale, scale]

#     def augment_frame_dropout(self, pose_data, drop_prob=0.1):
#         """ Randomly drops full frames. """
#         T = pose_data.shape[0]
#         mask = np.random.rand(T) > drop_prob
    
#         return pose_data * mask[:, np.newaxis, np.newaxis]

#     def augment_data(self, data, angle=None):
#         if np.random.rand() < 0.5:
#                 data = np.array([self.rotate((0.5, 0.5), frame, angle) for frame in data])
#         if np.random.rand() < 0.5:
#                 data = self.augment_jitter(data)
#         if np.random.rand() < 0.5:
#                 data = self.augment_scale(data)
#         if np.random.rand() < 0.5:
#                 data = self.augment_dropout(data)
#         return data


#     def normalize(self, pose):
#         #set coordinate frame as the wrist
#         pose[:,:] -= pose[0]  
#         pose[:,:] -= np.min(pose, axis=0)
        
#         #scale them to a box of 1x1
#         max_vals = np.max(pose, axis=0)
#         pose[:,:] /= max(max_vals)

#         # Subtract the mean from each element and divide by the maximum absolute value
#         # The values are then [-0.5,0.5] spread over zero 
#         pose[:,:] = pose[:,:] - np.mean(pose[:,:])
#         pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
#         pose[:,:] = pose[:,:] * 0.5

#         return pose

#     def normalize_face(self, pose):
#         #set coordinate frame as the lip
#         pose[:,:] -= pose[0]  
#         pose[:,:] -= np.min(pose, axis=0)
        
#         #scale them to a box of 1x1
#         max_vals = np.max(pose, axis=0)
#         pose[:,:] /= max(max_vals)

#         # Subtract the mean from each element and divide by the maximum absolute value
#         # The values are then [-0.5,0.5] spread over zero 
#         pose[:,:] = pose[:,:] - np.mean(pose[:,:])
#         pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
#         pose[:,:] = pose[:,:] * 0.5

#         return pose

    
#     def normalize_body(self, pose ):
#         #set coordinate frame as the neck
#         pose[:,:] -= pose[0]
#         pose[:,:] -= np.min(pose, axis=0)
        
#         #scale them to a box of 1x1
#         max_vals = np.max(pose, axis=0)
#         pose[:,:] /= max(max_vals)

#         # Subtract the mean from each element and divide by the maximum absolute value
#         # The values are then [-0.5,0.5] spread over zero 
#         pose[:,:] = pose[:,:] - np.mean(pose[:,:])
#         pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
#         pose[:,:] = pose[:,:] * 0.5

#         return pose

    

#     def __init__(self, dataset_name2, label_csv, split_type, target_enc_df, transform=None, augmentations=True, augmentations_prob=0.5, additional_joints=True):

#         self.dataset_name = dataset_name2
#         self.split_type = split_type  # "train", "dev", "test"
#         self.transform = transform
#         self.augmentations = augmentations
#         self.augmentations_prob = augmentations_prob
#         self.additional_joints = additional_joints
        
#         with open(os.path.join("./data/pose_data_isharah1000_hands_lips_body_May12.pkl"), 'rb') as f:
#             self.pose_dict = pickle.load(f)

#         self.files = []
#         self.labels = []

#         self.all_data = pd.read_csv(label_csv, delimiter="|")
#         if "isharah" in self.dataset_name:
#             self.all_data = self.all_data[self.all_data["id"].notna()]
#             self.all_data = self.all_data[self.all_data["gloss"].notna()]

#         for _, row in self.all_data.iterrows():
#             sample_id = str(row["id"])
#             enc_label = target_enc_df[target_enc_df["id"] == sample_id]["enc"]

#             if not enc_label.empty and sample_id in self.pose_dict.keys():
#                 self.files.append(sample_id)
#                 self.labels.append(enc_label.iloc[0])

#         print(f"Loaded {len(self.files)} samples for split: {split_type}")

#     def __len__(self):
#         return len(self.files)

#     def get_file_path(self, idx):
#         return self.files[idx]

#     def readPose(self, sample_id):
#         pose_data = self.pose_dict[sample_id]['keypoints']

#         if pose_data is None or pose_data.shape[0] == 0:
#             raise ValueError(f"Error loading pose data for {sample_id}")

#         T, J, D = pose_data.shape
#         aug = False

#         if self.augmentations and np.random.rand() < self.augmentations_prob:
#             aug = True
#             angle = np.radians(np.random.uniform(-13, 13))
#             pose_data = self.augment_time_warp(pose_data)
#             pose_data = self.augment_frame_dropout(pose_data)
        
#         right_hand = pose_data[:, 0:21, :2]
#         left_hand = pose_data[:, 21:42, :2]
#         lips = pose_data[:, 42:42+NUM_LIPS, :2]
#         body = pose_data[:,42+NUM_LIPS:]

#         right_joints, left_joints, face_joints, body_joints = [], [], [], []

#         for ii in range(T):
#             rh = right_hand[ii]
#             lh = left_hand[ii]
#             fc = lips[ii]
#             bd = body[ii]

#             if rh.sum() == 0:
#                 rh[:] = right_joints[-1] if ii != 0 else np.zeros((21, 2))
#             else:
#                 if aug:
#                     rh = self.augment_data(rh, angle)
#                 rh = self.normalize(rh)

#             if lh.sum() == 0:
#                 lh[:] = left_joints[-1] if ii != 0 else np.zeros((21, 2))
#             else:
#                 if aug:
#                     lh = self.augment_data(lh, angle)
#                 lh = self.normalize(lh)

#             if fc.sum() == 0:
#                 fc[:] = face_joints[-1] if ii != 0 else np.zeros((len(fc), 2))
#             else:
#                 fc = self.normalize_face(fc)
            
#             if bd.sum() == 0:
#                 bd[:] = body_joints[-1] if ii != 0 else np.zeros((len(bd), 2))
#             else:
#                 bd = self.normalize_body(bd)

#             right_joints.append(rh)
#             left_joints.append(lh)
#             face_joints.append(fc)
#             body_joints.append(bd)

#         for ljoint_idx in range(len(left_joints) - 2, -1, -1):
#             if left_joints[ljoint_idx].sum() == 0:
#                 left_joints[ljoint_idx] = left_joints[ljoint_idx + 1].copy()

#         for rjoint_idx in range(len(right_joints) - 2, -1, -1):
#             if right_joints[rjoint_idx].sum() == 0:
#                 right_joints[rjoint_idx] = right_joints[rjoint_idx + 1].copy()

#         concatenated_joints = np.concatenate((right_joints, left_joints), axis=1)
#         if  self.additional_joints :
#             concatenated_joints = np.concatenate((concatenated_joints, face_joints,body_joints), axis=1)
#         return concatenated_joints


#     def __len__(self):
#         return len(self.files)
    
#     def get_file_path(self, idx):
#         return self.files[idx]
    
#     def pad_or_crop_sequence(self, sequence, min_len=32, max_len=1000):
#         T, J, D = sequence.shape
#         if T < min_len:
#             pad_len = min_len - T
#             pad = np.zeros((pad_len, J, D))
#             sequence = np.concatenate((sequence, pad), axis=0)
#             T = sequence.shape[0]  # update T
#         if sequence.shape[0] > max_len:
#             sequence = sequence[:max_len]

#         return sequence
        
#     def __getitem__(self, idx):
#         file_path = os.path.join(self.files[idx])
#         pose = self.readPose(file_path)
#         pose = self.pad_or_crop_sequence(pose, min_len=32, max_len=1000)
#         pose = torch.from_numpy(pose).float()

#         if self.transform:
#             pose = self.transform(pose)

#         label = self.labels[idx]
#         return file_path, pose, torch.as_tensor(label)



import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
import math
import random
from random import randrange
import cv2
import pickle

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lips = sorted(set(lipsUpperOuter + lipsLowerOuter))
NUM_LIPS = 19

class PoseDatasetV2(Dataset):
    def rotate(self, origin, point, angle):
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def augment_jitter(self, keypoints, std_dev=0.01):
        noise = np.random.normal(loc=0, scale=std_dev, size=keypoints.shape)
        return keypoints + noise

    def augment_time_warp(self, pose_data, max_shift=2):
        T = pose_data.shape[0]
        new_data = np.zeros_like(pose_data)
        for i in range(T):
            shift = np.random.randint(-max_shift, max_shift + 1)
            new_idx = np.clip(i + shift, 0, T - 1)
            new_data[i] = pose_data[new_idx]
        return new_data

    def augment_dropout(self, keypoints, drop_prob=0.1):
        mask = np.random.rand(*keypoints.shape[:1]) > drop_prob
        keypoints *= mask[:, np.newaxis]
        return keypoints

    def augment_scale(self, keypoints, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(*scale_range)
        return keypoints * [scale, scale]

    def augment_frame_dropout(self, pose_data, drop_prob=0.1):
        T = pose_data.shape[0]
        mask = np.random.rand(T) > drop_prob
        return pose_data * mask[:, np.newaxis, np.newaxis]

    def augment_data(self, data, angle=None):
        if np.random.rand() < 0.5:
            data = np.array([self.rotate((0.5, 0.5), frame, angle) for frame in data])
        if np.random.rand() < 0.5:
            data = self.augment_jitter(data)
        if np.random.rand() < 0.5:
            data = self.augment_scale(data)
        if np.random.rand() < 0.5:
            data = self.augment_dropout(data)
        return data

    def normalize(self, pose):
        pose[:, :] -= pose[0]
        pose[:, :] -= np.min(pose, axis=0)
        max_vals = np.max(pose, axis=0)
        pose[:, :] /= max(max_vals)
        pose[:, :] = pose[:, :] - np.mean(pose[:, :])
        pose[:, :] = pose[:, :] / np.max(np.abs(pose[:, :]))
        pose[:, :] = pose[:, :] * 0.5
        return pose

    def normalize_face(self, pose):
        pose[:, :] -= pose[0]
        pose[:, :] -= np.min(pose, axis=0)
        max_vals = np.max(pose, axis=0)
        pose[:, :] /= max(max_vals)
        pose[:, :] = pose[:, :] - np.mean(pose[:, :])
        pose[:, :] = pose[:, :] / np.max(np.abs(pose[:, :]))
        pose[:, :] = pose[:, :] * 0.5
        return pose

    def normalize_body(self, pose):
        pose[:, :] -= pose[0]
        pose[:, :] -= np.min(pose, axis=0)
        max_vals = np.max(pose, axis=0)
        pose[:, :] /= max(max_vals)
        pose[:, :] = pose[:, :] - np.mean(pose[:, :])
        pose[:, :] = pose[:, :] / np.max(np.abs(pose[:, :]))
        pose[:, :] = pose[:, :] * 0.5
        return pose

    def __init__(self, dataset_name2, label_csv, split_type, target_enc_df, transform=None, augmentations=True, augmentations_prob=0.5, additional_joints=True, max_frames=512):
        self.dataset_name = dataset_name2
        self.split_type = split_type
        self.transform = transform
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.additional_joints = additional_joints
        self.max_frames = max_frames

        with open(os.path.join("./data/pose_data_isharah1000_hands_lips_body_May12.pkl"), 'rb') as f:
            self.pose_dict = pickle.load(f)

        self.files = []
        self.labels = []

        self.all_data = pd.read_csv(label_csv, delimiter="|")
        if "isharah" in self.dataset_name:
            self.all_data = self.all_data[self.all_data["id"].notna()]
            self.all_data = self.all_data[self.all_data["gloss"].notna()]

        for _, row in self.all_data.iterrows():
            sample_id = str(row["id"])
            enc_label = target_enc_df[target_enc_df["id"] == sample_id]["enc"]
            if not enc_label.empty and sample_id in self.pose_dict.keys():
                self.files.append(sample_id)
                self.labels.append(enc_label.iloc[0])

        print(f"Loaded {len(self.files)} samples for split: {split_type}")

    def readPose(self, sample_id):
        pose_data = self.pose_dict[sample_id]['keypoints']
        if pose_data is None or pose_data.shape[0] == 0:
            raise ValueError(f"Error loading pose data for {sample_id}")

        T, J, D = pose_data.shape
        aug = False
        if self.augmentations and np.random.rand() < self.augmentations_prob:
            aug = True
            angle = np.radians(np.random.uniform(-13, 13))
            pose_data = self.augment_time_warp(pose_data)
            pose_data = self.augment_frame_dropout(pose_data)

        right_hand = pose_data[:, 0:21, :2]
        left_hand = pose_data[:, 21:42, :2]
        lips = pose_data[:, 42:42+NUM_LIPS, :2]
        body = pose_data[:, 42+NUM_LIPS:]

        right_joints, left_joints, face_joints, body_joints = [], [], [], []
        for ii in range(T):
            rh = right_hand[ii]
            lh = left_hand[ii]
            fc = lips[ii]
            bd = body[ii]
            if rh.sum() == 0:
                rh[:] = right_joints[-1] if ii != 0 else np.zeros((21, 2))
            else:
                if aug:
                    rh = self.augment_data(rh, angle)
                rh = self.normalize(rh)
            if lh.sum() == 0:
                lh[:] = left_joints[-1] if ii != 0 else np.zeros((21, 2))
            else:
                if aug:
                    lh = self.augment_data(lh, angle)
                lh = self.normalize(lh)
            if fc.sum() == 0:
                fc[:] = face_joints[-1] if ii != 0 else np.zeros((len(fc), 2))
            else:
                fc = self.normalize_face(fc)
            if bd.sum() == 0:
                bd[:] = body_joints[-1] if ii != 0 else np.zeros((len(bd), 2))
            else:
                bd = self.normalize_body(bd)
            right_joints.append(rh)
            left_joints.append(lh)
            face_joints.append(fc)
            body_joints.append(bd)

        for ljoint_idx in range(len(left_joints) - 2, -1, -1):
            if left_joints[ljoint_idx].sum() == 0:
                left_joints[ljoint_idx] = left_joints[ljoint_idx + 1].copy()
        for rjoint_idx in range(len(right_joints) - 2, -1, -1):
            if right_joints[rjoint_idx].sum() == 0:
                right_joints[rjoint_idx] = right_joints[rjoint_idx + 1].copy()

        concatenated_joints = np.concatenate((right_joints, left_joints), axis=1)
        if self.additional_joints:
            concatenated_joints = np.concatenate((concatenated_joints, face_joints, body_joints), axis=1)
        return concatenated_joints

    def pad_or_crop_sequence(self, sequence, min_len=32, max_len=512):
        T, J, D = sequence.shape
        if T < min_len:
            pad_len = min_len - T
            pad = np.zeros((pad_len, J, D))
            sequence = np.concatenate((sequence, pad), axis=0)
            T = sequence.shape[0]
        if sequence.shape[0] > max_len:
            sequence = sequence[:max_len]
        return sequence

    def __getitem__(self, idx):
        file_path = self.files[idx]
        pose = self.readPose(file_path)
        pose = self.pad_or_crop_sequence(pose, min_len=32, max_len=self.max_frames)
        pose = torch.from_numpy(pose).float()
        if self.transform:
            pose = self.transform(pose)
        label = self.labels[idx]
        return file_path, pose, torch.as_tensor(label)

    def __len__(self):
        return len(self.files)

    def get_file_path(self, idx):
        return self.files[idx]