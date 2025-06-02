import os
import torch
import numpy as np
import argparse
import pickle
import csv
import pandas as pd
import zipfile
from collections import Counter
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.decode import Decode
from utils.text_ctc_utils import convert_text_for_ctc
from utils.datasetv2 import PoseDatasetV2  # Modified dataset for test evaluation

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

class TestPoseDataset(PoseDatasetV2):
    """Modified dataset for test evaluation without ground-truth labels"""
    def __init__(self, mode, max_frames=512):
        self.mode = mode
        self.max_frames = max_frames
        self.split_type = "test"
        self.additional_joints = True
        
        # Load test data based on mode
        pkl_path = {
            "SI": "/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/data/public_si_dat/pose_data_isharah1000_SI_test/pose_data_isharah1000_SI_test.pkl",
            "US": "/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/data/public_us_dat/pose_data_isharah1000_US_test/pose_data_isharah1000_US_test.pkl"
        }[mode]
        
        with open(pkl_path, 'rb') as f:
            self.pose_dict = pickle.load(f)
        
        self.files = list(self.pose_dict.keys())
        print(f"Loaded {len(self.files)} test samples for {mode} mode")

    def __getitem__(self, idx):
        file_id = self.files[idx]
        pose_data = self.pose_dict[file_id]['keypoints']
        
        # Preprocess pose
        processed_pose = self.preprocess_pose(pose_data)
        processed_pose = self.pad_or_crop_sequence(processed_pose)
        return file_id, torch.from_numpy(processed_pose).float()

    def preprocess_pose(self, pose_data):
        """Preprocess pose data without augmentation"""
        T, J, D = pose_data.shape
        right_hand = pose_data[:, 0:21, :2]
        left_hand = pose_data[:, 21:42, :2]
        lips = pose_data[:, 42:42+19, :2]
        body = pose_data[:, 42+19:]
        
        # Normalize and process joints
        right_joints = [self.normalize_joints(rh) for rh in right_hand]
        left_joints = [self.normalize_joints(lh) for lh in left_hand]
        face_joints = [self.normalize_joints(fc) for fc in lips]
        body_joints = [self.normalize_joints(bd) for bd in body]
        
        # Handle missing data
        for i in range(1, T):
            if np.sum(right_joints[i]) == 0:
                right_joints[i] = right_joints[i-1]
            if np.sum(left_joints[i]) == 0:
                left_joints[i] = left_joints[i-1]
                
        return np.concatenate((right_joints, left_joints, face_joints, body_joints), axis=1)

    def normalize_joints(self, joints):
        """Normalize joint coordinates"""
        if np.sum(joints) == 0:
            return joints
        joints = joints - joints[0]
        joints = joints - np.min(joints, axis=0)
        max_val = np.max(joints)
        if max_val > 0:
            joints = joints / max_val
        return joints

    def pad_or_crop_sequence(self, sequence):
        """Ensure consistent sequence length"""
        T, J, D = sequence.shape
        if T < 32:
            padded = np.zeros((32, J, D))
            padded[:T] = sequence
            return padded
        return sequence[:self.max_frames]
    
def process_submission_file(submission_file_path, mode="SI"):
    """
    Cleans and prepares the submission CSV file for final evaluation.
    - Removes 'tensor(...)' wrappers in ID fields.
    - Fills blank or missing gloss fields with the most frequent gloss.
    - Saves as 'test.csv' and compresses into 'test.zip'.
    
    Args:
        submission_file_path (str): Path to the raw submission CSV file.
    """
    # Load the submission CSV
    df = pd.read_csv(submission_file_path)

    # Convert 'id' column from tensor format to integer
    df["id"] = df["id"].astype(str).str.extract(r"tensor\((\d+)\)").astype(int)

    # Fill blank or NaN glosses with the most common valid gloss
    non_empty_glosses = df["gloss"].dropna().astype(str)
    non_empty_glosses = non_empty_glosses[non_empty_glosses.str.strip() != ""]
    
    if len(non_empty_glosses) == 0:
        most_common_gloss = "default_gloss"
    else:
        most_common_gloss = Counter(non_empty_glosses).most_common(1)[0][0]

    df["gloss"] = df["gloss"].apply(lambda x: most_common_gloss if pd.isna(x) or str(x).strip() == "" else x)

    # Save the cleaned CSV
    if mode == "SI":
        test_csv_path = "./submission/task-1/test.csv"
    elif mode == "US":
        test_csv_path = "./submission/task-2/test.csv"
    df.to_csv(test_csv_path, index=False)

    # Create a ZIP archive
    if mode == "SI":
        zip_path = "./submission/task-1/test.zip"
    elif mode == "US":
        zip_path = "./submission/task-2/test.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(test_csv_path, arcname="test.csv")

    print(f"Submission file cleaned and zipped at: {zip_path}")
    print("===================================================================")

def main(args):
    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") 

    # Rebuild vocabulary from training data
    train_csv = f"./annotations_v2/{args.mode}/train.txt"
    dev_csv = f"./annotations_v2/{args.mode}/dev.txt"
    _, _, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc(
        "isharah", train_csv, dev_csv
    )
    
    # Initialize test dataset and loader
    test_dataset = TestPoseDataset(mode=args.mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    # model = AdvancedSlowFastLLMCSLR(
    #     input_dim=86*2,
    #     num_classes=len(vocab_map)
    # ).to(device)

    model = MODELS[args.model](
        input_dim=86*2, 
        num_classes=len(vocab_map)
    ).to(device)
    
    # Load trained weights
    model_path = os.path.join(args.work_dir, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize decoder
    decoder = Decode(vocab_map, len(vocab_list), 'beam')
    
    # Output files
    submission_file = os.path.join(args.work_dir, f"{args.mode}_test_submission.csv")
    predictions_file = os.path.join(args.work_dir, f"{args.mode}_test_predictions.txt")
    
    with open(submission_file, 'w') as sub_f, open(predictions_file, 'w') as pred_f:
        csv_writer = csv.writer(sub_f)
        csv_writer.writerow(['id', 'gloss'])
        
        for file_ids, poses in tqdm(test_loader, desc="Evaluating"):
            poses = poses.to(device)
            
            # Model inference
            with torch.no_grad():
                logits = model(poses)
                vid_lgt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(device)
                decoded_list = decoder.decode(logits, vid_lgt=vid_lgt, batch_first=True, probs=False)
            
            # Process predictions
            for file_id, pred in zip(file_ids, decoded_list):
                gloss_seq = ' '.join([gloss for gloss, _ in pred])
                csv_writer.writerow([file_id, gloss_seq])
                pred_f.write(f"{file_id}|{gloss_seq}\n")
    
    print(f"Test predictions saved to {submission_file}")
    print(f"Detailed results saved to {predictions_file}")

    # Process submission file
    process_submission_file(submission_file, args.mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True, help="Directory containing trained model")
    parser.add_argument('--mode', choices=['SI', 'US'], required=True, help="Evaluation mode")
    parser.add_argument('--model', default="llm_advslowfast", help="Model architecture")
    parser.add_argument('--device', dest='device', default="0")
    args = parser.parse_args()
    
    main(args)

""" 

# Evaluate SI test set
python test_evaluate.py --work_dir ./work_dir/llm_advslowfast_SI --model llm_advslowfast --mode SI --device 0

# Evaluate US test set
python test_evaluate.py --work_dir ./work_dir/llm_advslowfast_US --model llm_advslowfast --mode US --device 1

"""