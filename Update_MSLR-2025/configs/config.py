import torch

class SIConfig:
    MODE = 'SI'
    # Data Paths
    TRAIN_POSE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/data/public_si_dat/pose_data_isharah1000_hands_lips_body_May12.pkl'
    TRAIN_ANNOTATION_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/SI/train.txt'
    DEV_POSE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/data/public_si_dat/pose_data_isharah1000_hands_lips_body_May12.pkl'
    DEV_ANNOTATION_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/SI/dev.txt'
    TEST_POSE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/SI/pose_data_isharah1000_SI_test.pkl'
    TEST_ANNOTATION_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/SI/SI_test.txt'

    # Model Parameters
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    VOCAB_SIZE = 1000  # Will be updated by dataset

    # Training Parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 200   ## <<==================
    BATCH_SIZE = 16
    LR = 0.001
    MAX_FRAMES = 256
    USE_MULTI_GPU = True
    MIXED_PRECISION = True
    GRAD_ACCUM_STEPS = 2
    FP16_DTYPE = torch.float16
    CTC_LOSS_FP32 = True

    # GPU Settings
    GPU_IDS = [0, 1]
    NUM_WORKERS = 8

    # Output Paths
    MODEL_SAVE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/models/SI/SI_best_model.pth'
    PREDICTION_OUTPUT_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/predictions/SI/SI_test_predictions.txt'
    DEV_PREDICTION_OUTPUT_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/predictions/SI/SI_dev_predictions.txt'


class USConfig:
    MODE = 'US'
    # Data Paths
    TRAIN_POSE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/data/public_us_dat/pose_data_isharah1000_hands_lips_body_May12.pkl'
    TRAIN_ANNOTATION_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/US/train.txt'
    DEV_POSE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/data/public_us_dat/pose_data_isharah1000_hands_lips_body_May12.pkl'
    DEV_ANNOTATION_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/US/dev.txt'
    TEST_POSE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/US/pose_data_isharah1000_US_test.pkl'
    TEST_ANNOTATION_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Pose86K-CSLR-Isharah/annotations_v2/US/US_test.txt'

    # Model Parameters
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    VOCAB_SIZE = 1000

    # Training Parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 250   ## <<==================
    BATCH_SIZE = 16
    LR = 0.001
    MAX_FRAMES = 256
    USE_MULTI_GPU = True
    MIXED_PRECISION = True
    GRAD_ACCUM_STEPS = 2
    FP16_DTYPE = torch.float16
    CTC_LOSS_FP32 = True

    # GPU Settings
    GPU_IDS = [0, 1]
    NUM_WORKERS = 8

    # Output Paths
    MODEL_SAVE_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/models/US/US_best_model.pth'
    PREDICTION_OUTPUT_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/predictions/US/US_test_predictions.txt'
    DEV_PREDICTION_OUTPUT_PATH = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/predictions/US/US_dev_predictions.txt'


    #####
    USE_MOTION = True          # Enable motion features
    SPATIAL_DIM = 128          # Joint embedding dimension
    TEMPORAL_DIM = 512          # Frame embedding dimension
    TRANSFORMER_DIM = 1024      # Transformer feature size
    TRANSFORMER_FF = 2048       # Transformer FF dimension
    SPATIAL_HEADS = 8           # Spatial attention heads
    TEMPORAL_HEADS = 8           # Temporal attention heads
    TRANSFORMER_HEADS = 8        # Transformer attention heads
    NUM_LAYERS_TMP = 6              # Transformer layers
    MAX_SEQ_LEN = 300
    #####

###--------------------------------------------------
## Usage example:
# def get_config(mode):
#     if mode == "SI":
#         return SIConfig
#     elif mode == "US":
#         return USConfig
#     else:
#         raise ValueError(f"Unknown mode: {mode}")

# ## Example usage:
# mode = "SI" # or "US"
# cfg = get_config(mode)
# print(cfg.TRAIN_POSE_PATH)
###--------------------------------------------------- 

