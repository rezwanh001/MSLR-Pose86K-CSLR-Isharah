# Import key components to make them available at package level
from .dataset import PoseDataset, collate_fn
from .model import SignLanguageRecognizer, SignLanguageConformer
# from .mamba_model import ArabicMamba
from .train import train
from .inference import generate_predictions
from .utils import save_checkpoint, load_checkpoint, setup_environment

# Optional: Define what gets imported with 'from src import *'
__all__ = [
    'PoseDataset',
    'collate_fn',
    'SignLanguageRecognizer',
    'SignLanguageConformer',
    # 'ArabicMamba',
    'train',
    'generate_predictions',
    'save_checkpoint',
    'load_checkpoint',
    'setup_environment'
]

# Package version
__version__ = '1.0.0'