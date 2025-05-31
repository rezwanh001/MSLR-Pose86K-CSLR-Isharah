import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import DistilBertModel, DistilBertConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class LLMEnhancedPoseCSLR(nn.Module):
    def __init__(self, input_dim=84, hidden_dim=768, num_classes=100, dropout=0.1):
        super().__init__()
        # Pose Feature Extractor
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # LLM Backbone (DistilBERT)
        bert_config = DistilBertConfig(
            vocab_size=1,  # Not used for embeddings
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=hidden_dim * 4,
            dropout=dropout,
            max_position_embeddings=512
        )
        self.llm = DistilBertModel(bert_config)

        # Output Layer for CTC
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, poses):
        # poses: [B, T, 42, 2]
        B, T, N, F = poses.shape
        x = poses.view(B, T, N * F)  # [B, T, 84]

        # Feature Extraction
        x = self.input_proj(x)  # [B, T, hidden_dim]
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)  # [B, hidden_dim, T]
        x = self.conv_encoder(x)  # [B, hidden_dim, T]
        x = x.permute(0, 2, 1)  # [B, T, hidden_dim]

        # LLM Processing
        outputs = self.llm(inputs_embeds=x)
        sequence_output = outputs.last_hidden_state  # [B, T, hidden_dim]

        # CTC Output
        logits = self.output_mlp(sequence_output)  # [B, T, num_classes]
        return logits