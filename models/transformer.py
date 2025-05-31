import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class PoseCSLRTransformer(nn.Module):
    def __init__(self, input_dim=84, hidden_dim=512, num_layers=2, num_heads=8, conv_channels=512, mlp_hidden=512, num_classes=100, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_pooling = nn.Sequential(
            nn.Conv1d(hidden_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),           # [B, C, T] → [B, C, T/2]
            nn.Conv1d(hidden_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),           # [B, C, T/2] → [B, C, T/4]
        )

        self.mlp = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, poses):
        # poses: [B, T, 42, 2]
        B, T, N, F = poses.shape
        x = poses.view(B, T, N * F)  # [B, T, 84]

        x = self.input_proj(x)  # [B, T, hidden_dim]
        x = self.pos_encoder(x)  # [B, T, hidden_dim]
        x = self.transformer(x)  # [B, T, hidden_dim]

        x = x.permute(0, 2, 1)  # [B, hidden_dim, T]
        x = self.temporal_pooling(x)  # [B, conv_channels, T']
        x = x.permute(0, 2, 1)  # [B, T', conv_channels]

        logits = self.mlp(x)  # [B, T', num_classes]
        return logits


class SlowFastCSLR(nn.Module):
    def __init__(self, input_dim=84, slow_hidden=2048, fast_hidden=256,
                 num_layers=2, num_heads=8, conv_channels_slow=512,
                 conv_channels_fast=128, mlp_hidden=512, num_classes=100,
                 dropout=0.1, alpha=2):
        super().__init__()
        self.alpha = alpha

        # Fast Pathway
        self.fast_proj = nn.Linear(input_dim, fast_hidden)
        self.fast_pos_encoder = PositionalEncoding(fast_hidden)
        fast_encoder_layer = nn.TransformerEncoderLayer(d_model=fast_hidden, nhead=4, dropout=dropout, batch_first=True)
        self.fast_transformer = nn.TransformerEncoder(fast_encoder_layer, num_layers=num_layers)

        self.fast_temporal_pooling = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(fast_hidden, conv_channels_fast, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        # Slow Pathway
        self.slow_proj = nn.Linear(input_dim, slow_hidden)
        self.slow_pos_encoder = PositionalEncoding(slow_hidden)
        slow_encoder_layer = nn.TransformerEncoderLayer(d_model=slow_hidden, nhead=num_heads, dropout=dropout, batch_first=True)
        self.slow_transformer = nn.TransformerEncoder(slow_encoder_layer, num_layers=num_layers)

        self.slow_temporal_pooling = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(slow_hidden, conv_channels_slow, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        # Fusion layers
        self.slow_to_fast_fusion = nn.Linear(slow_hidden, fast_hidden)
        self.fast_to_slow_fusion = nn.Linear(fast_hidden, slow_hidden)

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels_fast + conv_channels_slow, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, poses):
        # poses: [B, T, 42, 2]
        B, T, N, F = poses.shape
        x = poses.view(B, T, N * F)  # [B, T, 84]

        # Fast stream
        x_fast = self.fast_proj(x)  # [B, T, fast_hidden]
        x_fast = self.fast_pos_encoder(x_fast)
        x_fast = self.fast_transformer(x_fast)

        # Slow stream
        x_slow = x[:, ::self.alpha, :]  # [B, T//alpha, 84]
        x_slow = self.slow_proj(x_slow)
        x_slow = self.slow_pos_encoder(x_slow)
        x_slow = self.slow_transformer(x_slow)  # [B, T//alpha, slow_hidden]

        # Bidirectional fusion
        x_slow_expanded = torch.repeat_interleave(x_slow, self.alpha, dim=1)[:, :T, :]  # [B, T, slow_hidden]
        x_fast = x_fast + self.slow_to_fast_fusion(x_slow_expanded)

        pool = nn.AvgPool1d(kernel_size=self.alpha, stride=self.alpha, ceil_mode=True)
        x_fast_down = pool(x_fast.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T//alpha, fast_hidden]
        x_slow = x_slow + self.fast_to_slow_fusion(x_fast_down)

        # Temporal pooling
        x_fast = self.fast_temporal_pooling(x_fast.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', conv_channels_fast]
        x_slow = self.slow_temporal_pooling(x_slow.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', conv_channels_slow]

        # Length match before concatenation
        if x_fast.shape[1] != x_slow.shape[1]:
            min_len = min(x_fast.shape[1], x_slow.shape[1])
            x_fast = x_fast[:, :min_len, :]
            x_slow = x_slow[:, :min_len, :]

        x = torch.cat([x_fast, x_slow], dim=-1)  # [B, T', conv_fast + conv_slow]
        logits = self.mlp(x)  # [B, T', num_classes]
        return logits

