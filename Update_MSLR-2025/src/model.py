# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

###====================================================================
#### Sign Language Recognition Model Implementations
class SignLanguageRecognizer(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.input_size = 86 * 2  # 86 joints x 2 coordinates

        # Add +2 for <unk> and <pad> tokens
        self.output_size = vocab_size + 2  
        
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(self.input_size, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),
            
            nn.Conv1d(512, 768, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.MaxPool1d(2),
            
            nn.Conv1d(768, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(2)
        )

        # Sequence modeling
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if cfg.NUM_LAYERS > 1 else 0
        )
        
        # Update classifier output size
        self.classifier = nn.Sequential(
            nn.Linear(2*cfg.HIDDEN_SIZE, 2*cfg.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(2*cfg.HIDDEN_SIZE, self.output_size)
        )

    def forward(self, x):
        # x: (B, T, J*2)
        x = x.permute(0, 2, 1)  # (B, C, T)
        
        # Temporal features
        x = self.temporal_encoder(x)  # (B, 1024, T/8)
        x = x.permute(0, 2, 1)  # (B, T/8, 1024)
        
        # Sequence features
        x, _ = self.lstm(x)  # (B, T/8, 2*HIDDEN_SIZE)

        # print(f"Shape after LSTM: {x.
        # print(f"\033[91mShape after LSTM: {x.shape}\033[0m") # torch.Size([8, 32, 1024])

        # Classification
        return self.classifier(x)  # (B, T/8, vocab_size+1)
    
####====================================================================
##### Advanced Sign Language Recognition Model with Joint Attention
class JointAttention(nn.Module):
    """Attention mechanism to weigh important joints."""
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        query = self.query(x)  # (B, T, C)
        key = self.key(x)
        # (B, C, T)
        value = self.value(x)    # (B, T, C)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (C ** 0.5)  # (B, T, T)
        attn_weights = self.softmax(scores)  # (B, T, T)
        out = torch.matmul(attn_weights, value)  # (B, T, C)
        return out + x  # Residual connection

class AdvancedSignLanguageRecognizer(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.input_size = 86 * 2  # 86 joints x 2 coordinates
        self.output_size = vocab_size + 2  # +2 for <unk> and <pad> tokens

        # Initial embedding layer to project joint coordinates
        self.input_embedding = nn.Linear(self.input_size, 512)

        # Joint attention to weigh important joints
        self.joint_attention = JointAttention(512)

        # Temporal encoder with residual connections
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Conv1d(512, 768, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(768, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.MaxPool1d(2)
        )

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers= 7 # cfg.NUM_LAYERS  # Use same number of layers as before
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, self.output_size)
        )

    def forward(self, x):
        # x: (B, T, J*2)
        B, T, _ = x.shape

        # Project input to embedding space
        x = self.input_embedding(x)  # (B, T, 512)

        # Apply joint attention
        x = self.joint_attention(x)  # (B, T, 512)

        # Temporal encoding
        x = x.permute(0, 2, 1)  # (B, 512, T)
        x = self.temporal_encoder(x)  # (B, 1024, T/4)
        x = x.permute(0, 2, 1)  # (B, T/4, 1024)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T/4, 1024)

        # Classification
        x = self.classifier(x)  # (B, T/4, vocab_size+2)
        return x
    
###========================================================================== 
#### SignLanguageConformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ConvSubsampling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvSubsampling, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T/4, C)
        return self.layer_norm(x)

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Project to query, key, value
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # Final projection
        return self.linear_out(context)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, conv_kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        # Feed-forward module 1
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn1_norm = nn.LayerNorm(d_model)
        
        # Multi-headed self-attention
        self.attention = MultiHeadedSelfAttention(d_model, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
       
        # Feed-forward module 2
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn2_norm = nn.LayerNorm(d_model)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # FFN module 1
        residual = x
        x = self.ffn1_norm(x)
        x = 0.5 * self.ffn1(x) + residual
        
        # Attention module
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, mask) + residual
        
        # Convolution module
        residual = x
        x = self.conv_norm(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = x + residual
        
        # FFN module 2
        residual = x
        x = self.ffn2_norm(x)
        x = 0.5 * self.ffn2(x) + residual
        
        x = self.final_norm(x)
        return self.dropout(x)

class SignLanguageConformer(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        input_dim = 86 * 2
        d_model = 256
        num_heads = 8
        num_layers = 12
        conv_kernel_size = 31
        
        # Add +2 for <unk> and <pad> tokens
        self.output_size = vocab_size + 2
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Convolutional subsampling
        self.subsampling = ConvSubsampling(d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=0.1)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, self.output_size)
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Subsampling
        x = self.subsampling(x)
        
        # Positional encoding
        x = x.permute(1, 0, 2)  # (T, B, C) for positional encoding
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)  # (B, T, C)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Classifier
        return self.classifier(x)