import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        return x + self.pe[:, :x.size(1)]

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        # Register adjacency matrix as buffer
        self.register_buffer('adj', adj_matrix.float())
        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)  # Changed to BatchNorm2d
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        # x: [B, T, V, C]
        B, T, V, C = x.shape
        
        # Normalize adjacency matrix
        adj = self.adj + torch.eye(V, device=x.device)  # Add self-connections
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        # Apply graph convolution
        x = x.reshape(B*T, V, C)
        x = torch.matmul(norm_adj, x)       # Spatial aggregation
        x = torch.matmul(x, self.w)         # Feature transformation
        x = x.reshape(B, T, V, -1)
        
        # Change dimension order for BatchNorm2d: [B, T, V, C] -> [B, C, T, V]
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        x = F.relu(x)
        
        # Return to original dimension order: [B, T, V, C]
        return x.permute(0, 2, 3, 1)

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x, key_padding_mask=None):
        # x: [B, T, D]
        return self.transformer(x, src_key_padding_mask=key_padding_mask)

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, adj_matrix,
                 spatial_channels=64, embed_dim=256,
                 num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Graph convolution layers
        self.spatial_conv1 = SpatialGraphConv(input_dim, spatial_channels, adj_matrix)
        self.spatial_conv2 = SpatialGraphConv(spatial_channels, spatial_channels*2, adj_matrix)
        
        # Projection to embedding dimension
        self.proj = nn.Linear(spatial_channels*2, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: [B, T, V, C] (batch, time, joints, features)
        B, T, V, C = x.shape
        
        # Spatial processing
        x = self.spatial_conv1(x)  # [B, T, V, spatial_channels]
        x = self.spatial_conv2(x)  # [B, T, V, spatial_channels*2]
        
        # Aggregate spatial information
        x = x.mean(dim=2)  # [B, T, spatial_channels*2]
        
        # Project to embedding dimension
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Create mask for padded positions
        key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(B, max_len) >= lengths.unsqueeze(1)
            key_padding_mask = mask
        
        # Temporal processing
        x = self.temporal_transformer(x, key_padding_mask)
        
        # Classification
        logits = self.classifier(x)
        return logits

def get_body_adjacency_matrix(num_joints=86):
    """Create a simple chain adjacency matrix for body joints"""
    adj = torch.zeros(num_joints, num_joints)
    for i in range(num_joints-1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    return adj