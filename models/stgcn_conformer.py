import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import einsum
from einops import rearrange, repeat

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

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.adj_matrix = adj_matrix
        self.weight = nn.Parameter(torch.Tensor(adj_matrix.size(0), in_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: [B, C, T, V]
        x = torch.einsum('bctv, vwc->bctw', x, self.adj_matrix)
        x = torch.einsum('bctv, vwc->bwtc', x, self.weight)
        x = self.bn(x)
        return F.relu(x)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, stride=1, residual=True):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, adj_matrix)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)) if residual else None

    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = self.tcn(x)
        if self.residual:
            res = self.residual(res)
        return F.relu(x + res) if self.residual else F.relu(x)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, 2*d_model, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # FFN 1
        x = x + 0.5 * self.ffn1(x)
        
        # Attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn_out
        
        # Convolution
        conv_input = x.transpose(1, 2)
        conv_out = self.conv(conv_input).transpose(1, 2)
        x = x + conv_out
        
        # FFN 2
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)

class STGCNConformer(nn.Module):
    def __init__(self, input_dim, num_classes, adj_matrix, 
                 gcn_channels=[64, 128, 256], embed_dim=256,
                 num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        
        # ST-GCN Encoder
        self.gcn_input = nn.Conv2d(input_dim, gcn_channels[0], kernel_size=1)
        self.gcn_blocks = nn.ModuleList()
        
        for i in range(len(gcn_channels) - 1):
            self.gcn_blocks.append(STGCNBlock(gcn_channels[i], gcn_channels[i+1], adj_matrix, stride=2))
        
        # Conformer Encoder
        self.temporal_proj = nn.Linear(gcn_channels[-1], embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize adjacency matrix
        self.register_buffer('adj', adj_matrix)

    def forward(self, x, lengths=None):
        # x: [B, T, V, C] -> [B, C, T, V]
        x = x.permute(0, 3, 1, 2)
        
        # ST-GCN Processing
        x = self.gcn_input(x)
        for block in self.gcn_blocks:
            x = block(x)
        
        # Prepare for temporal processing
        x = x.permute(0, 2, 3, 1)  # [B, T, V, C]
        B, T, V, C = x.shape
        x = x.reshape(B, T, -1)  # Flatten spatial features
        
        # Temporal processing
        x = self.temporal_proj(x)
        x = self.pos_enc(x)
        
        # Create mask for padded positions
        key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
            key_padding_mask = mask
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask)
        
        # Classification
        logits = self.classifier(x)
        return logits

# Define adjacency matrix for body joints
def get_body_adjacency_matrix(num_joints=86):
    # Create a simple chain adjacency matrix
    adj = torch.eye(num_joints)
    for i in range(num_joints - 1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    return adj 

