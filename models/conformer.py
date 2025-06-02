import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSubsampling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T//4, F)
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, conv_kernel=31, dropout=0.1, layer_drop=0.1):
        super().__init__()
        self.layer_drop = layer_drop
        self.ffn1 = FeedForward(dim, dropout=dropout)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            # Remove LayerNorm here since we're processing time dimension
            nn.Conv1d(dim, 2*dim, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout)
        )
        self.ffn2 = FeedForward(dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        if self.training and torch.rand(1, device=x.device) < self.layer_drop:
            return x
            
        res = x
        x = 0.5 * self.ffn1(x) + res
        
        # Attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = attn_out + x
        
        # Convolution
        conv_input = x.transpose(1, 2)  # (B, F, T)
        conv_out = self.conv(conv_input)
        x = conv_out.transpose(1, 2) + x  # (B, T, F)
        
        # FFN
        x = 0.5 * self.ffn2(x) + x
        return self.norm(x)

class ConformerPoseCSLR(nn.Module):
    def __init__(self, input_dim, num_classes, dim=256, num_layers=8, num_heads=4, 
                 conv_kernel=31, dropout=0.1, layer_drop=0.1):
        super().__init__()
        # Calculate feature dimension (joints * coordinates)
        self.feature_dim = input_dim
        self.subsample = ConvSubsampling(self.feature_dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=dim,
                num_heads=num_heads,
                conv_kernel=conv_kernel,
                dropout=dropout,
                layer_drop=layer_drop
            ) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, lengths=None):
        # x shape: (B, T, J, D)
        B, T, J, D = x.shape
        
        # Flatten joints and coordinates dimensions
        x = x.reshape(B, T, J * D)  # (B, T, F)
        
        x = self.subsample(x)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Create mask for padded positions
        key_padding_mask = None
        if lengths is not None:
            # Adjust lengths for subsampling (4x reduction)
            lengths = torch.div(lengths + 1, 2, rounding_mode='floor')
            lengths = torch.div(lengths + 1, 2, rounding_mode='floor')
            lengths = torch.clamp(lengths, min=1)
            
            max_len = x.size(1)
            # Create mask on the same device as x
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
            key_padding_mask = mask
        
        for layer in self.layers:
            x = layer(x, key_padding_mask)
            
        return self.classifier(x)