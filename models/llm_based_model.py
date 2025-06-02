import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import DistilBertModel, DistilBertConfig
from transformers import BertModel, BertConfig 
from transformers import RobertaModel, RobertaConfig
from transformers import DebertaModel, DebertaConfig
from transformers import AutoModel
from transformers import XLMRobertaModel

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

        # # LLM Backbone
        # self.llm = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')

        # self.llm = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        # # LLM Backbone (DEBERTA)
        # self.llm = DebertaModel.from_pretrained('microsoft/deberta-base')

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

class SlowFastLLMCSLR(nn.Module):
    def __init__(self, input_dim=84, slow_hidden=2048, fast_hidden=256,
                 num_layers=2, num_heads=8, conv_channels_slow=512,
                 conv_channels_fast=128, mlp_hidden=512, num_classes=100,
                 dropout=0.1, alpha=2):
        super().__init__()
        self.alpha = alpha
        self.hidden_dim = 768  # For LLM compatibility

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

        # Bidirectional Fusion
        self.slow_to_fast_fusion = nn.Linear(slow_hidden, fast_hidden)
        self.fast_to_slow_fusion = nn.Linear(fast_hidden, slow_hidden)

        # Projection to LLM hidden dimension
        self.fusion_to_llm = nn.Linear(conv_channels_fast + conv_channels_slow, self.hidden_dim)

        # LLM Backbone
        self.llm = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')

        # Output MLP for CTC
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, poses):
        # poses: [B, T, 42, 2]
        B, T, N, F = poses.shape
        x = poses.view(B, T, N * F)  # [B, T, 84]

        # Fast Pathway
        x_fast = self.fast_proj(x)  # [B, T, fast_hidden]
        x_fast = self.fast_pos_encoder(x_fast)
        x_fast = self.fast_transformer(x_fast)

        # Slow Pathway
        x_slow = x[:, ::self.alpha, :]  # [B, T//alpha, 84]
        x_slow = self.slow_proj(x_slow)
        x_slow = self.slow_pos_encoder(x_slow)
        x_slow = self.slow_transformer(x_slow)  # [B, T//alpha, slow_hidden]

        # Bidirectional Fusion
        x_slow_expanded = torch.repeat_interleave(x_slow, self.alpha, dim=1)[:, :T, :]  # [B, T, slow_hidden]
        x_fast = x_fast + self.slow_to_fast_fusion(x_slow_expanded)

        pool = nn.AvgPool1d(kernel_size=self.alpha, stride=self.alpha, ceil_mode=True)
        x_fast_down = pool(x_fast.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T//alpha, fast_hidden]
        x_slow = x_slow + self.fast_to_slow_fusion(x_fast_down)

        # Temporal Pooling
        x_fast = self.fast_temporal_pooling(x_fast.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', conv_channels_fast]
        x_slow = self.slow_temporal_pooling(x_slow.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', conv_channels_slow]

        # Length Match
        if x_fast.shape[1] != x_slow.shape[1]:
            min_len = min(x_fast.shape[1], x_slow.shape[1])
            x_fast = x_fast[:, :min_len, :]
            x_slow = x_slow[:, :min_len, :]

        # Combine Pathways
        x = torch.cat([x_fast, x_slow], dim=-1)  # [B, T', conv_fast + conv_slow]
        x = self.fusion_to_llm(x)  # [B, T', hidden_dim]

        # LLM Processing
        outputs = self.llm(inputs_embeds=x)
        sequence_output = outputs.last_hidden_state  # [B, T', hidden_dim]

        # CTC Output
        logits = self.output_mlp(sequence_output)  # [B, T', num_classes]
        return logits
    

class AdvancedSlowFastLLMCSLR(nn.Module):
    def __init__(self, input_dim=84, slow_hidden=1024, fast_hidden=256,
                 num_layers=2, num_heads=8, conv_channels_slow=512,
                 conv_channels_fast=128, mlp_hidden=512, num_classes=100,
                 dropout=0.1, alpha=2, llm_hidden=768, llm_model_name='xlm-roberta-base'):
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

        # LLM Backbone (XLM-RoBERTa for Arabic/multilingual)
        self.llm = XLMRobertaModel.from_pretrained(llm_model_name)
        self.llm_proj = nn.Linear(llm_hidden, conv_channels_slow + conv_channels_fast)

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels_fast + conv_channels_slow + conv_channels_fast + conv_channels_slow, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, poses, llm_input_ids=None, llm_attention_mask=None):
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

        pose_features = torch.cat([x_fast, x_slow], dim=-1)  # [B, T', conv_fast + conv_slow]

        # LLM features (sequence-level, e.g., gloss or dummy tokens if no text)
        if llm_input_ids is not None and llm_attention_mask is not None:
            llm_outputs = self.llm(input_ids=llm_input_ids, attention_mask=llm_attention_mask)
            llm_feat = llm_outputs.last_hidden_state[:, 0, :]  # [B, llm_hidden] (CLS token)
        else:
            # If no text, use zeros
            llm_feat = torch.zeros(B, self.llm.config.hidden_size, device=poses.device)

        llm_feat_proj = self.llm_proj(llm_feat)  # [B, conv_channels_fast + conv_channels_slow]
        llm_feat_proj = llm_feat_proj.unsqueeze(1).expand(-1, pose_features.size(1), -1)  # [B, T', ...]
        fused = torch.cat([pose_features, llm_feat_proj], dim=-1)  # [B, T', all_features]

        logits = self.mlp(fused)  # [B, T', num_classes]
        return logits