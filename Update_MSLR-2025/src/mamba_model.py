###======================================= MAMBA =================================
"""ArabicMamba implementation.
Authors
-------
* Rezwan (Year: 2025)
"""
import torch
import torch.nn as nn
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.normalization import LayerNorm

import os
import sys
if os.environ.get("CONDA_DEFAULT_ENV") == "ArbMamba":
    # Mamba
    from mamba_ssm import Mamba
    from .mamba.bimamba import Mamba as BiMamba 
    from .mamba.mm_bimamba import Mamba as MMBiMamba 
else:
    print("Not importing mamba_ssm (not in ArbMamba environment).", file=sys.stderr)

# # Mamba
# from mamba_ssm import Mamba
# from .mamba.bimamba import Mamba as BiMamba 
# from .mamba.mm_bimamba import Mamba as MMBiMamba 

class MMMambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config is not None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
        self,
        x,
        inference_params=None
    ):
        out1 = self.mamba(x, inference_params)
        out = x + self.norm1(out1)
        return out

class MMCNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.bn, self.relu, self.drop)

        if input_size != output_size:
            self.skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.skipconv is not None:
            x = self.skipconv(x)
        out = out + x
        return out

class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def forward(
        self,
        x, inference_params = None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out

class CNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_size, output_size, 5, padding=2, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_size)
        # self.relu2 = nn.ReLU()

        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out+x
        return out


class CoSSM(nn.Module):
    """This class implements the CoSSM encoder for one input branch."""
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256, 512, 512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        kernel_size=3,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []

        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                input_size=input_size if i < 1 else output_sizes[i - 1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(MMMambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = nn.ModuleList(mamba_list)
        self.cnn_layers = nn.ModuleList(cnn_list)

    def forward(self, x, inference_params=None):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out = cnn_layer(out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            out = mamba_layer(out, inference_params=inference_params)

        return out

class EnSSM(nn.Module):
    """This class implements the EnSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(self, x, inference_params=None):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out = cnn_layer(out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            out = mamba_layer(
                out,
                inference_params = inference_params,
            )

        return out

class ArabicMamba(nn.Module):
    """
    ArabicMamba model refactored in a clean style similar to SignLanguageRecognizer.
    """

    def __init__(self, output_size, cfg, mamba_config=None):
        super().__init__()

        self.input_size = 86 * 2  # 86 joints x 2 coordinates
        # Add +2 for <unk> and <pad> tokens
        self.output_size = output_size + 2

        self.mm_input_size = 128
        self.mm_output_sizes = [256, 512, 512]
        self.d_ffn = 1024
        self.num_layers = 8
        self.dropout = 0.1
        self.activation = 'Swish'
        self.causal = False
        self.mamba_config = mamba_config

        # Input projection layer
        self.input_proj = nn.Conv1d(self.input_size, self.mm_input_size, kernel_size=1, bias=False)

        # CoSSM encoder
        self.cossm_encoder = CoSSM(
            num_layers=self.num_layers,
            input_size=self.mm_input_size,
            output_sizes=self.mm_output_sizes,
            d_ffn=self.d_ffn,
            activation=self.activation,
            dropout=self.dropout,
            causal=self.causal,
            mamba_config=self.mamba_config
        )

        # EnSSM encoder
        self.enssm_encoder = EnSSM(
            num_layers=self.num_layers,
            input_size=self.mm_output_sizes[-1],
            output_sizes=[self.mm_output_sizes[-1]],
            d_ffn=self.d_ffn,
            activation=self.activation,
            dropout=self.dropout,
            causal=self.causal,
            mamba_config=self.mamba_config
        )

        # Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.mm_output_sizes[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_size)
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.input_proj.weight.data)

    def forward(self, x, padding_mask=None, inference_params=None):
        """
        Args:
            x: (B, T, input_size)
            padding_mask: (B, T) optional
            inference_params: optional for Mamba
        Returns:
            logits: (B, output_size)
        """
        # Input projection
        x = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, mm_input_size)

        # CoSSM encoder
        x = self.cossm_encoder(x, inference_params=inference_params)

        # EnSSM encoder
        x = self.enssm_encoder(x, inference_params=inference_params)
        # print(f"\033[91mShape after CoSSM and EnSSM: {x.shape}\033[0m") ## torch.Size([8, 256, 512])

        # # Mask or pooling
        # if padding_mask is not None:
        #     x = x * padding_mask.unsqueeze(-1).float()
        #     x = x.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        # else:
        #     x = self.pool(x.permute(0, 2, 1)).squeeze(-1)

        # print(f"\033[91mShape after pooling: {x.shape}\033[0m") # torch.Size([8, 512])

        # Classifier
        logits = self.classifier(x)

        return logits
