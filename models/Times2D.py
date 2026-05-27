__all__ = ['Times2D']

# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.fft import rfft
import numpy as np
import math
from typing import Callable, Optional
from einops import rearrange
from layers.RevIN import RevIN
from layers.Times2D.Conv_Blocks import Inception_Block_V1
from layers.Times2D.encoders import TSTiEncoder, TSTEncoder, TSTEncoderLayer

def compute_derivative_heatmaps(x):
    # Calculate first derivative manually
    first_derivative = x[:, 1:] - x[:, :-1]
    # Pad the first_derivative to maintain original length
    first_derivative = torch.cat([torch.zeros_like(first_derivative[:, :1, :]), first_derivative], dim=1)
    
    # Calculate second derivative manually
    second_derivative = first_derivative[:, 1:] - first_derivative[:, :-1]
    # Pad the second_derivative to maintain original length
    second_derivative = torch.cat([torch.zeros_like(second_derivative[:, :1, :]), second_derivative], dim=1)

    # Stack the derivatives along a new dimension
    heatmap = torch.stack([first_derivative, second_derivative], dim=-1)
    heatmap = heatmap.permute(0,2,1,3)  # Adjust shape to [Batch, Channels, Time, Derivative]
    return heatmap

class Times2DBackbone(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Times2DBackbone, self).__init__()

        # Load parameters from configs
        self.data = configs.data
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.add = configs.add
        self.affine = configs.affine
        self.head_dropout = configs.head_dropout
        self.subtract_last = configs.subtract_last
        self.revin_layer = RevIN(self.enc_in, affine=self.affine, subtract_last=self.subtract_last)
        self.conv_blocks = nn.ModuleList()
        self.backbone = nn.ModuleList()
        self.n_layers = configs.e_layers
        self.wo_conv = configs.wo_conv
        self.serial_conv = configs.serial_conv
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.attn_dropout = configs.attn_dropout
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(self.seq_len, self.pred_len).to(self.device)
        self.dropout = configs.dropout
        self.act = 'gelu'
        self.norm = 'BatchNorm'
        self.key_padding_mask = 'auto'
        self.padding_var = None
        self.attn_mask = None
        self.res_attention = True
        self.pre_norm = False
        self.store_attn = False
        self.pe = 'zeros'
        self.learn_pe = True
        self.verbose = False
        self.patch_len  = configs.patch_len
        self.batch = configs.batch_size
        # Define period_list and period_len
      
        #self.period_list = [720, 360, 140, 70, 48]  # weather 
        self.period_list = [720, 360, 110, 96, 48]  # M4 yearly EETT (h and m)
        

        #self.period_list = configs.period_list
        self.top_k = len(self.period_list)
        self.period_len = [math.ceil(self.seq_len / i) for i in self.period_list]
        # Define kernel_list and stride_list
        self.kernel_list = [(n, self.patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = self.kernel_list

        # Define dim_list and tokens_list
        self.dim_list = [k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.enc_in, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.enc_in,
                               num_kernels=configs.num_kernels))
        
        self.conv2D = nn.ModuleList([
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s).to(self.device)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ])
        

        self.head = Head(self.seq_len, self.top_k, self.pred_len, head_dropout=self.head_dropout, Concat=not self.add).to(self.device)

        self.backbone = nn.ModuleList([nn.Sequential(
            TSTiEncoder(self.enc_in, patch_num=token, patch_len=self.dim_list[i], max_seq_len=self.seq_len,
                        n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, d_k=None, d_v=None,
                        d_ff=self.d_ff, norm=self.norm, attn_dropout=self.attn_dropout, head_dropout=self.head_dropout, dropout=self.dropout,
                        act=self.act,
                        key_padding_mask=self.key_padding_mask, padding_var=self.padding_var, attn_mask=self.attn_mask, 
                        res_attention=self.res_attention, pre_norm=self.pre_norm, store_attn=self.store_attn, pe=self.pe, 
                        learn_pe=self.learn_pe, verbose=self.verbose, **self.kwargs).to(self.device),
            nn.Flatten(start_dim=-2).to(self.device),
            nn.Linear(self.tokens_list[i] * self.d_model, self.seq_len).to(self.device)
            if self.tokens_list[i] * self.d_model != self.seq_len else nn.Identity().to(self.device)
        ) for i, token in enumerate(self.tokens_list)])

        self.to(self.device)  # Move the entire model to the specified device
        self.weights = nn.Parameter(torch.randn(self.batch, 2))  # Random initialization
        self.heatmap_to_pred = nn.Conv1d(in_channels=self.seq_len, out_channels=self.pred_len, kernel_size=1).to(self.device)
        
    def forward(self, x):  # x: [B, N, T]

        x = x.to(self.device)
        B, N, T = x.size()
        x = x.permute(0, 2, 1)  # [B, T, N]
        #Period_list, Period_length, weights = FFT_for_Period(x, self.top_k)
        # Forward pass through the model
        x = self.revin_layer(x, 'norm').to(self.device)
        # heatmap
        heatmap     = compute_derivative_heatmaps(x).permute(0,1,2,3)                          # [B, T, N, 2]
        heatmap_features = self.conv(heatmap).permute(0,2,1,3)                      # torch.Size([B, T, N, 2])
        
        if self.data == 'm4':
            weights = nn.Parameter(torch.randn(B, 2)).to(self.device)  # Random initialization
            weights = weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        else:   
            weights = self.weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
            weights = weights[:B,...]
        # only takes B,T,N
        Final_heatmap = torch.sum(heatmap_features * weights, -1)           #  torch.Size([B, T, N])
        Final_heatmap = Final_heatmap.permute(0,2,1)   
        #  torch.Size([B,N, T])
        # Transform heatmap to match Pred_len
        Final_heatmap = Final_heatmap.permute(0, 2, 1)  # [B, T, N]
        Final_heatmap = self.heatmap_to_pred(Final_heatmap)  # [B, Pred_len, N]
        Final_heatmap = Final_heatmap.permute(0, 2, 1)  # [B, N, Pred_len]
        
        
        x = x.permute(0, 2, 1)  # [B, N, T]
        res = []

        for i, (period, (kernel_height, kernel_width)) in enumerate(zip(self.period_list, self.kernel_list)):
            if self.seq_len % period != 0:
                pad1 = nn.ConstantPad1d((0, period - self.seq_len % period), 0)
                padded_X = pad1(x).to(self.device)
                padded_X = padded_X.reshape(padded_X.shape[0], padded_X.shape[1], padded_X.shape[2] // period, period)
            else:
                padded_X = x.reshape(B, N, T // period, period)
             
            if period % kernel_width != 0:
                pad2 = nn.ConstantPad1d((0, kernel_width - period % kernel_width), 0)
                out = pad2(padded_X).to(self.device)
            else:
                out = padded_X  # [B, N, patch, periods]

            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])       # 
            out = out.unsqueeze(-3)
            out = self.conv2D[i](out)
            out = self.flatten(out)
            out = rearrange(out, '(b n) d p -> b n d p', b=x.size(0))  # Reshape back to [B, N, dim_list[i], tokens_list[i]]
            glo = self.backbone[i](out)                                # [B, N, T]
            
            res.append(glo)
        res = [r.to(self.device) for r in res]
      
        z = self.head(res)                       # [B, N, Pred_lem]
        #combined  = z  
        combined  = z  + Final_heatmap            # [B, N, Pred_lem]
        combined = combined.permute(0, 2, 1)                            # [B,Pred_len, N]      
        combined = self.revin_layer(combined, 'denorm')
        combined = combined.permute(0, 2, 1)
        
        return combined
        

class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0,
                 Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(context_window * (num_period if Concat else 1), target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x
    
class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = Times2DBackbone(configs, **kwargs)# x: [B, N, T]

    def forward(self, x_enc, **kwargs):
        x = x_enc.permute(0, 2, 1)# x: [B, N, T]
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        # 👇 核心秘籍：开启“覆盖模式”。遇到冲突时，新的覆盖旧的！
        parent_parser.conflict_handler = 'resolve'
        parser = parent_parser.add_argument_group('MMPD Model Specific Arguments')
        parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')

        parser.add_argument('--add', action='store_true', default=False, help='add')
        parser.add_argument('--wo_conv', action='store_true', default=False, help='without convolution')
        parser.add_argument('--serial_conv', action='store_true', default=False, help='serial convolution')

        parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9], help='kernel size list')
        parser.add_argument('--patch_len', type=int, nargs='+', default=[16], help='patch high')
        parser.add_argument('--period_list', type=int, nargs='+', default=[24, 12], help='period list')
        parser.add_argument('--stride', type=int, nargs='+', default=None, help='stride')

        parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
        parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
        parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
        parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

        # Formers 
        parser.add_argument('--embed_type', type=int, default=0,
                            help='0: default 1: value patch_embedding + temporal patch_embedding + positional patch_embedding 2: value '
                                'patch_embedding + temporal patch_embedding 3: value patch_embedding + positional patch_embedding 4: value patch_embedding')
        parser.add_argument('--enc_in', type=int, default=7,
                            help='global_encoder input size')  # DLinear with --individual, use this hyperparameter as the number of
        # channels
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of global_encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in global_encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn_dropout', type=float, default=0.05, help='attention dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        return parent_parser
    
    
