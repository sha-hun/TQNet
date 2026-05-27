import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    
    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x) # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x



class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1) # L

        # Filter
        # self.alpha = 0.1 if configs.alpha is None else configs.alpha
        # self.top_p = 0.5 if configs.top_p is None else configs.top_p
        self.alpha = getattr(configs, 'alpha', 0.1)
        self.top_p = getattr(configs, 'top_p', 0.5)

        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter Backbone
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                  configs.n_heads, configs.e_layers, self.top_p, configs.dropout, self.seq_len * self.n_vars // self.patch_len)
        
        # head
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        # Without RevIN
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

        self.register_buffer('fixed_masks', self._get_mask())

    def _get_mask(self):
        dtype = torch.float32
        # 使用初始化时算好的 N 和 L
        N = self.seq_len // self.patch_len
        L = self.n_vars * N  # 等同于 seq_len * c_out // patch_len
        
        masks = []
        for k in range(L):
            # S: 空间掩码 (同一时间，不同变量)
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype)
            # T: 时间掩码 (同一变量，不同时间)
            T = ((torch.arange(L) >= (k // N) * N) & (torch.arange(L) < (k // N) * N + N) & (torch.arange(L) != k)).to(dtype)
            # ST: 时空掩码 (不同变量，不同时间)
            ST = torch.ones(L).to(dtype) - S - T
            ST[k] = 0.0
            
            masks.append(torch.stack([S, T, ST], dim=0))
        print("N是", N,"L是", L,"n_vars是", self.n_vars, "patch_len", self.patch_len)
        # print("masks[0]是",masks[0].shape)
        masks = torch.stack(masks, dim=0)
        return masks
    
    def forward(self, x_enc,**kwargs):
        is_training = self.training
        target=None
        x = x_enc
        masks = self.fixed_masks
        # x: [B, T, C]
        B, T, C = x.shape
        # Normalization
        x = self.norm(x, 'norm')
        # x: [B, C, T]
        x = x.permute(0, 2, 1).reshape(-1, C*T) # [B, C*T]
        x = self.patch_embed(x) # [B, N, D]  N = [C*T / P]
        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)

        # [B, C, T/P, D]
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)) # [B, C, T]
        x = x.permute(0, 2, 1)
        # De-Normalization
        x = self.norm(x, 'denorm')

        return x, moe_loss
        