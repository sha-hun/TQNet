import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math

def sin_round(x):
    return x - torch.sin(2 * np.pi * x) / (2 * np.pi)
def sin_sin_round(x):
    return sin_round(sin_round(x))


class ResidualDecomposition(nn.Module):
    def __init__(self, config):
        super().__init__()

        C = config['enc_in']
        self.max_season_length = config['max_season_length']
        self.num_seasonal_components = config['num_seasonal_components']
        self.pred_len = config['pred_len']

        self.E_trend = nn.Parameter(torch.randn(config['trend_length'], C))
        self.E_seasonal = nn.Parameter(torch.randn(config['num_seasonal_components'], config['max_season_length'], C))
        # 每个周期的可学习长度参数（浮点数）
        self.raw_lengths = nn.Parameter(config['max_season_length'] * torch.rand(config['num_seasonal_components'], 1))

        # 从周期捕获对结果的影响
        self.E_seasonal_linear = nn.Sequential(
            nn.Linear(config['seq_len'] * C , config['pred_len'] * C),
            nn.Dropout(p = config['dropout']))

        self.unique_lengths = set()  # 用于记录实际使用的周期长度，辅助调试
        self.pre_len = 0
    
    def forward(self, x_enc, time_idx):
        """

        :param x_enc: [batch_size, seq_len, n_vars]
        :param time_idx: [batch_size, seq_len, 1]
        :return: 
            X_Residual (batch, seq_len, num_seasonal_components * 2, C)
            Y_seasonal (batch, pred_len, data_dim)
        """
        
        batch, T, C = x_enc.shape
        
        trend_length = self.E_trend.shape[0]

        # 使用高级索引
        time_idx_mod = (time_idx.squeeze(-1) % trend_length).long()  # (batch, seq_len)
        trend_seq = self.E_trend[time_idx_mod]  # (batch, seq_len, C)
        x_detrended = x_enc - trend_seq

        # -----------------------------
        # 多周期残差 很难训练 self.raw_lengths
        # -----------------------------
        X_Residual = []
        Y_seasonal_tot = 0
        for i in range(self.num_seasonal_components):
            # -----------------------------
            # round + STE (可微分近似整数)
            # -----------------------------
            raw_val = self.raw_lengths[i]
            if torch.isnan(raw_val) or torch.isinf(raw_val):
                # print(f"Warning: raw_lengths[{i}] is {raw_val}, resetting to 2.0")
                # raw_val = torch.tensor(2.0, device=self.raw_lengths.device)
                raw_val = torch.rand(1, device=self.raw_lengths.device) * (self.max_season_length - 2) + 2  # 随机初始化在合理范围内
                self.raw_lengths.data[i] = raw_val  # 更新参数，避免后续出现

            length_float = torch.clamp(raw_val, 2.0, self.max_season_length)
            # STE 近似 round
            length_ste = (length_float.round() - length_float).detach() + length_float
            Nsi = length_ste  # 保留浮点数，后续插值使用
            # print(f"Component {i}: float={length_float.item():.2f}, STE round={Nsi.item():.2f}")

            # -----------------------------
            # 循环取模索引，浮点索引
            # -----------------------------
            t_idx = time_idx.squeeze(-1).float() % Nsi  # (batch, seq_len), float

            # 插值索引
            left_idx = t_idx.floor().long()                        # (batch, seq_len)
            right_idx = (left_idx + 1) % self.max_season_length   # 循环
            weight = (t_idx - t_idx.floor()).unsqueeze(-1)        # (batch, seq_len, 1)

            self.unique_lengths.add(length_ste.long().item()) # 记录实际使用的周期长度，辅助调试
            # 如果发生变化，就输出
            if len(self.unique_lengths) > self.pre_len:
                # print(f"Component {i}: Updated unique lengths: {self.unique_lengths}")
                # print(f"Component {i}: length_float={length_float.item():.2f}, STE round={Nsi.item():.2f}")
                self.pre_len = len(self.unique_lengths)  # 记录实际使用的周期长度，辅助调试

            # 如果发生变化，就输出
            if len(self.unique_lengths) > self.pre_len:
                print(f"Component {i}: Updated unique lengths: {self.unique_lengths}")
                self.pre_len = len(self.unique_lengths)

            seasonal_left = self.E_seasonal[i, left_idx]   # (batch, seq_len, C)
            seasonal_right = self.E_seasonal[i, right_idx] # (batch, seq_len, C)
            seasonal_aligned = (1 - weight) * seasonal_left + weight * seasonal_right  # (batch, seq_len, C)

            # -----------------------------
            # 残差
            # -----------------------------
            residual_i = x_detrended - seasonal_aligned
            X_Residual.append(residual_i)

            Y_seasonal = self.E_seasonal_linear(seasonal_aligned.reshape(batch, -1)).reshape(batch, self.pred_len, -1)  # (batch, pred_len, C)
            Y_seasonal_tot = Y_seasonal_tot + Y_seasonal

        X_Residual = torch.stack(X_Residual, dim=2)  # (batch, seq_len, num_seasonal_components, C)
        return X_Residual, Y_seasonal


class HybridDecomposition(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.C = config['enc_in']
        self.S = config['num_seasonal_components']
        self.T = config['seq_len']


        # -----------------------------
        # 步骤1 ： 两个卷积分别处理 (T,2) 和 (T,S) 的特征，捕获时间维度的局部模式和跨组件的模式，并将它们融合到一起得到Hx，作为残差的增强特征
        # -----------------------------
        # 卷积1: (T,2)
        # 输入 (B, C*S, T, 2) -> 输出 (B, hidden_dim, T, 1)
        self.conv_time_dup = nn.Conv2d(
            in_channels=self.C * self.S,
            out_channels=config['hidden_dim'],
            kernel_size=(3, 2)  # 时间维度卷积3，复制列2
        )
        self.out1_dropout = nn.Dropout(p = config['dropout'])
        # 卷积2: (T,S)
        # 输入 (B, C, T, S) -> 输出 (B, hidden_dim, T, S)
        self.conv_time_season = nn.Conv2d(
            in_channels=self.C,
            out_channels=config['hidden_dim'],
            kernel_size=(3, self.S),  # 时间3，组件全卷积          
        )
        self.out2_dropout = nn.Dropout(p = config['dropout'])

        # -----------------------------
        # 步骤2 ： 使用掩码信息进行邻域卷积，捕获缺失值周围的局部模式，得到Hm_mask特征
        # -----------------------------
        # Embed (1-M)
        self.embed_m = nn.Linear(self.C, config['hidden_dim_mask'])

        # 一共 2^C 种组合，每个组合对应一个 embedding
        # self.embed = nn.Embedding(2**self.C, config['hidden_dim_mask'])

        # 邻域卷积
        self.conv_neighborhood = nn.Conv1d(
            in_channels=self.C,
            out_channels=config['hidden_dim_mask'],
            kernel_size=3,
        )

        self.Hm_attn = nn.MultiheadAttention(embed_dim=config['hidden_dim_mask'], num_heads=4, batch_first=True)
        self.Hm_attn_dropout = nn.Dropout(p = config['dropout'])

        # -----------------------------
        # 步骤3 ： 使用time_dif 跟 time_idx，捕获时间差分和时间索引的模式的模式，得到Ht特征和Hdelta特征
        # -----------------------------
        # 可学习的 tau 参数，正值
        self.log_tau_idx = nn.Parameter(torch.zeros(1))  # 对 time_idx
        self.log_tau_dif = nn.Parameter(torch.zeros(1))  # 对 time_dif

        self.A_idx_linear = nn.Linear(self.T, config['hidden_dim_mask'])
        self.A_dif_linear = nn.Linear(self.C * self.T, config['hidden_dim_mask'])

        self.mlp_gm = nn.Sequential(
            nn.Linear(3 * config['hidden_dim_mask'], config['hidden_dim_mask']),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(config['hidden_dim_mask'], config['hidden_dim_mask'])  # 输出可以是任意维度，后面再做 Gm = X @ X^T
        )

        print("C:", self.C)
        print("hidden_dim_mask:", config['hidden_dim_mask'])

    def forward(self, X_Residual, mask, time_dif, time_idx):
        """
        X_Residual: (B, T, S, C)
        mask : (B, T, C)
        time_dif : (B, T, C)
        time_idx : (B, T, 1)

        return :
        Hx: (B, hidden_dim, T, 1) 通过卷积捕获的增强特征
        Gm: (B, T, T) 通过掩码和时间特征捕获的注意力偏置
        """
        B, T, S, C = X_Residual.shape

        # 这里为了 (B, C_in, H, W)

        # -----------------------------
        # 步骤1
        # -----------------------------
        # 卷积1: (T,2)  复制时间列:
        # (B, T, 2, S, C)-> (B, T, 2, S*C) -> (B, S*C, T, 2)
        X_dup = X_Residual.unsqueeze(2).repeat(1,1,2,1,1).reshape(B, T, 2, S * C).permute(0, 3, 1, 2)  
        X_padded = F.pad(X_dup, pad=(0, 0, 2, 0),value=0)               # pad = (W_left, W_right, H_top, H_bottom)
        out1 = F.gelu(self.conv_time_dup(X_padded) )                    # (B, S*C, T, 2) -> (B, hidden_dim, T, 1)
        out1 = self.out1_dropout(out1)  # 卷积1的输出增加 dropout

        # 卷积2: (T,S)
        X_conv2 = X_Residual.permute(0, 3, 1, 2)                        # (B, C, T, S)
        X_conv2_padded = F.pad(X_conv2, pad=(0, 0, 2, 0), value=0)      # pad = (W_left, W_right, H_top, H_bottom)
        out2 = F.gelu(self.conv_time_season(X_conv2_padded))            # (B, C, T, S) -> (B, hidden_dim, T, 1)
        out2 = self.out2_dropout(out2)  # 卷积2的输出增加 dropout

        Hx = out1 + out2  # (B, hidden_dim, T, 1)
        # -----------------------------
        # 步骤2
        # -----------------------------
        inv_M = 1.0 - mask  # (B, T, C)

        Hm_init = self.embed_m(inv_M)  # (B, T, hidden_dim_mask)
        # powers_of_two = 2**torch.arange(C-1, -1, -1, device=inv_M.device)
        # indices = (inv_M.long() * powers_of_two).sum(dim=-1)  # [B, T]
        # inv_M_long = inv_M.long()  # nn.Embedding 需要 Long 类型
        # Hm_init = self.embed_m(inv_M_long)

        # print(f"inv_M shape: {inv_M.shape}, Hm_init shape: {Hm_init.shape}")
        # print("Weight NaN:", torch.isnan(self.embed_m.weight).any())
        # print("Bias NaN:", torch.isnan(self.embed_m.bias).any())
        # print("Weight inf:", torch.isinf(self.embed_m.weight).any())
        # print("Bias inf:", torch.isinf(self.embed_m.bias).any())

        if torch.isnan(Hm_init).any():
            print("Warning: Hm_init contains NaN values!")
        inv_M_pad = F.pad(inv_M.permute(0, 2, 1), pad=(2, 0), value=0)  # (B, C, T) -> (B, C, T+2)
        mask_conv = F.gelu(self.conv_neighborhood(inv_M_pad))          # (B, hidden_dim_mask, T)   

        Hm_init = Hm_init + mask_conv.permute(0, 2, 1)  # (B, T, hidden_dim_mask)
        Hm, _ = self.Hm_attn(Hm_init, Hm_init, Hm_init)  # (B, T, hidden_dim_mask)
        Hm = self.Hm_attn_dropout(Hm)  # 注意力输出增加 dropout

        if torch.isnan(Hm).any():
            print("Warning: Hm self-attention output contains NaN values!")

        # -----------------------------
        # 步骤3
        # -----------------------------
        # 计算时间衰减矩阵
        tau_idx = torch.exp(self.log_tau_idx)
        tau_dif = torch.exp(self.log_tau_dif)

        # 处理 A_idx
        t_i = time_idx.float()  # (B, T, 1)
        A_idx = torch.exp(-abs(t_i - t_i.transpose(1, 2)) / tau_idx)  # (B, T, T)
        if torch.isnan(A_idx).any():
            print("注意: A_idx contains NaN values!")
        # 处理 A_dif ，使用广播计算每个通道的衰减矩阵
        t_d = time_dif.float()  # (B, T, C)
        t_d_i = t_d.unsqueeze(2)  # (B, T, 1, C)
        t_d_j = t_d.unsqueeze(1)  # (B, 1, T, C)
        A_dif = torch.exp(-abs(t_d_j - t_d_i) / tau_dif)  # (B, T, T, C)

        # 计算 H_idx 和 H_dif
        H_idx = self.A_idx_linear(A_idx)  # (B, T, T) -> (B, T, hidden_dim_mask)
        # print(f"A_dif shape: {A_dif.shape}")
        H_dif = self.A_dif_linear(A_dif.reshape(B, T, T * C))  # (B, T, T*C) -> (B, T, hidden_dim_mask)
        if torch.isnan(H_idx).any():
            print("Warning: H_idx contains NaN values!")
        if torch.isnan(H_dif).any():
            print("Warning: H_dif contains NaN values!")
        # 融合三个表示 MLP 生成注意力 bias Gm
        H_concat = torch.cat([Hm, H_idx, H_dif], dim=-1)  # (B, T, 3*D)
        if torch.isnan(H_concat).any():
            print("Warning: H_concat contains NaN values!")
        Gm = self.mlp_gm(H_concat)  # (B, T, hidden_dim_mask)
        if torch.isnan(Gm).any():
            print("Warning: Gm after MLP contains NaN values!")

        return Hx, Gm

class HighFreqFFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.keep_ratio = config.get('high_freq_ratio', 0.2)  # 保留高频比例
        self.S = config['num_seasonal_components']
        self.C = config['enc_in']  # 每个组件的特征维度

        # 投影到隐藏维度
        self.freq_proj = nn.Linear(self.S * self.C, self.hidden_dim)
        self.dropout = nn.Dropout(p = config['dropout'])

    def forward(self, X_Residual):
        """
        X_Residual: (B, T, S, C)
        输出:
        Hf: (B, T, hidden_dim)
        """
        B, T, S, C = X_Residual.shape

        # 将 S,C 维度合并为通道
        X_flat = X_Residual.reshape(B, T, S * C)  # (B, T, S*C)

        # FFT 到频域
        X_fft = torch.fft.rfft(X_flat, dim=1)  # (B, Freq_len, S*C)
        Freq_len = X_fft.shape[1]

        # 只保留高频部分
        keep_k = max(1, int(Freq_len * self.keep_ratio))
        X_fft_filtered = torch.zeros_like(X_fft)
        X_fft_filtered[:, -keep_k:, :] = X_fft[:, -keep_k:, :]  # 高频保留

        # IFFT 回到时间域
        X_ifft = torch.fft.irfft(X_fft_filtered, n=T, dim=1)  # (B, T, S*C)

        # 投影到隐藏维度
        Hf = self.freq_proj(X_ifft)  # (B, T, hidden_dim)
        Hf = F.relu(Hf)
        Hf = self.dropout(Hf)  # 高频特征增加 dropout
        return Hf

class BiasedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        self.hidden_dim = hidden_dim
        self.num_heads = config['num_heads']
        T = config['seq_len']
        assert hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.d_k = hidden_dim // self.num_heads


        # Gm 线性映射
        self.linear = nn.Linear(config['hidden_dim_mask'], T)
        # Q,K,V 线性映射
        self.q_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # Hx + Hf
        self.k_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, config['enc_in'])  # 输出维度为 enc_in，后续会补齐到 data_dim
        self.dropout = nn.Dropout(p = config['dropout'])

    def forward(self, Hx, Hf, Gm):
        """
        Hx: (B, hidden_dim, T, 1)
        Hf: (B, T, hidden_dim)
        Gm: (B, num_heads, T, T) or (B, T, T) 注意力偏置
        """
        Hx = Hx.squeeze(-1)  # (B, hidden_dim, T)
        Hx = Hx.permute(0, 2, 1)  # (B, T, hidden_dim)
        B, T, _ = Hx.shape
    
        # 拼接 Hx,Hf
        H_cat = torch.cat([Hx, Hf], dim=-1)  # (B, T, hidden_dim*2)

        # 线性映射
        Q = self.q_proj(H_cat)  # (B, T, hidden_dim)
        K = self.k_proj(H_cat)
        V = self.v_proj(Hx)

        # 分头
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1,2)  # (B, heads, T, d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1,2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)  # (B, heads, T, T)

        # 加入偏置
        Gm = self.linear(Gm)  # (B, T, T)
        Gm = Gm.unsqueeze(1)  # (B, 1, T, T)
        scores = scores + Gm  # (B, heads, T, T)

        attn = F.softmax(scores, dim=-1)  # (B, heads, T, T)

        # 输出
        out = torch.matmul(attn, V)  # (B, heads, T, d_k)
        out = out.transpose(1,2).contiguous().view(B, T, self.hidden_dim)  # (B, T, hidden_dim)
        out = self.out_proj(out)  # (B, T, hidden_dim)
        out = self.dropout(out)  # 注意力输出增加 dropout

        return out


# def init_weights(module):
#     """
#     自动初始化模块权重  在HybridDecomposition的self.embed_m = nn.Linear(self.C, config['hidden_dim_mask'])中，Weight跟Bias经过训练后出现了nan
#     """
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight)  # Xavier uniform
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)         # bias 全部置零
#     elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
#         nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, mean=0, std=0.01)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # print("初始化 ZWF model with config:", config)
        # for key, value in vars(config).items():
        #     print(f"{key}: {value}")
        config = vars(args)
        T = config['seq_len']
        C = config['enc_in']
        config['pred_len'] = config['pred_len']
        # print(f"ZWF model initialized with T={T}, C={C}")
        self.T = T
        self.C = C

        self.pred_len = config['pred_len']

        self.Linear = nn.Linear( 2 * config['num_seasonal_components'] * C, config['enc_in'])

        self.residual_decomposition = ResidualDecomposition(config=config)

        self.hybrid_decomposition = HybridDecomposition(config=config)

        self.high_freq_fft = HighFreqFFT(config=config)

        self.Y_atten = BiasedMultiheadAttention(config=config)

        self.Y_linear = nn.Linear(config['seq_len'], config['pred_len'])

        # 递归初始化权重，确保所有子模块都被正确初始化
        # self.apply(init_weights)

    def forward(self, input, x_mark, dec_inp, y_mark):  # 这里输入的y是init特征,第二维为label_len + pred_len
        # x_enc: [batch_size, seq_len, c]
        # mask: [batch_size, seq_len, c]
        # time_dif: [batch_size, seq_len, c]
        # time_idx: [batch_size, seq_len, 1]

        batch, T, data_dim = input.shape
        # print(f"输入的形状 : {input.shape}")
        x_enc = input[:, :, :self.C]  # (B, T, C)
        mask = input[:, :, self.C:2*self.C]  # (B, T, C)
        time_dif = input[:, :, 2*self.C:3*self.C]  #
        time_idx = input[:, :, 3*self.C:3*self.C+1]  # (B, T, 1)

        # 看x_enc里有没有nan
        if torch.isnan(input).any():
            print("注意: input 包含 NaN !")



        # print(f"x_enc shape: {x_enc.shape}")
        # print(f"mask shape: {mask.shape}")
        # print(f"time_dif shape: {time_dif.shape}")
        # print(f"time_idx shape: {time_idx.shape}")
        


        X_Residual, Y_seasonal_tot = self.residual_decomposition(x_enc, time_idx)  # (batch, seq_len, num_seasonal_components, C)
        # 看X_Residual,Y_seasonal_tot里有没有nan
        if torch.isnan(X_Residual).any():
            print("Warning: X_Residual contains NaN values!")
        if torch.isnan(Y_seasonal_tot).any():
            print("Warning: Y_seasonal_tot contains NaN values!")

        Hx, Gm = self.hybrid_decomposition(X_Residual, mask, time_dif, time_idx)   # (B, hidden_dim, T, 1), (B, T, hidden_dim_mask)
        # 看Hx,Gm里有没有nan        Gm出现了nan
        if torch.isnan(Hx).any():
            print("Warning: Hx contains NaN values!")
        if torch.isnan(Gm).any():
            print("Warning: Gm contains NaN values!")
        Hf = self.high_freq_fft(X_Residual)                                         # (B, T, hidden_dim)
        if torch.isnan(Hf).any():
            print("Warning: Hf contains NaN values!")
        Y = self.Y_atten(Hx, Hf, Gm)  # (B, T, C)
        if torch.isnan(Y).any():
            print("Warning: Y self.Y_atten contains NaN values!")
        # print(f"Y shape after linear: {Y.shape}")
        Y = self.Y_linear(Y.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, c)
        if torch.isnan(Y).any():
            print("Warning: Y after Y_linear contains NaN values!")
        # print(f"Y shape after linear2: {Y.shape}")

        output = Y  + Y_seasonal_tot  # [batch_size, pred_len, c]
        # 看output,Y,Y_seasonal_tot里有没有nan
        if torch.isnan(output).any():
            print("Warning: output contains NaN values!")

        return output
