import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class ResidualDecomposition(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 输出config的值
        for key, value in config.items():
            print(f"{key}: {value}")

        C = config['enc_in']
        self.max_season_length = config['max_season_length']
        self.num_seasonal_components = config['num_seasonal_components']
        self.pred_len = config['pred_len']
        self.seq_len = config['seq_len']

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
    
    def forward(self, x_enc, cycle_index):
        """

        :param x_enc: [batch_size, seq_len, n_vars]
        :param time_idx: [batch_size, seq_len, 1]
        :return: 
            X_Residual (batch, seq_len, num_seasonal_components * 2, C)
            Y_seasonal (batch, pred_len, data_dim)
        """
        # 这里的cycle_index是输入序列中每个时间步在周期中的位置索引，只有一个值
        time_idx = cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)

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
        Seasonal_aligneds = []
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
            # print('seasonal_aligned shape:', seasonal_aligned.shape)
            Seasonal_aligneds.append(seasonal_aligned)
        
        # Seasonal_aligneds_tensor = torch.stack(Seasonal_aligneds, dim=0)
        # return Seasonal_aligneds_tensor.mean(dim=0)  # (batch, seq_len, C)
            Y_seasonal_tot = Y_seasonal_tot + seasonal_aligned


        Y_seasonal_tot = Y_seasonal_tot / self.num_seasonal_components
        X_Residual = torch.stack(X_Residual, dim=2).mean(dim=2)  # (batch, seq_len, num_seasonal_components, C) 
        return X_Residual, Y_seasonal_tot # (batch, seq_len, C), (batch, pred_len, C)

class InputMaskGenerator(nn.Module):
    def __init__(self,config):
        """
        input_dim: 输入特征维度 C
        hidden_dim: 投影到的隐藏维度 d
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.9))  # 可学习的 alpha 参数，控制缺失信息权重的下限
        input_dim = config['enc_in']
        hidden_dim = config['hidden_dim_mask']
        seq_len = config['seq_len']
        # 特征相似度投影
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        # 局部统计投影
        self.local_proj = nn.Linear(input_dim, hidden_dim)
        # 融合 S, L 的小 MLP
        self.fusion_layer = nn.Sequential(
            nn.Linear(2*seq_len , seq_len),
            nn.Sigmoid()
        )
    
    def forward(self, X, mask=None, training=True):
        """
        X: (B, T, C)
        mask: (B, T, C) 原始 mask，可选
        return: M_final (B, T, C)
        """
        B, T, C = X.shape
        
        # -----------------------------
        # 特征投影
        # -----------------------------
        X_feat = self.feature_proj(X)      # (B, T, d)
        X_local = self.local_proj(X)       # (B, T, d)

        # -----------------------------
        # 特征相似度矩阵 S
        # -----------------------------
        S = F.cosine_similarity(
            X_feat.unsqueeze(2),  # (B, T, 1, d)
            X_feat.unsqueeze(1),  # (B, 1, T, d)
            dim=-1
        )  # (B, T, T)

        # -----------------------------
        # 局部统计矩阵 L
        # -----------------------------
        L = torch.abs(X_local.unsqueeze(2) - X_local.unsqueeze(1)).mean(-1)  # (B, T, T)

        fused = torch.cat([S, L], dim=-1)  # (B, T, 2*T)
        P = self.fusion_layer(fused) # (B, T, T)

        # -----------------------------
        # 缺失信息权重 W
        # -----------------------------
        if mask is not None:
            v = mask.float().mean(-1)  # (B, T), 每个时间步可见比例
            W = self.alpha + (1 - self.alpha) * (v.unsqueeze(2) * v.unsqueeze(1))  # (B, T, T)
        else:
            W = 1.0
        
        # print("W的形状",W.shape)
        # -----------------------------
        # 融合得到概率矩阵 P
        # -----------------------------

        P = P * W  # 考虑缺失信息权重

        # -----------------------------
        # 采样得到 mask
        # -----------------------------
        # if training:
        #     M_final = torch.bernoulli(P)
        # else:
        #     M_final = (P > 0.5).float()
        M_final = P
        return M_final



class HybridDecomposition(nn.Module):
    def __init__(self,config):
        super().__init__()


        self.C = config['enc_in']
        self.T = config['seq_len']

        # -----------------------------
        # 步骤1 ： 两个卷积分别处理 (T,2) 和 (C,2) 的特征，捕获时间维度的局部模式和跨组件的模式，并将它们融合到一起得到Hx，作为残差的增强特征
        # -----------------------------
        # 卷积1: (T,2)
        # 输入 (B, C, T) -> 复制时间列 -> (B, C, T, 2) -> 输出 (B, hidden_dim, T, 1)
        self.conv_time_dup = nn.Conv2d(
            in_channels=self.C,
            out_channels=self.C, #config['hidden_dim'],
            kernel_size=(3, 2)  # 时间维度卷积3，复制列2
        )
        self.out1_dropout = nn.Dropout(p = config['dropout'])
        self.out1_linear = nn.Linear(config['seq_len'], config['d_model'])  # 卷积1的输出维度 -> 输入维度，方便后续融合

        # -----------------------------
        # 步骤2 ： 使用掩码信息进行邻域卷积，捕获缺失值周围的局部模式，得到Hm_mask特征
        # -----------------------------
        # Embed (1-M)
        self.embed_m = nn.Linear(self.C, config['hidden_dim_mask'])

        # 一共 2^C 种组合，每个组合对应一个 embedding
        # self.embed = nn.Embedding(2**self.C, config['hidden_dim_mask'])

        # 邻域卷积 输入(B, C, T+2)
        self.conv_neighborhood = nn.Conv1d(
            in_channels=self.C,
            out_channels=self.C,
            kernel_size=5,
        )
        self.conv_Linear = nn.Linear(self.C, config['hidden_dim_mask'])  # 卷积输出维度 -> 注意力输入维度

        self.Hm_attn = nn.MultiheadAttention(embed_dim=config['hidden_dim_mask'], num_heads=4, batch_first=True)
        self.Hm_attn_dropout = nn.Dropout(p = config['dropout'])
        self.Hm_linear = nn.Linear(config['hidden_dim_mask'], self.C)  
        self.Hm_liner2 = nn.Linear(self.T, config['hidden_dim_mask']) 

        # -----------------------------
        # 步骤3 ： 使用time_dif 跟 time_idx，捕获时间差分和时间索引的模式的模式，得到Ht特征和Hdelta特征
        # -----------------------------
        # 可学习的 tau 参数，正值
        self.log_tau_idx = nn.Parameter(torch.zeros(1))  # 对 time_idx
        self.log_tau_dif = nn.Parameter(torch.zeros(1))  # 对 time_dif

        self.A_idx_to_C = nn.Linear(self.T, self.C)
        self.A_idx_linear = nn.Linear(self.T, config['hidden_dim_mask'])
        self.A_dif_linear = nn.Linear(self.T, config['hidden_dim_mask'])

        self.mlp_gm = nn.Sequential(
            nn.Linear(3 * config['hidden_dim_mask'], config['hidden_dim_mask']),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(config['hidden_dim_mask'], config['d_model'])  # 输出可以是任意维度，后面再做 Gm = X @ X^T
        )

        seq_len = config['seq_len']
        self.a_1 = nn.Parameter(torch.zeros(1, seq_len))
        self.a_2 = nn.Parameter(torch.zeros(seq_len, 1))
        self.b_1 = nn.Parameter(torch.zeros(1, seq_len))
        self.b_2 = nn.Parameter(torch.zeros(seq_len, 1))
        self.seq_len = seq_len
        distance = torch.abs(torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0))
        self.diff = nn.Parameter(torch.abs(torch.min(distance%seq_len, (-distance)%seq_len)).float(),
                                 requires_grad=False)
        
        self.input_mask_generator = InputMaskGenerator(config)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.logger =  logging.getLogger(f"logger_{config['logger_uique_id']}")
        self._logged_log_func_once = False
    def func(self):
        a = torch.sigmoid((self.a_1@self.a_2))
        b = torch.sigmoid((self.b_1@self.b_2)) * self.seq_len
        return 1/(1+torch.exp(a*(self.diff-b))) + torch.exp(-self.diff)/(1+torch.exp(a*b))



    def forward(self, X_Residual, mask, time_dif, time_idx):
        """
        X_Residual: (B, T, C)
        mask : (B, T, C)
        time_dif : (B, T, C)
        time_idx : (B, T, 1)

        return :
        Hx: (B, hidden_dim, T, 1) 通过卷积捕获的增强特征
        Gm: (B, T, T) 通过掩码和时间特征捕获的注意力偏置
        """
        B, T, C = X_Residual.shape

        # 这里为了 (B, C_in, H, W)

        # -----------------------------
        # 步骤1
        # -----------------------------
        # 卷积1: (T,2)  复制时间列:
        # (B, T, C) -> (B, C, T) -> (B, C, T, 2)
        X_dup = X_Residual.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, 2)  # -> (B, C, T, 2)

        X_padded = F.pad(X_dup, pad=(0, 0, 2, 0),value=0)               # pad = (W_left, W_right, H_top, H_bottom)
        out1 = F.gelu(self.conv_time_dup(X_padded) )                    # (B, C, T, 2) -> (B, C, T, 1)
        out1 = self.out1_dropout(out1)  # 卷积1的输出增加 dropout
        Hx = out1.squeeze(3).permute(0, 2, 1)  # (B, C, T, 1) -> (B, C, T) -> (B, T, C)
        # Hx = self.out1_linear(out1.squeeze(3))  # (B, C, T) -> (B, C, d_model)

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
        inv_M_pad = F.pad(inv_M.permute(0, 2, 1), pad=(4, 0), value=0)  # (B, C, T) -> (B, C, T+4)
        mask_conv = F.gelu(self.conv_neighborhood(inv_M_pad))          # (B, C, T)
        mask_conv = self.conv_Linear(mask_conv.permute(0, 2, 1))       # (B, C, T) ->(B, T, C) -> (B, T, hidden_dim_mask)

        Hm_init = Hm_init + mask_conv  # (B, T, hidden_dim_mask)

        # Hm, _ = self.Hm_attn(Hm_init, Hm_init, Hm_init)  # (B, T, hidden_dim_mask)
        # Hm = self.Hm_attn_dropout(Hm)  # 注意力输出增加 dropout
        # Hm = F.gelu(self.Hm_linear(Hm))  # (B, T, hidden_dim_mask) -> (B, T, C) # 将emb转回特征维度，并将T转到hid维度
        
        Hm = F.gelu(self.Hm_linear(Hm_init))  # (B, T, hidden_dim_mask) -> (B, T, C) # 将emb转回特征维度，并将T转到hid维度
        Hm = self.Hm_liner2(Hm.permute(0, 2, 1))  # (B, T, C) ->(B, C, T)-> (B, C, hidden_dim_mask)

        # -----------------------------
        # 步骤3
        # -----------------------------
        # 计算时间衰减矩阵
        input_mask = self.input_mask_generator(X_Residual, mask=mask, training=self.training)  # (B, T, T)
        tau_idx = torch.exp(self.log_tau_idx)
        tau_dif = torch.exp(self.log_tau_dif)

        # 处理 H_idx
        t_i = time_idx.float()  # (B, T, 1)
        func_val = self.func()  # (T, T)
        log_func = torch.log(func_val)  # (T, T)

        t1 = torch.exp(-abs(t_i - t_i.transpose(1, 2)) / tau_idx) # (B,T,T)


        print("tau_idx的值是", tau_idx.item())
        print("tau_dif的值是", tau_dif.item())
        import matplotlib.pyplot as plt
        matrix = t1[0].detach().numpy()  # (T, T)

        # 绘制热力图
        plt.figure(figsize=(8,6))
        plt.imshow(matrix, cmap='viridis', origin='lower')  # origin='lower' 保证左下是(0,0)
        plt.colorbar(label='exp(-|t_i - t_j| / tau_idx)')
        plt.xlabel('t_j')
        plt.ylabel('t_i')
        plt.title('时间衰减热力图')
        plt.show()

        A_idx = torch.exp(-abs(t_i - t_i.transpose(1, 2)) / tau_idx) +log_func  # (B, T, T)


        # a_scalar = torch.sigmoid((self.a_1 @ self.a_2)).item()
        # b_scalar = (torch.sigmoid((self.b_1 @ self.b_2)) * self.seq_len).item()
        # self.logger.info(
        #     "log_func stats shape=%s min=%.6g max=%.6g mean=%.6g std=%.6g | func min=%.6g max=%.6g | a=%.6g b=%.6g",
        #     tuple(log_func.shape),
        #     log_func.min().item(), log_func.max().item(), log_func.mean().item(), log_func.std().item(),
        #     func_val.min().item(), func_val.max().item(),
        #     a_scalar, b_scalar,
        # )

        
        A_idx = A_idx * (1 + self.alpha * input_mask)  # 考虑缺失信息权重
        H_idx = self.A_idx_linear(A_idx)  # (B, T, T) -> (B, T, hidden_dim_mask)
        # (B, T, hidden_dim_mask) -> (B, hidden_dim_mask, T) -> (B, hidden_dim_mask, C) -> (B, C, hidden_dim_mask)
        H_idx = self.A_idx_to_C(F.gelu(H_idx).permute(0, 2, 1)).permute(0, 2, 1)

        # 处理 H_dif ，使用广播计算每个通道的衰减矩阵
        t_d = time_dif.float()  # (B, T, C)
        t_d_i = t_d.unsqueeze(2)  # (B, T, 1, C)
        t_d_j = t_d.unsqueeze(1)  # (B, 1, T, C)
        A_dif = torch.exp(-abs(t_d_j - t_d_i) / tau_dif) + log_func.unsqueeze(-1) # (B, T, T, C)
        A_dif = A_dif * (1 + self.alpha * input_mask.unsqueeze(-1)) # 考虑缺失信息权重

        A_dif = A_dif.permute(0, 3, 1, 2).mean(dim=3)  #(B, T, T, C) -> (B, C, T, T) -> (B, C, T)
        H_dif = self.A_dif_linear(A_dif)  # (B, C, T) -> (B, C, hidden_dim_mask)


        # 融合三个表示 MLP 生成注意力 bias Gm
        H_concat = torch.cat([Hm, H_idx, H_dif], dim=-1)  # (B, C, 3*D)

        Gm = self.mlp_gm(H_concat)  # (B, C, hidden_dim_mask)

        return Hx, Gm   # (B, T, C), (B, C, d_model)
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.C = configs.enc_in

        self.use_tq = True  # ablation parameter, default: True
        self.channel_aggre = True   # ablation parameter, default: True

        config_zwf = vars(configs)
        self.residual_decomposition = ResidualDecomposition(config=config_zwf)
        

        if self.use_tq:
            # self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True) # [24,7]
            self.hybird_decomposition = HybridDecomposition(config=config_zwf)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        self.seasonal_weight = nn.Parameter(torch.tensor(0.1))

        self.seq_len_to_d_1 = nn.Linear(self.seq_len, self.d_model)
        self.seq_len_to_d_2 = nn.Linear(self.seq_len, self.d_model)


    def forward(self, all_data, x_mark, dec_inp, y_mark, y_true, cycle_index):  # (batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y,batch_cycle)

        x = all_data[:, :, :self.C]  # (B, T, C)
        mask = all_data[:, :, self.C:2*self.C]  # (B, T, C)
        time_dif = all_data[:, :, 2*self.C:3*self.C]  # (B, T, C)
        time_idx = all_data[:, :, 3*self.C:3*self.C+1] # (B, T, 1)

        # cycle_index shape: (b,) 代表每个样本对应的cycle索引，范围是[0, cycle_len-1]，表示输入序列的起始时间点在周期中的位置
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)


        # b,T,c -> b,c,T    c是enc_in
        x_input = x.permute(0, 2, 1)

        # print("cycle_index shape:", cycle_index.shape)
        # print("值是:", cycle_index)
        query_input ,Y_seasonal = self.residual_decomposition(x_input.permute(0, 2, 1), cycle_index=cycle_index)
        # query_input ,Y_seasonal = self.residual_decomposition(x_input.permute(0, 2, 1), cycle_index=time_idx[:, 0, 0]) # (b, t, c), (b, t, c)

        query_input = query_input.permute(0, 2, 1)  # (b, s, c) -> (b, c, s)

        channel_information = self.channelAggregator( query=query_input, key=query_input, value=query_input )[0]

        Hx, Gm = self.hybird_decomposition(x, mask=mask, time_dif=time_dif, time_idx=time_idx) # (b, t, c), (b, C, d_model)
        
        Hx_d = self.seq_len_to_d_1(Hx.permute(0, 2, 1))  #(b,t,c) -> (b, c, t) -> (b, c, d_model)
        Y_seasonal_d = self.seq_len_to_d_2(Y_seasonal.permute(0, 2, 1)) #(b,t,c) -> (b, c, t) -> (b, c, d_model)
        attention_bias = Hx_d * Y_seasonal_d  # element-wise 融合 -> (b, c, d_model)


        input = self.input_proj(query_input + channel_information) # (b, c, T) -> (b, C, d_model)
        # print("Hx的形状", Hx.shape,"Gm的形状", Gm.shape, "input的形状", input.shape)
            
        # print("input的形状",input.shape,"Gm的形状", Gm.shape, "attention_bias的形状", attention_bias.shape)
        hidden = self.model(input + Gm + attention_bias) # (b, C, d_model)


        output = self.output_proj(hidden+input).permute(0, 2, 1) #(b, C, d_model) -> (b, C, pred_len) -> (b, pred_len, C)

        # instance denorm
        if self.use_revin:
            output_main = output * torch.sqrt(seq_var) + seq_mean

        output = output_main
        # output = output_main + nn.Tanh(Hx) * Y_seasonal

        return output, None