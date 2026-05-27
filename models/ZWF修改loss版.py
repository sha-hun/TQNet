import torch
import torch.nn as nn


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


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        import logging
        self.logger =logging.getLogger(f"logger_{configs.logger_uique_id}")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.use_tq = True  # ablation parameter, default: True
        self.channel_aggre = True   # ablation parameter, default: True

        config_zwf = vars(configs)
        if self.use_tq:
            # self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True) # [24,7]
            self.residual_decomposition = ResidualDecomposition(config=config_zwf)

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

    def forward(self, x, x_mark, dec_inp, y_mark, y_true, cycle_index):  # (batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y,batch_cycle)
        # cycle_index shape: (b,) 代表每个样本对应的cycle索引，范围是[0, cycle_len-1]，表示输入序列的起始时间点在周期中的位置
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        # print(f"输入模型的x的形状是{x.shape}，cycle_index的形状是{cycle_index.shape}")
        # b,s,c -> b,c,s    这里s是seq_len也就是T，c是enc_in
        x_input = x.permute(0, 2, 1)

        # gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        # # gather_index [B,T] 代表每个时间步对应的周期索引，范围是[0, cycle_len-1]，表示输入序列中每个时间步在周期中的位置
        # query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
        query_input ,Y_seasonal = self.residual_decomposition(x_input.permute(0, 2, 1), cycle_index=cycle_index)  # 传入b,s,c返回b,s,c
        query_input = query_input.permute(0, 2, 1)  # (b, s, c) -> (b, c, s)

        channel_information = self.channelAggregator( query=query_input, key=query_input, value=query_input )[0]

        input = self.input_proj(query_input + channel_information)
        hidden = self.model(input)
        output_main = self.output_proj(hidden + input).permute(0, 2, 1) # (b, pred_len, c)

        
        if self.use_revin:
            output_main = output_main * torch.sqrt(seq_var) + seq_mean
        
        output = output_main + self.seasonal_weight * Y_seasonal
        

        return output, None