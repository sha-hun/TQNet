import torch
import torch.nn as nn

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

        self.use_tq = True  # ablation parameter, default: True
        self.channel_aggre = True   # ablation parameter, default: True

        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True) # [24,7]

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


    def forward(self, x_enc, cycle_index, **kwargs):
        # cycle_index shape: (b,) 代表每个样本对应的cycle索引，范围是[0, cycle_len-1]，表示输入序列的起始时间点在周期中的位置
        # instance norm
        x = x_enc
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        # print(f"输入模型的x的形状是{x.shape}，cycle_index的形状是{cycle_index.shape}")
        # b,s,c -> b,c,s    这里s是seq_len也就是T，c是enc_in
        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            # gather_index [B,T] 代表每个时间步对应的周期索引，范围是[0, cycle_len-1]，表示输入序列中每个时间步在周期中的位置
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        input = self.input_proj(x_input+channel_information)

        hidden = self.model(input)

        output = self.output_proj(hidden+input).permute(0, 2, 1)

        # instance denorm
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean
        
        return output,None


