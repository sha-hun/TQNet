import torch
import torch.nn as nn
from layers.Embed import PatchEmbed
from layers.TimeBridge_EncDec import CointAttention, IntAttention, PatchSampling, ResAttention, TSEncoder, TSMixer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin  # long-term with temporal

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if not configs.num_p:
            configs.num_p = self.num_p

        self.embedding = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )

    def layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_enc = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)

        return dec_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    @staticmethod
    def add_model_specific_args(parent_parser):
        # 👇 核心秘籍：开启“覆盖模式”。遇到冲突时，新的覆盖旧的！
        parent_parser.conflict_handler = 'resolve'
        
        parser = parent_parser.add_argument_group('DistDF-TimeBridge Model Specific Arguments')
        parser.add_argument('--ia_layers', type=int, default=1, help='num of integrated attention layers')
        parser.add_argument('--pd_layers', type=int, default=1, help='num of patch downsampled layers')
        parser.add_argument('--ca_layers', type=int, default=0, help='num of cointegrated attention layers')
        parser.add_argument('--stable_len', type=int, default=6, help='length of moving average in patch norm')
        parser.add_argument('--num_p', type=int, default=0, help='num of down sampled patches')
        parser.add_argument('--attn_dropout', type=float, default=0.15, help='dropout rate of attention map')
        parser.add_argument('--period', type=int, default=24, help='length of patches')

        parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')

        parser.add_argument('--stride', type=int, default=8, help='stride')
        parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
        parser.add_argument('--auxi_lambda', type=float, default=1, help='weight of auxilary function')
        parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
        from utils.tools import EvalAction
        parser.add_argument('--var_weight', type=float, default=1.0, help="variance weight", action=EvalAction)
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
        parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
        parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        parser.add_argument('--joint_forecast', type=int, default=0, help='joint forecast; True 1 False 0')
        parser.add_argument('--ot_type', type=str, default='emd1d_h', help="type of ot distance, choices in ['emd1d_h']")
        parser.add_argument('--normalize', type=int, default=1, help="normalize ot distance matrix")
        parser.add_argument('--distance', type=str, default="time", help="distance metric for ot")
        parser.add_argument('--mask_factor', type=float, default=0.01, help="mask factor for mask matrix")
        parser.add_argument('--reg_sk', type=float, default=0.1, help="strength of entropy regularization in Sinkhorn")
        parser.add_argument('--eps', type=float, default=1e-9, help='epsilon for numerical stability in CCA')

                
        parser.add_argument('--cf_dim',         type=int, default=48)   #feature dimension
        parser.add_argument('--cf_drop',        type=float, default=0.2)#dropout
        parser.add_argument('--cf_depth',       type=int, default=2)    #Transformer layer
        parser.add_argument('--cf_heads',       type=int, default=6)    #number of multi-heads
        #parser.add_argument('--cf_patch_len',  type=int, default=16)   #patch length
        parser.add_argument('--cf_mlp',         type=int, default=128)  #ff dimension
        parser.add_argument('--cf_head_dim',    type=int, default=32)   #dimension for single head
        parser.add_argument('--cf_weight_decay',type=float, default=0)  #weight_decay
        parser.add_argument('--cf_p',           type=int, default=1)    #patch_type
        parser.add_argument('--use_nys',           type=int, default=0)    #use nystrom
        parser.add_argument('--mlp_drop',           type=float, default=0.3)    #output type
        parser.add_argument('--ablation',       type=int, default=0)    #ablation study 012.
        parser.add_argument('--mlp_hidden', type=int, default=64, help='hidden layer dimension of model')

        parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

        parser.add_argument('--patch_len', type=int, default=16, help='patch length')

        return parent_parser