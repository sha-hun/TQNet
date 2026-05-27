import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SSformer.Embed import DataEmbedding_wo_pos, DataEmbedding
from layers.SSformer.StandardNorm import Normalize
from layers.SSformer.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SSformer.SelfAttention_Family import FullAttention, AttentionLayer
import numpy as np
from einops import rearrange


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence
        self.topk = configs.topk

        self.patch_sizes = [3, 4, 6, 10, 16, 24, 32, 48, 64]
        self.out_padding = [2, 2, 2, 4, 2, 2, 2, 2, 2]
        self.up_len = [192, 64, 48, 32, 19, 12, 8, 6, 4, 3]


        self.enc_in = configs.enc_in
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.normalize_layers = Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)

        self.Conv_list = torch.nn.ModuleList(
            [
                nn.Conv1d(in_channels=self.configs.d_model, out_channels=self.configs.d_model,
                                 kernel_size=self.patch_sizes[i], padding=1, stride=int(self.patch_sizes[i]),
                                 padding_mode='zeros', bias=False)
                for i in range(len(self.patch_sizes))
            ]
        )

        self.DConv_list = torch.nn.ModuleList(
            [
                nn.ConvTranspose1d(in_channels=self.configs.d_model, out_channels=self.configs.d_model,
                          kernel_size=self.patch_sizes[i], padding=1, stride=int(self.patch_sizes[i]), output_padding=self.out_padding[i],
                          padding_mode='zeros', bias=False)
                for i in range(len(self.patch_sizes))
            ]
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(torch.nn.ModuleList(
                    [AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads)
                        for i in range(configs.topk + 1)]),
                    configs.d_model,
                    self.up_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for i in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.predict_layers = torch.nn.Linear(configs.seq_len, configs.pred_len)
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def Conv_Tokens(self, enc_embed, patch_idx):
        x_ori = enc_embed.permute(0, 2, 1)
        x_conv_list = []
        x_conv_list.append(enc_embed)
        for i in range(self.topk):
            x_conv = self.Conv_list[patch_idx[i]](x_ori)
            x_conv_list.append(x_conv.permute(0, 2, 1))

        return x_conv_list

    def DConv_Tokens(self, dec_out_list, patch_idx):
        dec_out = [dec_out_list[0]]
        for i in range(self.topk):
            dec = dec_out_list[i + 1].permute(0, 2, 1)
            dec = self.DConv_list[patch_idx[i]](dec).permute(0, 2, 1)
            dec_out.append(dec)

        dec_out = torch.mean(torch.stack(dec_out, dim=0), dim=0)
        return dec_out

    def compute_patch_similarity(self, x, patch_sizes):

        B, L, D = x.shape
        results = torch.zeros(B, len(patch_sizes), device=x.device)

        def create_mask(N):
            mask = torch.ones(N, N, device=x.device)
            mask.fill_diagonal_(0)
            return mask

        for idx, p_size in enumerate(patch_sizes):

            num_patches = L // p_size

            patches = x[:, :num_patches * p_size, :]
            patches = patches.reshape(B, num_patches, p_size * D)
            # patches = patches.view(B, num_patches, p_size * D)

            norms = torch.norm(patches, dim=2, keepdim=True)
            norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            normalized = patches / norms

            cos_sim = torch.bmm(normalized, normalized.transpose(1, 2))

            dist_matrix = torch.sqrt(torch.clamp(2 - 2 * cos_sim, min=0.0))

            mask = create_mask(num_patches)
            non_diag_sum = (dist_matrix * mask).sum(dim=(1, 2))
            non_diag_mean = non_diag_sum / (num_patches * (num_patches - 1))

            similarity = 1.0 / (1.0 + non_diag_mean)
            results[:, idx] = similarity

        return results

    def select_topk_patches(self, similarities, patch_sizes, k):

        valid_sims = torch.where(
            torch.isnan(similarities),
            torch.tensor(-float('inf'), device=similarities.device),
            similarities
        )

        topk_values, topk_indices = torch.topk(valid_sims, k, dim=1)

        topk_indices_np = topk_indices.cpu().numpy()

        topk_patches = [
            [patch_sizes[idx] for idx in row]
            for row in topk_indices_np
        ]

        return topk_values, topk_indices, topk_patches

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        print("调用forecast")
        similarities = self.compute_patch_similarity(x_enc, self.patch_sizes)

        _, topk_idxs, _ = self.select_topk_patches(similarities, self.patch_sizes, self.topk)
        topk_idxs = topk_idxs.cpu().numpy()
        topk_idxs = np.sort(topk_idxs, axis=1)


        B, T_enc, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        print("归一化完成")
        dec_out = []
        for i in range(B):
            print(f"第{i}层循环")
            if self.channel_independence == 1:
                x_b = x_enc[i].reshape(1, T_enc, N)
                x_b = x_b.permute(0, 2, 1).contiguous().reshape(N, T_enc, 1)
                if x_mark_enc is not None:
                    x_mark_b = x_mark_enc[i].reshape(1, T_enc, -1).repeat(N, 1, 1)
                else:
                    x_mark_b = None
            else:
                x_b = x_enc[i].reshape(1, T_enc, N)
                if x_mark_enc is not None:
                    x_mark_b = x_mark_enc[i].reshape(1, T_enc, N)
                else:
                    x_mark_b = None

            enc = self.enc_embedding(x_b, x_mark_b)
            print("embedding完成")
            enc_list = self.Conv_Tokens(enc, topk_idxs[i])
            print("Conv_Tokens完成")
            enc_list, _ = self.encoder(enc_list, topk_idxs[i])
            print("encoder完成")
            dec = self.DConv_Tokens(enc_list, topk_idxs[i])
            print("DConv_Tokens完成")
            dec = dec + enc
            dec_out.append(dec)

        dec_out = torch.cat(dec_out, dim=0)
        dec_out = self.predict_layers(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        print("预测层完成")
        dec_out = self.projection_layer(dec_out)
        dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        print("projection完成")
        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out[:, -self.pred_len:, :]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        # x_enc = rearrange(x_enc, 'b t n -> b n t')
        # x_mark_enc = rearrange(x_mark_enc, 'b t n -> b n t')
        # x_dec = rearrange(x_dec, 'b t n -> b n t')
        # x_mark_dec = rearrange(x_mark_dec, 'b t n -> b n t')
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.conflict_handler = 'resolve'
        parser = parent_parser.add_argument_group('Model Specific Arguments')

        parser.add_argument('--topk', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=32, help='dimension of f`c`n')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default='avg',
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                            help='whether to use future_temporal_feature; True 1 False 0')

        parser.add_argument('--dilated_layers', type=int, default=5, help='num of dilated layers')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')       
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        
        return parent_parser