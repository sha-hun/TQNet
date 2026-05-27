from torch import nn
from torch.nn import Transformer
from einops import rearrange
import torch
import math
from utils.MMPD.mmpd_loss import MMPD_Loss

from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
import torch
from math import ceil

class learnable_position_embedding(nn.Module):
    def __init__(self, d_model, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model), requires_grad=True) #add one more for zero-padding

    def forward(self, seq_idxs):
        """Positional encoding

        Args:
            seq_idxs shape: [batch_size, patch_num]

        Returns:
            torch.tensor: output position embedding with shape [batch_size, patch_num, d_model]
        """
        batch_size = seq_idxs.shape[0]

        position_embedding_expand = self.position_embedding[None, :, :].expand(batch_size, -1, -1)
        idxs_expand = seq_idxs[:, :, None].expand(-1, -1, self.d_model)
        pe = position_embedding_expand.gather(1, idxs_expand)

        return pe


class Transformer(nn.Module):
    def __init__(self, n_layers=3, d_model=256, n_heads=4,  d_ff=512, dropout=0.):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first = True)
        self.encoder_layers = TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, seqs):

        encoded_seqs = self.encoder_layers(seqs)

        return encoded_seqs
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, configs):
        super(DecoderOnlyTransformer, self).__init__()
        self.in_len = configs.in_len
        self.out_len = configs.out_len
        self.patch_size = configs.patch_size  
        self.d_model = configs.d_model        
        self.d_ff = configs.d_ff              
        self.n_heads = configs.n_heads        
        self.d_layers = configs.d_layers      
        self.dropout = configs.dropout        

        self.patch_embedding = nn.Linear(self.patch_size, self.d_model, bias=False)

        self.position_embedding = learnable_position_embedding(self.d_model)
        self.learnable_patch = nn.Parameter(torch.randn(self.d_model))
        

        self.decoder = Transformer(self.d_layers, self.d_model, self.n_heads, self.d_ff, self.dropout)

    def forward(self, x_seq, *args, **kwargs):
        """Forward pass

        Args:
            x_seq scaled ts, shape: [batch_size, data_dim, in_len]

        Returns:
            torch.tensor: tokens of [batch_size, out_patch_num, d_model]
        """
        
        batch_size, data_dim, seq_len = x_seq.shape
        x_seq = rearrange(x_seq, 'b d l -> (b d) l')

        #pad if not multiple of patch_size
        point_to_pad = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if point_to_pad > 0:
            x_seq = torch.cat([x_seq[:, 0:1].expand(-1, point_to_pad), x_seq], dim=1)
            seq_len = x_seq.shape[1]

        input_patch_num = seq_len // self.patch_size
        output_patch_num = ceil(self.out_len / self.patch_size)
        flatten_batch_size = x_seq.shape[0]
        
        #patchify
        patch_seq = rearrange(x_seq, 'b (n p) -> b n p', p = self.patch_size)

        #patch embedding
        patch_embed = self.patch_embedding(patch_seq) # [batch_size, input_patch_num, d_model]
        in_idxs = torch.arange(input_patch_num)[None, :].expand(flatten_batch_size, -1).to(x_seq.device)
        in_pos_embed = self.position_embedding(in_idxs)
        input_embed = patch_embed + in_pos_embed

        #decoder
        out_idxs = torch.arange(input_patch_num, input_patch_num + output_patch_num)[None, :].expand(flatten_batch_size, -1).to(x_seq.device)
        out_pos_embed = self.position_embedding(out_idxs)
        out_patch_embed = out_pos_embed + self.learnable_patch[None, None, :]
        dec_in = torch.cat([input_embed, out_patch_embed], dim = 1)
        dec_out = self.decoder(dec_in)[:, -output_patch_num:, :]

        dec_out = rearrange(dec_out, '(batch_size data_dim) out_len d_model -> batch_size data_dim out_len d_model', data_dim = data_dim)
        return dec_out  # [batch_size, data_dim, out_patch_num, d_model]

        
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        configs.in_len = configs.seq_len
        configs.out_len = configs.pred_len
        configs.data_dim = configs.enc_in

        self.model = DecoderOnlyTransformer(configs)
        self.loss_func = MMPD_Loss(configs)  # 假设 MMPD_Loss 接收 configs 初始化
        self.args = configs  # 假设 configs 中包含 point_weight 和 weighted 等参数
        self.scale = getattr(configs, 'scale', None)  # 如果 configs 中没有 scale，就默认为 None

        self.mode_now = "train" if getattr(configs, 'is_training', False) else "test"

    def predict(self, x_seq, *args, **kwargs):

        batch_size = x_seq.shape[0]
        dec_condition = self.model(x_seq, *args, **kwargs)
        print(f"--> dec_condition device: {dec_condition.device}")
        try:
            print(f"--> loss_func params device: {next(self.loss_func.parameters()).device}")
        except StopIteration:
            pass # 说明 loss_func 里面没有可学习的参数

        sample_result = self.loss_func.predict(dec_condition, *args, **kwargs)
        deterministic_pred, multi_mode_pred, raw_samples = sample_result
        print(f"--> deterministic_pred device: {deterministic_pred.device}")
        return deterministic_pred, multi_mode_pred, raw_samples
        
    def forward(self, x_enc, batch_y, *args, **kwargs):

        """
        x_enc: 输入序列, shape B,T,C
        batch_y: 目标序列, shape B,T,C
        x_scale: 对应输入的 scale, 用于加权loss
        """
        # 模型预测
        # 传入的x_seq scaled ts, shape: [batch_size, data_dim, in_len]
        # y_seq: scaled target ts, shape: [batch_size, data_dim, out_len]
        x_enc_to_model = rearrange(x_enc, 'b t c -> b c t')
        if self.mode_now in ["train", "vali"]:
            Y = self.model(x_enc_to_model)
            batch_loss = self.loss_func.compute_loss(rearrange(batch_y, 'b t c -> b c t'), Y, *args, **kwargs)  # [batch, dim, out_len]

            # 根据是否加权处理
            if getattr(self.args, 'weighted', False) and self.scale is not None:
                additional_loss = (self.scale ** 2) * batch_loss
            else:
                additional_loss = batch_loss
            additional_loss = additional_loss.mean()  # 对所有维度求平均，得到一个标量损失值

            return batch_y, additional_loss
        elif self.mode_now == "test":
            deterministic_pred, _, _ = self.predict(x_enc_to_model, prob_pred=self.args.prob_pred, 
                        sample_num = self.args.sample_num, temperature = self.args.temperature, \
                        gmm=True, gmm_components=self.args.gmm_components, prior_pi_decay=self.args.prior_pi_decay, prior_precision_shape=self.args.prior_precision_shape, \
                        gmm_iterations=self.args.gmm_iterations)
            
            # 转回 [batch, seq_len, dim]
            Y_final = rearrange(deterministic_pred, 'b d l -> b l d')
            return Y_final, None
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        # 👇 核心秘籍：开启“覆盖模式”。遇到冲突时，新的覆盖旧的！
        parent_parser.conflict_handler = 'resolve'
        
        parser = parent_parser.add_argument_group('MMPD Model Specific Arguments')
        from utils.tools import str2bool
        parser.add_argument('--backbone', type=str, default='Decoder', help='backbone model')
        parser.add_argument('--patch_size', type=int, default=12, help='segment length (L_seg)')
        parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
        parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers (N)')
        parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers (M)')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

        #parameters for loss function
        parser.add_argument('--loss_func', type=str, default='MMPD', help='loss function')
        parser.add_argument('--point_weight', type=float, default=0.01, help='weight for point loss')
        parser.add_argument('--weighted', type=str2bool, default=True, help='weighted loss')
        parser.add_argument('--d_diffusion', type=int, default=256, help='dimension for MLP in diffusion projector')
        parser.add_argument('--diffusion_layers', type=int, default=1, help='num of diffusion layers')
        parser.add_argument('--max_diffusion_steps', type=int, default=1000, help='max denosing steps')
        parser.add_argument('--beta_schedule', type=str, default='linear', help='beta schedule for diffusion process')
        parser.add_argument('--radius', type=int, default=3, help='radius for adjacent patches')

        #parameters for training
        parser.add_argument('--training', type=str2bool, default=True, help='training process')
        parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
        parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
        parser.add_argument('--lradj', type=str, default='cosine',help='adjust learning rate')

        #parameters for testing
        parser.add_argument('--test_batch_num', type=int, default=-1, help='test batch number')

        #for diffusion sampling
        parser.add_argument('--testing', type=str2bool, default=True, help='testing process')
        parser.add_argument('--prob_pred', type=str2bool, default=True, help='sample from diffusion')
        parser.add_argument('--sample_num', type=int, default=100, help='sample number to compute expectation')
        parser.add_argument('--num_sampling_steps', type=str, default='20', help='number of sampling steps for diffusion process')
        parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')

        #for diffusion gaussian mixture
        parser.add_argument('--gmm_components', type=int, default=10, help='maximum number of components in GMM')
        parser.add_argument('--prior_pi_decay', type=float, default=0.5, help='prior for weight decay, in the range of [0, 1], smaller value activates less components')
        parser.add_argument('--prior_precision_shape', type=float, default=1e2, help='prior for variance')
        parser.add_argument('--gmm_iterations', type=int, default=10, help='number of EM iterations for GMM at each diffusion step')
        return parent_parser