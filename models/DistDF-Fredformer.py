__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.Fredformer_backbone import Fredformer_backbone



class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0.3, 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        output = 0#configs.output
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        use_nys = configs.use_nys
        ablation = configs.ablation


        cf_dim = configs.cf_dim
        cf_depth = configs.cf_depth
        cf_heads = configs.cf_heads
        cf_mlp = configs.cf_mlp
        cf_head_dim = configs.cf_head_dim
        cf_drop = configs.cf_drop
        mlp_hidden = configs.mlp_hidden
        mlp_drop = configs.mlp_drop
        
        self.model = Fredformer_backbone(ablation=ablation,mlp_drop=mlp_drop, use_nys=use_nys,output=output,mlp_hidden=mlp_hidden,c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose,cf_dim=cf_dim,cf_depth =cf_depth,cf_heads=cf_heads,cf_mlp=cf_mlp,cf_head_dim=cf_head_dim,cf_drop=cf_drop, **kwargs)
    
    
    def forward(self, x_enc, *args,**kwargs):           # x: [Batch, Input length, Channel]
        x = x_enc
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x #,oz,t,attn
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        # 👇 核心秘籍：开启“覆盖模式”。遇到冲突时，新的覆盖旧的！
        parent_parser.conflict_handler = 'resolve'
        parser = parent_parser.add_argument_group('DistDF-Fredformer Model Specific Arguments')
        
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
        parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')
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


        return parent_parser