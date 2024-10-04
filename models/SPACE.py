import torch
import torch.nn as nn
from layers.SPACE_layers import SingleTe

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        # Revin
        nvars = configs.nvars
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        # Embedding
        d_model = configs.d_model
        d_mutual = configs.d_mutual
        pre_embedding = configs.pre_embedding
        stride_list = configs.stride
        if isinstance(stride_list, int):
            stride_list = [stride_list]
        stride_list.sort()
        seq_len = configs.seq_len
        patch_len_list = configs.patch_len
        if isinstance(patch_len_list, int):
            patch_len_list = [patch_len_list]
        patch_len_list.sort()
        
        if not pre_embedding:
            patch_num = int((seq_len - patch_len_list[-1]) / stride_list[-1] + 1)
        else:
            patch_num = int((d_model - patch_len_list[-1]) / stride_list[-1] + 1)
        
        padding_patch = configs.padding_patch
        use_se = configs.use_se
        
        # Encoder
        lag = configs.lag
        model_order = configs.model_order
        fast = bool(configs.use_fast),
        n_heads = configs.n_heads
        
        n_heads_forward = configs.n_heads_forward
        dropout = configs.dropout
        d_ff = configs.d_ff
        store_attn = False
        mutual_type = configs.mutual_type
        mutual_individual = configs.mutual_individual
        activation = configs.activation
        res_attention = configs.res_attention
        e_layers = configs.e_layers
        
        use_entropy = bool(configs.use_entropy)
        
        self.lenth = len(patch_len_list)
        # self.stride = stride
        self.res_attention = res_attention
        
        # self.use_multiscale = configs.use_multiscale
        # kernel_sizes = configs.kernel_sizes
        
        # Decoder
        individual = configs.head_individual
        seq_len = configs.seq_len
        target_window = configs.pred_len

        if len(patch_len_list) > 1:
            # self.agg = nn.Conv1d(in_channels=len(patch_len_list), out_channels=1, kernel_size=1)
            self.agg = nn.AvgPool1d(3)
        
        self.model = nn.ModuleList()
        for i in range(len(patch_len_list)):
            patch_len = patch_len_list[i]
            stride = stride_list[i]

            model = SingleTe(
                seq_len=seq_len, n_heads=n_heads, d_model=d_model, d_mutual=d_mutual, patch_len=patch_len, patch_num=patch_num,
                n_heads_forward=n_heads_forward, nvars=nvars, dropout=dropout, d_ff=d_ff, store_attn=store_attn, stride=stride, mutual_type=mutual_type,
                mutual_individual=mutual_individual, activation=activation, res_attention=res_attention, e_layers=e_layers,lag=lag, use_se=use_se,
                model_order=model_order, head_individual=individual, target_window=target_window, pre_embedding=pre_embedding,
                padding_patch=padding_patch, revin=revin, affine=affine, subtract_last=subtract_last, fast=fast, use_entropy=use_entropy
            )
            self.model.append(model)
    
    def forward(self, z):                                                                   # [bs, seq_len, nvars]
        # whether to use multi-scale
        v_list = []
        for model in self.model:
            if self.res_attention:
                v, self.attn_scores, self.entropy_scores = model(z)
            
            else:
                v = model(z)
            v_list.append(v)

        if len(v_list) == 1:

            return v.permute(0, 2, 1)
        
        v_list = torch.stack(v_list, dim=2)                                                  # [bs, nvars, pl, seq_len]
        bs, nvars, pl, seq_len = v_list.shape
        
        v_list = self.agg(v_list.view(bs * nvars, pl, -1))
        v = v_list.view(bs, nvars, seq_len)
        
        v = v.permute(0, 2, 1)

        return v
