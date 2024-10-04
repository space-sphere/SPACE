import torch
import torch.nn as nn
import numpy as np
from utils.masking import EntropyMask
from utils.tools import time_series_difference_with_repeat

def standardize(a, axis=-1):
    means = torch.mean(a, axis=axis, keepdims=True)
    stds = torch.std(a, axis=axis, unbiased=False, keepdims=True)
    return (a - means) / stds

def remove_trend(data):
    bs, nvars, dimension, t = data.shape
    
    time = torch.arange(t, dtype=torch.float32, device=data.device) 
    time_with_bias = torch.stack([time, torch.ones_like(time)], dim=1)  # [t, 2]
    time_with_bias = time_with_bias.unsqueeze(0)
    
    data_reshaped = data.contiguous().view(-1, t)  # [bs * nvars * dimension, t]
    
    coeffs= torch.linalg.lstsq(time_with_bias, data_reshaped.unsqueeze(-1)).solution

    trend = torch.matmul(time_with_bias, coeffs)
    
    detrended_data = data_reshaped - trend.squeeze(-1)

    detrended_data = detrended_data.view(bs, nvars, dimension, t)
    
    return detrended_data

def embed_data(x: torch.tensor, order, lag, device):
    batch_size, feature_num, d, N = x.shape
    x = x.reshape(batch_size, -1, N)
    hidx = torch.arange(start=0, end=order * lag, step=lag)
    Nv = N - (order - 1) * lag
    u = torch.zeros((batch_size, feature_num * d, order, Nv)).to(device=device)
    for i in range(order):
        u[:, :, i, :] = x[:, :, hidx[i]:hidx[i]+Nv]
    return u.reshape(batch_size, feature_num, d, order, Nv).permute(0, 1, 3, 2, 4)  # [b, feature_num, order, d, time]

def self_cov(z: torch.tensor, device):
    # [b, feature_num, time]
    b, f, t = z.shape
    x = z - torch.mean(z, axis=-1, keepdims=True).to(device=device)
    return torch.bmm(x, x.permute(0, 2, 1)) / (z.shape[-1] - 1)

def cov_xy(feature_id: list, t: list, time, feature_num, device):  # time=order + 1
    '''
    feature_id : the feature we choose
    t          : time we get from that feature
    time       : whole time in embed data
    no longer use in new cov function
    '''
    # b, t_features, _ = c.shape
    if isinstance(feature_id, int):
        feature_id = [feature_id]
    if isinstance(t, int):
        t = [t] * len(feature_id)
    if isinstance(time, int):
        time = [time] * len(feature_id)
    assert(len(feature_id) == len(t))
    assert(len(time) == len(t))
    
    feature_id, t, time = torch.tensor(feature_id, device=device), torch.tensor(t, device=device), torch.tensor(time, device=device)
    # assert (feature_num * time).sum() == c.shape[-1]
    start = feature_id * time
    end = start + t
    sl = []
    for k in range(len(feature_id)):
        sl.append(torch.arange(start[k] + k * time[0] * feature_num, end[k] + k * time[0] * feature_num))
    sl = torch.concat(sl)
    return sl

def cov(c, feature_num_queries, feature_num_keys=None, order=None, d=None, num_matrix=None, device=None):
    '''这里order表示每个feature经过embedding之后的时间步'''
    batch_size, _, _ = c.shape
    feature_queries = torch.arange(feature_num_queries, device=device)
    if feature_num_keys is not None:
        feature_keys = torch.arange(feature_num_keys, device=device)
    if num_matrix == 1:  # XtYt
        sig = torch.zeros([batch_size, feature_num_queries, feature_num_keys, 2 * d * (order - 1), 2 * d * (order - 1)], device=device)
        # feature2 = torch.arange(feature_num)
        feature1, feature2 = torch.meshgrid(feature_queries, feature_keys, indexing='ij')
        time_step = torch.arange((order - 1) * d, device=device)
        feature1, feature2 = feature1.flatten(), feature2.flatten()
        feature1_ = order * d * feature1
        feature2_ = feature_num_queries * d * order + feature2 * d * (order - 1)
        # meshgrid = torch.stack((feature1, feature2), dim=1)  # [feature_num * feature_num, 2]
        
        feature1_ = feature1_.unsqueeze(1) + time_step
        feature2_ = feature2_.unsqueeze(1) + time_step
        indices = torch.concat([feature1_, feature2_], dim=1)
        sig[:, feature1, feature2, :, :] = c[:, indices[:, :, None], indices[:, None, :]]
        
    elif num_matrix == 2:
        sig = torch.zeros([batch_size, feature_num_queries, order * d, order * d], device=device)
        time_step = torch.arange(order * d, device=device)
        feature_ = feature_queries * order *d
        feature_ = feature_.unsqueeze(1) + time_step
        sig[:, feature_queries, :, :] = c[:, feature_[:, :, None], feature_[:, None, :]]
        
    elif num_matrix == 3:
        sig = torch.zeros((batch_size, feature_num_queries, feature_num_keys, 2 * d * order - d, 2 * d * order - d), device=device)
        feature1, feature2 = torch.meshgrid(feature_queries, feature_keys, indexing='ij')
        time_step1 = torch.arange(order * d, device=device)
        time_step2 = torch.arange((order - 1) * d, device=device)
        feature1, feature2 = feature1.flatten(), feature2.flatten()
        feature1_ = feature1 * order * d
        feature2_ = order * feature_num_queries * d + feature2 * d * (order - 1)
        feature1_ = feature1_.unsqueeze(1) + time_step1
        feature2_ = feature2_.unsqueeze(1) +time_step2
        indices = torch.concat([feature1_, feature2_], dim=1)
        sig[:, feature1, feature2, :, :] = c[:, indices[:, :, None], indices[:, None, :]]
        
    else:
        sig = torch.zeros([batch_size, feature_num_queries, (order-1) * d, (order-1) * d], device=device)
        time_step = torch.arange((order - 1) * d, device=device)
        feature_ = feature_queries * order * d
        feature_ = feature_.unsqueeze(1) + time_step
        sig[:, feature_queries, :, :] = c[:, feature_[:, :, None], feature_[:, None, :]]
    return sig

def torch_det(x):
    if x.shape[-1] == 1:
        return x
    return torch.abs(torch.det(x))  # TODO remove

def entro(queries, keys, eps=0, lag=1, model_order=1, fast=False):
    batch_size, feature_num, d, T = queries.shape  # [batch_size, feature_num, time]
    bk, feature_num_k, d_k, Tk = keys.shape

    if fast:
        lag = d
        queries, keys = queries.permute(0, 1, 3, 2), keys.permute(0, 1, 3, 2)
        # [bs, nvars, 1, t]
        queries, keys = queries.reshape(batch_size, feature_num, 1, -1), keys.reshape(bk, feature_num_k, 1, -1)
        # queries = remove_trend(queries)
        # keys = remove_trend(keys)
        T = d * T
        Tk = d_k * Tk
        d = d_k = 1

    queries = standardize(time_series_difference_with_repeat(queries))
    keys = standardize(time_series_difference_with_repeat(keys))
    assert batch_size == bk and T == Tk and d == d_k
    device = queries.device
    
    '''using self_cov to get the covariance'''
    embed_queries = embed_data(queries, lag=lag, order=model_order + 1, device=device).reshape(batch_size, feature_num, (model_order + 1) * d, -1)  # [b, feature, (model_order(时间步) + 1) * d, Nv]
    embed_keys = embed_data(keys, lag=lag, order=model_order + 1, device=keys.device)[:, :, :-1, :, :].reshape(
        batch_size, feature_num_k, model_order * d, -1)
    # [b, feature_k, model_order * d, Nv]
    covariance = self_cov(torch.concat([embed_queries.reshape(batch_size, -1, T - (model_order ) *lag), 
                                    embed_keys.reshape(batch_size, -1, T - (model_order) *lag)], dim=1), device=device)
    
    
    '''get cov in the shape of [b, f, f] and [b, f]'''
    cov_XtYt = cov(covariance, feature_num_queries=feature_num, feature_num_keys=feature_num_k, d=d, order=model_order+1, num_matrix=1, device=device)  # [b, f, f, 2* m, 2 * m]
    cov_YYt = cov(covariance, feature_num_queries=feature_num, order=model_order+1, d=d, num_matrix=2, device=device)  # [b, f, m+1, m+1]
    cov_YYtXt = cov(covariance, feature_num_queries=feature_num, feature_num_keys=feature_num_k, order=model_order+1, d=d, num_matrix=3, device=device)  # [b, f, f, (2m + 1), (2m + 1)]
    cov_Yt = cov(covariance, feature_num_queries=feature_num, order=model_order+1, d=d, num_matrix=4, device=device)  # [b, f, 1]
    '''computing det'''
    H_XtYt = torch_det(cov_XtYt.reshape(-1, 2 * model_order * d, 2 * model_order * d)).reshape(batch_size, feature_num, feature_num_k)
    H_YYt = torch_det(cov_YYt.reshape(-1, (model_order + 1) *d, (model_order + 1) * d)).reshape(batch_size, feature_num, 1)
    H_YYtXt = torch_det(cov_YYtXt.reshape(-1, 2 * d * model_order + d, 2 * d * model_order + d)).reshape(batch_size, feature_num, feature_num_k)
    H_Yt = torch_det(cov_Yt.reshape(-1, model_order * d, model_order * d)).reshape(batch_size, feature_num, 1)
    
    pte = 0.5 * (torch.log(H_XtYt / (H_YYtXt + eps) + 1) - torch.log(H_Yt / (H_YYt + eps) + 1))
    # pte = 0.5 * (torch.log(H_XtYt / H_YYtXt) - torch.log(H_Yt / H_YYt))
    
    return pte  # [b, y, x] Tx->y

def transpose(X, n_heads):                                          # [batch_size, nvars, patch_num, d_model]
    # [b, f, t, d]
    X = X.reshape(X.shape[0], X.shape[1],X.shape[2] , n_heads, -1)        # [b, f, t, h, d / h]
    X = X.permute(0, 3, 1, 4, 2)                                          # [b, h, f, d/h, t]
    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])              # [-1, f, d/h, t]

def transpose_output(X, n_heads):
    # [-1, f, d / h, t]
    dshape = X.shape
    X = X.reshape(-1, n_heads, dshape[1], dshape[2], dshape[3])           # [b, h, f, d / h, t]
    # [b, f, h, d / h, t]
    return X.permute(0, 2, 4,1, 3).reshape(-1, dshape[1], dshape[3], n_heads * dshape[2])  # [b, f, t, d]

def attention(queries, keys, fast=False):
    batch_size, feature_num, d, T = queries.shape
    bk, feature_num_k, d_k, Tk = keys.shape

    queries, keys = queries.permute(0, 1, 3, 2), keys.permute(0, 1, 3, 2)
    # [bs, nvars, t]
    queries, keys = queries.reshape(batch_size, feature_num, -1), keys.reshape(bk, feature_num_k, -1)

    keys = keys.permute(0, 2, 1)                                                                    # [bs, t, nvars]
    attention = torch.matmul(queries, keys) / np.sqrt(T * d)                                                 # [bs, nvars, nvars]

    return attention

    # if fast:
    #     queries, keys = queries.permute(0, 1, 3, 2), keys.permute(0, 1, 3, 2)
    #     # [bs, nvars, t]
    #     queries, keys = queries.reshape(batch_size, feature_num, -1), keys.reshape(bk, feature_num_k, -1)

    # assert batch_size == bk and T == Tk and d == d_k
    # device = queries.device



class TransferEntropy(nn.Module):
    def __init__(self, mask_flag=False, dropout=0.1, lag=1, model_order=1, output_attention=False, n_vars=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lag = lag
        self.model_order = model_order
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        
        if n_vars is None:
            self.k = nn.Parameter(torch.tensor(0.), requires_grad=True)
        else:
            # assert isinstance(int, n_vars), print('n_vars should be int')
            self.k = nn.Parameter(torch.zeros((n_vars, 1, 1)), requires_grad=True)
        
        # self.b = nn.Parameter(torch.tensor(b), requires_grad=True)
    
    def forward(self, queries, keys, values, attn_mask, eps=1e-6):
        '''[b, f, d, t]'''
        if queries.dim() == 3:
            queries.unsqueeze_(2)
        if keys.dim() == 3:
            keys.unsqueeze_(2)
        pte = entro(queries, keys, eps=eps, lag=self.lag, model_order=self.model_order)  # [b, q, k]
        
        # zero = torch.zeros_like(pte, device=pte.device)
        # pte = torch.where(torch.isnan(pte) | torch.isinf(pte), zero, pte)
        
        B, L, _ = pte.shape
        _, num_queries, _, _ = queries.shape
        bs, num_values, d, num_hiddens = values.shape
        assert B == bs
        if self.mask_flag:
            if attn_mask == None:
                attn_mask = EntropyMask(B, L)
            pte.masked_fill_(attn_mask.mask, -np.inf)
        self.pte = nn.functional.softmax(pte, dim=-1)
        
        k = nn.functional.sigmoid(self.k)
        output = torch.bmm(self.dropout(self.pte), values.reshape(B, num_values, -1)).reshape(B, num_queries, d, num_hiddens) * k + values * (1 - k)            
        
        if self.output_attention:
            return output, self.pte
        return output, None
    
if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.randn(200).reshape(10, 20)
    np.random.seed(1)
    y = np.random.randn(200).reshape(10, 20)
    x = torch.tensor(x)
    x, y = torch.tensor(x.reshape(2, 5, 20), dtype=torch.float32), torch.tensor(y.reshape(2, 5, 20), dtype=torch.float32)
    pte = TransferEntropy(dropout=0, lag=2, model_order=3)
    z = x
