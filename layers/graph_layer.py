import torch
import torch.nn as nn
from layers.Entropy_layers import get_activation_fn

class Gin(nn.Module):
    def __init__(self, patch_num, n_heads, d_forward, d_mutual, dropout, activation, nvars, individual, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.individual = individual

        if not individual:
            self.gnn = nn.Sequential(
                nn.Linear((d_forward // n_heads) * patch_num, d_mutual),
                get_activation_fn(activation),
                nn.Dropout(dropout),
                nn.Linear(d_mutual, (d_forward // n_heads) * patch_num)
            )
    
        else:
            
            self.gnns = nn.ModuleList()
            for i in range(nvars):
                gin = nn.Sequential(
                    nn.Linear((d_forward // n_heads) * patch_num, d_mutual),
                    get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(d_mutual, (d_forward // n_heads) * patch_num)
                )
                self.gnns.append(gin)
            
        self.dropout_entropy = nn.Dropout(dropout)
        self.output_linear = nn.Linear(patch_num * d_forward, patch_num * d_forward)
    
    def forward(self, z, adjacency_matrix):  # [bs, nvars, patch_num, d_forward]
        bs, nvars, patch_num, d_forward = z.shape
        # devide heads
        z = z.view(bs, nvars, patch_num, self.n_heads, -1).permute(0, 3, 1, 4, 2).flatten(-2)
        
        # graph layer
        if not self.individual:
            out = self.gnn(z) + z                                                                     # [bs, n_heads, nvars, d / h * patch_num]
        else:
            out = []
            for var in range(nvars):
                u = out[:, :, var]
                u = self.gnns[var](u)
                out.append(u)
            out = torch.stack(out, dim=2)
        out = torch.matmul(adjacency_matrix, out)                         # [bs, n_heads, nvars, p * d / h]

        out = out.view(bs, self.n_heads, nvars, patch_num, -1)
        out = out.permute(0, 2, 3, 1, 4).reshape(bs, nvars, -1)                      # [bs, nvars, patch_num, d_forward]
        out = self.output_linear(out).view(bs, nvars, patch_num, -1)

        return out

class TwodMixer(nn.Module):
    def __init__(self, patch_num, n_heads, d_forward, d_mutual, dropout, activation, nvars, individual, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.individual = individual
        self.n_heads = n_heads

        if not individual:
            self.cross_sect = nn.Linear(d_forward // n_heads, d_forward // n_heads)
            self.cross_time = nn.Sequential(
                nn.Linear(patch_num, d_mutual),
                get_activation_fn(activation),
                nn.Dropout(dropout),
                nn.Linear(d_mutual, patch_num)
            )
        
        else:
            self.cross_sect = nn.ModuleList()
            self.cross_time = nn.ModuleList()
            for i in range(nvars):
                cross_sect = nn.Linear(d_forward // n_heads, d_forward // n_heads)
                cross_time = nn.Sequential(
                    nn.Linear(patch_num, d_mutual),
                    get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(d_mutual, patch_num)
                )
                
                self.cross_sect.append(cross_sect)
                self.cross_time.append(cross_time)
        
        self.dropout_entropy = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_forward, d_forward)
    
    def forward(self, z, adjacency_matrix):                                                                 # [bs, nvars, patch_num, d_forward]
        bs, nvars, patch_num, d_forward = z.shape

        if not self.individual:
            out = self.cross_sect(z.view(bs, nvars, patch_num, self.n_heads, -1)).permute(0, 3, 1, 4, 2)    # [bs, n_heads, nvars, d / h, patch_num]
            out = self.cross_time(out) + out                                                                # [bs, n_heads, nvars, d / h, patch_num]

        else: 
            out = []
            for i in range(nvars):
                u = z[:, i]                                                                                 # [bs, patch_num, d_forward]
                u = self.cross_sect[i](u.view(bs, patch_num, self.n_heads, -1)).permute(0, 2, 3, 1)         # [bs, heads, d / h, patch_nu]
                u = self.cross_time[i](u)                                                                   # [bs, heads, d / h, patch_nu]
                out.append(u)
            out = torch.stack(out, dim=2)                                                                   # [bs, n_heads, nvars, d / h, patch_num]
        out = out.permute(0, 1, 2, 4, 3)                                                                    # [bs, n_heads, nvars, patch_num, d / h]
        out = torch.matmul(adjacency_matrix, out.reshape(bs, self.n_heads, nvars, -1))                      # [bs, n_heads, nvars, p * d / h]
        
        out = out.view(bs, self.n_heads, nvars, patch_num, d_forward // self.n_heads)
        out = out.permute(0, 2, 3, 1, 4).reshape(bs, nvars, patch_num, -1)                                  # [bs, nvars, patch_num, d_forward]
        out = self.output_linear(out)

        return out
