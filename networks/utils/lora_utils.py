
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from pdb import set_trace as bb

class Vector_MoRA(nn.Module):
    def __init__(self, w_qkv, mora_rank, lora_dropout):
        super().__init__()
        self.base_layer = w_qkv
        self.r = mora_rank  
        self.in_features = w_qkv.in_features
        self.out_features = w_qkv.out_features
        self.lora_dropout = nn.ModuleDict({
            'default': nn.Dropout(p=lora_dropout)
        })

        self.lora_A = nn.Linear(self.r, self.r, bias=False)
        nn.init.zeros_(self.lora_A.weight)

    def forward(self, x):
        in_f, out_f = self.in_features, self.out_features
        result = self.base_layer(x)
        x = self.lora_dropout['default'](x) 
        #########type6 compression
        sum_inter = in_f // self.r
        rb1 = in_f//self.r if in_f % self.r == 0 else in_f//self.r + 1
        if in_f % self.r != 0:
            pad_size = self.r - in_f % self.r
            x = torch.cat([x, x[..., :pad_size]], dim=-1)
            sum_inter += 1
        in_x = x.view(*x.shape[:-1], sum_inter, self.r)
        if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.r, 2).float() / self.r))
            t = torch.arange(rb1)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
            self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
        rh_in_x = torch.cat((-in_x[..., self.r//2:], in_x[..., :self.r//2]), dim=-1)
        in_x = in_x*self.cos + rh_in_x*self.sin
        ############################

        out_x = self.lora_A(in_x)

        ###############type6 decompression
        out_x = out_x.view(*x.shape[:-1], -1)[..., :out_f]
        if out_x.shape[-1] < out_f:
            repeat_time = out_f // out_x.shape[-1]
            if out_f % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :out_f]     
        # print("out_x",out_x)
        return result + out_x

class VectorMoRAInitializer:
    def __init__(self, model, r_enc=None, r_dec = None, lora_dropout=0.01):
        self.model = model
        self.lora_dropout = lora_dropout
        self.r_enc = r_enc
        self.r_dec = r_dec

    def initialize_mora(self):
        print("DAM mora rank for enc is ", self.r_enc)
        print("DAM mora rank for dec is ", self.r_dec)

        for t_layer_i, blk in enumerate(self.model.enc_blocks): 
            w_qkv = blk.attn.qkv
            blk.attn.qkv = Vector_MoRA(w_qkv, self.r_enc[t_layer_i], self.lora_dropout)

        for t_layer_i, blk in enumerate(self.model.dec_blocks): 
            w_qkv = blk.attn.qkv
            blk.attn.qkv = Vector_MoRA(w_qkv, self.r_dec[t_layer_i], self.lora_dropout)
        


        print("Vector MoRA params initialized!")
        return self.model


class DoRA(nn.Module):
    def __init__(self, m: nn.Module , lora_r= 1, lora_dropout = 0.0, lora_s = 1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.original_weight_matrix = m.weight.detach()
        self.weight_m = nn.Parameter(torch.empty((self.out_features, 1), **factory_kwargs),requires_grad=True)
        self.weight_v = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs),requires_grad=False)
        self.lora_r = lora_r
        if m.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)
        ### init weight_m and weight_v and bias
        with torch.no_grad():
            m = nn.utils.weight_norm(m, dim=0)
            copy_weight_m = m.weight_g.detach()
            copy_weight_v = m.weight_v.detach()
            self.weight_m.copy_(copy_weight_m)
            self.weight_v.copy_(copy_weight_v)
            if m.bias is not None:
                copy_bias = m.bias.detach()
                self.bias.copy_(copy_bias)

        self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora_r, self.in_features)))
        self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora_r)))
        self.scaling = lora_s  ## don't know if this is really needed as tining scaling is essentially the same as tuning learning rate

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        new_weight_v = self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling
        weight = ( self.weight_m / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * (self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling)

        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, lora_dim={}, lora_scale={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora_r, self.scaling
        )

class DoRAInitializer:
    def __init__(self, model, r_enc=None, r_dec = None, lora=None, lora_alpha=32, lora_dropout=0.1):
        if r_enc is None:
            r_enc = [18, 18, 16, 16, 16, 14, 14, 14, 14, 12, 12, 12, 12, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8]
        if r_dec is None:
            r_dec = [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]
        if lora is None:
            lora = ['q', 'v']

        self.model = model
        self.r_enc = r_enc
        self.r_dec = r_dec
        self.lora = lora
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.w_As = []
        self.w_Bs = []

    def reset_parameters(self):
        for w_A, w_B in zip(self.w_As, self.w_Bs):
            # normal distribution init for w_A
            nn.init.normal_(w_A.weight, mean=0.0, std=0.02)
            nn.init.zeros_(w_B.weight)  # zero init for w_B

    def initialize_dora(self):
        for param in self.model.enc_blocks.parameters():
            param.requires_grad = False  
            
        for param in self.model.dec_blocks.parameters():
            param.requires_grad = False  

        for param in self.model.enc_norm.parameters():
            param.requires_grad = False  

        for param in self.model.decoder_embed.parameters():
            param.requires_grad = False  

        # for block in self.model.dec_blocks:
        #     for param in block.cross_attn.parameters():
        #         param.requires_grad = True
            

        for t_layer_i, blk in enumerate(self.model.enc_blocks): 
            
            w_qkv = blk.attn.qkv
            blk.attn.qkv = DoRA(w_qkv, lora_r = self.r_enc[t_layer_i])


        for t_layer_i, blk in enumerate(self.model.dec_blocks): 
            
            w_qkv = blk.attn.qkv
            blk.attn.qkv = DoRA(w_qkv, lora_r = self.r_dec[t_layer_i])

        self.reset_parameters()
        print("cross attention layers are frozen!")
        # print("cross attention layers are frozen!")
        print("DoRA params initialized!")
        print("encoder rank is", self.r_enc)
        print("decoder rank is", self.r_dec)
        return self.model