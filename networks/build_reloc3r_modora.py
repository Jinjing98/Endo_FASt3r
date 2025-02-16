from reloc3r.reloc3r_relpose import load_model
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DepthAnythingForDepthEstimation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import requests
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
import ipdb

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


# def build_reloc3r_model(path: str
#                        ):
#     print("MoDoRA in Pose model")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     reloc3r_relpose = load_model(ckpt_path=path, img_size=512, device=device)
#     reloc3r_relpose = DoRAInitializer(reloc3r_relpose, [20, 20, 20, 18, 18, 18, 18, 18, 16, 16, 16, 16, 14,14,12,12,10,10,8,8,8,8,8,8], [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]).initialize_dora()
#     reloc3r_relpose = VectorMoRAInitializer(reloc3r_relpose, [20, 20, 20, 18, 18, 18, 18, 18, 16, 16, 16, 16, 14,14,12,12,10,10,8,8,8,8,8,8], [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]).initialize_mora()
#     reloc3r_relpose.pose_head = PoseHead(net=reloc3r_relpose)
#     reloc3r_relpose.head = transpose_to_landscape(reloc3r_relpose.pose_head, activate=True)
#     return reloc3r_relpose


class Endo_FASt3r_pose(nn.Module):
    def __init__(self, path: str):
        super(Endo_FASt3r_pose, self).__init__()
        print("MoDoRA in Pose model")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.model = load_model(ckpt_path=path, img_size=512, device=self.device)
        
        # Apply DoRA initialization
        self.model = DoRAInitializer(
            self.model,
            [20, 20, 20, 18, 18, 18, 18, 18, 16, 16, 16, 16, 14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8],
            [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]
        ).initialize_dora()
        
        # Apply VectorMoRA initialization
        self.model = VectorMoRAInitializer(
            self.model,
            [20, 20, 20, 18, 18, 18, 18, 18, 16, 16, 16, 16, 14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8],
            [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]
        ).initialize_mora()
        
        # Set up pose head and head
        self.model.pose_head = PoseHead(net=self.model)
        self.model.head = transpose_to_landscape(self.model.pose_head, activate=True)

    def forward(self, x):
        return self.model(x)  # Forward pass through the initialized model


# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


# parts of the code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L193'
class PoseHead(nn.Module):
    """ 
    pose regression head
    """
    def __init__(self, 
                 net, 
                 num_resconv_block=2,
                 rot_representation='axis-angle'):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.num_resconv_block = num_resconv_block
        self.rot_representation = rot_representation  

        output_dim = 4*self.patch_size**2

        self.proj = nn.Linear(net.dec_embed_dim, output_dim)
        self.res_conv = nn.ModuleList([copy.deepcopy(ResConvBlock(output_dim, output_dim)) 
            for _ in range(self.num_resconv_block)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        # if self.rot_representation=='9D':
        #     self.fc_rot = nn.Linear(output_dim, 9)
        # else:
        #     self.fc_rot = nn.Linear(output_dim, 6)
        self.fc_rot = nn.Linear(output_dim, 3)
        
    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r
    
    def rot_from_axisangle(self, vec):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be Bx1x3
        """
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)

        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca

        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)

        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1

        return rot

    def transformation_from_parameters(self, axisangle, translation, invert=False):
        """Convert the network's (axisangle, translation) output into a 4x4 matrix
        """
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()

        if invert:
            R = R.transpose(1, 2)
            t *= -1

        T = self.get_translation_matrix(t)

        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M
    
    def get_translation_matrix(self, translation_vector):
        """Convert a translation vector into a 4x4 transformation matrix
        """
        T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

        t = translation_vector.contiguous().view(-1, 3, 1)

        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t

        return T

    def rotation_6d_to_matrix(self, d6):  # code from pytorch3d
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    
    # def convert_pose_to_4x4(self, B, out_r, out_t, device):
    #     out_r = 0.001 * out_rout
    #     out_t = 0.001 * out_t
    #     # if self.rot_representation=='9D':
    #     #     out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
    #     # else:
    #     #     out_r = self.rotation_6d_to_matrix(out_r)
        
    #     out_r = self.rot_from_axisangle(out_r.unsqueeze(1)) 
    #     ipdb.set_trace()
    #     pose = torch.zeros((B, 4, 4), device=device)
    #     pose[:, :3, :3] = out_r 
    #     pose[:, :3, 3] = out_t
    #     pose[:, 3, 3] = 1.
    #     return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = 0.001 * out_r
        out_t = 0.001 * out_t
        # if self.rot_representation=='9D':
        #     out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        # else:
        #     out_r = self.rotation_6d_to_matrix(out_r)
        
        
        # ipdb.set_trace()
        # pose = torch.zeros((B, 4, 4), device=device)
        # pose[:, :3, :3] = out_r 
        # pose[:, :3, 3] = out_t
        # pose[:, 3, 3] = 1.
        pose = self.transformation_from_parameters(out_r.unsqueeze(1), out_t)
        return pose

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        out_t = self.fc_t(feat)  # [B,3]
        out_r = self.fc_rot(feat)  # [B,3]
        pose = self.convert_pose_to_4x4(B, out_r, out_t, tokens.device)
        res = {"pose": pose}

        return res






def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait),  (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no
