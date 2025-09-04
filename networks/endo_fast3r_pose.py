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

from reloc3r.reloc3r_relpose import load_model
from networks.utils.lora_utils import Vector_MoRA, DoRA, DoRAInitializer, VectorMoRAInitializer
from networks.models.endofast3r_posehead import PoseHead, transpose_to_landscape

def Reloc3rX(path: str
                       ):
    print("DoMoRA in Reloc3rX")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reloc3r_relpose = load_model(ckpt_path=path, img_size=512, device=device)
    reloc3r_relpose = DoRAInitializer(reloc3r_relpose, [20, 20, 20, 18, 18, 18, 18, 18, 16, 16, 16, 16, 14,14,12,12,10,10,8,8,8,8,8,8], [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]).initialize_dora()
    reloc3r_relpose = VectorMoRAInitializer(reloc3r_relpose, [20, 20, 20, 18, 18, 18, 18, 18, 16, 16, 16, 16, 14,14,12,12,10,10,8,8,8,8,8,8], [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]).initialize_mora()
    reloc3r_relpose.pose_head = PoseHead(net=reloc3r_relpose)
    reloc3r_relpose.head = transpose_to_landscape(reloc3r_relpose.pose_head, activate=True)
    return reloc3r_relpose



# # code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
# class ResConvBlock(nn.Module):
#     """
#     1x1 convolution residual block
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
#         self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
#         self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
#         self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

#     def forward(self, res):
#         x = F.relu(self.res_conv1(res))
#         x = F.relu(self.res_conv2(x))
#         x = F.relu(self.res_conv3(x))
#         res = self.head_skip(res) + x
#         return res


# # parts of the code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L193'
# class PoseHead(nn.Module):
#     """ 
#     pose regression head
#     """
#     def __init__(self, 
#                  net, 
#                  num_resconv_block=2,
#                  rot_representation='axis-angle'):
#         super().__init__()
#         self.patch_size = net.patch_embed.patch_size[0]
#         self.num_resconv_block = num_resconv_block
#         self.rot_representation = rot_representation  

#         output_dim = 4*self.patch_size**2

#         self.proj = nn.Linear(net.dec_embed_dim, output_dim)
#         self.res_conv = nn.ModuleList([copy.deepcopy(ResConvBlock(output_dim, output_dim)) 
#             for _ in range(self.num_resconv_block)])
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.more_mlps = nn.Sequential(
#             nn.Linear(output_dim,output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim,output_dim),
#             nn.ReLU()
#             )
#         self.fc_t = nn.Linear(output_dim, 3)
#         # if self.rot_representation=='9D':
#         #     self.fc_rot = nn.Linear(output_dim, 9)
#         # else:
#         #     self.fc_rot = nn.Linear(output_dim, 6)
#         self.fc_rot = nn.Linear(output_dim, 3)
        
#     def svd_orthogonalize(self, m):
#         """Convert 9D representation to SO(3) using SVD orthogonalization.

#         Args:
#           m: [BATCH, 3, 3] 3x3 matrices.

#         Returns:
#           [BATCH, 3, 3] SO(3) rotation matrices.
#         """
#         if m.dim() < 3:
#             m = m.reshape((-1, 3, 3))
#         m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
#         u, s, v = torch.svd(m_transpose)
#         det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
#         # Check orientation reflection.
#         r = torch.matmul(
#             torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
#             u.transpose(-2, -1)
#         )
#         return r
    
#     def rot_from_axisangle(self, vec):
#         """Convert an axisangle rotation into a 4x4 transformation matrix
#         (adapted from https://github.com/Wallacoloo/printipi)
#         Input 'vec' has to be Bx1x3
#         """
#         angle = torch.norm(vec, 2, 2, True)
#         axis = vec / (angle + 1e-7)

#         ca = torch.cos(angle)
#         sa = torch.sin(angle)
#         C = 1 - ca

#         x = axis[..., 0].unsqueeze(1)
#         y = axis[..., 1].unsqueeze(1)
#         z = axis[..., 2].unsqueeze(1)

#         xs = x * sa
#         ys = y * sa
#         zs = z * sa
#         xC = x * C
#         yC = y * C
#         zC = z * C
#         xyC = x * yC
#         yzC = y * zC
#         zxC = z * xC

#         rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

#         rot[:, 0, 0] = torch.squeeze(x * xC + ca)
#         rot[:, 0, 1] = torch.squeeze(xyC - zs)
#         rot[:, 0, 2] = torch.squeeze(zxC + ys)
#         rot[:, 1, 0] = torch.squeeze(xyC + zs)
#         rot[:, 1, 1] = torch.squeeze(y * yC + ca)
#         rot[:, 1, 2] = torch.squeeze(yzC - xs)
#         rot[:, 2, 0] = torch.squeeze(zxC - ys)
#         rot[:, 2, 1] = torch.squeeze(yzC + xs)
#         rot[:, 2, 2] = torch.squeeze(z * zC + ca)
#         rot[:, 3, 3] = 1

#         return rot

#     def transformation_from_parameters(self, axisangle, translation, invert=False):
#         """Convert the network's (axisangle, translation) output into a 4x4 matrix
#         """
#         R = self.rot_from_axisangle(axisangle)
#         t = translation.clone()

#         if invert:
#             R = R.transpose(1, 2)
#             t *= -1

#         T = self.get_translation_matrix(t)

#         if invert:
#             M = torch.matmul(R, T)
#         else:
#             M = torch.matmul(T, R)

#         return M
    
#     def get_translation_matrix(self, translation_vector):
#         """Convert a translation vector into a 4x4 transformation matrix
#         """
#         T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

#         t = translation_vector.contiguous().view(-1, 3, 1)

#         T[:, 0, 0] = 1
#         T[:, 1, 1] = 1
#         T[:, 2, 2] = 1
#         T[:, 3, 3] = 1
#         T[:, :3, 3, None] = t

#         return T

#     def rotation_6d_to_matrix(self, d6):  # code from pytorch3d
#         """
#         Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
#         using Gram--Schmidt orthogonalization per Section B of [1].
#         Args:
#             d6: 6D rotation representation, of size (*, 6)

#         Returns:
#             batch of rotation matrices of size (*, 3, 3)

#         [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
#         On the Continuity of Rotation Representations in Neural Networks.
#         IEEE Conference on Computer Vision and Pattern Recognition, 2019.
#         Retrieved from http://arxiv.org/abs/1812.07035
#         """
#         a1, a2 = d6[..., :3], d6[..., 3:]
#         b1 = F.normalize(a1, dim=-1)
#         b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
#         b2 = F.normalize(b2, dim=-1)
#         b3 = torch.cross(b1, b2, dim=-1)
#         return torch.stack((b1, b2, b3), dim=-2)
    
#     # def convert_pose_to_4x4(self, B, out_r, out_t, device):
#     #     out_r = 0.001 * out_rout
#     #     out_t = 0.001 * out_t
#     #     # if self.rot_representation=='9D':
#     #     #     out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
#     #     # else:
#     #     #     out_r = self.rotation_6d_to_matrix(out_r)
        
#     #     out_r = self.rot_from_axisangle(out_r.unsqueeze(1)) 
#     #     ipdb.set_trace()
#     #     pose = torch.zeros((B, 4, 4), device=device)
#     #     pose[:, :3, :3] = out_r 
#     #     pose[:, :3, 3] = out_t
#     #     pose[:, 3, 3] = 1.
#     #     return pose

#     def convert_pose_to_4x4(self, B, out_r, out_t, device):
#         out_r = 0.001 * out_r
#         out_t = 0.001 * out_t
#         # if self.rot_representation=='9D':
#         #     out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
#         # else:
#         #     out_r = self.rotation_6d_to_matrix(out_r)
        
        
#         # ipdb.set_trace()
#         # pose = torch.zeros((B, 4, 4), device=device)
#         # pose[:, :3, :3] = out_r 
#         # pose[:, :3, 3] = out_t
#         # pose[:, 3, 3] = 1.
#         pose = self.transformation_from_parameters(out_r.unsqueeze(1), out_t)
#         return pose

#     def forward(self, decout, img_shape):
#         H, W = img_shape
#         tokens = decout[-1]
#         B, S, D = tokens.shape
        
#         feat = self.proj(tokens)  # B,S,D
#         feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
#         for i in range(self.num_resconv_block):
#             feat = self.res_conv[i](feat)

#         feat = self.avgpool(feat)
#         feat = feat.view(feat.size(0), -1)

#         feat = self.more_mlps(feat)  # [B, D_]
#         out_t = self.fc_t(feat)  # [B,3]
#         out_r = self.fc_rot(feat)  # [B,3]
#         pose = self.convert_pose_to_4x4(B, out_r, out_t, tokens.device)
#         res = {"pose": pose}

#         # print('endofast3r pose_head returned pose:', pose)

#         return res






# def freeze_all_params(modules):
#     for module in modules:
#         try:
#             for n, param in module.named_parameters():
#                 param.requires_grad = False
#         except AttributeError:
#             # module is directly a parameter
#             module.requires_grad = False


# def transpose_to_landscape(head, activate=True):
#     """ Predict in the correct aspect-ratio,
#         then transpose the result in landscape 
#         and stack everything back together.
#     """
#     def wrapper_no(decout, true_shape):
#         B = len(true_shape)
#         assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
#         H, W = true_shape[0].cpu().tolist()
#         res = head(decout, (H, W))
#         return res

#     def wrapper_yes(decout, true_shape):
#         B = len(true_shape)
#         # by definition, the batch is in landscape mode so W >= H
#         H, W = int(true_shape.min()), int(true_shape.max())

#         height, width = true_shape.T
#         is_landscape = (width >= height)
#         is_portrait = ~is_landscape

#         # true_shape = true_shape.cpu()
#         if is_landscape.all():
#             return head(decout, (H, W))
#         if is_portrait.all():
#             return transposed(head(decout, (W, H)))

#         # batch is a mix of both portraint & landscape
#         def selout(ar): return [d[ar] for d in decout]
#         l_result = head(selout(is_landscape), (H, W))
#         p_result = transposed(head(selout(is_portrait),  (W, H)))

#         # allocate full result
#         result = {}
#         for k in l_result | p_result:
#             x = l_result[k].new(B, *l_result[k].shape[1:])
#             x[is_landscape] = l_result[k]
#             x[is_portrait] = p_result[k]
#             result[k] = x

#         return result

#     return wrapper_yes if activate else wrapper_no
