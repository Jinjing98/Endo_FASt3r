import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from pdb import set_trace as bb


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
                 rot_representation='9D',
                 patch_size = None,
                 dec_embed_dim = None,
                 output_dim = None,
                 trans_emb_scale = 1.0,
                 rot_emb_scale = 1.0,
                 debug_later_avgpool=False,
                 debug_scale_down_opt_trans = 1.0,
                 debug_scale_down_opt_rot_ang = 1.0,
                 ):
        super().__init__()
        
        self.trans_emb_scale = trans_emb_scale
        self.rot_emb_scale = rot_emb_scale
        # self.patch_size = net.patch_embed.patch_size[0]

        if patch_size is None:
            assert dec_embed_dim is None, 'dec_embed_dim must be provided'
            assert output_dim is None, 'output_dim must be provided'
            self.patch_size = net.patch_embed.patch_size[0]
            # used for reloc3r and endofast3r
            output_dim = 4*self.patch_size**2
            dec_embed_dim = net.dec_embed_dim
        else:
            assert dec_embed_dim is not None, 'dec_embed_dim must be provided'
            assert output_dim is not None, 'output_dim must be provided'
            assert net is None, 'net must be provided'
            # used for resnet feature
            self.patch_size = patch_size
            # output_dim = 4*dec_embed_dim #4*self.patch_size**2
            output_dim = output_dim
            dec_embed_dim = dec_embed_dim

        
        self.num_resconv_block = num_resconv_block
        self.rot_representation = rot_representation  

        # output_dim = 4*self.patch_size**2

        self.proj = nn.Linear(dec_embed_dim, output_dim)
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
        if self.rot_representation=='9D':
            self.fc_rot = nn.Linear(output_dim, 9)
        else:
            self.fc_rot = nn.Linear(output_dim, 6)

        self.debug_scale_down_opt_trans = debug_scale_down_opt_trans
        self.debug_scale_down_opt_rot_ang = debug_scale_down_opt_rot_ang
        self.debug_later_avgpool = debug_later_avgpool
        if self.debug_later_avgpool:
            self.mlp_reduce_dim = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 12),
                nn.ReLU()
            )
        
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
    
    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        if self.rot_representation=='9D':
            out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        else:
            out_r = self.rotation_6d_to_matrix(out_r)
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose


    def axis_angle_to_matrix(self, axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
        """
        Convert rotations given as axis/angle to rotation matrices.

        Args:
            axis_angle: Rotations given as a vector in axis angle form,
                as a tensor of shape (..., 3), where the magnitude is
                the angle turned anticlockwise in radians around the
                vector's direction.
            fast: Whether to use the new faster implementation (based on the
                Rodrigues formula) instead of the original implementation (which
                first converted to a quaternion and then back to a rotation matrix).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        # if not fast:
            # return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

        shape = axis_angle.shape
        device, dtype = axis_angle.device, axis_angle.dtype

        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

        rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
        zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
        cross_product_matrix = torch.stack(
            [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
        ).view(shape + (3,))
        cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

        identity = torch.eye(3, dtype=dtype, device=device)
        angles_sqrd = angles * angles
        angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
        return (
            identity.expand(cross_product_matrix.shape)
            + torch.sinc(angles / torch.pi) * cross_product_matrix
            + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
        )

    def matrix_to_axis_angle(self, matrix: torch.Tensor, fast: bool = False) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to axis/angle.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            fast: Whether to use the new faster implementation (based on the
                Rodrigues formula) instead of the original implementation (which
                first converted to a quaternion and then back to a rotation matrix).

        Returns:
            Rotations given as a vector in axis angle form, as a tensor
                of shape (..., 3), where the magnitude is the angle
                turned anticlockwise in radians around the vector's
                direction.

        """
        # if not fast:
            # return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        omegas = torch.stack(
            [
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ],
            dim=-1,
        )
        norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
        traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
        angles = torch.atan2(norms, traces - 1)

        zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
        omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

        near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

        axis_angles = torch.empty_like(omegas)
        axis_angles[~near_pi] = (
            0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
        )

        # this derives from: nnT = (R + 1) / 2
        n = 0.5 * (
            matrix[near_pi][..., 0, :]
            + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
        )
        axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

        return axis_angles

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)
            # print('feat shape after res_conv_{}'.format(i), feat.shape)
        
        if self.debug_later_avgpool:
            #
            feat = feat.permute(0, 2, 3, 1)
            # print('feat shape after permute', feat.shape)
            feat = self.mlp_reduce_dim(feat)
            # print('feat shape after mlp_reduce_dim', feat.shape)
            feat = feat.permute(0, 3, 1, 2)
            # print('feat shape after permute back', feat.shape)
            # pool reduce  dim
            feat = self.avgpool(feat)
            # print('feat shape after avgpool', feat.shape)
            feat = feat.view(feat.size(0), -1)
            # print('feat shape after view', feat.shape)
            out_r = feat[:,:9]
            # print('out_r shape after view', out_r.shape)
            out_t = feat[:,9:]
            # print('out_t shape after view', out_t.shape)
        
        else:



            feat = self.avgpool(feat)
            # print('feat shape after avgpool', feat.shape)
            feat = feat.view(feat.size(0), -1)
            # print('feat shape after view', feat.shape)

            feat = self.more_mlps(feat)  # [B, D_]
            # print('feat shape after more_mlps', feat.shape)

            # out_r = self.fc_rot(feat)  # [B,9]
            # out_t = self.fc_t(feat)  # [B,3]

            # following the implementation of Endo_FASt3r; but we only scale trans
            out_r = self.fc_rot(feat * self.rot_emb_scale)  # [B,9]
            print('out_r shape after fc_rot', out_r.shape)
            out_t = self.fc_t(feat * self.trans_emb_scale)  # [B,3]
            print('out_t shape after fc_t', out_t.shape)

        pose = self.convert_pose_to_4x4(B, out_r, out_t, tokens.device)

        # scale down the trans and rot angle
        pose[:, :3, 3] *= self.debug_scale_down_opt_trans #scale down the trans
        angle_axis = self.matrix_to_axis_angle(pose[:, :3, :3])
        angle_axis *= self.debug_scale_down_opt_rot_ang # scale down the rot angle
        pose[:, :3, :3] = self.axis_angle_to_matrix(angle_axis)



        res = {"pose": pose}

        return res

