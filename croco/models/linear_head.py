# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch
# from dust3r.heads.postprocess import postprocess
# from reloc3r_uni.utils.head_postprocess import mask_postprocess, postprocess

def mask_postprocess(out, mask_conf_mode):
    """
    extract 3D mask from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  
    if mask_conf_mode is not None:
        # extract 3D mask
        res = dict(mask_conf=reg_dense_conf(fmap[:, :, :, 0], mode=mask_conf_mode))
    return res

def postprocess(out, depth_mode, conf_mode, default_key='pts3d'):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    if depth_mode is not None:
        res = dict({default_key: reg_dense_depth(fmap[:, :, :, 0:-1], mode=depth_mode)})
    else:
        res = dict()

    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[:, :, :, -1], mode=conf_mode)
    return res

def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')



class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False, output_mode='pts3d'):
        super().__init__()
        assert output_mode in ['pts3d','flow2d','disp1d'], output_mode
        if output_mode == 'pts3d':
            out_nchan = 3
        elif output_mode == 'flow2d':
            out_nchan = 2
        elif output_mode == 'disp1d':
            out_nchan = 1
        else:
            assert 0, f'{output_mode}'
        self.output_mode = output_mode

        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf

        self.proj = nn.Linear(net.dec_embed_dim, (out_nchan + has_conf)*self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # post_process with dict_key: output_mode + 'conf'
        assert self.output_mode in ['pts3d', 'flow2d', 'disp1d'], self.output_mode
        postprocess_func = lambda out, depth_mode, conf_mode: postprocess(out, depth_mode, conf_mode, default_key=self.output_mode)

        # permute + norm depth
        return postprocess_func(feat, self.depth_mode, self.conf_mode)



class Linear_MaskEsti (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.mask_conf_mode = net.mask_conf_mode

        self.proj = nn.Linear(net.dec_embed_dim, self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return mask_postprocess(feat, mask_conf_mode=self.mask_conf_mode)
