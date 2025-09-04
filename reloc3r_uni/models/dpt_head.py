# adapt from das3r
import torch.nn as nn
import torch
from einops import rearrange
from typing import List
from models.dpt_block import DPTOutputAdapter  # noqa
from reloc3r_uni.utils.head_postprocess import mask_postprocess, postprocess

class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    copy from easi3r

    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        # [torch.Size([2, 768, 1024]), torch.Size([2, 768, 768]), torch.Size([2, 768, 768]), torch.Size([2, 768, 768])]
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation [torch.Size([2, 768, 1024]), torch.Size([2, 768, 768]), torch.Size([2, 768, 768]), torch.Size([2, 768, 768])]
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]
        # [torch.Size([2, 1024, 24, 32]), torch.Size([2, 768, 24, 32]), torch.Size([2, 768, 24, 32]), torch.Size([2, 768, 24, 32])]


        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # [torch.Size([2, 96, 96, 128]), torch.Size([2, 192, 48, 64]), torch.Size([2, 384, 24, 32]), torch.Size([2, 768, 12, 16])]

        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        # [torch.Size([2, 256, 96, 128]), torch.Size([2, 256, 48, 64]), torch.Size([2, 256, 24, 32]), torch.Size([2, 256, 12, 16])]


        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        # torch.Size([2, 256, 18, 32])

        path_3 = self.scratch.refinenet3(path_4, layers[2])
        # torch.Size([2, 256, 36, 64])

        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out

class PixelwiseTaskWithDPT(nn.Module):
    """ 
    copy from easi3r
    DPT module for dust3r, can return 3D points + confidence for all pixels
    """

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out

class PixelwiseTaskWithDPT_MaskEsti(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, 
                 mask_conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT_MaskEsti, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.mask_conf_mode = mask_conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, mask_conf_mode = self.mask_conf_mode)
        return out

def create_dpt_head(net, has_conf=False, output_mode='pts3d'):
    """
    copy from easi3r.dust3r.head implementaiton
    extend for 2d OF and 1D disparity
    """
    assert output_mode in ['pts3d','flow2d','disp1d'], output_mode

    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    if output_mode == 'pts3d':
        out_nchan = 3
    elif output_mode == 'flow2d':
        out_nchan = 2
    elif output_mode == 'disp1d':
        out_nchan = 1
    else:
        assert 0, f'{output_mode}'
    

    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    
    # post_process with dict_key: output_mode + 'conf'
    assert output_mode in ['pts3d', 'flow2d', 'disp1d'], output_mode
    postprocess_func = lambda out, depth_mode, conf_mode: postprocess(out, depth_mode, conf_mode, default_key=output_mode)

    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')

def create_motion_mask_dpt_head(net):
    """
    return PixelwiseTaskWithDPT_MaskEsti for given net params
    """
    assert net.dec_depth > 9
    return PixelwiseTaskWithDPT_MaskEsti(num_channels=1,
                                feature_dim=256,
                                last_dim=128,
                                hooks_idx=[0, net.dec_depth*2//4, net.dec_depth*3//4, net.dec_depth],
                                dim_tokens=[net.enc_embed_dim, net.dec_embed_dim, net.dec_embed_dim, net.dec_embed_dim],
                                postprocess=mask_postprocess,
                                mask_conf_mode=net.mask_conf_mode,
                                head_type='semseg')