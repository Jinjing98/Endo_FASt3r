
from copy import deepcopy
import os
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial
# we still reuse the other func under Endo_FASt3r/croco; while when refer module 'models', we refer easi3r croco
from croco.stereoflow.datasets_flow import flowToColor 
import sys
import reloc3r_uni.utils.path_to_croco
# from patch_embed import ManyAR_PatchEmbed
from reloc3r_uni.patch_embed import ManyAR_PatchEmbed
from models.pos_embed import RoPE2D 
# from models.blocks_unireloc3r import Block, DecoderBlock
from models.blocks_unireloc3r import Block
import kornia



# from baselines.Easi3R.croco.models.blocks import DecoderBlock # facilicate use decoder attn
# from reloc3r.pose_head import PoseHead
# from reloc3r.utils.misc import freeze_all_params, transpose_to_landscape,transpose_to_landscape_with_mask
# from reloc3r_uni.pose_head import PoseHead


from pdb import set_trace as bb
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange

# import sys
# # risky
# sys.path.append('/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/')

# from models.croco_dec import DecoderBlock
from models.blocks_unireloc3r import DecoderBlock
# from models.utils.attn_mask import resize_mask, get_attn_k
from reloc3r_uni.models.linear_head import Linear_MaskEsti, LinearPts3d
from reloc3r_uni.models.dpt_head import create_motion_mask_dpt_head, create_dpt_head

# from mvp3r.utils.process_mask import save_dynamic_conf_masks
# from mvp3r.models.utils.corr import global_correlation_softmax # for corr computation
from reloc3r_uni.utils.vis import flow3DToColor

from reloc3r_uni.pose_head import PoseHead
from reloc3r_uni.utils.misc import freeze_all_params, transpose_to_landscape, transpose_to_landscape_with_mask
from reloc3r_uni.utils.geometry import inv, geotrf

import numpy as np
import cv2
import argparse
import torch.nn.functional as F
inf = float('inf')


# parts of the code adapted from 
# 'https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/croco.py#L21'
# 'https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/model.py#L46'
class Reloc3rRelpose(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=512,          # input image size
                 patch_size=16,         # patch_size; smaller, denser 
                 enc_embed_dim=1024,    # encoder feature dimension
                 enc_depth=24,          # encoder depth 
                 enc_num_heads=16,      # encoder number of heads in the transformer block 
                 dec_embed_dim=768,     # decoder feature dimension 
                 dec_depth=12,          # decoder depth 
                 dec_num_heads=12,      # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',   # positional embedding (either cosine or RoPE100)
                # easi3r extension
                #  ret_atten_mask=True,  # only faciliatet save it in the opt, not have to use in the 2nd inference; if set False, is the original reloc3r
                 ret_atten_mask=False,  #disable for memory
                 dec_use_atten_mask=False,  # only actually valid in the 2nd, as 1st masks are None
                 reweight_both_when_use_mask=True,  # only actually valid in the 2nd, as 1st masks are None
                # das3r extension
                 init_dynamic_mask_estimator=False,
                 shared_dynamic_mask_estimator = False,
                 dynamic_mask_estimator_type='linear',  # 'linear' or 'dpt'  
                # 3d motion flow for masked obj
                 init_3d_scene_flow=False,# the flow contains: motion+ego # for mask area
                 init_3d_ego_flow=False, # for whole area
                 init_3d_motion_flow=False,# the motion flow contains: motion # for mask area---most challenging
                 scene_flow_estimator_type='linear', 
                 ego_flow_estimator_type='linear',  
                 motion_flow_estimator_type='linear', 
                 shared_scene_flow_estimator = True,
                 shared_motion_flow_estimator = True,
                 shared_ego_flow_estimator = True,
                 #extend for depth
                 init_3d_depth=False,  # whether to init the depth head
                 depth_estimator_type='linear',  # 'dpt' or 'linear'
                 shared_depth_estimator = True,  # whether to share the depth estimator for all flow heads
                 init_another_dec_for_depth = False,  #use another decoder!
                #extend for optic flow
                init_2d_optic_flow=False,  # whether to init the optic flow head
                optic_flow_estimator_type='linear',  # 'dpt' or 'linear'
                shared_optic_flow_estimator = True,  # whether to share the optic flow estimator for all flow heads
                #extend for pose_head
                pose_head_seperate_scale = False,  # whether to use separate scale for pose regression
                #extend for diff pose solver
                unireloc3r_pose_estimation_mode = 'vanilla_pose_head_regression',
                pose_regression_with_mask = False,
                pose_regression_which_mask = 'gt',
                pose_regression_head_input = 'default',
                mapero_pixel_pe_scheme = 'focal_norm',
                #
                landscape_only = True,  # always landscape only; not sure
                output_mode = 'pts3d',  # always pts3d
                #  vis=False,  # whether to visualize the trn/infer process
                 vis=True,  # whether to visualize the trn/infer process
                 exp_id=None, # experiment id for saving visualizations
                 output_dir=None,
                ):   
        super(Reloc3rRelpose, self).__init__()


        print('uni Reloc3rRelpose init.....')
        print('dynamic_mask: ', init_dynamic_mask_estimator)

        self.exp_id = exp_id # used only for distingusible visulistion saved dir
        self.output_dir = output_dir # used only for distingusible visulistion saved dir

        # patchify and positional embedding
        self.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)
        # ViT encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        # ViT decoder
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from encoder to decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)

        # save attention weights for visualization
        self.ret_atten_mask = ret_atten_mask
        self.dec_use_atten_mask = dec_use_atten_mask 
        self.reweight_both_when_use_mask = reweight_both_when_use_mask # only used when dec_use_atten_mask on during the 2nd inference

        # for visualization
        self.vis = vis
        # hard code for vis mapero PE
        self.do_vis_pe = True
        self.do_vis_pe = False
        # self.vis_pe_freq = 10000
        self.vis_all_freq = 1000  
        self.vis_all_freq = 100
        self.vis_all_freq = 100
        self.vis_pe_freq = -1
        # self.vis_all_freq = -1
        self.iter_cnt = 0

        # Initialize clamp_stats to None to avoid memory leaks
        self.clamp_stats = None

        #///hard code entry for : elborate pose_reg
        self.dec_use_motion_mask = False 
        self.dec_use_which_motion_mask = 'gt'  
        if self.dec_use_motion_mask:
            assert self.reweight_both_when_use_mask, f'reweight_both_when_use_mask should be True when dec_use_motion_mask is True'
        assert self.dec_use_which_motion_mask in ['gt'], f'Unknown dec_use_which_motion_mask {self.dec_use_which_motion_mask}, should be one of [esti, gt]'
        #/////////////////
        self.init_dynamic_mask_estimator = init_dynamic_mask_estimator
        self.shared_dynamic_mask_estimator = shared_dynamic_mask_estimator
        self.dynamic_mask_estimator_type = dynamic_mask_estimator_type

        self.init_3d_scene_flow = init_3d_scene_flow
        self.scene_flow_estimator_type = scene_flow_estimator_type
        self.shared_scene_flow_estimator = shared_scene_flow_estimator
        #///hard code entry for : elborate ego_flow,motion_flow
        self.init_3d_ego_flow = init_3d_ego_flow
        self.ego_flow_estimator_type = ego_flow_estimator_type
        self.shared_ego_flow_estimator = shared_ego_flow_estimator

        self.init_3d_motion_flow = init_3d_motion_flow
        self.motion_flow_estimator_type = motion_flow_estimator_type
        self.shared_motion_flow_estimator = shared_motion_flow_estimator

        #hard code pts3d prediction
        self.init_3d_depth = init_3d_depth # not used in reloc3r, but can be used for depth estimation
        self.depth_estimator_type = depth_estimator_type # 'dpt' or 'linear'
        self.shared_depth_estimator = shared_depth_estimator # whether to share the depth estimator for all flow heads
        #use a second non shared decoder for depth estimation
        self.init_another_dec_for_depth = init_another_dec_for_depth
        if self.init_another_dec_for_depth:
            assert self.init_3d_depth, f'init_another_dec_for_depth={self.init_another_dec_for_depth} requires init_3d_depth to be True'
            # self.decoder_embed2 = nn.Linear(enc_embed_dim, self.dec_embed_dim, bias=True)
            self.dec_blocks_2nd = nn.ModuleList([
                DecoderBlock(self.dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
                for i in range(self.dec_depth)])


        #///hard code entry for : elborate OF
        self.init_2d_optic_flow = init_2d_optic_flow
        self.optic_flow_estimator_type = optic_flow_estimator_type
        self.shared_optic_flow_estimator = shared_optic_flow_estimator

        #///hard code entry for : mased pose regression. Controlled via pose_regression_with_mask
        self.detach_token_grad_for_dyn_mask = True
        # self.detach_token_grad_for_dyn_mask = False #default before

        self.unireloc3r_pose_estimation_mode = unireloc3r_pose_estimation_mode #'vanilla_pose_head_regression'
        # self.unireloc3r_pose_estimation_mode = 'epropnp'
        assert self.unireloc3r_pose_estimation_mode in ['vanilla_pose_head_regression', 'epropnp', 'geoaware_pnet'], f'Unknown unireloc3r_pose_estimation_mode {self.unireloc3r_pose_estimation_mode}, should be one of [vanilla_pose_head_regression, epropnp, geoaware_pnet]'
        if self.unireloc3r_pose_estimation_mode == 'epropnp':
            assert self.init_3d_scene_flow, f'unireloc3r_pose_estimation_mode={self.unireloc3r_pose_estimation_mode} requires init_3d_scene_flow to be True'
            # todo: implemented the flow based matching
            # assert self.init_2d_optic_flow, f'unireloc3r_pose_estimation_mode={self.unireloc3r_pose_estimation_mode} requires init_2d_optic_flow to be True'
            self.initialize_epropnp() # init: log_weight_scale, camera, cost_fun, epropnp
            self._base_grid = None

        self.pose_regression_with_mask = pose_regression_with_mask#False
        # self.pose_regression_with_mask = True
        self.pose_regression_which_mask = pose_regression_which_mask #'gt'#'esti' 'detached_esti' 'gt' 
        assert self.pose_regression_which_mask in ['detached_esti','esti','gt'], f'Unknown pose_regression_which_mask {self.pose_regression_which_mask}, should be one of [gt, esti]'
        self.pose_regression_head_input = pose_regression_head_input#'default'#'mapero' 'optic_flow' 'default' 'corr' 'cat_feats' 'add_feats' 'optic_flow' default is raw_feat
        # self.pose_regression_head_input = 'mapero'#'mapero_2Dbv''default' 'corr' 'cat_feats' 'add_feats' 'optic_flow' 'optic_flow_detach' default is raw_feat
        self.mapero_pixel_pe_scheme = mapero_pixel_pe_scheme #'focal_norm' #focal_norm  focal_norm_OF_warped 
        #self.mapero_pixel_pe_scheme = 'focal_norm_OF_warped' #focal_norm  focal_norm_OF_warped 
        # 'focal_norm',  # use the pts3d_in_other_view+2D
        # 'focal_norm_OF_warped',  # use the pts3d in itself view+Flow_corrsponded_Based_2D
        if 'mapero' in self.pose_regression_head_input:
            if self.mapero_pixel_pe_scheme == 'focal_norm_OF_warped':
                assert self.init_2d_optic_flow, f'pose_regression_head_input={self.pose_regression_head_input} requires init_2d_optic_flow to be True'
                assert self.init_3d_depth, f'pose_regression_head_input={self.pose_regression_head_input} requires init_3d_depth to be True'
            elif self.mapero_pixel_pe_scheme == 'focal_norm':
                assert self.init_3d_scene_flow, f'pose_regression_head_input={self.pose_regression_head_input} requires init_3d_scene_flow to be True'
            else:
                assert 0, f'Unknown mapero_pixel_pe_scheme {self.mapero_pixel_pe_scheme}, should be one of [focal_norm_OF_warped, focal_norm]'
        # assert self.pose_regression_head_input in ['default', 'mapero', 'corr', 'cat_feats', 'add_feats', 'optic_flow','optic_flow_detach','cat_with_optic_flow','cat_with_optic_flow_detach'], f'Unknown pose_regression_head_input {self.pose_regression_head_input}, should be one of [default, corr, cat_feats, add_feats]'       

        assert self.pose_regression_head_input in ['default', 
                                                   'mapero',
                                                   'cat_feats', 'add_feats',
                                                   'corr', 
                                                   'optic_flow','optic_flow_detach',
                                                   'cat_with_optic_flow','cat_with_optic_flow_detach',\
                                                    ], f'Unknown pose_regression_head_input {self.pose_regression_head_input}, should be one of [default, corr, cat_feats, add_feats]'
        #init the learned PE for mapero
        if 'mapero' in self.pose_regression_head_input:
            #/////////////////
            config_PE_detail = {
                "d_model": 32,
                # "default_img_HW": [384, 512],
                "default_img_HW": [None, None],
                "nerf_frequency_band": 5,
                "oor": 50
            }
            # assert of1.shape[2] == config_PE_detail['default_img_HW'][0], f'optic flow shape {of1.shape[2:]} should be {config_PE_detail["default_img_HW"]}'
            # assert of1.shape[3] == config_PE_detail['default_img_HW'][1], f'optic flow shape {of1.shape[2:]} should be {config_PE_detail["default_img_HW"]}'
            # assert pts3d_1.shape[2] == config_PE_detail['default_img_HW'][0], f'pts3d shape {pts3d_1.shape[2:]} should be {config_PE_detail["default_img_HW"]}'
            # assert pts3d_1.shape[3] == config_PE_detail['default_img_HW'][1], f'pts3d shape {pts3d_1.shape[2:]} should be {config_PE_detail["default_img_HW"]}'
            config_PE = {
                'config': config_PE_detail,
                'max_shape': (0, 0),
                'in_ch_dim': 3,
                'sc_pe': 'nerf',  # 'default', 'nerf'
                # 'sc_pe': '3dbv_lofter_depth_nerf',  # 'nerf' 'default' '3dbv_lofter_depth_nerf'
                # 'pixel_pe': 'focal_norm',  # use the pts3d_in_other_view+2D
                # 'pixel_pe': 'focal_norm_OF_warped',  # use the pts3d in itself view+Flow_corrsponded_Based_2D
                'pixel_pe': self.mapero_pixel_pe_scheme,  # use the pts3d in itself view+Flow_corrsponded_Based_2D
                # 'pixel_pe': '2d_bearing_vector',  # use the pts3d in itself view+Flow_corrsponded_Based_2D
                'mag_heuristic': 400,
                'scenecoord_img_ratio_H': 4,  # 8 for 384/48
                'scenecoord_img_ratio_W': 4,  # 8 for 512/64
                # 'scenecoord_img_ratio_H': 1,  # 1 for 384/384
                # 'scenecoord_img_ratio_W': 1,  # 1 for 512/512
            }
            assert config_PE['scenecoord_img_ratio_W'] == config_PE['scenecoord_img_ratio_H'], f'scenecoord_img_ratio_W {config_PE["scenecoord_img_ratio_W"]} should be the same as scenecoord_img_ratio_H {config_PE["scenecoord_img_ratio_H"]}'
            self.marepo_scenecoord_img_ratio = config_PE['scenecoord_img_ratio_H']
            self.set_mapero_pos_encoding(config_PE)
        # marepo_pose_head_ipt_patch_size will affect the param dim of regression MLP
        self.marepo_pose_head_ipt_patch_size = 4 #further reduce based on the downsampled PE (scenecoord_img_ratio)
        self.marepo_pose_head_ipt_patch_size = 1

        # debug old model
        # self.marepo_pose_head_ipt_patch_size = 16

        self.pose_head_seperate_scale = pose_head_seperate_scale # whether to use separate scale for pose regression

        self.patch_size = patch_size# used for masked_pose_regress: resize the mask to be consistent with token dim
        self.img_size = img_size# used for corr computation
        self.landscape_only = landscape_only # use to reshape the dec for corr computation
        self.set_downstream_head_pose()

        if self.init_dynamic_mask_estimator:
            self.set_downstream_head_motion_mask(mask_conf_mode=('sigmoid', 0, 1))
        # Concise flow head setup using a dictionary mapping flow_head_names to their output modes
        flow_configs = {
            'optic_flow': 'flow2d',
            'scene_flow': 'pts3d',
            'motion_flow': 'pts3d',
            'ego_flow': 'pts3d',
            'depth': 'pts3d',  
        }
        for flow_head_name, head_output_mode in flow_configs.items():
            head_type = getattr(self, f'{flow_head_name}_estimator_type')
            self.set_downstream_head_general_flow(patch_size=patch_size, img_size=img_size,
                                              output_mode=head_output_mode, 
                                              head_type=head_type,
                                              landscape_only=landscape_only,
                                              depth_mode=('exp', -inf, inf), 
                                              conf_mode=('exp', 1, inf),
                                              flow_head_name=flow_head_name)
        
        self.initialize_weights() 

    def initialize_epropnp(self):
        '''
        init: log_weight_scale, camera, cost_fun, epropnp
        '''
        from lib.ops.pnp.epropnp import EProPnP6DoF
        from lib.ops.pnp.levenberg_marquardt import LMSolver, RSLMSolver
        from lib.ops.pnp.camera import PerspectiveCamera
        from lib.ops.pnp.cost_fun import AdaptiveHuberPnPCost
        
        self.log_weight_scale = nn.Parameter(torch.zeros(2))# Here we use static weight_scale because the data noise is homoscedastic
        self.camera = PerspectiveCamera()
        self.cost_fun = AdaptiveHuberPnPCost(relative_delta=0.5)
        self.epropnp = EProPnP6DoF(
            mc_samples=512,
            num_iter=4,
            solver=LMSolver(
                dof=6,
                num_iter=10,
                init_solver=RSLMSolver(
                    dof=6,
                    num_points=8,
                    num_proposals=128,
                    num_iter=5)))

    def epropnp_pose_head(self, x3d, x2d, w2d, cam_mats, pose_init, matches_num = 64):
        '''
        adapt from epropnp demo notebook.
        '''
        # if x3d.shape[1] > matches_num:
        #     # randomly sample matches_num matches for x3d, x2d, w2d
        #     # 1. randomly sample matches_num matches
        #     matches_num = x3d.shape[1]
        #     x3d = x3d[torch.randperm(x3d.shape[0])[:matches_num]]
        #     x2d = x2d[torch.randperm(x2d.shape[0])[:matches_num]]
        #     w2d = w2d[torch.randperm(w2d.shape[0])[:matches_num]]

        # x3d, x2d, w2d = self.forward_correspondence(in_pose)
        self.camera.set_param(cam_mats)
        self.cost_fun.set_param(x2d.detach(), w2d)  # compute dynamic delta
        pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(
            x3d,
            x2d,
            w2d,
            self.camera,
            self.cost_fun,
            pose_init=pose_init,
            force_init_solve=True,
            with_pose_opt_plus=True)  # True for derivative regularization loss
        # norm_factor = model.log_weight_scale.detach().exp().mean()
        norm_factor = self.log_weight_scale.detach().exp().mean()
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt, norm_factor


    def flow_head_factory(self, head_type, output_mode, has_conf):
        if head_type == 'linear':
            # assert output_mode == 'pts3d', output_mode
            assert output_mode in ['pts3d','flow2d','disp1d'], output_mode
            # return LinearPts3d(self, has_conf)
            return LinearPts3d(self, has_conf, output_mode=output_mode)
        elif head_type == 'dpt':
            assert output_mode in ['pts3d','flow2d','disp1d'], output_mode
            return create_dpt_head(self, has_conf=has_conf, output_mode=output_mode)
        else:
            assert 0, NotImplementedError

    def set_mapero_pos_encoding(self,config_PE):
        from mvp3r.models.utils.position_encoding import PositionEncodingSine
        # config_PE['scenecoord_img_ratio_H'] = 1#int(of1.shape[2] / pts3d_1.shape[2])# our of and pts3d are in the same resolution
        # config_PE['scenecoord_img_ratio_W'] = 1#int(of1.shape[3] / pts3d_1.shape[3])
        # assert config_PE['scenecoord_img_ratio_H'] == 1 and config_PE['scenecoord_img_ratio_W'] == 1, f'optic flow shape {of1.shape[2:]} should be the same as pts3d shape {pts3d_1.shape[2:]}'
        self.mapero_pos_encoding = PositionEncodingSine(**config_PE)
        # self.mapero_pos_encoding.marepo_scenecoord_img_ratio = config_PE['scenecoord_img_ratio_H']

    def set_downstream_head_pose(self):
        # always init pose head
        # pose regression head
        self.pose_head = PoseHead(net=self, seperate_scale=self.pose_head_seperate_scale)
        # magic wrapper
        if self.pose_regression_with_mask:
            self.head = transpose_to_landscape_with_mask(self.pose_head, activate=True)
        else:
            self.head = transpose_to_landscape(self.pose_head, activate=True)

    def set_downstream_head_motion_mask(self,mask_conf_mode):
        # add dynamic mask head
        self.mask_conf_mode = mask_conf_mode
        if self.init_dynamic_mask_estimator:
            assert self.dynamic_mask_estimator_type in ['dpt', 'linear'], f'Unknown dynamic mask estimator type {self.dynamic_mask_estimator_type}'
            if self.dynamic_mask_estimator_type == 'dpt':
                mask_estimator = create_motion_mask_dpt_head
            elif self.dynamic_mask_estimator_type == 'linear':
                mask_estimator = Linear_MaskEsti
            else:
                assert 0, f'{self.dynamic_mask_estimator_type} is not a valid dynamic mask estimator type'

            if self.shared_dynamic_mask_estimator:
                self.downstream_head_dynamic_mask = mask_estimator(net=self)
                # magic wrapper
                self.head_dynamic_mask = transpose_to_landscape(self.downstream_head_dynamic_mask, activate=True)
            else:
                self.downstream_head_dynamic_mask1 = mask_estimator(net=self)
                self.downstream_head_dynamic_mask2 = mask_estimator(net=self)
                # magic wrapper
                self.head_dynamic_mask1 = transpose_to_landscape(self.downstream_head_dynamic_mask1, activate=True)
                self.head_dynamic_mask2 = transpose_to_landscape(self.downstream_head_dynamic_mask2, activate=True)
 
    def set_downstream_head_general_flow(self, 
                            patch_size, img_size,
                            output_mode = 'pts3d', head_type = 'linear', 
                            landscape_only = True, 
                            depth_mode = ('exp', -inf, inf), conf_mode = ('exp', 1, inf), 
                            flow_head_name = None,
                            **kw):
        # add scene_flow head
        # def: scene flow contains: motion+ego
        # construct an attribute with an variable_name self.haha_{flow_head_name}
        assert flow_head_name in ['scene_flow', 'ego_flow', 'motion_flow', 'optic_flow', 'depth'], f'Unknown flow head name {flow_head_name}, should be one of [optic_flow, scene_flow, ego_flow, motion_flow]'
        '''
        follow the depth head in dust3r
        '''
        if isinstance(img_size, tuple):
            assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
                f'{img_size=} must be multiple of {patch_size=}'
        else:
            assert img_size % patch_size == 0, \
                f'{img_size=} must be multiple of {patch_size=}'
        
        # they are needed once when init the heads, once init, it is saved within the head
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        if flow_head_name == 'scene_flow':
            if self.init_3d_scene_flow:
                self.downstream_head_scene_flow = self.flow_head_factory(head_type, output_mode, has_conf=bool(conf_mode))
                self.head_scene_flow = transpose_to_landscape(self.downstream_head_scene_flow, activate=landscape_only)
        elif flow_head_name == 'ego_flow':
            if self.init_3d_ego_flow:
                self.downstream_head_ego_flow = self.flow_head_factory(head_type, output_mode, has_conf=bool(conf_mode))
                self.head_ego_flow = transpose_to_landscape(self.downstream_head_ego_flow, activate=landscape_only)
        elif flow_head_name == 'motion_flow':
            if self.init_3d_motion_flow:
                self.downstream_head_motion_flow = self.flow_head_factory(head_type, output_mode, has_conf=bool(conf_mode))
                self.head_motion_flow = transpose_to_landscape(self.downstream_head_motion_flow, activate=landscape_only)
        elif flow_head_name == 'depth':
            if self.init_3d_depth:
                self.downstream_head_depth = self.flow_head_factory(head_type, output_mode, has_conf=bool(conf_mode))
                self.head_depth = transpose_to_landscape(self.downstream_head_depth, activate=landscape_only)
        elif flow_head_name == 'optic_flow':
            if self.init_2d_optic_flow:
                self.downstream_head_optic_flow = self.flow_head_factory(head_type, output_mode, has_conf=bool(conf_mode))
                self.head_optic_flow = transpose_to_landscape(self.downstream_head_optic_flow, activate=landscape_only)
        else:
            assert 0, f'Unknown flow head name {flow_head_name}, should be one of [optic_flow, scene_flow, ego_flow, motion_flow]'

        # allocate heads
        # self.downstream_head_scene_flow = self.flow_head_factory(head_type, output_mode, has_conf=bool(conf_mode))
        # we can reset to None, wont affect once init is done
        self.depth_mode = None
        self.conf_mode = None
        # magic wrapper
        # self.head_scene_flow = transpose_to_landscape(self.downstream_head_scene_flow, activate=landscape_only)

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_encoder(self):
        freeze_all_params([self.patch_embed, self.enc_blocks])

    def set_freeze(self, freeze):  # this is for use by downstream models
        '''
        copy from das3r
        '''
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],

            # 'encoder': [self.mask_token, self.patch_embed, self.enc_blocks], #mast3r
            # 'mask': [self.mask_token], #dust3r
            # 'encoder': [self.mask_token, self.patch_embed, self.enc_blocks], #dust3r

            'encoder':  [self.patch_embed, self.enc_blocks],
            'encoder_decoder': [self.patch_embed, self.enc_blocks, self.dec_blocks],
            'encoder_decoder_pose_head': [self.patch_embed, self.enc_blocks, self.dec_blocks, self.pose_head],
            'encoder_pose_head': [self.patch_embed, self.enc_blocks, self.pose_head],
        }
        assert freeze in to_be_frozen, f'Unknown freeze option {freeze}, should be one of {list(to_be_frozen.keys())}'
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def load_state_dict(self, ckpt, **kw):
        import copy
        # new_ckpt = dict(ckpt)
        new_ckpt = copy.deepcopy(dict(ckpt))

        if any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks2'):
                    new_ckpt[key.replace('dec_blocks2', 'dec_blocks')] = value

        list_of_heads_reusing_pretrained_regression_head = []
        # if args.use_pretrained_head_ego_flow:
        # list_of_heads_reusing_pretrained_regression_head.append('downstream_head_ego_flow')
        # if args.use_pretrained_head_motion_flow:
        # list_of_heads_reusing_pretrained_regression_head.append('downstream_head_motion_flow')
        if self.init_3d_scene_flow and self.scene_flow_estimator_type == 'dpt':
            list_of_heads_reusing_pretrained_regression_head.append('downstream_head_scene_flow')
        # if args.use_pretrained_head_depth:
        # list_of_heads_reusing_pretrained_regression_head.append('downstream_head_depth')

        for head_name in list_of_heads_reusing_pretrained_regression_head:
            if any(k.startswith('downstream_head2') for k in ckpt):
                for key, value in ckpt.items():
                    if key.startswith('downstream_head2') and head_name != 'downstream_head_depth':
                        new_ckpt[key.replace('downstream_head2',head_name)] = value 
                        print(f'{key} loaded to {head_name}......')
            if any(k.startswith('downstream_head1') for k in ckpt):
                for key, value in ckpt.items():
                    if key.startswith('downstream_head1') and head_name == 'downstream_head_depth':
                        new_ckpt[key.replace('downstream_head1',head_name)] = value
                        print(f'{key} loaded to {head_name}......')

        del ckpt        # in case it occupies memory


        return super().load_state_dict(new_ckpt, **kw)


    # def load_state_dict(self, ckpt, **kw):
    #     new_ckpt = dict(ckpt)
    #     if any(k.startswith('dec_blocks2') for k in ckpt):
    #         for key, value in ckpt.items():
    #             if key.startswith('dec_blocks2'):
    #                 new_ckpt[key.replace('dec_blocks2', 'dec_blocks')] = value
    #     if any(k.startswith('head4') for k in ckpt):
    #         assert 0, "head4 is not supported"
    #         for key, value in ckpt.items():
    #             if key.startswith('head4'):
    #                 new_ckpt[key.replace('head4', 'head')] = value
    #     return super().load_state_dict(new_ckpt, **kw)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encoder(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, mask1, mask2, swap_to_decoder2 = False):
        final_output = [(f1, f2)]  # before projection
        attention_maps = []

        # project to decoder dim
        f1 = self.decoder_embed(f1) #if not swap_to_decoder2 else self.decoder_embed2(f1)
        f2 = self.decoder_embed(f2) #if not swap_to_decoder2 else self.decoder_embed2(f2)

        final_output.append((f1, f2))
        for blk in (self.dec_blocks if not swap_to_decoder2 else self.dec_blocks_2nd):
            # img1 side
            # f1, _ = blk(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            # f2, _ = blk(*final_output[-1][::-1], pos2, pos1)
            f1, _, self_attn1, cross_attn1 = blk(*final_output[-1][::+1], pos1, pos2, mask1, mask2, None) # mask1, mask2, mask1
            # mvp3r masked attn implementaion: ((qmask > 0.5) | (kmask > 0.5))
            # old easi3r: attention_mask = ((qmask < 0.5) & (kmask > 0.5))
            if self.reweight_both_when_use_mask:
                f2, _, self_attn2, cross_attn2 = blk(*final_output[-1][::-1], pos2, pos1, mask2, mask1, None) # mask2, mask1, mask2
            else:
                f2, _, self_attn2, cross_attn2 = blk(*final_output[-1][::-1], pos2, pos1, None, None, None) # mask2, mask1, mask2            
            # store the result
            final_output.append((f1, f2))
            attention_maps.append((self_attn1, cross_attn1, self_attn2, cross_attn2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1])) if not swap_to_decoder2 \
            else tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, head_id, decout, img_shape, mask = None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_id}')
        if mask is not None:
            return head(decout, img_shape, mask)
        else:
            return head(decout, img_shape)

    def _resize_mask(self, view, shape, query_key='atten_mask'):
        '''
        copy from easi3r for attn mask obtaining
        '''
        return resize_mask(view=view, shape=shape, query_key = query_key)  # if 'atten_mask' not in view, return None

    def _get_attn_k(self, attn, shape):
        '''
        copy from easi3r for attn mask obtaining
        '''
        return get_attn_k(attn=attn, shape=shape)    
    
    def _get_mask_for_decoder(self, view1, view2, shape1, shape2):
        # compute mask if needed(only for the 2nd inference)
        if self.dec_use_atten_mask or self.dec_use_motion_mask:
            if self.dec_use_atten_mask:
                query_key = 'atten_mask'
            else:
                assert self.dec_use_motion_mask
                if self.dec_use_which_motion_mask == 'esti':
                    query_key = 'motion_mask'
                    assert 0,'no motion mask esti...'
                elif self.dec_use_which_motion_mask == 'gt':
                    query_key = 'dynamic_mask'
                else:
                    assert 0, f'Unknown dec_use_which_motion_mask {self.dec_use_which_motion_mask}, should be one of [esti, gt]'
            mask1 = self._resize_mask(view1, shape1, query_key=query_key)  # if 'atten_mask' not in view1, return None
            mask2 = self._resize_mask(view2, shape2, query_key=query_key)  # if 'atten_mask' not in view2, return None
            # assert mask1 is None and mask2 is None,f'mask should be none for the 1st inference model forward ' 
        else:
            mask1 = None
            mask2 = None
            assert mask1 is None and mask2 is None,f'mask should be none when not dec_use_atten_mask ' 

        return mask1, mask2

    def inference_motion_mask(self, dec1, dec2, shape1, shape2):
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            if self.shared_dynamic_mask_estimator:
                # use the same dynamic mask estimator for both images
                # detach_token_grad_for_dyn_mask = True
                # detach_token_grad_for_dyn_mask = False 
                if self.detach_token_grad_for_dyn_mask:
                    mask_1 = self._downstream_head('_dynamic_mask', [tok.float().detach() for tok in dec1], shape1)
                    mask_2 = self._downstream_head('_dynamic_mask', [tok.float().detach() for tok in dec2], shape2)
                else:
                    mask_1 = self._downstream_head('_dynamic_mask', [tok.float() for tok in dec1], shape1)
                    mask_2 = self._downstream_head('_dynamic_mask', [tok.float() for tok in dec2], shape2)
            else:
                mask_1 = self._downstream_head('_dynamic_mask1', [tok.float() for tok in dec1], shape1)
                mask_2 = self._downstream_head('_dynamic_mask2', [tok.float() for tok in dec2], shape2)
        return mask_1, mask_2

    def inference_optic_flow(self, dec1, dec2, shape1, shape2):
        # self.optic_flow_name_appendix = 'optic_flow'
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            if self.shared_optic_flow_estimator:
                # use the same optic flow estimator for both images
                optic_flow1 = self._downstream_head('_optic_flow', [tok.float() for tok in dec1], shape1)
                optic_flow2 = self._downstream_head('_optic_flow', [tok.float() for tok in dec2], shape2)
            else:
                assert 0, NotImplementedError
        return optic_flow1, optic_flow2

    def _get_mask_for_pose_head(self, mask_1, mask_2, view1, view2, dec1):

        if self.pose_regression_with_mask:
            B, S, D = dec1[-1].shape #1 768 768
            # self.patch_size = net.marepo_pose_head_ipt_patch_size*net.marepo_scenecoord_img_ratio
            # downsample_patch_size = self.marepo_pose_head_ipt_patch_size*self.marepo_scenecoord_img_ratio
            # reshape mask to B S 1
            assert self.pose_regression_which_mask in ['esti','detached_esti', 'gt'], f'Unknown pose_regression_which_mask {self.pose_regression_which_mask}, should be one of [esti, gt]'
            if self.pose_regression_which_mask in ['esti', 'detached_esti']:
                try:
                    pose_esti_mask1 = mask_1['mask_conf']
                    pose_esti_mask2 = mask_2['mask_conf']
                except Exception as e:
                    print(f'Fail for masked_pose_regression! Error when getting mask_conf from mask_1 and mask_2: {e}')
                    pose_esti_mask1 = None
                    pose_esti_mask2 = None
            elif self.pose_regression_which_mask == 'gt':
                try:
                    pose_esti_mask1 = view1['dynamic_mask']#view1.get('dynamic_mask', None)
                    pose_esti_mask2 = view2['dynamic_mask']#view2.get('dynamic_mask', None)
                except Exception as e:
                    print(f'Fail for masked_pose_regression! Error when getting mask_conf from mask_1 and mask_2: {e}')
                    pose_esti_mask1 = None
                    pose_esti_mask2 = None
            else:
                assert 0, f'Unknown pose_regression_which_mask {self.pose_regression_which_mask}, should be one of [esti, gt]'

            if pose_esti_mask1 is not None:
                assert pose_esti_mask2 is not None, f'pose_esti_mask1 and pose_esti_mask2 should be both from gt or esti to be consistent'
                _, H, W = pose_esti_mask1.shape
                mask_resize_patch = self.pose_head.patch_size_final
                # print('the current patch size is:', self.pose_head.patch_size_final)
                # resize to be consistent with token num S 
                if self.pose_regression_which_mask == 'gt':
                    # used fro binary
                    pose_esti_mask1 = F.interpolate(pose_esti_mask1.unsqueeze(1), size=(H//mask_resize_patch, W//mask_resize_patch), mode='nearest').squeeze(1)
                    pose_esti_mask2 = F.interpolate(pose_esti_mask2.unsqueeze(1), size=(H//mask_resize_patch, W//mask_resize_patch), mode='nearest').squeeze(1)
                elif self.pose_regression_which_mask in ['esti','detached_esti']:
                    if self.pose_regression_which_mask == 'detached_esti':
                        # detach the mask for pose regression
                        pose_esti_mask1 = pose_esti_mask1.detach()
                        pose_esti_mask2 = pose_esti_mask2.detach()
                    # used for float
                    pose_esti_mask1 = F.interpolate(pose_esti_mask1.unsqueeze(1), size=(H//mask_resize_patch, W//mask_resize_patch), mode='bilinear', align_corners=False).squeeze(1)
                    pose_esti_mask2 = F.interpolate(pose_esti_mask2.unsqueeze(1), size=(H//mask_resize_patch, W//mask_resize_patch), mode='bilinear', align_corners=False).squeeze(1)
                else:
                    assert 0, f'Unknown pose_regression_which_mask {self.pose_regression_which_mask}, should be one of [esti, gt]'     
                # print('the current pose_esti_mask1 shape is:', pose_esti_mask1.shape)
                # print('the current pose_esti_mask2 shape is:', pose_esti_mask2.shape)
                # view to be consistent with token shape
                # convert to bool of 0 or 1
                pose_esti_mask1 = pose_esti_mask1.view(B, -1, 1) 
                pose_esti_mask2 = pose_esti_mask2.view(B, -1, 1)
                # convert from motion mask to static mask
                pose_esti_mask1 = 1 - pose_esti_mask1
                pose_esti_mask2 = 1 - pose_esti_mask2 
                # # hard threshold will kill grad
                use_STE = True
                # use_STE = False
                if not use_STE:
                    scale = 10# min(1 + epoch * 2, 50)
                    # scale = 0.1# min(1 + epoch * 2, 50)
                    print('Before Pose Esti Mask, scale:', scale, 'max:', pose_esti_mask1.max(), 'min:', pose_esti_mask1.min())
                    
                    # Store clamping statistics for logging
                    self.clamp_stats = {}
                    log_clamp_stats = True
                    # log_clamp_stats = False
                    if log_clamp_stats:                     
                        # Before clamping
                        before_clamp1 = (pose_esti_mask1.detach() - 0.5) * scale
                        before_clamp2 = (pose_esti_mask2.detach() - 0.5) * scale
                    
                    # After clamping
                    pose_esti_mask1 = torch.clamp((pose_esti_mask1 - 0.5) * scale, 0, 1)
                    pose_esti_mask2 = torch.clamp((pose_esti_mask2 - 0.5) * scale, 0, 1)
                    
                    # Store statistics
                    # log_clamp_stats = True
                    # log_clamp_stats = False
                    if log_clamp_stats:                    
                        # Calculate clamping statistics
                        total_pixels1 = pose_esti_mask1.numel()
                        total_pixels2 = pose_esti_mask2.numel()
                        
                        # Count pixels clamped to 0
                        clamped_to_zero1 = (pose_esti_mask1 == 0).sum().item()
                        clamped_to_zero2 = (pose_esti_mask2 == 0).sum().item()
                        
                        # Count pixels clamped to 1
                        clamped_to_one1 = (pose_esti_mask1 == 1).sum().item()
                        clamped_to_one2 = (pose_esti_mask2 == 1).sum().item()
                        
                        # Calculate ratios
                        ratio_clamped_to_zero1 = clamped_to_zero1 / total_pixels1
                        ratio_clamped_to_zero2 = clamped_to_zero2 / total_pixels2
                        ratio_clamped_to_one1 = clamped_to_one1 / total_pixels1
                        ratio_clamped_to_one2 = clamped_to_one2 / total_pixels2
                    

                        self.clamp_stats = {
                            '/pose_mask1_clamped_to_zero_ratio': ratio_clamped_to_zero1,
                            # '/pose_mask2_clamped_to_zero_ratio': ratio_clamped_to_zero2,
                            '/pose_mask1_clamped_to_one_ratio': ratio_clamped_to_one1,
                            # '/pose_mask2_clamped_to_one_ratio': ratio_clamped_to_one2,
                            '/pose_mask1_clamped_to_zero_count': clamped_to_zero1,
                            # '/pose_mask2_clamped_to_zero_count': clamped_to_zero2,
                            '/pose_mask1_clamped_to_one_count': clamped_to_one1,
                            # '/pose_mask2_clamped_to_one_count': clamped_to_one2,
                            # 'pose_mask1_total_pixels': total_pixels1,
                            # 'pose_mask2_total_pixels': total_pixels2,
                            '/pose_mask1_before_clamp_min': before_clamp1.min().item(),
                            '/pose_mask1_before_clamp_max': before_clamp1.max().item(),
                            # '/pose_mask2_before_clamp_min': before_clamp2.min().item(),
                            # '/pose_mask2_before_clamp_max': before_clamp2.max().item(),
                            # 'pose_mask1_after_clamp_min': pose_esti_mask1.min().item(),
                            # 'pose_mask1_after_clamp_max': pose_esti_mask1.max().item(),
                            # 'pose_mask2_after_clamp_min': pose_esti_mask2.min().item(),
                            # 'pose_mask2_after_clamp_max': pose_esti_mask2.max().item(),
                            '/clamp_scale': scale
                        }
                        
                        print('After Pose Esti Mask, scale:', scale, 'max:', pose_esti_mask1.max(), 'min:', pose_esti_mask1.min())
                        print(f'Clamping stats - Mask1: {ratio_clamped_to_zero1:.3f} to 0, {ratio_clamped_to_one1:.3f} to 1')
                        print(f'Clamping stats - Mask2: {ratio_clamped_to_zero2:.3f} to 0, {ratio_clamped_to_one2:.3f} to 1')
                        
                        # Clear intermediate tensors to free memory
                        del before_clamp1, before_clamp2
                        torch.cuda.empty_cache()
                        # print('ORI is used for pose regression mask',pose_esti_mask1.requires_grad)
                else:
                    pose_esti_mask1_hard = (pose_esti_mask1 > 0.5).float()
                    pose_esti_mask2_hard = (pose_esti_mask2 > 0.5).float()
                    pose_esti_mask1 = pose_esti_mask1_hard+pose_esti_mask1-pose_esti_mask1.detach()
                    pose_esti_mask2 = pose_esti_mask2_hard+pose_esti_mask2-pose_esti_mask2.detach()
        else:
            pose_esti_mask1 = None
            pose_esti_mask2 = None
        return pose_esti_mask1, pose_esti_mask2

    def _init_base_grid(self, H, W, device, reallocate=False):
        """use for construct the 2D in 2D-3D matches, assume 3D are regressed SC."""
        if self._base_grid is None or reallocate:
            hh, ww = torch.meshgrid(torch.arange(
                H).float(), torch.arange(W).float())
            coord = torch.zeros([1, H, W, 2])
            coord[0, ..., 0] = ww
            coord[0, ..., 1] = hh
            self._base_grid = coord.to(device)

    def prepare_epropnp_input(self, pts3d_2in1view, pts3d_2in1view_conf, 
                                K_2, B, H, W, device, 
                                # matches_num = 8,
                                matches_num = 48,
                                ):
        '''
        return pose_init: xyz_wxyz
        '''
        assert pts3d_2in1view.shape[-1] == 3, f'pts3d_2in1view.shape[-1] should be 3, but got {pts3d_2in1view.shape[-1]}'
        assert pts3d_2in1view_conf.shape[-1] == 1, f'pts3d_2in1view_conf.shape[-1] should be 1, but got {pts3d_2in1view_conf.shape[-1]}'
        
        x3d = pts3d_2in1view.reshape(B, -1, 3)
        x3d_conf = pts3d_2in1view_conf.reshape(B, -1, 1)
        # reset self._base_grid for various image size
        self._init_base_grid(H=H, W=W, device=device)
        x2d = self._base_grid.repeat(B, 1, 1, 1)# B H W 2
        x2d = x2d.reshape(B, -1, 2)
        w2d = x3d_conf.repeat(1, 1, 2).detach()
        # sample  matches_num tuples (x3d_1_i,x2d_1_i,w2d_1_i)
        # Generate a random permutation of indices
        
        if matches_num < x3d.shape[0]:
            idx = torch.randperm(x3d.shape[0])[:matches_num]
            x3d = x3d[idx]
            x2d = x2d[idx]
            w2d = w2d[idx]

        cam_mats = K_2[:,:3,:3] 
        # in Epropnp: pose are in XYZ_quat 7 dim format
        # pose_init_1 = torch.eye(4).repeat(B, 1, 1).to(view1['img'].device) 
        # xyz_wxyz
        pose_init = torch.zeros([B, 7], device=device)
        # set quat as 1,0,0,0
        pose_init[:, 3:] = torch.tensor([0,0,0,1], device=device).repeat(B, 1)
        pose_init = None

        return x3d, x2d, w2d, cam_mats, pose_init


    def inference_pose(self, view1, view2, mask_1, mask_2, dec1, dec2, shape1, shape2, 
                       optic_flow1, optic_flow2,
                       depth1, depth2,
                       scene_flow1, scene_flow2,):
        # inference_pose
        with torch.cuda.amp.autocast(enabled=False):
            # prepare according to pose_regression_head_input
            # assert 0, view1.keys()
            if self.unireloc3r_pose_estimation_mode == 'vanilla_pose_head_regression':
                pose_esti_mask1, pose_esti_mask2 = self._get_mask_for_pose_head(mask_1, mask_2, view1, view2, dec1)
                pose_regress_input1, pose_regress_input2 = self.prepare_pose_reg_input(dec1, dec2, 
                                                                                    of1=optic_flow1['flow2d'], of2=optic_flow2['flow2d'],
                                                                                    pts3d_1=depth1['pts3d'], pts3d_2=depth2['pts3d'],
                                                                                    pts3d_1in2view=scene_flow1['pts3d'], pts3d_2in1view=scene_flow2['pts3d'],
                                                                                    K_1=view1.get('camera_intrinsics', None), K_2=view2.get('camera_intrinsics', None),
                                                                                    ) 

                pose1 = self._downstream_head('', [tok.float() for tok in pose_regress_input1], shape1, pose_esti_mask1)  
                pose2 = self._downstream_head('', [tok.float() for tok in pose_regress_input2], shape2, pose_esti_mask2)  # relative camera pose from 2 to 1. 
                return pose1, pose2
            elif self.unireloc3r_pose_estimation_mode == 'epropnp':
                '''
                # return pose1(pose1to2), pose2(pose2to1)

                leverage matched 3D scene_flow to establish the 3D-2D correspondences
                then use the epropnp to estimate the pose.
                when compute pose1to2: as pnp, we need 3d pts w.r.t cam1, 
                2d and K from cam2;
                '''
                B, C, H, W = view1['img'].shape
                pose1, pose2 = {}, {}
                self._init_base_grid(H=H, W=W, device=view1['img'].device)

                pts3d_1 = depth1['pts3d']
                pts3d_2 = depth2['pts3d']
                pts3d_1in2view = scene_flow1['pts3d']
                pts3d_2in1view = scene_flow2['pts3d']# B H W 3
                pts3d_1in2view_conf = scene_flow1['conf']
                pts3d_2in1view_conf = scene_flow2['conf']

                # print('key in scene_flow1:', scene_flow1.keys())
                # print('key in scene_flow2:', scene_flow2.keys())
                K_1 = view1['camera_intrinsics'] # B 4 4
                K_2 = view2['camera_intrinsics'] # B 4 4

                # def quaternion_wxyz_to_matrix_pytorch3d(quaternions: torch.Tensor) -> torch.Tensor:
                #     """
                #     Convert rotations given as quaternions to rotation matrices.

                #     Args:
                #         quaternions(wxyz): quaternions with real part first,
                #             as tensor of shape (..., 4).

                #     Returns:
                #         Rotation matrices as tensor of shape (..., 3, 3).
                #     """
                #     r, i, j, k = torch.unbind(quaternions, -1)
                #     # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
                #     two_s = 2.0 / (quaternions * quaternions).sum(-1)

                #     o = torch.stack(
                #         (
                #             1 - two_s * (j * j + k * k),
                #             two_s * (i * j - k * r),
                #             two_s * (i * k + j * r),
                #             two_s * (i * j + k * r),
                #             1 - two_s * (i * i + k * k),
                #             two_s * (j * k - i * r),
                #             two_s * (i * k - j * r),
                #             two_s * (j * k + i * r),
                #             1 - two_s * (i * i + j * j),
                #         ),
                #         -1,
                #     )
                #     return o.reshape(quaternions.shape[:-1] + (3, 3))

                # def xyz_quat_to_matrix_kornia(
                #     xyz_quat: torch.Tensor, 
                #     quat_format: str = 'wxyz',
                #     soft_clamp_quat: bool = False,
                #     max_angle_rad: float = 0.1,
                # ) -> torch.Tensor:
                #     """
                #     Convert [B, 7] tensor (tx,ty,tz,qx,qy,qz,qw or wxyz) to [B,4,4] homogeneous matrices.
                    
                #     Args:
                #         xyz_quat: [B, 7] tensor, where the last 4 are quaternion components.
                #         quat_format: 'xyzw' if input quaternions are (x,y,z,w), 
                #                     'wxyz' if input is (w,x,y,z)
                                    
                #     Returns:
                #         T: [B,4,4] homogeneous transformation matrices
                #     """
                #     assert xyz_quat.dim() == 2, f'xyz_quat.dim() should be 3, but got {xyz_quat.dim()}'
                #     assert xyz_quat.shape[1] == 7, f'xyz_quat.shape[2] should be 7, but got {xyz_quat.shape[2]}'
                #     assert quat_format in ('xyzw', 'wxyz'), "quat_format must be 'xyzw' or 'wxyz'"
                #     B = xyz_quat.shape[0]
                #     t = xyz_quat[:, :3]   # [B,3]
                #     quat = xyz_quat[:, 3:]  # [B,4]
                #     # Reorder quaternion for Kornia (expects w,x,y,z)
                #     if quat_format == 'xyzw':
                #         quat_wxyz = torch.cat([quat[:, 3:], quat[:, :3]], dim=-1)  # x,y,z,w -> w,x,y,z
                #     elif quat_format == 'wxyz':  # already wxyz
                #         quat_wxyz = quat
                #     else:
                #         assert 0, f'Unknown quat_format {quat_format}, should be one of [xyzw, wxyz]'

                #     def clamp_quaternion_angle(q: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
                #         """
                #         Clamp quaternion rotations by maximum angle.
                        
                #         Args:
                #             q: (B, 4) tensor of quaternions (w, x, y, z), not necessarily normalized
                #             max_angle_rad: float, maximum allowed rotation angle in radians

                #         Returns:
                #             (B, 4) tensor of clamped, normalized quaternions
                #         """
                #         # normalize in case of drift
                #         q = q / q.norm(dim=-1, keepdim=True)

                #         w, xyz = q[:, 0], q[:, 1:]
                #         theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # (B,)

                #         # scale factor = sin(new_theta/2) / sin(theta/2)
                #         scale = torch.ones_like(theta)
                #         mask = theta > max_angle_rad
                #         if mask.any():
                #             new_theta = torch.full_like(theta[mask], max_angle_rad)
                #             scale_val = torch.sin(new_theta / 2) / torch.sin(theta[mask] / 2)
                #             scale[mask] = scale_val

                #         # apply scaling to xyz
                #         xyz = xyz * scale.unsqueeze(-1)

                #         # recompute w for clamped ones
                #         w = torch.where(mask, torch.cos(max_angle_rad / 2).expand_as(w), w)

                #         q_clamped = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
                #         # normalize again for safety
                #         return q_clamped / q_clamped.norm(dim=-1, keepdim=True)

                #     def soft_clamp_quaternion_angle(q: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
                #         """
                #         Softly clamp quaternion rotations by max_angle_rad using a smooth squash.
                        
                #         Args:
                #             q: (B, 4) quaternions (w, x, y, z), not necessarily normalized
                #             max_angle_rad: float, maximum allowed rotation angle in radians
                        
                #         Returns:
                #             (B, 4) softly clamped, normalized quaternions
                #         """
                #         # normalize in case of drift
                #         q = q / q.norm(dim=-1, keepdim=True)

                #         w, xyz = q[:, 0], q[:, 1:]
                #         theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # (B,)

                #         # avoid div by zero: if theta ~ 0, just keep axis as xyz
                #         axis = torch.zeros_like(xyz)
                #         mask = theta > 1e-8
                #         axis[mask] = xyz[mask] / torch.sin(theta[mask] / 2).unsqueeze(-1)

                #         # squash angle smoothly
                #         theta_clamped = max_angle_rad * torch.tanh(theta / max_angle_rad)

                #         # rebuild quaternion
                #         w_new = torch.cos(theta_clamped / 2)
                #         xyz_new = axis * torch.sin(theta_clamped / 2).unsqueeze(-1)

                #         q_new = torch.cat([w_new.unsqueeze(-1), xyz_new], dim=-1)
                #         return q_new / q_new.norm(dim=-1, keepdim=True)


                #     if soft_clamp_quat:
                #         quat_wxyz = soft_clamp_quaternion_angle(quat_wxyz, max_angle_rad=max_angle_rad)

                #     # Convert quaternion to rotation matrix: [B,3,3]
                #     # R = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat_wxyz) # seems have issue for kornia implementation
                #     R = quaternion_wxyz_to_matrix_pytorch3d(quat_wxyz)

                #     # Build homogeneous matrices
                #     T = torch.eye(4, device=xyz_quat.device, dtype=xyz_quat.dtype).unsqueeze(0).repeat(B,1,1)

                #     T[:, :3, :3] = R
                #     T[:, :3, 3] = t

                #     return T

                from reloc3r_uni.utils.epropnp_utils import xyz_quat_to_matrix_kornia,quaternion_wxyz_to_matrix_pytorch3d
                x3d_1, x2d_1, w2d_1, cam_mats_1, pose_init_1 = self.prepare_epropnp_input(pts3d_2in1view,pts3d_2in1view_conf, 
                                                                                     K_2, B, H, W, view1['img'].device)
                x3d_2, x2d_2, w2d_2, cam_mats_2, pose_init_2 = self.prepare_epropnp_input(pts3d_1in2view,pts3d_1in2view_conf,
                                                                                     K_1, B, H, W, view2['img'].device)

                # print the max and min dpeth in x3d_2
                print('ori x3d_2 min z:', x3d_2[:, :, 2].min())
                print('ori x3d_2 max z:', x3d_2[:, :, 2].max())
                print('ori x3d_2 mean x:', x3d_2[:, :, 0].mean())
                print('ori x3d_2 mean y:', x3d_2[:, :, 1].mean())
                print('ori x3d_2 mean z:', x3d_2[:, :, 2].mean())

                # print('ori x3d_1 mean depth:', x3d_1[:, :, 2].mean())
                debug_only = True
                if debug_only:
                    # better contrain in range 0.2-0.3---be consistent with trained DAM?
                    scale_x3d = 0.25 #mag smaller, safer; but can not be too small--seems get epropnp can get insance if the mag too smaller
                    
                    x3d_1 = x3d_1 * scale_x3d
                    x3d_2 = x3d_2 * scale_x3d

                    print('scaled x3d_2 min z:', x3d_2[:, :, 2].min())
                    print('scaled x3d_2 max z:', x3d_2[:, :, 2].max())
                    print('scaled x3d_2 mean x:', x3d_2[:, :, 0].mean())
                    print('scaled x3d_2 mean y:', x3d_2[:, :, 1].mean())
                    print('scaled x3d_2 mean z:', x3d_2[:, :, 2].mean())
                    # print('scaled x3d_1 mean depth:', x3d_1[:, :, 2].mean())



                _, _, pose_opt_plus_1, _, pose_sample_logweights_1, cost_tgt_1, norm_factor_1 = self.epropnp_pose_head(
                                                                                                x3d_1, x2d_1, w2d_1, 
                                                                                                cam_mats_1, 
                                                                                                pose_init_1,
                                                                                                )
                # conver B 1 7 xyz_quat to B 4 4 matrix
                # pose1['pose'] = pose_opt_plus_1
                
                #setting up
                opt_quat_format='wxyz'
                soft_clamp_quat = True
                max_angle_rad = 0.01

                pose1['pose'] = xyz_quat_to_matrix_kornia(pose_opt_plus_1, 
                                                          quat_format=opt_quat_format,
                                                          soft_clamp_quat=soft_clamp_quat,
                                                          max_angle_rad=max_angle_rad)
                pose1['pose_sample_logweights'] = pose_sample_logweights_1
                pose1['cost_tgt'] = cost_tgt_1
                pose1['norm_factor'] = norm_factor_1

                _, _, pose_opt_plus_2, _, pose_sample_logweights_2, cost_tgt_2, norm_factor_2 = self.epropnp_pose_head(
                                                                                                x3d_2, x2d_2, w2d_2, 
                                                                                                cam_mats_2, 
                                                                                                pose_init_2,
                                                                                                )
                # pose2['pose'] = pose_opt_plus_2
                pose2['pose'] = xyz_quat_to_matrix_kornia(pose_opt_plus_2, 
                                                          quat_format=opt_quat_format,
                                                          soft_clamp_quat=soft_clamp_quat,
                                                          max_angle_rad=max_angle_rad)
                pose2['pose_sample_logweights'] = pose_sample_logweights_2
                pose2['cost_tgt'] = cost_tgt_2
                pose2['norm_factor'] = norm_factor_2

                # inspect the translation part in the solved pose
                print('pose1 translation from epropnp:', pose1['pose'][:,:3,3])
                print('pose2 translation from epropnp:', pose2['pose'][:,:3,3])

                return pose1, pose2

            elif self.unireloc3r_pose_estimation_mode == 'geoaware_pnet':
                assert 0, NotImplementedError
            else:
                assert 0, f'Unknown unireloc3r_pose_estimation_mode {self.unireloc3r_pose_estimation_mode}, should be one of [vanilla_pose_head_regression, epropnp, geoaware_pnet]'
        

    def _wrap_output(self, pose1, pose2, 
                     scene_flow1, scene_flow2, 
                     depth1, depth2,
                     optic_flow1, optic_flow2, 
                     mask_1, mask_2, 
                     cross_attn1, cross_attn2,
                     shape1, shape2):
        # wrap up the opt dict            
        if self.init_3d_scene_flow:
            pose1[f'pts3d_in_other_view_{self.scene_flow_name_appendix}'] = scene_flow1['pts3d'] #may term it as scene_flow_in_other_view
            pose2[f'pts3d_in_other_view_{self.scene_flow_name_appendix}'] = scene_flow2['pts3d']
            pose1[f'conf_{self.scene_flow_name_appendix}'] = scene_flow1['conf']  # pts3d in view1's frame
            pose2[f'conf_{self.scene_flow_name_appendix}'] = scene_flow2['conf']  # pts3d in view1's frame
        if self.init_3d_depth:
            pose1[f'pts3d_{self.depth_name_appendix}'] = depth1['pts3d']
            pose2[f'pts3d_{self.depth_name_appendix}'] = depth2['pts3d']
            pose1[f'conf_{self.depth_name_appendix}'] = depth1['conf']  # pts3d in view1's frame
            pose2[f'conf_{self.depth_name_appendix}'] = depth2['conf']  # pts3d in view1's frame
        if self.init_dynamic_mask_estimator:
            pose1['dynamic_mask'] = mask_1['mask_conf']  # pts3d in view1's frame
            pose2['dynamic_mask'] = mask_2['mask_conf']  # pts3d in view1's frame
        if self.init_2d_optic_flow:
            pose1[f'of_itself2other'] = optic_flow1['flow2d']
            pose2[f'of_itself2other'] = optic_flow2['flow2d']
            pose1[f'conf_of'] = optic_flow1['conf'].unsqueeze(-1)  # conf is B 1 H W  
            pose2[f'conf_of'] = optic_flow2['conf'].unsqueeze(-1)  # conf is B 1 H W
        if self.ret_atten_mask or self.dec_use_atten_mask:
            pose1['cross_atten_maps_k'] = self._get_attn_k(torch.cat(cross_attn1), shape1)
            pose2['cross_atten_maps_k'] = self._get_attn_k(torch.cat(cross_attn2), shape2)
        
        return pose1, pose2

    def forward(self, view1, view2, ret_vis=False):
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encoder(view1, view2)  # B,S,D
        # get mask for decoder
        mask1, mask2 = self._get_mask_for_decoder(view1, view2, shape1, shape2)
        # decoding with mask
        (dec1, dec2), (self_attn1, cross_attn1, self_attn2, cross_attn2) = self._decoder(feat1, pos1, feat2, pos2, mask1, mask2)
        # decoding for the secodn time for dpeth
        if self.init_another_dec_for_depth:
            (dec1_2nd, dec2_2nd), _ = self._decoder(
                feat1, pos1, feat2, pos2, mask1, mask2, swap_to_decoder2=True)

        # inference motion mask first!
        mask_1, mask_2 = {}, {}
        if self.init_dynamic_mask_estimator:
            mask_1, mask_2 = self.inference_motion_mask(dec1, dec2, shape1, shape2)
        
        # obatin optic_flow early!
        optic_flow1, optic_flow2 = {'flow2d': None},{'flow2d': None}  # default to None
        if self.init_2d_optic_flow:
            optic_flow1, optic_flow2 = self.inference_optic_flow(dec1, dec2, shape1, shape2)

        # ifnerence scene flow
        scene_flow1, scene_flow2 = {'pts3d': None},{'pts3d': None}  # default to None
        if self.init_3d_scene_flow:
            self.scene_flow_name_appendix = 'scene_flow'
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                if self.shared_scene_flow_estimator:
                    # use the same scene flow estimator for both images
                    scene_flow1 = self._downstream_head('_scene_flow', [tok.float() for tok in dec1], shape1)
                    scene_flow2 = self._downstream_head('_scene_flow', [tok.float() for tok in dec2], shape2)
                else:
                    assert 0, NotImplementedError

        # ifnerence depth
        depth1, depth2 = {'pts3d': None},{'pts3d': None}  # default to None
        if self.init_3d_depth:
            self.depth_name_appendix = 'depth'
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                if self.shared_depth_estimator:
                    # use the same depth estimator for both images
                    if self.init_another_dec_for_depth:
                        # use the second decoder for depth
                        depth1 = self._downstream_head('_depth', [tok.float() for tok in dec1_2nd], shape1)
                        depth2 = self._downstream_head('_depth', [tok.float() for tok in dec2_2nd], shape2)
                    else:
                        depth1 = self._downstream_head('_depth', [tok.float() for tok in dec1], shape1)
                        depth2 = self._downstream_head('_depth', [tok.float() for tok in dec2], shape2)
                else:
                    assert 0, NotImplementedError

        # ifnerence pose
        pose1, pose2 = self.inference_pose(view1, view2, 
                                        mask_1, mask_2, 
                                        dec1, dec2, shape1, shape2, 
                                        optic_flow1, optic_flow2,
                                        depth1, depth2,
                                        scene_flow1, scene_flow2
                                        )
        # wrap up the output
        pose1, pose2 = self._wrap_output( 
                                    pose1, pose2,
                                    scene_flow1, scene_flow2,
                                    depth1, depth2,
                                    optic_flow1, optic_flow2, 
                                    mask_1, mask_2, 
                                    cross_attn1, cross_attn2,
                                    shape1, shape2)
        # visualise the estimated dynamic mask
       #/////////////////
        all_vis_imgs_fused_gt_esti = None
        if self.vis and self.iter_cnt % self.vis_all_freq == 0:
        # if self.vis:
            all_vis_imgs_fused_gt_esti = self.do_vis(pose1, pose2, view1, view2)
            # all_vis_imgs_fused_gt_esti = self.do_vis(pose1, pose2, view1, view2, online_vis = True)
        #/////////////////
        # is_train = pose1['pose'].requires_grad
        # if is_train:
        self.iter_cnt += 1

        # Clear visualization memory if not needed
        if not ret_vis and all_vis_imgs_fused_gt_esti is not None:
            del all_vis_imgs_fused_gt_esti
            all_vis_imgs_fused_gt_esti = None

        if ret_vis:
            return pose1, pose2, all_vis_imgs_fused_gt_esti 
        else:
            return pose1, pose2

    def prepare_pose_reg_input(self, dec1, dec2, 
                               of1 = None, of2 = None, 
                               pts3d_1 = None, pts3d_2 = None, 
                               pts3d_1in2view = None, pts3d_2in1view = None,
                               K_1 = None, K_2 = None):
        if self.pose_regression_head_input == 'default':
            pose_regress_input1 = dec1
            pose_regress_input2 = dec2
        elif self.pose_regression_head_input in ['mapero',
                                                #  'mapero_detach_of',
                                                #  'mapero_detach_pts3d',
                                                #  'mapero_detach_of_detach_pts3d',
                                                 ]:
            if 'detach_of' in self.pose_regression_head_input:
                of1 = of1.detach()
                of2 = of2.detach()
            if 'detach_pts3d' in self.pose_regression_head_input:
                pts3d_1 = pts3d_1.detach() # B H W C
                pts3d_2 = pts3d_2.detach()

            #////////////////////
            assert K_1.dim() == 3 and K_2.dim() == 3, f'K_1 and K_2 should be 3D tensors, got {K_1.shape} and {K_2.shape}'
            # intrinsic1 = torch.eye(3).unsqueeze(0).repeat(1, 1, 1).to(pts3d_1.device)  # Example intrinsic matrix
            # intrinsic2 = torch.eye(3).unsqueeze(0).repeat(1, 1, 1).to(pts3d_1.device)  # Example intrinsic matrix
            if self.mapero_pos_encoding.pixel_pe == 'focal_norm_OF_warped':
                print('todo  to be visual checked')
                # 2D-3D
                of1 = of1.permute(0, 3, 1, 2)  # (B, 2, H, W) B 2 384 512
                of2 = of2.permute(0, 3, 1, 2)  # (B, 2, H, W)
                pts3d_1 = pts3d_1.permute(0, 3, 1, 2)  # (B, 3, H, W) B 3 384 512
                pts3d_2 = pts3d_2.permute(0, 3, 1, 2)  # (B, 3, H, W)
                assert of1 is not None and of2 is not None, f'optic flow should be provided when pose_regression_head_input is mapero'
                assert pts3d_1 is not None and pts3d_2 is not None, f'optic flow should be provided when pose_regression_head_input is mapero'
                pts3d_1_mapero_PE, pts3d_1_pixel_pe = self.mapero_pos_encoding(pts3d_2, K_1, of2)# B 32 H W #3d is from view2
                pts3d_2_mapero_PE, pts3d_2_pixel_pe = self.mapero_pos_encoding(pts3d_1, K_2, of1)#
            # elif self.mapero_pos_encoding.pixel_pe == 'focal_norm':
            elif self.mapero_pos_encoding.pixel_pe in ['focal_norm', '2d_bearing_vector']:
                # 2D-3D
                assert pts3d_1in2view is not None and pts3d_2in1view is not None, f'pts3d should be provided when pose_regression_head_input is mapero'
                pts3d_1in2view = pts3d_1in2view.permute(0, 3, 1, 2)  # (B, 3, H, W) B 3 384 512
                pts3d_2in1view = pts3d_2in1view.permute(0, 3, 1, 2)  # (B, 3, H, W)
                pts3d_1_mapero_PE, pts3d_1_pixel_pe = self.mapero_pos_encoding(pts3d_1in2view, K_1)# B 32 H W #3d is from view2
                pts3d_2_mapero_PE, pts3d_2_pixel_pe = self.mapero_pos_encoding(pts3d_2in1view, K_2)# 3d is from view2
            else:
                assert 0, f'Unknown mapero_pos_encoding {self.mapero_pos_encoding.pixel_pe}, should be one of [focal_norm_OF_warped, focal_norm_OF_warped]'

            #vis pe for debug
            if self.output_dir is not None and self.do_vis_pe and self.iter_cnt % self.vis_pe_freq == 0:
                # visualize the pts3d_1_mapero_PE and pts3d_2_mapero_PE
                # Use exp_id for default folder if not provided
                if self.exp_id is None:
                    save_dir = os.path.join(self.output_dir,'other_PE_mapero')
                else:
                    save_dir = os.path.join(self.output_dir,self.exp_id, 'PE_mapero')
                os.makedirs(save_dir, exist_ok=True)
                self.vis_pe(pts3d_1_pixel_pe,pts3d_2_pixel_pe, folder = save_dir)

            if self.marepo_scenecoord_img_ratio!=1:
                # per_point_patch
                # use the skip resolution
                pass
            def reshape_based_on_patch_size(pts3d_1_mapero_PE, patch_size):
                pts3d_1_mapero_PE_resized = pts3d_1_mapero_PE.view(pts3d_1_mapero_PE.shape[0], pts3d_1_mapero_PE.shape[1],
                                                                    pts3d_1_mapero_PE.shape[2] // patch_size,
                                                                    patch_size,
                                                                    pts3d_1_mapero_PE.shape[3] // patch_size,
                                                                    patch_size)
                pts3d_1_mapero_PE_resized = pts3d_1_mapero_PE_resized.permute(0, 2, 4, 3, 5, 1)  # (B, H//p, W//p, p, p, C)
                pts3d_1_mapero_PE_resized = pts3d_1_mapero_PE_resized.reshape(pts3d_1_mapero_PE.shape[0],
                                                                                (pts3d_1_mapero_PE.shape[2] // patch_size) * (pts3d_1_mapero_PE.shape[3] // patch_size),
                                                                                -1)
                return pts3d_1_mapero_PE_resized

            pts3d_1_mapero_PE_resized = reshape_based_on_patch_size(pts3d_1_mapero_PE, self.marepo_pose_head_ipt_patch_size,)#self.patch_size)
            pts3d_2_mapero_PE_resized = reshape_based_on_patch_size(pts3d_2_mapero_PE, self.marepo_pose_head_ipt_patch_size,)#self.patch_size)
            print('marepo_pose_head_ipt_patch_size', self.marepo_pose_head_ipt_patch_size, pts3d_1_mapero_PE_resized.shape, pts3d_2_mapero_PE_resized.shape)
            # pts3d_1_mapero_PE_resized = reshape_based_on_patch_size(pts3d_1_mapero_PE, self.patch_size)
            # pts3d_2_mapero_PE_resized = reshape_based_on_patch_size(pts3d_2_mapero_PE, self.patch_size)
 
            # tok1: B h_w C (h_w: 768    C:1024->768)
            # of1: B 2 H W
            pose_regress_input1 = [torch.cat([tok1, of1_resized], dim=-1) for tok1 in dec1] \
                if 'cat_with' in self.pose_regression_head_input else [pts3d_1_mapero_PE_resized for _ in dec1]
            pose_regress_input2 = [torch.cat([tok2, of2_resized], dim=-1) for tok2 in dec2] \
                if 'cat_with' in self.pose_regression_head_input else [pts3d_2_mapero_PE_resized for _ in dec2]

        elif self.pose_regression_head_input in ['optic_flow','optic_flow_detach',
                                                'cat_with_optic_flow', 'cat_with_optic_flow_detach']:
            assert of1 is not None and of2 is not None, f'optic flow should be provided when pose_regression_head_input is optic_flow'
            # resize of1 as the token dim, then concat
            print('todo need to leverage various resolution flow')
            if self.pose_regression_head_input in ['optic_flow_detach', 'cat_with_optic_flow_detach']:
                of1 = of1.detach()
                of2 = of2.detach()
            of1 = of1.permute(0, 3, 1, 2)  # (B, 2, H, W)
            of2 = of2.permute(0, 3, 1, 2)  # (B, 2, H, W)

            # # Convert flow from HxWx2 to BxCxHxW format for interpolation
            # if of1.dim() == 3:  # (H, W, 2)
            #     of1 = of1.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
            # elif of1.dim() == 4 and of1.shape[-1] == 2:  # (B, H, W, 2)
            #     of1 = of1.permute(0, 3, 1, 2)  # (B, 2, H, W)
            # if of2.dim() == 3:  # (H, W, 2)
            #     of2 = of2.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
            # elif of2.dim() == 4 and of2.shape[-1] == 2:  # (B, H, W, 2)
            #     of2 = of2.permute(0, 3, 1, 2)  # (B, 2, H, W)

            B,_, H, W = of1.shape  # B: batch size, _ : 2 for flow channels, H: height, W: width
            # Interpolate to target size
            # target_size = (int(H/self.patch_size), int(W/self.patch_size))
            
            # Resize the optical flow to match the target size: use view rahter interploation
            of1_resized = of1.view(B, 2, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
            of1_resized = of1_resized.permute(0, 2, 4, 3, 5, 1)  # (B, 2, H//p, W//p, p, p)
            of1_resized = of1_resized.reshape(B, (H//self.patch_size) * (W//self.patch_size), -1)  # [B, 2, H//patch_size, W//patch_size]

            of2_resized = of2.view(B, 2, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
            of2_resized = of2_resized.permute(0, 2, 4, 3, 5, 1)  # (B, 2, H//p, W//p, p, p)
            of2_resized = of2_resized.reshape(B, (H//self.patch_size) * (W//self.patch_size), -1)  # [B, 2, H//patch_size, W//patch_size]

            # of1_resized = of1.view(B, 2, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size).mean(dim=(3,5))  # [B, 2, H//patch_size, W//patch_size]
            # of2_resized = of2.view(B, 2, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size).mean(dim=(3,5))  # [B, 2, H//patch_size, W//patch_size]
            # of1_resized = F.interpolate(of1, size=target_size, mode='bilinear', align_corners=False)
            # of2_resized = F.interpolate(of2, size=target_size, mode='bilinear', align_corners=False)
            # of1_resized = of1_resized.permute(0, 2, 3, 1).view(B,-1, 2)  # [B, H, W, 2]
            # of2_resized = of2_resized.permute(0, 2, 3, 1).view(B,-1, 2)  # [B, H, W, 2]

            # tok1: B h_w C (h_w: 768    C:1024->768)
            # of1: B 2 H W
            pose_regress_input1 = [torch.cat([tok1, of1_resized], dim=-1) for tok1 in dec1] \
                if 'cat_with' in self.pose_regression_head_input else [of1_resized for _ in dec1]
            pose_regress_input2 = [torch.cat([tok2, of2_resized], dim=-1) for tok2 in dec2] \
                if 'cat_with' in self.pose_regression_head_input else [of2_resized for _ in dec2]
        elif self.pose_regression_head_input == 'corr':
            b,s,d = dec1[-1].shape
            '''
            i = i.view(b,d,self.img_size/self.patch_size)
            global_correlation_softmax takes input with shape:b c h w
            opt with shape b 2 h w
            B, S, D = dec1[-1].shape #1 768 768
            '''
            assert self.landscape_only, f'the view operation below only for landscape'
            # only take the flow esti, and b 2 h w => b h_w 2
            pose_regress_input1 = [ global_correlation_softmax(tok1.permute(0,2,1).contiguous().view(b,d,-1,int(self.img_size/self.patch_size)), 
                                                               tok2.permute(0,2,1).contiguous().view(b,d,-1,int(self.img_size/self.patch_size)))[0].view(b,2,-1).permute(0,2,1) for tok1,tok2 in zip(dec1,dec2)]
            pose_regress_input2 = [ global_correlation_softmax(tok2.permute(0,2,1).contiguous().view(b,d,-1,int(self.img_size/self.patch_size)), 
                                                               tok1.permute(0,2,1).contiguous().view(b,d,-1,int(self.img_size/self.patch_size)))[0].view(b,2,-1).permute(0,2,1) for tok1,tok2 in zip(dec1,dec2)]
        elif self.pose_regression_head_input == 'cat_feats':
            pose_regress_input1 = [ torch.cat([tok1, tok2], dim=-1) for tok1,tok2 in zip(dec1,dec2)] 
            pose_regress_input2 = [ torch.cat([tok2, tok1], dim=-1) for tok1,tok2 in zip(dec1,dec2)] 
        elif self.pose_regression_head_input == 'add_feats':
            pose_regress_input1 = [ tok1.float() + tok2.float() for tok1,tok2 in zip(dec1,dec2)]
            pose_regress_input2 = [ tok2.float() + tok1.float() for tok1,tok2 in zip(dec1,dec2)]
        else:
            assert 0, f'Unknown pose_regression_head_input {self.pose_regression_head_input}, should be one of [default, corr, cat_feats, add_feats]'
        return pose_regress_input1, pose_regress_input2

    def do_vis(self, pose1, pose2, view1, view2, online_vis=False):
        # dict_keys(['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'z_far', 'pts3d', 'valid_mask', 'dynamic_mask', 'rng'])
        all_vis_imgs_esti = []
        all_vis_imgs_gt = []
        self.vis_placeholder_shape = view1['img'].shape #(view1['img'].shape[0], view1['img'].shape[1], 3) if 'img' in view1 else None
        vis_img = self._get_pair_vis(view1, view2, vis_which=['img'], vis_title='gt')
        all_vis_imgs_gt.append(vis_img)
        # vis_img = self._get_pair_vis(view1, view2, vis_which=['depthmap'], vis_title='gt')
        # all_vis_imgs_gt.append(vis_img)
        vis_img = self._get_pair_vis(pose1, pose2, vis_which=['pts3d_depth'], vis_title='esti')
        all_vis_imgs_esti.append(vis_img)
        vis_img = self._get_pair_vis(view1, view2, vis_which=['dynamic_mask'], vis_title='gt')
        all_vis_imgs_gt.append(vis_img)
        vis_img = self._get_pair_vis(pose1, pose2, vis_which=['conf_scene_flow'], vis_title='esti')
        all_vis_imgs_esti.append(vis_img)
        vis_img = self._get_pair_vis(view1, view2, vis_which=['dynamic_mask'], vis_title='gt')
        all_vis_imgs_gt.append(vis_img)
        vis_img = self._get_pair_vis(pose1, pose2, vis_which=['dynamic_mask'], vis_title='esti')
        all_vis_imgs_esti.append(vis_img)
        vis_img = self._get_pair_vis(view1, view2, vis_which=['gt_flow'], vis_title='of_gt')
        all_vis_imgs_gt.append(vis_img)
        vis_img = self._get_pair_vis(pose1, pose2, vis_which=['of_itself2other'], vis_title='of_esti')
        all_vis_imgs_esti.append(vis_img)
        vis_img = self._get_pair_vis(view1, view2, vis_which=['pts3d'], vis_title='gt')
        all_vis_imgs_gt.append(vis_img)
        # vis_img = self._get_pair_vis(pose2, pose1, vis_which=[f'pts3d_in_other_view_{self.scene_flow_name_appendix}'], vis_title=f'esti_{self.scene_flow_name_appendix}')
        # vis_img = self._get_pair_vis(pose2, pose1, vis_which=[f'pts3d_in_other_view_scene_flow'], 
                                    #  vis_title=f'esti_scene_flow')
        vis_img = self._get_pair_vis(pose1, pose2, vis_which=[f'pts3d_in_other_view_scene_flow'], 
                                     vis_title=f'esti_scene_flow')
        all_vis_imgs_esti.append(vis_img)

        all_vis_imgs_fused_gt = np.vstack(all_vis_imgs_gt)
        all_vis_imgs_fused_esti = np.vstack(all_vis_imgs_esti)
        # resize all_vis_imgs_fused
        all_vis_imgs_fused_gt = cv2.resize(all_vis_imgs_fused_gt, (0, 0), fx=0.5, fy=0.5)  # resize to half size
        all_vis_imgs_fused_esti = cv2.resize(all_vis_imgs_fused_esti, (0, 0), fx=0.5, fy=0.5)  # resize to half size
        # cv2.imshow(f'esti', all_vis_imgs_fused_esti)
        # cv2.imshow('gt', all_vis_imgs_fused_gt)
        # show the fused image
        all_vis_imgs_fused_gt_esti = np.hstack((all_vis_imgs_fused_gt, all_vis_imgs_fused_esti))
        if online_vis:
            cv2.imshow('fused', all_vis_imgs_fused_gt_esti)
            cv2.waitKey(1)
        #save the fused image base on self.output_dir
        # if self.output_dir is not None and self.iter_cnt % self.vis_all_freq == 0:
        if self.output_dir is not None:
            is_train = pose1['pose'].requires_grad
            save_dir_name = 'ALL_vis_train' if is_train else 'ALL_vis_test'
            output_root = os.path.join(self.output_dir, self.exp_id, save_dir_name)
            os.makedirs(output_root, exist_ok=True)
            #///////
            # record esti_conf max and min if exists
            try:
                # esti_conf = pose1['conf_scene_flow']
                esti_conf = pose1['dynamic_mask']
                esti_conf_max = esti_conf.max().item()
                esti_conf_min = esti_conf.min().item()
                esti_conf_min = f'{esti_conf_min:.2f}'
                esti_conf_max = f'{esti_conf_max:.2f}'
            except:
                esti_conf_max = ''
                esti_conf_min = ''
            #///////
            # output_path = os.path.join(output_root, f'vis_it{self.iter_cnt}.png')
            output_path = os.path.join(output_root, f'vis_it{self.iter_cnt}_{esti_conf_min}_{esti_conf_max}.png')
            cv2.imwrite(output_path, all_vis_imgs_fused_gt_esti*255)
            print('All img saved in...', output_path)

        return all_vis_imgs_fused_gt_esti

    def vis_pe(self, pe, pe2=None, folder=None):
        """
        For each call, save every channel of each PE map as a separate image.
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        def prepare_pe(pe):
            if pe is None:
                return None
            if torch.is_tensor(pe):
                pe = pe.cpu().detach().numpy()
            if pe.ndim == 4:
                pe = pe[0]  # BxCxHxW -> CxHxW
            return pe

        pe = prepare_pe(pe)
        pe2 = prepare_pe(pe2)



        os.makedirs(folder, exist_ok=True)

        pes = [pe]
        titles = ['PE']
        if pe2 is not None:
            pes.append(pe2)
            titles.append('PE2')

        for i, pe_map in enumerate(pes):
            if pe_map is None:
                continue
            for ch in range(pe_map.shape[0]):
                fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
                im = ax.imshow(pe_map[ch, :, :])
                ax.set_title(f'{titles[i]} (ch {ch})')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(f'{folder}/pe_it{self.iter_cnt}_map{i}_ch{ch}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
        print(f"Visualized all PE channels to {folder}/pe_it{self.iter_cnt}_map*_ch*.png")

    def _get_pair_vis(self, view1, view2, vis_which=['dynamic_mask'], hstack=True, 
                        # vis = True, 
                        vis = False, 
                        vis_title=''):
        vis_img = None
        for which in vis_which:
            if which not in view1 :
                assert which not in view2, f'view1 and view2 should have are both from gt or esti to be consistent'
                # print('', f'Warning: {which} not in view, set as black')
                #use self.img_size to create a black image
                vis_img1 = torch.zeros(self.vis_placeholder_shape).detach().cpu().numpy()
                vis_img2 = torch.zeros(self.vis_placeholder_shape).detach().cpu().numpy()
            else:
                # print(f'exist and plot: {vis_title}:', which)
                if which == 'pts3d':
                    # need to project the pts3d based on cam---the raw pts3d is w.r.t world
                    assert 'gt' in vis_title, f'vis_title should contain gt for pts3d, but got {vis_title}'
                    in_camera1 = inv(view1['camera_pose'])# ->world to 1
                    in_camera2 = inv(view2['camera_pose'])# ->world to 2
                    gt_pts2_wrtC1 = geotrf(in_camera1, view1['pts3d'])
                    gt_pts1_wrtC2 = geotrf(in_camera2, view2['pts3d'])
                    vis_img1 = gt_pts2_wrtC1.detach().cpu().numpy()  # Bx3xHxW
                    vis_img2 = gt_pts1_wrtC2.detach().cpu().numpy()  # Bx3xHxW
                else:
                    vis_img1 = view1[which].detach().cpu().numpy()
                    vis_img2 = view2[which].detach().cpu().numpy()
                    if which == 'pts3d_depth':
                        # only vis the Z value
                        vis_img1 = vis_img1[...,-1]
                        vis_img2 = vis_img2[...,-1]
                    vis_img1 = vis_img1[:,None,...] if vis_img1.ndim == 3 else vis_img1  # BxHxW -> BXCxHxW
                    vis_img2 = vis_img2[:,None,...] if vis_img2.ndim == 3 else vis_img2  # BxHxW -> BXCxHxW

            vis_img1 = vis_img1[0] if vis_img1.ndim == 4 else vis_img1
            vis_img2 = vis_img2[0] if vis_img2.ndim == 4 else vis_img2
            def process_vis_img(vis_img1, is_pts3d_conf = False):
                if vis_img1.ndim == 3:
                    if vis_img1.shape[0] == 1:
                        #repeat as 3 dim
                        vis_img1 = np.repeat(vis_img1, 3, axis=0)
                        # norm to range (0,1)
                        if is_pts3d_conf:
                            print('raw: 1D OPT max and min and sum:', vis_img1.max(), vis_img1.min(), vis_img1.sum())
                            # vis_img1 = vis_img1 - 1 #conf is 0-1
                            # vis_img1 = np.log(vis_img1)
                            # print('raw after log: 1D OPT max and min and sum:', vis_img1.max(), vis_img1.min(), vis_img1.sum())
                            # vis_img1 = (vis_img1 - vis_img1.min()) / (vis_img1.max() - vis_img1.min() + 1e-6)
                            # vis_img1 = vis_img1#already in range (0,1)
                            vis_img1 = (vis_img1 - vis_img1.min()) / (vis_img1.max() - vis_img1.min() + 1e-6)

                        else:
                            vis_img1 = (vis_img1 - vis_img1.min()) / (vis_img1.max() - vis_img1.min() + 1e-6)
                        # print('after: 1D OPT max and min and sum:', vis_img1.max(), vis_img1.min(), vis_img1.sum())
                    else:
                        if vis_img1.shape[-1] == 3:
                            # it is pts3d
                            vis_img1 = np.transpose(vis_img1, (2, 0, 1))  # to CxHxW
                            # Convert 3D points to color visualization
                            vis_img1 = flow3DToColor(vis_img1.transpose(1, 2, 0))  # Convert to HxWx3 for flow3DToColor
                            vis_img1 = np.transpose(vis_img1, (2, 0, 1))  # Convert back to CxHxW
                            vis_img1 = vis_img1 / 255.0  # Normalize to [0,1] range
                        elif vis_img1.shape[0] == 2  or vis_img1.shape[-1] == 2:
                            # of it is gt_optical flow
                            if  vis_img1.shape[-1] == 2:
                                vis_img1 = np.transpose(vis_img1, (2, 0, 1))
                            # Convert flow to color visualization
                            vis_img1 = flowToColor(vis_img1.transpose(1, 2, 0))  # Convert to HxWx2 for flowToColor
                            vis_img1 = np.transpose(vis_img1, (2, 0, 1))  # Convert back to CxHxW
                            vis_img1 = vis_img1 / 255.0  # Normalize to [0,1] range
                        elif vis_img1.shape[0] == 3:
                            # it is RGB img
                            vis_img1 = (vis_img1 + 1) / 2 #torch image is range (-1,1)
                        else:
                            assert 0, f'Unknown vis_img1 shape {vis_img1.shape}, should be 3 or 1 channel'
                return vis_img1
            
            # is_pts3d_conf = which in ['conf_scene_flow'] # depth and conf are plot differently
            is_pts3d_conf = 'conf' in which or 'mask' in which  # depth and conf are plot differently
            is_pts3d_conf = 'dynamic_mask' in which  # depth and conf are plot differently
            vis_img1 = process_vis_img(vis_img1, is_pts3d_conf)
            vis_img2 = process_vis_img(vis_img2, is_pts3d_conf)

            if hstack:
                if len(vis_img1.shape) == 3 and vis_img1.shape[0] == 3:  # RGB image
                    vis_img = np.hstack((vis_img1.transpose(1, 2, 0), vis_img2.transpose(1, 2, 0)))
                else:  # Single channel image
                    vis_img = np.hstack((vis_img1[0, :, :], vis_img2[0, :, :]))
            else:
                if len(vis_img1.shape) == 3 and vis_img1.shape[0] == 3:  # RGB image
                    vis_img = np.vstack((vis_img1.transpose(1, 2, 0), vis_img2.transpose(1, 2, 0)))
                else:  # Single channel image
                    vis_img = np.vstack((vis_img1[0, :, :], vis_img2[0, :, :]))

            if vis:
                cv2.imshow(f'{vis_title}_View1view2 {"_".join(vis_which)}', vis_img)
                cv2.waitKey(1)

        return vis_img   
            


def load_UniReloc3r_model(ckpt_path, img_size, device, opt, output_dir = None):
    model = Reloc3rRelpose(init_dynamic_mask_estimator=getattr(opt, 'init_dynamic_mask_estimator', False),
                        shared_dynamic_mask_estimator=getattr(opt, 'shared_dynamic_mask_estimator', False),
                        dynamic_mask_estimator_type=getattr(opt, 'dynamic_mask_estimator_type', False),
                        #////////
                        init_3d_scene_flow=getattr(opt, 'init_3d_scene_flow', False),
                        scene_flow_estimator_type=getattr(opt, 'scene_flow_estimator_type', 'linear'),
                        init_3d_depth=getattr(opt, 'init_3d_depth', False),
                        init_2d_optic_flow=getattr(opt, 'init_2d_optic_flow', False),
                        init_another_dec_for_depth=getattr(opt, 'init_another_dec_for_depth', False),
                        pose_head_seperate_scale=getattr(opt, 'pose_head_seperate_scale', False),
                        #/////////
                        unireloc3r_pose_estimation_mode=getattr(opt, 'unireloc3r_pose_estimation_mode', 'vanilla_pose_head_regression'),
                        pose_regression_with_mask=getattr(opt, 'pose_regression_with_mask', False),
                        pose_regression_which_mask=getattr(opt, 'pose_regression_which_mask', 'gt'),
                        pose_regression_head_input=getattr(opt, 'pose_regression_head_input', 'default'),
                        mapero_pixel_pe_scheme=getattr(opt, 'mapero_pixel_pe_scheme', 'focal_norm'),
                        #/////////
                        img_size=img_size,
                        output_dir=output_dir,
                        exp_id='tmp_id',
                        )  # pass required args if any
    # model = Reloc3rRelpose(img_size=img_size)
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # /////////
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt  # Assume the checkpoint is directly the state dict
    
    print('model scene_flow_estimator_type:', model.scene_flow_estimator_type)
    # report keys not exist in the state_dict or not matching
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # reloc3r_relpose.load_state_dict(state_dict)  # or adjust key if needed
    if missing_keys:
        missing_keys_sim = {k.split('.')[0].split('[')[0] for k in missing_keys}
        print(f'Warning: Missing keys in the checkpoint: {missing_keys_sim}')
    if unexpected_keys:
        unexpected_keys_sim = {k.split('.')[0].split('[')[0] for k in unexpected_keys}
        print(f'Warning: Unexpected keys in the checkpoint: {unexpected_keys_sim}')
    #/////////
    
    print('Model loaded from ', ckpt_path)
    del ckpt  # in case it occupies memory.
    model.eval()
    return model

 
if __name__ == '__main__':
    model = Reloc3rRelpose()
    model.eval()
    model.to('cuda')
    # model.forward(view1, view2)




