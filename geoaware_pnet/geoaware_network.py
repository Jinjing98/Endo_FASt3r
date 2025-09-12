
# the code is adapted from https://github.com/nianticlabs/marepo/blob/main/marepo/marepo_network.py
# Copyright © Niantic, Inc. 2024.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoaware_pnet.transformer.transformer import Transformer_Head

_logger = logging.getLogger(__name__)


# from geoaware_pnet.geoaware_network import PoseRegressor
import json
import os


def load_geoaware_pose_head(geoaware_cfg_path, load_geoaware_pretrain_model, px_resample_rule_dict_scale_step):
    assert os.path.exists(geoaware_cfg_path), f"geoaware_cfg_path {geoaware_cfg_path} does not exist"
    f = open(geoaware_cfg_path)
    config = json.load(f)
    f.close()
    mean_cam_center = torch.tensor([0.0, 0.0, 0.0])
    default_img_H = config["default_img_H"]#480
    default_img_W = config["default_img_W"]#640
    # default_img_H = 256
    # default_img_W = 320    

    # extend: not included in the json
    # transformer_pose_mean will be applied internnally in pose_regression_head
    config["transformer_pose_mean"] = mean_cam_center # for us, we set to zero as we predict only the relative pose.
    config["default_img_HW"] = [default_img_H, default_img_W]
    config["px_resample_rule_dict_scale_step"] = px_resample_rule_dict_scale_step


    if load_geoaware_pretrain_model:

        pose_model = PoseRegressor(config)

        transformer_root = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/geoaware_pnet/trained_ckpt2/paper_model'
        if config["rotation_representation"] == "9D":
            transformer_path = os.path.join(transformer_root, 
                                            "marepo_9D/marepo_9D.pt",
                                            )
        else:
            transformer_path = os.path.join(transformer_root, 
                                            "marepo/marepo.pt",
                                            )
        assert os.path.exists(transformer_path), f"transformer_path {transformer_path} does not exist"
        print('loading pretrained GeoAwarePNet pose regressor from with default img HW {}'.format(config["default_img_HW"]), transformer_path)
        pose_model.load_pose_regressor_from_state_dict(transformer_path)
    else:
        pose_model = PoseRegressor(config)
        print('init GeoAwarePNet pose regressor from scratch with default img HW {}'.format(config["default_img_HW"]))
    
    return pose_model


class Encoder(nn.Module):
    """
    mapero HW default is 480,640

    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x

class Head(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=512):
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

    def forward(self, res):
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean

        return sc

class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=512, config={}, transformer_head_only=False):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.transformer_head_only = transformer_head_only
        self.feature_dim = num_encoder_features
        self.encoder = Encoder(out_channels=self.feature_dim)
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)
        self.config=config
        self.transformer_head = Transformer_Head(config)

    
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous, num_encoder_features):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        # num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    
    def create_from_state_dict(cls, state_dict, config):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # # Whether the network uses homogeneous coordinates.
        # use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4
        use_homogeneous = config["use_homogeneous"]

        # # Number of output channels of the last encoder layer.
        # num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]
        num_encoder_features = config["num_encoder_features"]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features, config)

        # Load all weights.
        regressor.load_state_dict(state_dict, strict=False)

        # Done.
        return regressor

    
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict, config):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        return cls.create_from_state_dict(merged_state_dict, config)

    
    def load_marepo_from_state_dict(cls, encoder_state_dict, head_state_dict, transformer_state_dict, config):
        """
        Load Marepo networks including weights from encoder, head, and transformer head
        """
        network = cls.create_from_split_state_dict(encoder_state_dict, head_state_dict, config)
        network.transformer_head.load_state_dict(transformer_state_dict['aceformer_head'], strict=True)
        return network

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def load_head(self, head_dict_file):
        """
        Load weights into the heads network.
        """
        self.heads.load_state_dict(torch.load(head_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    # def get_pose(self, sc, intrinsics_B33=None, sc_mask=None, random_rescale_sc=False):
    #     return self.transformer_head(sc, intrinsics_B33, sc_mask, random_rescale_sc)
    

    def forward(self, sc, intrinsics_B33=None, sc_mask=None, random_rescale_sc=False):
        # exact the same as
        return self.transformer_head(sc, intrinsics_B33, sc_mask, random_rescale_sc)
    


class PoseRegressor(nn.Module):
    def __init__(self, config):
        super(PoseRegressor, self).__init__()
        self.config = config
        self.transformer_head = Transformer_Head(config)
    
    def forward(self, sc, intrinsics_B33=None, sc_mask=None, random_rescale_sc=False, sample_level = 3):
        return self.transformer_head(sc, intrinsics_B33, sc_mask, random_rescale_sc, sample_level)


    
    def load_pose_regressor_from_state_dict(self, transformer_path):
        """
        Load Marepo networks including weights from encoder, head, and transformer head
        """
        transformer_state_dict = torch.load(transformer_path, map_location="cpu")
        _logger.info(f"Loaded transformer weights from: {transformer_path}")

        # network = cls.create_from_split_state_dict(encoder_state_dict, head_state_dict, config)
        self.transformer_head.load_state_dict(transformer_state_dict['aceformer_head'], strict=True)
        print('loaded pretrained TR pose regressor from marepo...')

if __name__ == "__main__":
    import json
    transformer_json = "/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/geoaware_pnet/transformer/config/nerf_focal_12T1R_256_homo_c2f_geoaware.json"
    # some configuration for the transformer
    f = open(transformer_json)
    config = json.load(f)
    f.close()
    mean_cam_center = torch.tensor([0.0, 0.0, 0.0])
    default_img_H = config["default_img_H"]#480
    default_img_W = config["default_img_W"]#640

    # default_img_H = 256
    # default_img_W = 320    

    # extend: not included in the json
    # transformer_pose_mean will be applied internnally in pose_regression_head
    config["transformer_pose_mean"] = mean_cam_center # for us, we set to zero as we predict only the relative pose.
    config["default_img_HW"] = [default_img_H, default_img_W]

    # test original pose_regressor 
    # model = Regressor.create_from_split_state_dict(encoder_state_dict = {}, 
    #                                                    head_state_dict = {}, 
    #                                                    config = config)

    # from torchsummary import summary

    # # test the regressor model with simulated input data
    # input_data = torch.randn(1, 1, default_img_H, default_img_W)
    # intrinsics_B33 = torch.randn(1, 3, 3)
    # sc_mask = torch.randn(1, 1, int(default_img_H/8), int(default_img_W/8)) # use during training, else set to None
    # random_rescale_sc = True # on during traning


    # features = model.get_features(input_data)
    # sc = model.get_scene_coordinates(features)
    # print('sc.shape',sc.shape)
    # print('intrinsics_B33.shape',intrinsics_B33.shape)

    # # test pose regressor only
    # pose = model(sc.repeat(2, 1, 1, 1), intrinsics_B33.repeat(2, 1, 1),
    #                     sc_mask=sc_mask.repeat(2, 1, 1, 1), random_rescale_sc=random_rescale_sc)
    # print('pose. len/shape',len(pose), pose[0].shape)
    
 
    #test pose regressor only
    model = PoseRegressor(config)
    sc = torch.randn(1, 3, int(default_img_H/8), int(default_img_W/8))
    # sc = torch.randn(1, 3, 64, 80)
    intrinsics_B33 = torch.randn(1, 3, 3)
    print('sc.shape',sc.shape)
    print('default HW',default_img_H, default_img_W)
    print('intrinsics_B33.shape',intrinsics_B33.shape)
    sc_mask = torch.randn(1, 1, int(default_img_H/8), int(default_img_W/8)) # use during training, else set to None
    random_rescale_sc = True # on during traning
    pose = model(sc.repeat(2, 1, 1, 1), intrinsics_B33.repeat(2, 1, 1),
                        sc_mask=sc_mask.repeat(2, 1, 1, 1), random_rescale_sc=random_rescale_sc)
    print('pose. len/shape',len(pose), pose[0].shape)
    