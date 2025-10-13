from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
from networks import Customised_DAM
import numpy as np
import torch.optim as optim
import torch.nn as nn

from utils import *
from layers import *
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from metrics import compute_pose_error_v2

# from networks import DINOEncoder
import random
# import ipdb
# from networks import RelocerX
# from torch.cuda.amp import autocast, GradScaler
import PIL
import PIL.Image
from torchvision.transforms import ToPILImage
import torchvision.transforms as tvf
from PIL import Image
from utils import color_to_cv_img, gray_to_cv_img, flow_to_cv_img
from networks.utils.endofas3r_data_utils import prepare_images, resize_pil_image



AF_PRETRAINED_ROOT = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/DARES/af_sfmlearner_weights"
RELOC3R_PRETRAINED_ROOT = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512"


def clamp_pose(pose, min_value=-1.0, max_value=1.0):
    return torch.clamp(pose, min=min_value, max=max_value)



def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed for a single GPU
    torch.cuda.manual_seed_all(seed)  # Seed for all GPUs (if you are using multi-GPU)
    
    # Set deterministic option
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, options):



        # update options for debug purposes
        if options.debug:
            print("DEBUG MODE")
            print('update options for debug purposes...')
            options.num_epochs = 50000
            options.batch_size = 1
            # options.accumulate_steps = 4  # Effective batch size = 1 * 12 = 12
            options.log_frequency = 10
            options.save_frequency = 100000# no save
            # options.log_dir = "/mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm_debug"


            # options.pose_model_type = "geoaware_pnet"
            # options.pose_model_type = "posetr_net"
            # options.pose_model_type = "separate_resnet"
            # options.model_name = "debug_tr_posenet"


            # options.shared_MF_OF_network = True

            options.enable_motion_computation = True
            # options.use_MF_network = True
            # options.shared_MF_OF_network = True
            # options.enable_mutual_motion = True
            # options.reg_mutual_raw_disp_based_OF_for_consistency_and_correctness = True
            
            # options.use_loss_reproj2_nomotion = True
            # # options.use_soft_motion_mask = True

            # # options.disable_pose_head_overwrite = True
            # # options.backbone_pretrain_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/crocoflow.pth"
            # # options.backbone_pretrain_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            # options.backbone_pretrain_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            
            # # options.pose_model_type = "uni_reloc3r"
            # # options.init_3d_scene_flow = True


            # # options.scene_flow_estimator_type = "linear"
            # # options.init_2d_optic_flow = True
            # # options.optic_flow_estimator_type = "dpt"


            # # options.unireloc3r_pose_estimation_mode = "epropnp"
            # # options.pose_regression_with_mask = True
            # # options.pose_regression_which_mask = "esti"

            # options.enable_grad_flow_motion_mask = True
            # options.use_loss_motion_mask_reg = True



            # options.zero_pose_debug = True

            # options.freeze_depth_debug = True

            # options.ignore_motion_area_at_calib = True

            # options.use_raft_flow = True
            # options.use_raft_flow = False

            # # options.zero_pose_flow_debug = True
            # # options.reproj_supervised_with_which = "raw_tgt_gt"
            # # options.reproj_supervised_which = "color_MotionCorrected"

            # # options.flow_reproj_supervised_with_which = "raw_tgt_gt"

            # # options.transform_constraint = 0.0
            # # options.transform_smoothness = 0.0
            # # options.disparity_smoothness = 0.0

            # options.freeze_as_much_debug = True #save mem # need to be on for OF exp

            options.of_samples = True
            # options.of_samples_num = 100
            # options.of_samples_num = 8
            # options.of_samples_num = 2
            # options.of_samples_num = 1
            # options.is_train = True
            # options.is_train = False # no augmentation

            # # big step might lead to inf?
            # options.frame_ids = [0, -1, 1]
            # options.frame_ids = [0, -3, 3]
            # options.frame_ids = [0, -14, 14]

            # # # # not okay to use: we did not adjust the init_K accordingly yet
            # DYNASCARED IS FINE?
            # options.height = 192
            # options.width = 224


            # # raft can use this?
            # options.height = 192
            # options.width = 192

            options.dataset = "StereoMIS"
            options.data_path = "/mnt/nct-zfs/TCO-All/SharedDatasets/StereoMIS_DARES_test/"
            options.split_appendix = ""

            # options.dataset = "endovis"
            # options.data_path = "/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/"
            # options.split_appendix = ""

            # options.dataset = "DynaSCARED"
            # options.data_path = "/mnt/cluster/datasets/Surg_oclr_stereo/"
            # options.split_appendix = "_CaToTi000"


            # options.split_appendix = "_CaToTi001"
            # options.split_appendix = "_CaToTi010"
            # options.split_appendix = "_CaToTi110"
            # options.split_appendix = "_CaToTi101"
            # options.split_appendix = "_CaToTi011"

            # options.split_appendix = "_CaToTi000" #critical
            # options.split_appendix = "_CaToTi001" #critical reason for nan raft flow


            # #debug nan present in geoaware with static scene traning
            # # options.model_name = "debug_geoaware_in_unireloc3r"
            # options.pose_model_type = "uni_reloc3r"
            # options.init_3d_scene_flow = True
            # options.scene_flow_estimator_type = "dpt"
            # options.backbone_pretrain_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            # options.backbone_pretrain_ckpt_path = '/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
            # # options.use_soft_motion_mask = True
            # options.unireloc3r_pose_estimation_mode = "epropnp"
            # options.unireloc3r_pose_estimation_mode = "geoaware_pnet"
            # options.unireloc3r_pose_estimation_mode = "vanilla_pose_head_regression"
            # options.geoaware_cfg_path = "/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/geoaware_pnet/transformer/config/config_geoaware_in_unireloc3r.json"
            # options.load_geoaware_pretrain_model = True
            # options.depth_model_type = "endofast3r_depth_trained_dbg" #critical! we better init with optimized DAM


            # options.pose_model_type = "pcrnet"
            # options.enable_all_depth = True

            # options.pose_model_type = "separate_resnet"

            # # # debug trained fast3r (understand its learned scale)
            # options.pose_model_type = "endofast3r_pose_trained_dbg"
            # options.pose_model_type = "endofast3r"
            # options.use_raft_flow = True
            # options.depth_model_type = "endofast3r_depth_trained_dbg" #critical! we better init with optimized DAM
            # options.gt_metric_rel_pose_as_estimates_debug = True
            options.min_depth = 0.1 # bigger safer
            options.max_depth = 150.0 # bigger safer
            # options.enable_motion_computation = False
            # options.enable_motion_computation = True

            # a new pose model varient:
            # options.pose_model_type = "diffposer_epropnp" # not possible ?

            # options.pose_model_type = "geoaware_pnet" #not possible ?



            # options.enable_mutual_motion = True
            # options.enable_mutual_motion = False
            # options.use_soft_motion_mask = True
            # options.use_soft_motion_mask = False
            # options.use_MF_network = False
            
            options.num_workers = 0

            # options.shared_MF_OF_network = False # still not working



            # #////////////////// default endofast3r setting //////////////////
            # options.use_raft_flow = False # here is the only issue? raft especially non robust when no scene motion for both pose models when enable_motion_etc (we only test on b1)?
            # # options.pose_model_type = "separate_resnet"

            # # options.enable_motion_computation = False
            # options.use_MF_network = False
            # options.ignore_motion_area_at_calib = False
            # options.reproj_supervised_which = 'color'
            # options.use_loss_motion_mask_reg = False #not checked this factor
            # options.use_loss_reproj2_nomotion = False #not checked this factor
            # options.enable_grad_flow_motion_mask = False #not checked this factor

            # # motion_flow net predict all nan after grads update when the there is no scene dynamics?

            options.datasets = [
                                # 'endovis', 
                                # 'DynaSCARED',
                                'StereoMIS',
                                ]
            options.split_appendixes = [
                                        # '', 
                                        # '_CaToTi000',
                                        # '_CaToTi100',
                                        '_offline',
                                        ]
            options.data_paths = [
                # '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/', 
                # '/mnt/cluster/datasets/Surg_oclr_stereo/',
                '/mnt/nct-zfs/TCO-All/SharedDatasets/StereoMIS_DARES_test/',
                ]
            options.dataset_configs = [{'dataset': options.datasets[i],
                                       'split_appendix': options.split_appendixes[i],
                                       'data_path': options.data_paths[i]} for i in range(len(options.datasets))]

        # '--datasets', 'endovis', 'DynaSCARED',
        # '--split_appendixes', '', '000_00597',
        # '--data_paths', '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/', '/mnt/cluster/datasets/Surg_oclr_stereo/'
    


        options.px_resample_rule_dict_scale_step = {
                                                         0: 2**0,
                                                         1: 2**1,
                                                         2: 2**2,
                                                         3: 2**3,}

        self.opt = options
        
        # sanity check some params early
        if self.opt.use_loss_motion_mask_reg:
            assert self.opt.enable_grad_flow_motion_mask, "enable_grad_flow_motion_mask must be True when use_loss_motion_mask_reg is True"
        if self.opt.reg_mutual_raw_disp_based_OF_for_consistency_and_correctness:
            assert self.opt.enable_mutual_motion, "enable_mutual_motion must be True when reg_mutual_depth_based_OF_for_consistency_and_correctness is True"
        if self.opt.pose_model_type == "pcrnet":
            assert self.opt.enable_all_depth, "enable_motion_computation must be True when pose_model_type is pcrnet"

        # #/////
        # if not self.opt.debug:
        #     # update the launched job if it is on queue
        #     self.opt.freeze_depth_debug = True
        #     self.opt.use_soft_motion_mask = False
        #     self.opt.flow_reproj_supervised_with_which = "raw_tgt_gt" # should be safer considering the refined is not reasonable yet.
        #     self.opt.model_name = "colored_MotionCorrected_loss2nomo_000_fzD_b4_hardRegMF_contrasiveFlowMag_baseline"
        #     #/////

        # SANITY CHECK BEFORE TRANING START
        if self.opt.ignore_motion_area_at_calib:
            assert self.opt.enable_motion_computation, "enable_motion_computation must be True when ignore_motion_area_at_calib is True"

        from datetime import datetime
        
        # Handle log path for resume training
        if self.opt.resume_training and self.opt.load_weights_folder and not self.opt.new_tensorboard:
            # Use the same log path as the original training (default behavior)
            self.log_path = self.opt.load_weights_folder
            print(f"Resuming tensorboard logging in existing directory: {self.log_path}")
        else:
            # Create new log directory with timestamp
            timestamp = datetime.now().strftime("%m%d-%H%M%S")
            self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name, timestamp)
            if self.opt.resume_training and self.opt.new_tensorboard:
                print(f"Creating new tensorboard logging directory (--new_tensorboard specified): {self.log_path}")
            else:
                print(f"Creating new tensorboard logging directory: {self.log_path}")
        
        os.makedirs(self.log_path, exist_ok=True)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        # assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        set_seed(self.opt.seed)
        # self.scaler = GradScaler()

        print(self.opt.seed, "is the seed!")

        print("learning rate:", self.opt.learning_rate)

        # print("learning rate is",self.opt.learning_rate)
        print("batch size is:",self.opt.batch_size)
        print("accumulate steps is:",self.opt.accumulate_steps)
        print("effective batch size is:",self.opt.batch_size * self.opt.accumulate_steps)

        self.models = {}  
        self.parameters_to_train = []
        self.parameters_to_train_0 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        # Initialize learnable motion mask threshold if enabled
        if self.opt.enable_learned_motion_mask_thre_px:
            # Initialize with the default threshold value
            self.learned_motion_mask_thre_px = torch.nn.Parameter(
                torch.tensor(self.opt.motion_mask_thre_px, dtype=torch.float32, device=self.device)
            )
            self.parameters_to_train.append(self.learned_motion_mask_thre_px)
            print(f"Initialized learnable motion mask threshold with value: {self.opt.motion_mask_thre_px}")
        else:
            self.learned_motion_mask_thre_px = None

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        print("Using DoMoRA")
        if self.opt.depth_model_type == "dam":
            self.models["depth_model"] = networks.Endo_FASt3r_depth()
        elif self.opt.depth_model_type == "endofast3r_depth_trained_dbg":
            # self.models["depth_model"] = networks.Endo_FASt3r_depth()
            # self.models["depth_model"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/depth_model.pth"))
            load_weights_folder = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/fast3r_ckpts/best_weights'
            depth_model_path = os.path.join(load_weights_folder, "depth_model.pth")
            depth_model_dict = torch.load(depth_model_path)
            depth_model = networks.Endo_FASt3r_depth()
            model_dict = depth_model.state_dict()
            depth_model.load_state_dict({k: v for k, v in depth_model_dict.items() if k in model_dict})
            self.models["depth_model"] = depth_model
            print('loaded endofast3r_depth_trained_dbg depth model...')
        else:
            assert 0, "Unknown depth model type: " + self.opt.depth_model_type
        self.models["depth_model"].to(self.device)



        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))

        # Initialize optical flow networks (either custom or RAFT)
        if self.opt.use_raft_flow:
            print("Using RAFT optical flow estimator instead of custom position networks")
            # Use RAFT as a direct replacement for both position_encoder and position
            from networks import RAFT
            self.models["raft_flow"] = RAFT(self.device).model
            # self.models["raft_flow"].to(self.device)
            # RAFT is trainable, so add its parameters to training
            self.parameters_to_train_0 += list(self.models["raft_flow"].parameters())
            print("RAFT flow estimator initialized (trainable)")
            
            # enable motion_flow_net
            if self.opt.use_MF_network:
                # self.models["motion_raft_flow"] = RAFT(self.device).model
                self.models["motion_raft_flow"] = RAFT(self.device).model if not self.opt.shared_MF_OF_network \
                    else self.models["raft_flow"]

                self.parameters_to_train_0 += list(self.models["motion_raft_flow"].parameters())
                print("Motion RAFT flow estimator initialized (trainable)")

        else:
            # Use original custom networks
            self.models["position_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
            
            self.models["position_encoder"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/position_encoder.pth"))
            self.models["position_encoder"].to(self.device)
            self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())

            self.models["position"] = networks.PositionDecoder(
                self.models["position_encoder"].num_ch_enc, self.opt.scales)
            self.models["position"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/position.pth"))
            
            self.models["position"].to(self.device)
            self.parameters_to_train_0 += list(self.models["position"].parameters())


            # enable motion_flow_net
            if self.opt.use_MF_network:
                if self.opt.shared_MF_OF_network:
                    self.models["motion_position_encoder"] = self.models["position_encoder"]
                else:
                    self.models["motion_position_encoder"] = networks.ResnetEncoder(
                        self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
                    # INTI FROM OF encoder
                    self.models["motion_position_encoder"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/position_encoder.pth"))
                    self.models["motion_position_encoder"].to(self.device)

                    self.parameters_to_train += list(self.models["motion_position_encoder"].parameters())

                if self.opt.shared_MF_OF_network:
                    self.models["motion_position"] = self.models["position"]
                else:
                    self.models["motion_position"] = networks.PositionDecoder(
                        self.models["motion_position_encoder"].num_ch_enc, self.opt.scales)
                    self.models["motion_position"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/position.pth"))
                    self.models["motion_position"].to(self.device)
                    self.parameters_to_train += list(self.models["motion_position"].parameters())

        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.models["transform_encoder"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/transform_encoder.pth"))
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())


        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.models["transform"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/transform.pth"))
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                pose_encoder_path = os.path.join('/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/', "af_sfmlearner_weights", "pose_encoder.pth")
                pose_decoder_path = os.path.join('/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/', "af_sfmlearner_weights", "pose.pth")
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                assert os.path.exists(pose_encoder_path), f"pose_encoder_path {pose_encoder_path} does not exist"
                # self.models["pose_encoder"].load_state_dict(torch.load(pose_encoder_path))
                assert os.path.exists(pose_decoder_path), f"pose_decoder_path {pose_decoder_path} does not exist"
                # self.models["pose"].load_state_dict(torch.load(pose_decoder_path))
                print('loaded separate_resnet pose model...')

            elif self.opt.pose_model_type == "posetr_net":
                from functools import partial
                # already load pretrained resnet18 internally
                # self.models["pose"] = PoseTransformer(enc_embed_dim=512,
                #                  enc_depth=6,
                #                  enc_num_heads=8,
                #                  dec_embed_dim=384,
                #                  dec_depth=4,
                #                  dec_num_heads=6,
                #                  mlp_ratio=4,
                #                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                #                  norm_im2_in_dec=True,
                #                  pos_embed='RoPE100')
                
                debug_only = True
                # debug_only = False
                if debug_only:

                    from posetr.posetr_model_v2 import PoseTransformerV2
                    # self.models["pose"]  = PoseTransformerV2(
                    #     img_size=(256, 320),
                    #     patch_size=16,
                    #     embed_dim=384,              # Reduced from 768
                    #     vit_depth=6,                # Reduced from 12
                    #     vit_num_heads=6,            # 384/6 = 64 dim per head
                    #     attention_depth=3,          # Reduced from 6
                    #     attention_num_heads=6       # Keep consistent
                    # )
                    self.models["pose"] = PoseTransformerV2(
                        img_size=(256, 320),
                        patch_size=16,
                        # embed_dim=384,              # Match DeiT-Small dimension
                        attention_depth=4,          # 2 self + 2 cross attention
                        attention_num_heads=6,       # 384/6 = 64 dim per head
                        croco_vit=True,
                        # skip_sa_ca=True,
                        # use_vit = False,
                        embed_dim=512,              # enable when no_use_vit so that exactly resnet_seperate_embedding

                    )

                print('loaded posetr_net pose model...')

            elif self.opt.pose_model_type == "endofast3r":
                # reloc3r_ckpt_path = f"{RELOC3R_PRETRAINED_ROOT}/Reloc3r-512.pth"
                assert os.path.exists(self.opt.backbone_pretrain_ckpt_path), f"backbone_pretrain_ckpt_path {self.opt.backbone_pretrain_ckpt_path} does not exist"
                # assert self.opt.backbone_pretrain_ckpt_path == f"{RELOC3R_PRETRAINED_ROOT}/Reloc3r-512.pth", f"backbone_pretrain_ckpt_path {self.opt.backbone_pretrain_ckpt_path} is not correct"
                from networks import Reloc3rX
                self.models["pose"] = Reloc3rX(self.opt.backbone_pretrain_ckpt_path)
            elif self.opt.pose_model_type == "endofast3r_pose_trained_dbg":
                # load reloc3r pose model, then overwrite if saved in pose.pth
                load_weights_folder = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/fast3r_ckpts/best_weights'
                pose_model_path = os.path.join(load_weights_folder, "pose.pth")
                pose_model_dict = torch.load(pose_model_path)

                reloc3r_ckpt_path = f"{RELOC3R_PRETRAINED_ROOT}/Reloc3r-512.pth"
                pose_model = networks.Reloc3rX(reloc3r_ckpt_path)
                model_dict = pose_model.state_dict()
                
                # log in the layers that are overwritten or remain
                # only the 1st level name should be fine
                overwritten_layers = [k for k in pose_model_dict.keys() if k in model_dict]
                remain_layers = [k for k in pose_model_dict.keys() if k not in model_dict]
                print("Overwritten layers:", set([k.split(".")[0] for k in overwritten_layers]))
                print("Remaining layers:", set([k.split(".")[0] for k in remain_layers]))

                pose_model.load_state_dict({k: v for k, v in pose_model_dict.items() if k in model_dict})
                self.models["pose"] = pose_model#Reloc3rX(reloc3r_ckpt_path)
                
                print('loaded endofast3r_pose_trained_dbg pose model...')

            elif self.opt.pose_model_type == "uni_reloc3r":
                # reloc3r_ckpt_path = f"{RELOC3R_PRETRAINED_ROOT}/Reloc3r-512.pth"
                # backbone_pretrain_ckpt_path = self.opt.backbone_pretrain_ckpt_path#f"{RELOC3R_PRETRAINED_ROOT}/Reloc3r-512.pth"
                assert os.path.exists(self.opt.backbone_pretrain_ckpt_path), f"backbone_pretrain_ckpt_path {self.opt.backbone_pretrain_ckpt_path} does not exist"
                from networks import UniReloc3r
                print('init UniReloc3r from', self.opt.backbone_pretrain_ckpt_path)
                self.models["pose"] = UniReloc3r(self.opt.backbone_pretrain_ckpt_path, 
                                                 self.opt,
                                                 self.log_path
                                                 )
                print('loaded UniReloc3r...')


            elif self.opt.pose_model_type == "geoaware_pnet":
                from geoaware_pnet.geoaware_network import load_geoaware_pose_head
                self.models["pose"] = load_geoaware_pose_head(self.opt.geoaware_cfg_path, 
                                                              self.opt.load_geoaware_pretrain_model, 
                                                              self.opt.px_resample_rule_dict_scale_step)


            elif self.opt.pose_model_type == "diffposer_epropnp":
                # if self.unireloc3r_pose_estimation_mode == 'epropnp':
                self.initialize_epropnp() # init: log_weight_scale, camera, cost_fun, epropnp
                self._base_grid = None

                self.models["pose"] = torch.nn.Module()
                print('init epropnp  pose solver.....')
            elif self.opt.pose_model_type == "pcrnet":
                # load pcrnet pose model
                from pcrnet_integration import load_pcrnet_pose_head
                self.models["pose"] = load_pcrnet_pose_head(emb_dims=1024)
                print("loaded PCRNet pose estimator...")

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)
                self.models["pose"].load_state_dict(torch.load(pose_decoder_path))

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                self.models["pose"].load_state_dict(torch.load(pose_decoder_path))


            self.models["pose"].to(self.device)

            self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["pose"].parameters()))

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            print('CHECK: self.opt.predictive_mask')
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        else:
            print('CHECK: NO self.opt.predictive_mask')

        #///////////params stats//////////
        # report for total num of params and trainable params for each model in self.models
        # Each parameter is a 32-bit float, which is 4 bytes.
        bytes_per_param = 4

        for model_name, model in self.models.items():
            print(f"Model: {model_name}")
            total_params = sum(p.numel() for p in model.parameters())
            total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # show in MB
            print(f"  Total params: {total_params} ({total_params * bytes_per_param/ 1024 / 1024 :.2f} MB)")
            print(f"  Total params trainable: {total_params_trainable} ({total_params_trainable * bytes_per_param / 1024 / 1024:.2f} MB)")
            # print(f"  Total params non-trainable: {(total_params - total_params_trainable) / 1024 / 1024:.2f} MB")
        # sum over all the models
        total_params = sum(sum(p.numel() for p in model.parameters()) for model in self.models.values())
        total_params_trainable = sum(sum(p.numel() for p in model.parameters() if p.requires_grad) for model in self.models.values())
        print(f"Total params: {total_params} ({total_params * bytes_per_param / 1024 / 1024:.2f} MB)")
        print(f"Total params trainable: {total_params_trainable} ({total_params_trainable * bytes_per_param / 1024 / 1024:.2f} MB)")

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        
        # Handle resume training
        if self.opt.resume_training:
            self.resume_training()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # Multi-dataset loading
        print("Loading datasets...")
        

        if len(self.opt.dataset_configs) == 1:
            # Single dataset mode (existing logic)
            config = self.opt.dataset_configs[0]
            self._load_single_dataset(config)
        else:
            # Multi-dataset mode
            self._load_multi_datasets()
        
        # Calculate total samples and steps
        if isinstance(self.train_dataset, ConcatDataset):
            num_train_samples = sum(len(dataset) for dataset in self.train_dataset.datasets)
        else:
            num_train_samples = len(self.train_dataset)
        
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, self.opt.batch_size, 
            shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=True)
        
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            self.ms_ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)
        if self.opt.pose_model_type == "geoaware_pnet" or \
            (self.opt.pose_model_type == "uni_reloc3r" and self.opt.unireloc3r_pose_estimation_mode == "geoaware_pnet"):
            print('using spatial_transform_used_to_warp_sc_3d.....')
            self.spatial_transform_warp_sc3d_geoaware_pnet = {}
            for scale in self.opt.scales:
                sample_step = self.opt.px_resample_rule_dict_scale_step[scale]
                self.spatial_transform_warp_sc3d_geoaware_pnet_scale_i = SpatialTransformer((int(self.opt.height/sample_step), 
                                                                                int(self.opt.width/sample_step)))
                self.spatial_transform_warp_sc3d_geoaware_pnet_scale_i.to(self.device)
                self.spatial_transform_warp_sc3d_geoaware_pnet[scale] = self.spatial_transform_warp_sc3d_geoaware_pnet_scale_i
        elif self.opt.pose_model_type == "diffposer_epropnp":
            print('using spatial_transform_used_to_warp_sc_3d.....')
            self.spatial_transform_used_to_warp_sc_3d = SpatialTransformer((int(self.opt.height), 
                                                                            int(self.opt.width)))
            self.spatial_transform_used_to_warp_sc_3d.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        # self.project_3d_raw = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
            # self.project_3d_raw[scale] = Project3D_Raw(self.opt.batch_size, h, w)
            # self.project_3d_raw[scale].to(self.device)

            # not used?
            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # print("Using dataset:\n  ", self.opt.dataset)
        print("Using dataset:\n  ")
        for config in self.opt.dataset_configs:
            print(f"  {config['dataset']}: {config['split_appendix']}, {config['data_path']}")
        print("There are {:d} training items and {:d} validation items\n".format(
            len(self.train_dataset), len(self.val_dataset)))

        self.save_opts()
        # ipdb.set_trace()

    def _load_single_dataset(self, config):
        """Load single dataset (existing logic)"""
        dataset_name = config['dataset']
        split_appendix = config['split_appendix']
        data_path = config['data_path']
        
        # Validate data path
        if dataset_name == 'DynaSCARED':
            assert data_path == '/mnt/cluster/datasets/Surg_oclr_stereo/', f"data_path {data_path} is not correct"
            datasets_dict = {dataset_name: datasets.DynaSCAREDRAWDataset}
            fpath = os.path.join(os.path.dirname(__file__), "splits", dataset_name, "{}.txt")
            fpath_train = fpath.format(f"train{split_appendix}")
            fpath_val = fpath.format(f"val{split_appendix}")
            assert split_appendix in ['',] or '_CaToTi' in split_appendix, f"split_appendix {split_appendix} is not correct"
        elif dataset_name == 'endovis':
            assert data_path == '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/', f"data_path {data_path} is not correct"
            datasets_dict = {dataset_name: datasets.SCAREDRAWDataset}
            fpath = os.path.join(os.path.dirname(__file__), "splits", dataset_name, "{}_files.txt")
            assert split_appendix == '', "split_appendix should be empty for endovis"
            fpath_train = fpath.format(f"train{split_appendix}")
            fpath_val = fpath.format(f"val{split_appendix}")
        elif dataset_name == 'StereoMIS':
            assert data_path == '/mnt/nct-zfs/TCO-All/SharedDatasets/StereoMIS_DARES_test/', f"data_path {data_path} is not correct"
            datasets_dict = {dataset_name: datasets.StereoMISDataset}
            fpath = os.path.join(os.path.dirname(__file__), "splits", dataset_name, "{}_files.txt")
            fpath_train = fpath.format(f"train{split_appendix}")
            fpath_val = fpath.format(f"val{split_appendix}")
            import warnings
            warnings.warn(f"Using StereoMIS dataset with split_appendix {split_appendix}")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name} {data_path}")
        
        self.dataset = datasets_dict[dataset_name]
        
        train_filenames = readlines(fpath_train)
        val_filenames = readlines(fpath_val)
        img_ext = '.png'
        
        self.train_dataset = self.dataset(
            data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, 
            of_samples=self.opt.of_samples, of_samples_num=self.opt.of_samples_num)
        
        self.val_dataset = self.dataset(
            data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, 
            of_samples=self.opt.of_samples, of_samples_num=self.opt.of_samples_num)

    def _load_multi_datasets(self):
        """Load multiple datasets and combine them"""
        from torch.utils.data import ConcatDataset
        
        train_datasets = []
        val_datasets = []
        
        for config in self.opt.dataset_configs:
            dataset_name = config['dataset']
            split_appendix = config['split_appendix']
            data_path = config['data_path']
            
            print(f"Loading {dataset_name} from {data_path} with split {split_appendix}")
            
            # Load dataset (similar to single dataset logic)
            if dataset_name == 'DynaSCARED':
                dataset_class = datasets.DynaSCAREDRAWDataset
                fpath = os.path.join(os.path.dirname(__file__), "splits", dataset_name, "{}.txt")
                fpath_train = fpath.format(f"train{split_appendix}")
                fpath_val = fpath.format(f"val{split_appendix}")
            elif dataset_name == 'endovis':
                dataset_class = datasets.SCAREDRAWDataset
                fpath = os.path.join(os.path.dirname(__file__), "splits", dataset_name, "{}_files.txt")
                fpath_train = fpath.format(f"train{split_appendix}")
                fpath_val = fpath.format(f"val{split_appendix}")
            elif dataset_name == 'StereoMIS':
                dataset_class = datasets.StereoMISDataset
                fpath = os.path.join(os.path.dirname(__file__), "splits", dataset_name, "{}_files.txt")
                fpath_train = fpath.format(f"train{split_appendix}")
                fpath_val = fpath.format(f"val{split_appendix}")
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            train_filenames = readlines(fpath_train)
            val_filenames = readlines(fpath_val)
            img_ext = '.png'
            
            # Create datasets
            train_dataset = dataset_class(
                data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext,
                of_samples=self.opt.of_samples, of_samples_num=self.opt.of_samples_num)
            
            val_dataset = dataset_class(
                data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,
                of_samples=self.opt.of_samples, of_samples_num=self.opt.of_samples_num)
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        
        # Combine datasets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

        # hard code load gt_pose flag for all datasets
        self.train_dataset.load_gt_poses = True
        self.val_dataset.load_gt_poses = True
        
        print(f"Combined {len(train_datasets)} datasets:")
        for i, config in enumerate(self.opt.dataset_configs):
            print(f"  {config['dataset']}: {len(train_datasets[i])} train samples, {len(val_datasets[i])} val samples")




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

    def prepare_epropnp_input(self, pts3d_2in1view, pts3d_2in1view_conf, 
                                K_2, B, H, W, device, 
                                # matches_num = 8,
                                matches_num = 4800,
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
        
        if matches_num < x3d.shape[1]:
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

    def _init_base_grid(self, H, W, device, reallocate=False):
        """use for construct the 2D in 2D-3D matches, assume 3D are regressed SC."""
        if self._base_grid is None or reallocate:
            hh, ww = torch.meshgrid(torch.arange(
                H).float(), torch.arange(W).float())
            coord = torch.zeros([1, H, W, 2])
            coord[0, ..., 0] = ww
            coord[0, ..., 1] = hh
            self._base_grid = coord.to(device)
    
    def set_train_0(self):
        """Convert all models to training mode
        """
        if self.opt.use_raft_flow:
            # RAFT models are trainable
            for param in self.models["raft_flow"].parameters():
                param.requires_grad = True
            self.models["raft_flow"].train()
        else:
            # Custom position networks are trainable
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = True
            for param in self.models["position"].parameters():
                param.requires_grad = True
            self.models["position_encoder"].train()
            self.models["position"].train()

        for param in self.models["depth_model"].parameters():
            param.requires_grad = False
        # for param in self.models["pose_encoder"].parameters():
        #     param.requires_grad = False
        for param in self.models["pose"].parameters():
            param.requires_grad = False
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["transform"].parameters():
            param.requires_grad = False
        
        self.models["depth_model"].eval()
        # self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()

        # MF is not trainable during train_0()
        if self.opt.use_MF_network and not self.opt.shared_MF_OF_network:
            if self.opt.use_raft_flow:
                for param in self.models["motion_raft_flow"].parameters():
                    param.requires_grad = False
                self.models["motion_raft_flow"].eval()
            else:
                for param in self.models["motion_position_encoder"].parameters():
                    param.requires_grad = False
                for param in self.models["motion_position"].parameters():
                    param.requires_grad = False
                self.models["motion_position_encoder"].eval()
                self.models["motion_position"].eval()

    def freeze_params(self,keys = []):
        # no grad compuation: debug only
        # for all keys in self.models, set requires_grad to False
        for key in keys:
            if key not in self.models:
                continue
            for param in self.models[key].parameters():
                param.requires_grad = False


    def freeze_all_params(self, modules=[]):
        for module in modules:
            try:
                for n, param in module.named_parameters():
                    param.requires_grad = False
            except AttributeError:
                # module is directly a parameter
                module.requires_grad = False

    def set_train(self):
        """Convert all models to training mode
        # motion flow net is trainable druing train(); OF is not trainable 
        """
        if self.opt.use_raft_flow:
            # RAFT models are frozen during main training
            for param in self.models["raft_flow"].parameters():
                param.requires_grad = False
            self.models["raft_flow"].eval()
        else:
            # Custom position networks are frozen during main training
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = False
            for param in self.models["position"].parameters():
                param.requires_grad = False
            self.models["position_encoder"].eval()
            self.models["position"].eval()

        # for param in self.models["encoder"].parameters():
            # param.requires_grad = True
        for param in self.models["depth_model"].parameters():
            param.requires_grad = True
            #debug: freeze depth_model
            if self.opt.freeze_depth_debug:
                param.requires_grad = False

        # for param in self.models["pose_encoder"].parameters():
        #     param.requires_grad = True
        for param in self.models["pose"].parameters():
            param.requires_grad = True
        
        # enable trainable enc_norm for best adapt with trainable decoder
        if self.opt.pose_model_type == "posetr_net" and self.opt.freeze_posetr_encoder:
            print('freeze posetr_net encoder...')
            self.freeze_all_params(modules=[self.models["pose"].feature_extractor.patch_embed, 
                                            self.models["pose"].feature_extractor.enc_blocks])

        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["transform"].parameters():
            param.requires_grad = True

        self.models["depth_model"].train()
        # self.models["pose_encoder"].train()
        self.models["pose"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
    
        if self.opt.use_MF_network:
            # it will enable the OF net when shared_MF_OF_network is on
            # we need to manually detach all the OF estimation during train() statge
            if self.opt.use_raft_flow:
                for param in self.models["motion_raft_flow"].parameters():
                    param.requires_grad = True
                self.models["motion_raft_flow"].train()
            else:
                for param in self.models["motion_position_encoder"].parameters():
                    param.requires_grad = True
                for param in self.models["motion_position"].parameters():
                    param.requires_grad = True
                self.models["motion_position_encoder"].train()
                self.models["motion_position"].train()


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if self.opt.use_raft_flow:
            self.models["raft_flow"].eval()
            if self.opt.use_MF_network:
                self.models["motion_raft_flow"].eval()
        else:
            self.models["position_encoder"].eval()
            self.models["position"].eval()
            if self.opt.use_MF_network:
                self.models["motion_position_encoder"].eval()
                self.models["motion_position"].eval()
            
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        # self.models["pose_encoder"].eval()
        self.models["pose"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        # Initialize training state if not resuming
        if not hasattr(self, 'epoch') or self.epoch is None:
            self.epoch = 0
        if not hasattr(self, 'step') or self.step is None:
            self.step = 0
        if not hasattr(self, 'start_time') or self.start_time is None:
            self.start_time = time.time()
        
        # Start training from the current epoch
        for self.epoch in range(self.epoch, self.opt.num_epochs):

            self.run_epoch(self.epoch)
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self, epoch):
        """Run a single epoch of training and validation
        """

        print("Training")

        # Initialize gradient accumulation counters
        accumulate_step = 0
        
        for batch_idx, inputs in enumerate(self.train_loader):
            # print('step:', self.step)

            before_op_time = time.time()

            # position
            self.set_train_0()
            if self.opt.freeze_as_much_debug:
                if self.opt.use_raft_flow:
                    pass
                    # self.freeze_params(keys = ['raft_flow',])#debug only for save mem
                else:
                    self.freeze_params(keys = ['position_encoder',])#debug only for save mem
            _, losses_0 = self.process_batch_0(inputs)
            
            # Scale loss by accumulate_steps for gradient accumulation
            scaled_loss_0 = losses_0["loss"] / self.opt.accumulate_steps
            scaled_loss_0.backward()
            
            accumulate_step += 1
            
            # Only step optimizer when accumulation is complete
            if accumulate_step % self.opt.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train_0, max_norm=1.0)
                self.model_optimizer_0.step()
                self.model_optimizer_0.zero_grad()

            self.set_train()
            if self.opt.freeze_as_much_debug:
                pass
                # self.freeze_params(keys = ['depth_model', 'pose', 'transform_encoder'])#debug only
            outputs, losses = self.process_batch(inputs) # img_warped_from_pose_flow saved as "color"; img_warped_from_optic_flow saved as "registration"

            # Scale loss by accumulate_steps for gradient accumulation
            scaled_loss = losses["loss"] / self.opt.accumulate_steps
            scaled_loss.backward()
            
            # Only step optimizer when accumulation is complete
            if accumulate_step % self.opt.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                # Use the accumulated loss for logging (multiply back to show effective loss)
                # effective_loss = losses["loss"] * (self.opt.accumulate_steps / max(1, accumulate_step % self.opt.accumulate_steps))
                # self.log_time(batch_idx, duration, effective_loss.cpu().data)

                # losses_0
                losses_0_to_log = {
                    "loss_0": losses_0["loss"],
                }
                # losses
                losses_to_log = losses

                # catains the breakdown of losses
                scalers_to_log = {**losses_0_to_log, **losses_to_log}

                errs_to_log = {}
                metric_errs = {}
                if self.train_dataset.load_gt_poses:
                    # compute the pose errors
                    metric_errs = self.compute_pose_metrics(inputs, outputs) # dict
                    # print('trans_err:', trans_err)
                    # print('rot_err:', rot_err)
                    # metric_errs = {
                    #     'trans_err': trans_err,
                    #     'rot_err': rot_err
                    # }

                for k, v in metric_errs.items():
                    assert len(v) == len(self.opt.frame_ids)-1, f'{k}: {v}'
                    # mean over frames_ids (already mean over batches internally)
                    errs_to_log[f"{k}"] = torch.mean(torch.stack(v)).item()

                self.log_time(batch_idx, duration, 
                              scaled_loss.cpu().data, 
                              scaled_loss_0.cpu().data, 
                              errs_to_log)

                scalers_to_log = {**scalers_to_log, **errs_to_log}
                # add 'scalar/' prefix to the keys
                scalers_to_log = {f"scalar/{k}": v for k, v in scalers_to_log.items()}

                self.log("train", inputs, outputs, 
                         scalers_to_log, 
                         compute_vis=True)
                # self.log("train", inputs, outputs, losses, compute_vis=True, online_vis=True)

            self.step += 1
            
        # Step schedulers at the end of epoch
        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    def process_batch_0(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            # print('process_batch_0, key:', key)
            inputs[key] = ipt.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses_0(inputs, outputs))
        # detach compute_transfrom from predict_pose for better extention
        outputs.update(self.compute_brightness_transform_0(inputs, outputs))

        losses = self.compute_losses_0(inputs, outputs)

        return outputs, losses

    def reformat_raft_output(self, num_flow_udpates, outputs_raw, hard_detach_OF_grad = False):
        outputs = {}
        assert len(outputs_raw) == num_flow_udpates, f"outputs_raw length {len(outputs_raw)} != num_flow_udpates {num_flow_udpates}"
        # sanity check if outputs_raw contains nan or inf. if does,
        # give warning, and sanitize these flow values
        def sanitize_flow_values(outputs_raw):
            for scale, scale_raw in enumerate([11,8,5,2]):
                if torch.isnan(outputs_raw[scale_raw]).any() or torch.isinf(outputs_raw[scale_raw]).any():
                    print('!!!!**************!!!!')
                    # print the number of num/inf
                    num_nan = torch.isnan(outputs_raw[scale_raw]).sum()
                    num_inf = torch.isinf(outputs_raw[scale_raw]).sum()
                    print(f"Warning: outputs_raw contains nan or inf at scale {scale}: {num_nan} nan, {num_inf} inf")
                    assert 0, f"outputs_raw contains nan or inf at scale {scale}: {num_nan} nan, {num_inf} inf"
                    outputs_raw[scale_raw] = torch.nan_to_num(outputs_raw[scale_raw], nan=0.0, posinf=0.0, neginf=0.0)
            return outputs_raw
        outputs_raw = sanitize_flow_values(outputs_raw)


        for scale, scale_raw in enumerate([11,8,5,2]):
            #high to low
            # resize the resolution according to the scale
            # pyramid resolution
            pyramid_resolution_height = self.opt.height // (2 ** scale)
            pyramid_resolution_width = self.opt.width // (2 ** scale)
            if scale!=0:
                outputs[("position", scale)] = F.interpolate(
                        outputs_raw[scale_raw].detach() if hard_detach_OF_grad else outputs_raw[scale_raw], # manually detach for safety as when shared_MF_OF_network is on, OF net is trainable during train() stage
                        [pyramid_resolution_height, pyramid_resolution_width], mode="bilinear",
                        align_corners=True)
            else:
                outputs[("position", scale)] = outputs_raw[scale_raw]


        return outputs

    def predict_poses_0(self, inputs, outputs = {}):
        """Predict poses between input frames for monocular sequences.
        """
        # outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
            assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # ipdb.set_trace()


                    # position - handle both custom networks and RAFT
                    if self.opt.use_raft_flow:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]# tgt to src flow
                        inputs_all_reverse = [pose_feats[f_i], pose_feats[0]]
                        # RAFT expects separate image inputs, not concatenated features
                        # we reformat raft 12 resolution output to 4 resolution output
                        num_flow_udpates = 12
                        outputs_0_raw = self.models["raft_flow"](pose_feats[0], pose_feats[f_i])
                        outputs_1_raw = self.models["raft_flow"](pose_feats[f_i], pose_feats[0])
                        # print('inference optic flow in train_0()...')
                        outputs_0 = self.reformat_raft_output(num_flow_udpates, outputs_0_raw, hard_detach_OF_grad=False)# there is alreay no grad in OF net
                        outputs_1 = self.reformat_raft_output(num_flow_udpates, outputs_1_raw, hard_detach_OF_grad=False)# there is alreay no grad in OF net
                    else:
                        # take care! the author network were trained in such an format
                        inputs_all = [pose_feats[f_i], pose_feats[0]]# tgt to src flow
                        inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]
                        # Original custom networks
                        position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))# there is no grad in position_encoder
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))# there is no grad in position_encoder
                        outputs_0 = self.models["position"](position_inputs)# there is no grad in position
                        outputs_1 = self.models["position"](position_inputs_reverse)# there is no grad in position

                    # for k, v in outputs_0.items():
                    #     print(f"{k}: {v.shape}")

                    for scale in self.opt.scales:
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)

                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)],
                                                                                       outputs[(
                                                                                       "position", "high", scale, f_i)])

                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                            outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)])

            # transform
            # outputs = self.compute_brightness_transform(inputs, outputs)
        return outputs

    def compute_brightness_transform_0(self, inputs, outputs):
        """Compute transform for all frame_ids.
        
        Args:
            inputs: Dictionary containing input data
            outputs: Dictionary containing output data
            
        Returns:
            outputs: Updated outputs dictionary with transform results
        """
        return self.conduct_brightness_calib(inputs, outputs, existing_motion_mask=False)

    def compute_brightness_transform(self, inputs, outputs):
        """Compute transform for all frame_ids.
        
        Args:
            inputs: Dictionary containing input data
            outputs: Dictionary containing output data
            
        Returns:
            outputs: Updated outputs dictionary with transform results
        """
        return self.conduct_brightness_calib(inputs, outputs, 
                                      existing_motion_mask=self.opt.enable_motion_computation)


    def conduct_brightness_calib(self, inputs, outputs, existing_motion_mask):
        """Compute transform for all frame_ids.
        
        Args:
            inputs: Dictionary containing input data
            outputs: Dictionary containing output data
            
        Returns:
            outputs: Updated outputs dictionary with transform results
        """
        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # transform
                transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                outputs_2 = self.models["transform"](transform_inputs)

                for scale in self.opt.scales:
                    outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                    outputs[("transform", "high", scale, f_i)] = F.interpolate(
                        outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                        align_corners=True)
                                            
                    # TWO STRATEGY TO HANDLE MOTION:
                    calib_target_area = outputs[("occu_mask_backward", 0, f_i)].detach()

                    # strategy 1
                    if existing_motion_mask and self.opt.ignore_motion_area_at_calib:
                        calib_target_area = calib_target_area * outputs[("motion_mask_backward", 0, f_i)].detach()
                        if self.opt.enable_mutual_motion:
                            calib_target_area = calib_target_area * outputs[("motion_mask_s2t_backward", 0, f_i)].detach()

                    # strategy 2: use regularizer to enforce the calib_target_area and gt_tgt have same LPIPS; and enforce the calib net only predict spec differece---reduce its capacity for color prediction.
                    # to be implemented later: we potentialy need both; would be nice if in the end we can calibrate motion area too.



                    outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * calib_target_area + inputs[("color", 0, 0)])

                    outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0,
                                                                   max=1.0)
        return outputs

    def compute_losses_0(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += (get_smooth_loss(outputs[("position", scale, frame_id)], color))

                if self.opt.flow_reproj_supervised_with_which == "raw_tgt_gt":
                    reproj_loss_supervised_signal_color = inputs[("color", 0, 0)].detach()
                elif self.opt.flow_reproj_supervised_with_which == "detached_refined":
                    reproj_loss_supervised_signal_color = outputs[("refined", scale, frame_id)].detach()
                else:
                    raise ValueError(f"Invalid flow_reproj_supervised_with_which: {self.opt.flow_reproj_supervised_with_which}")

                loss_registration += (self.compute_reprojection_loss(outputs[("registration", scale, frame_id)], 
                                                                     reproj_loss_supervised_signal_color) * occu_mask_backward).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        # for scale in self.opt.scales:
            # print(f'disp dim: {outputs["disp", scale].shape}')

        if self.opt.enable_mutual_motion or self.opt.enable_all_depth:
            # extend the depth prediction for srcs
            for frame_id in self.opt.frame_ids[1:]:
                outputs_i = self.models["depth_model"](inputs["color_aug", frame_id, 0])
                for scale in self.opt.scales:
                    outputs["disp", scale, frame_id] = outputs_i["disp", scale]

        outputs.update(self.generate_depth_pred_from_disp(outputs))

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))# img is warp from optic_flow, save as "registration" 

        # img is warp from pose_flow('sample'), save as "color"
        # motion masks is computed below
        # outputs.update(self.generate_images_pred(inputs, outputs)) # break down and put depth early before predict poses

        outputs.update(self.generate_images_pred(inputs, outputs))

        if self.opt.enable_motion_computation:
            if self.opt.use_MF_network:
                outputs.update(self.predict_motion_flow_with_MF_net(inputs, outputs))
            else:
                outputs.update(self.generate_motion_flow(inputs, outputs)) 

        if self.use_pose_net:
            outputs.update(self.compute_brightness_transform(inputs, outputs))

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses


    def disp_to_sc_3d_v0(self, disp, intrinsics, ret_mutilscale, scale_depth = 1.0):
        assert isinstance(ret_mutilscale, bool), "ret_mutilscale must be a bool"
        if not ret_mutilscale:
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) #muti-scale
        # print('min depth', torch.min(depth))
        # print('max depth', torch.max(depth))
        depth = depth * scale_depth

        pts_3d = depth_to_3d(depth, intrinsics)
        
        return pts_3d, depth

    def compute_pose_per_scale_geoaware_pnet_v2(self, inputs, outputs, scale_3d, px_K_scale, f_i):
        '''
        different strategy---need to keep enable_mutual_motion on
        '''
        
        # we need t2s pose---what above compute s2t pose then inverse?
        # aim: s2t pose
        assert self.opt.enable_mutual_motion, "enable_mutual_motion must be True"
        # K is shared acroos all the frames per frame_ids
        sc_3d_fi, _ = self.disp_to_sc_3d_v0(
                                    outputs[("disp", scale_3d, f_i)], # use src fi
                                    inputs["K", scale_3d][:,:3,:3], # use src fi
                                    ret_mutilscale=True,
                                    scale_depth=1.0,
                                    )
        # of_fi_to_f0 = outputs[("position_reverse", scale_3d, f_i)].detach()# s2t flow
        of_f0_to_fi = outputs[("position", scale_3d, f_i)].detach()# t2s flow
        # sc_3d_fi_matched = sc_3d_fi
        sc_3d_fi_matched = self.spatial_transform_warp_sc3d_geoaware_pnet[scale_3d](sc_3d_fi,
                                                of_f0_to_fi)
        
        K_for_f0_2D = inputs["K", px_K_scale][:,:3,:3]

        if torch.isnan(sc_3d_fi_matched).any() or torch.isinf(sc_3d_fi_matched).any():
            assert 0, f"sc_3d_f0_matched contains nan or inf"

        debug_only = True
        debug_only = False
        if debug_only:
            # scale up sc_3d_f0_matched
            sc_3d_fi_matched = sc_3d_fi_matched * 50.0 

        debug_only = True
        # debug_only = False
        if debug_only:
            print('for frame_id: {} and scale: {}'.format(f_i, scale_3d))
            # print('input scene points 3d f0 matched shape:', sc_3d_fi_matched.shape)
            print('input scene points 3d f0 matched max min z:', sc_3d_fi_matched[:,2,:,:].max(), 
                  sc_3d_fi_matched[:,2,:,:].min())
            print('input scene points 3d f0 matched mean x:', sc_3d_fi_matched[:,0,:,:].mean())
            print('input scene points 3d f0 matched mean y:', sc_3d_fi_matched[:,1,:,:].mean())
            print('input scene points 3d f0 matched mean z:', sc_3d_fi_matched[:,2,:,:].mean())
            
        poses_list = self.models["pose"](sc_3d_fi_matched, 
                                        intrinsics_B33=K_for_f0_2D,
                                        sample_level=scale_3d)
        poses_list = [ torch.linalg.inv(pose) for pose in poses_list]# from s2t to t2s

        return poses_list[-1]

    def compute_pose_per_scale_geoaware_pnet(self, inputs, outputs, sc_scale, px_K_scale, f_i):

        # K is shared acroos all the frames per frame_ids
        sc_3d_f0, _ = self.disp_to_sc_3d_v0(
                                    outputs[("disp", sc_scale)], # use tgt f0
                                    inputs["K", sc_scale][:,:3,:3], # use tgt f0
                                    ret_mutilscale=True,
                                    scale_depth=1.0,
                                    )
        # critical: we need to warp sc_3d_f0 as matches of px_2d_fi
        # for key, value in outputs.items():
            # print('key: {}'.format(key))
            # print('value shape: {}'.format(value.shape))

        # of_fi_to_f0 = outputs[("position", sc_scale, f_i)].detach()# s2t flow
        of_fi_to_f0 = outputs[("position_reverse", sc_scale, f_i)].detach()# s2t flow
        sc_3d_f0_matched = self.spatial_transform_warp_sc3d_geoaware_pnet[sc_scale](sc_3d_f0,
                                                of_fi_to_f0)
        # in trinsics is from src img 
        # notice for 2D we provide the scale0 intrinsics, as internally it will gen px_pe_raw with 
        # raw_resolution then subsample the px_pe with 8
        K_for_fi_2D = inputs["K", px_K_scale][:,:3,:3]
        # sanity check if sc_3d_f0_matched contains nan or inf.
        if torch.isnan(sc_3d_f0_matched).any() or torch.isinf(sc_3d_f0_matched).any():
            print('!!!!**************!!!!')
            print('sc_3d_f0_matched contains nan or inf')
            print('!!!!**************!!!!')
            assert 0, f"sc_3d_f0_matched contains nan or inf"
            # sc_3d_f0_matched = torch.nan_to_num(sc_3d_f0_matched, nan=0.0, posinf=0.0, neginf=0.0)

        debug_only = True
        debug_only = False
        if debug_only:
            # scale up sc_3d_f0_matched
            sc_3d_f0_matched = sc_3d_f0_matched * 50.0 

        debug_only = True
        # debug_only = False
        if debug_only:
            print('for frame_id: {} and scale: {}'.format(f_i, sc_scale))
            # print('input scene points 3d f0 matched shape:', sc_3d_f0_matched.shape)
            print('input scene points 3d f0 matched max min z:', sc_3d_f0_matched[:,2,:,:].max(), sc_3d_f0_matched[:,2,:,:].min())
            print('input scene points 3d f0 matched mean x:', sc_3d_f0_matched[:,0,:,:].mean())
            print('input scene points 3d f0 matched mean y:', sc_3d_f0_matched[:,1,:,:].mean())
            print('input scene points 3d f0 matched mean z:', sc_3d_f0_matched[:,2,:,:].mean())
            


        poses_list = self.models["pose"](sc_3d_f0_matched, 
                                        intrinsics_B33=K_for_fi_2D,
                                        sample_level=sc_scale)


        # def scale_down_xyz(pose, scale_xyz = [1.0,1.0,1.0]):
        #     scale_down_x = scale_xyz[0]
        #     scale_down_y = scale_xyz[1]
        #     scale_down_z = scale_xyz[2]
        #     pose[:,:3,0] *= scale_down_x
        #     pose[:,:3,1] *= scale_down_y
        #     pose[:,:3,2] *= scale_down_z
        #     return pose
        # scale_xyz = [1.0,1.0,1.0]
        # poses_list = [scale_down_xyz(pose, scale_xyz) for pose in poses_list]
        
        # poses_list[-1] # extract the last pose with high quality
        return poses_list[-1]


    # @classmethod
    def predict_poses(self, inputs, outputs = {}):
        """Predict poses between input frames for monocular sequences.
        """
        # outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            scale_k = 0
            scale0_camera_intrinsics = {f_i: inputs["K", scale_k] for f_i in self.opt.frame_ids}

            assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
            assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":
                    # position - handle both custom networks and RAFT
                    if self.opt.use_raft_flow:
                        num_flow_udpates = 12
                        # inputs_all = [pose_feats[f_i], pose_feats[0]]
                        # inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]
                        outputs_0_raw = self.models["raft_flow"](pose_feats[0], pose_feats[f_i]) #t2s flow
                        outputs_1_raw = self.models["raft_flow"](pose_feats[f_i], pose_feats[0]) #s2t flow
                        # print('inference optic flow in train()...')

                        outputs_0 = self.reformat_raft_output(num_flow_udpates, outputs_0_raw, hard_detach_OF_grad=self.opt.shared_MF_OF_network)# there will be grad in OF net if shared_MF_OF_network is on, therfore we need to hard detach
                        outputs_1 = self.reformat_raft_output(num_flow_udpates, outputs_1_raw, hard_detach_OF_grad=self.opt.shared_MF_OF_network)# there will be grad in OF net if shared_MF_OF_network is on, therfore we need to hard detach
                        # RAFT expects separate image inputs, not concatenated features
                    else:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                        inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                        # # Original custom networks
                        # position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1)) if not self.opt.shared_MF_OF_network \
                        #     else self.models["position_encoder"](torch.cat(inputs_all, 1)).detach()
                        # position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1)) if not self.opt.shared_MF_OF_network \
                        #     else self.models["position_encoder"](torch.cat(inputs_all_reverse, 1)).detach()
                        
                        position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                        if self.opt.shared_MF_OF_network:
                            position_inputs = [item.detach() for item in position_inputs]
                            position_inputs_reverse = [item.detach() for item in position_inputs_reverse]

                        outputs_0 = self.models["position"](position_inputs)
                        outputs_1 = self.models["position"](position_inputs_reverse)
                        if self.opt.shared_MF_OF_network:
                            # detach the grad in outputs_0 and outputs_1
                            outputs_0 = {k: v.detach() for k, v in outputs_0.items()}
                            outputs_1 = {k: v.detach() for k, v in outputs_1.items()}

                    for scale in self.opt.scales:

                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]# no grad anyway due to freeze OF net
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)])
                    
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)],  outputs[("occu_map_backward", scale, f_i)]= self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(outputs[("position", "high", scale, f_i)],
                                                                                                          outputs[("position_reverse", "high", scale, f_i)])


                    def disp_to_sc_3d(inputs, outputs, scale, ret_mutilscale, scale_depth = 1.0):
                        disp = outputs[("disp", scale)]# for tgt frame
                        B, _, H, W = disp.shape
                        assert isinstance(ret_mutilscale, bool), "ret_mutilscale must be a bool"
                        if not ret_mutilscale:
                            assert 0, f'ret_mutilscale must be True'
                            disp = F.interpolate(
                                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) #muti-scale
                        print('min depth', torch.min(depth))
                        print('max depth', torch.max(depth))
                        depth = depth * scale_depth

                        # pts_3d = depth_to_3d(depth, intrinsics)
                        
                        from layers import BackprojectDepth
                        cam_points = self.backproject_depth[scale](
                            depth, inputs[("inv_K", scale)])
                        
                        
                        print('Used cam_points shape:')
                        print(cam_points.shape)
                        print('Used cam_points max min z:')
                        print(cam_points[:,2,:].max())
                        print(cam_points[:,2,:].min())
                        print('Used cam_points mean x:')
                        print(cam_points[:,0,:].mean())
                        print('Used cam_points mean y:')
                        print(cam_points[:,1,:].mean())
                        print('Used cam_points mean z:')
                        print(cam_points[:,2,:].mean())

                        # reshape back to B, 3, H, W
                        print('cam_points shape:', cam_points.shape)
                        cam_points = cam_points.contiguous().reshape(B, 4, H, W)

                        return cam_points, depth                       

                    if self.opt.pose_model_type == "geoaware_pnet":
                        from layers import disp_to_depth, depth_to_3d
                        # we need estimated t2s
                        # px_K_scale = 0
                        # sc_scale = 3 # lowest 

                        px_K_scale = 0 # 2d alwasy use high resolution matched with 8 grid
                        sc_scales = [3] # lowest 
                        # we need pose tgt2src, ie: pose2to1, i.e the pose2 in breif in reloc3r model.
                        for sc_scale in self.opt.scales:
                        # for sc_scale in sc_scales:
                            # outputs[("cam_T_cam", 0, f_i)] = compute_pose_per_scale(inputs, outputs, sc_scale, px_K_scale)
                            outputs[("cam_T_cam", sc_scale, f_i)] = self.compute_pose_per_scale_geoaware_pnet(inputs, outputs, sc_scale, px_K_scale, f_i)
                            
                            # print('For geoaware: we are able to get multi level resoutluion poses!')
                            debug_only = True
                            debug_only = False
                            if debug_only:
                                print('level of the poses for frame_id: {} and scale: {}'.format(f_i, sc_scale))
                                print(outputs[("cam_T_cam", sc_scale, f_i)][:,:3,3])

                    elif self.opt.pose_model_type == "pcrnet":
                        
                        for scale in self.opt.scales:
                            depth_fi = outputs[("depth", f_i, scale)].detach()# seems not matters much
                            depth_f0 = outputs[("depth", 0, scale)].detach()
                            # we are in fact use high res depth
                            inv_K_fi = inputs[("inv_K", 0)]
                            inv_K_f0 = inputs[("inv_K", 0)]

                            # we are in fact use high res depth
                            cam_points_fi = self.backproject_depth[0](depth_fi, inv_K_fi)
                            cam_points_f0 = self.backproject_depth[0](depth_f0, inv_K_f0) \
                                if outputs.get(tuple(["cam_points", 0, scale])) is None else outputs[("cam_points", 0, scale)]
                            
                            # # training is getting unstable--it is possible to ignore points where the x/y magnitude are too small?
                            # print('the range of magnitude of cam_points_fi:', torch.norm(cam_points_fi[:,:2,...], dim=1).min(), torch.norm(cam_points_fi[:,:2,...], dim=1).max())
                            # print('the range of magnitude of cam_points_f0:', torch.norm(cam_points_f0[:,:2,...], dim=1).min(), torch.norm(cam_points_f0[:,:2,...], dim=1).max())
                            # B, _, _ = cam_points_fi.shape
                            # cam_points_fi = cam_points_fi[torch.norm(cam_points_fi[:,:2,...].detach(), dim=1).repeat(1, 4, 1)>0.5].view(B, 4, -1)
                            # cam_points_f0 = cam_points_f0[torch.norm(cam_points_f0[:,:2,...].detach(), dim=1).repeat(1, 4, 1)>0.5].view(B, 4, -1)
                            # print('after ignoring', torch.norm(cam_points_fi[:,:2,...], dim=1).min(), torch.norm(cam_points_fi[:,:2,...], dim=1).max())
                            # print('after ignoring', torch.norm(cam_points_f0[:,:2,...], dim=1).min(), torch.norm(cam_points_f0[:,:2,...], dim=1).max())



                            debug_only = True
                            debug_only = False
                            if debug_only:
                                # seems to be critical ot be big: bigger than 100
                                cam_points_fi = 100*cam_points_fi
                                cam_points_f0 = 100*cam_points_f0



                            outputs[("cam_points", f_i, scale)] = cam_points_fi
                            if outputs.get(tuple(["cam_points", 0, scale])) is None:
                                outputs[("cam_points", 0, scale)] = cam_points_f0 

                            pcd_fi = cam_points_fi[:,:3,...].permute(0, 2, 1)
                            pcd_f0 = cam_points_f0[:,:3,...].permute(0, 2, 1)

                            # B, _, _ = pcd_fi.shape
                            # debug_only = True
                            # debug_only = False
                            # if debug_only:
                            #     # ignore pts where the xy magnitude are too small
                            #     # print('process scale {} frame_id {}'.format(scale, f_i))
                            #     # print('the range of magnitude of pcd_fi:', torch.norm(pcd_fi[:,:,:2], dim=-1).min(), torch.norm(pcd_fi[:,:,:2], dim=-1).max())
                            #     # print('the range of magnitude of pcd_f0:', torch.norm(pcd_f0[:,:,:2], dim=-1).min(), torch.norm(pcd_f0[:,:,:2], dim=-1).max())
                            #     pcd_fi = pcd_fi[torch.norm(pcd_fi[:,:,:2], dim=-1)>0.01].view(B, -1, 3)
                            #     pcd_f0 = pcd_f0[torch.norm(pcd_f0[:,:,:2], dim=-1)>0.01].view(B, -1, 3)
                            #     # print('after ignoring pcd_fi shape:', pcd_fi.shape)
                            #     # print('after ignoring', torch.norm(pcd_fi[:,:,:2], dim=-1).min(), torch.norm(pcd_fi[:,:,:2], dim=-1).max())
                            #     # print('after ignoring', torch.norm(pcd_f0[:,:,:2], dim=-1).min(), torch.norm(pcd_f0[:,:,:2], dim=-1).max())
                            
                            # subtraction: 
                            pcd_fi = pcd_fi - torch.mean(pcd_fi, dim=1, keepdim=True).detach()
                            pcd_f0 = pcd_f0 - torch.mean(pcd_f0, dim=1, keepdim=True).detach()
                             

                            outputs[("cam_T_cam", scale, f_i)] = self.models["pose"](pcd_fi, pcd_f0, 
                                                                                     max_iteration = self.opt.pcrnet_max_iteration)['est_T']

                            # print('check if requires_grad is on cam_T_cam')
                            # print(pcd_fi.requires_grad)
                            # print(pcd_f0.requires_grad)
                            # print(outputs[("cam_T_cam", scale, f_i)].requires_grad)
                            # assert outputs[("cam_T_cam", scale, f_i)].requires_grad , "cam_T_cam must have requires_grad"

                    elif self.opt.pose_model_type == "diffposer_epropnp":
                        from layers import disp_to_depth, depth_to_3d
                        # sc_scale = 3 # lowest 
                        # px_K_scale = 3
                        sc_scale = 0 # lowest 
                        px_K_scale = 0
                        assert sc_scale == px_K_scale, "sc_scale and px_K_scale must be the same"
                        k_for_f0_3D = inputs["K", sc_scale][:,:3,:3]
                        scale_depth = 1.0
                        # sc_3d_f0, _ = disp_to_sc_3d(
                        #                             outputs[("disp", sc_scale)], 
                        #                             k_for_f0_3D,
                        #                             ret_mutilscale=True,
                        #                             scale_depth=scale_depth,
                        #                             )
                        sc_3d_f0, _ = disp_to_sc_3d(
                                                    inputs,
                                                    outputs,
                                                    sc_scale,
                                                    ret_mutilscale=True,
                                                    scale_depth=scale_depth,
                                                    )
                        sc_3d_f0 = sc_3d_f0[:,0:3,:,:]
                        print('sc_3d_f0 shape:', sc_3d_f0.shape)
                        # sc_3d_f0_conf = outputs[("motion_mask_backward", sc_scale, f_i)]
                        sc_3d_f0_conf = torch.ones_like(sc_3d_f0[:,0:1,:,:])

                        of_fi_to_f0 = outputs[("position", sc_scale, f_i)].detach() 
                        sc_3d_f0_matched = self.spatial_transform_used_to_warp_sc_3d(sc_3d_f0,
                                                                 of_fi_to_f0)
                        sc_3d_f0_matched_conf = self.spatial_transform_used_to_warp_sc_3d(sc_3d_f0_conf,
                                                                 of_fi_to_f0)
                        
                        # print('Used cam_points shape:')
                        # print(sc_3d_f0.shape)
                        # print('Used cam_points max min z:')
                        # print(sc_3d_f0[:,2,:,:].max())
                        # print(sc_3d_f0[:,2,:,:].min())
                        # print(sc_3d_f0_matched[:,2,:,:].max())
                        # print(sc_3d_f0_matched[:,2,:,:].min())

                        # print('Used cam_points mean x:')
                        # print(sc_3d_f0[:,0,:,:].mean())
                        # print(sc_3d_f0_matched[:,0,:,:].mean())
                        # print('Used cam_points mean y:')
                        # print(sc_3d_f0[:,1,:,:].mean())
                        # print(sc_3d_f0_matched[:,1,:,:].mean())
                        # print('Used cam_points mean z:')
                        # print(sc_3d_f0[:,2,:,:].mean())                        
                        # print(sc_3d_f0_matched[:,2,:,:].mean())
                        # assert 0, f'above is problematic for sc pts'
                        
                        K_for_fi_2D = inputs["K", px_K_scale][:,:3,:3]
                        if torch.isnan(sc_3d_f0_matched).any() or torch.isinf(sc_3d_f0_matched).any():
                            print('!!!!**************!!!!')
                            print('sc_3d_f0_matched contains nan or inf')
                            print('!!!!**************!!!!')
                            assert 0, f"sc_3d_f0_matched contains nan or inf"
                            # sc_3d_f0_matched = torch.nan_to_num(sc_3d_f0_matched, nan=0.0, posinf=0.0, neginf=0.0)
                        # solve the pose
                        
                        B, _, _, _ = sc_3d_f0.shape
                        # H, W = int(self.opt.height / (2**sc_scale)), int(self.opt.width / (2**sc_scale))
                        # self._init_base_grid(H=H, W=W, device=k_for_f0_3D.device)

    
                        print('sc_3d_f0_matched shape:', sc_3d_f0_matched.shape)
                        print('sc_3d_f0_matched_conf shape:', sc_3d_f0_matched_conf.shape)
                        print('K_for_fi_2D shape:', K_for_fi_2D.shape)
                        # t2s mask
                        x3d_2, x2d_2, w2d_2, cam_mats_2, pose_init_2 = self.prepare_epropnp_input(
                            sc_3d_f0_matched.permute(0,2,3,1),
                            sc_3d_f0_matched_conf.permute(0,2,3,1),
                            K_for_fi_2D, 
                            B, 
                            int(self.opt.height / (2**sc_scale)), 
                            int(self.opt.width / (2**sc_scale)), 
                            self.device)
                        print('x3d_2 shape:', x3d_2.shape)
                        print('x2d_2 shape:', x2d_2.shape)
                        print('w2d_2 shape:', w2d_2.shape)
                        # print('cam_mats_2 shape:', cam_mats_2.shape)
                        # print('pose_init_2 shape:', pose_init_2.shape)
                        
                        # print the max and min dpeth in x3d_2
                        print('ori x3d_2 min z:', x3d_2[:, :, 2].min())
                        print('ori x3d_2 max z:', x3d_2[:, :, 2].max())
                        print('ori x3d_2 mean x:', x3d_2[:, :, 0].mean())
                        print('ori x3d_2 mean y:', x3d_2[:, :, 1].mean())
                        print('ori x3d_2 mean z:', x3d_2[:, :, 2].mean())

                        # assert 0, f'there is issuse for the x3d_2:  need to be zero for mean_x,mean_y?'


                        #setting up
                        opt_quat_format='wxyz'
                        soft_clamp_quat = True
                        max_angle_rad = 0.1
                        # max_angle_rad = 1

                        _, _, pose_opt_plus_2, _, pose_sample_logweights_2, cost_tgt_2, norm_factor_2 = self.epropnp_pose_head(
                                                                                                        x3d_2, x2d_2, w2d_2, 
                                                                                                        cam_mats_2, 
                                                                                                        pose_init_2,
                                                                                                        )
                        from reloc3r_uni.utils.epropnp_utils import xyz_quat_to_matrix_kornia
                        # pose2['pose'] = pose_opt_plus_2
                        pose2 = {}
                        pose2['pose'] = xyz_quat_to_matrix_kornia(pose_opt_plus_2, 
                                                                quat_format=opt_quat_format,
                                                                soft_clamp_quat=soft_clamp_quat,
                                                                max_angle_rad=max_angle_rad)
                        pose2['pose_sample_logweights'] = pose_sample_logweights_2
                        pose2['cost_tgt'] = cost_tgt_2
                        pose2['norm_factor'] = norm_factor_2
                        print('pose2 translation from epropnp:', pose2['pose'][:,:3,3])

                        outputs[("cam_T_cam", 0, f_i)] = pose2["pose"] # we need pose tgt2src, ie: pose2to1, i.e the pose2 in breif in reloc3r model.

                    elif self.opt.pose_model_type == 'separate_resnet':
                        # tran scale from af sfmlearner: [-5.2260e-06, -1.7639e-05, -3.6466e-04]
                        pose_inputs = [self.models["pose_encoder"](torch.cat([pose_feats[f_i], pose_feats[0]], 1))]

                        axisangle, translation = self.models["pose"](pose_inputs)

                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0])# only extract the 0 
                    elif self.opt.pose_model_type == "posetr_net":
                        pass
                        view1 = {'img':inputs["color_aug", f_i, 0]}
                        view2 = {'img':inputs["color_aug", 0, 0]}
                        # print('view1 shape:', view1['img'].shape)
                        # print('view2 shape:', view2['img'].shape)
                        _ , pose2 = self.models["pose"](view1,view2)
                        outputs[("cam_T_cam", 0, f_i)] = pose2["pose"] # we need pose tgt2src, ie: pose2to1, i.e the pose2 in breif in reloc3r model.

                    elif self.opt.pose_model_type in ["endofast3r",
                                                      'endofast3r_pose_trained_dbg',
                                                      "uni_reloc3r", 
                                                      ]:
                        # trans scale for endofast3r: [-2.4887e-05, -1.8644e-05,  1.3773e-03]
                        
                        resized_img1, adapted_K1 = prepare_images(inputs["color_aug", f_i, 0],self.device, size = 512, Ks=scale0_camera_intrinsics[f_i])
                        resized_img2, adapted_K2 = prepare_images(inputs["color_aug", 0, 0], self.device, size = 512, Ks=scale0_camera_intrinsics[0])
                        view1 = {'img':resized_img1, 'camera_intrinsics':adapted_K1}
                        view2 = {'img':resized_img2, 'camera_intrinsics':adapted_K2}

                        # compute mean_cam_center; we infact compute the mean map center for the 3 frames
                        


                        # # notice we save pose2to1 as usually saved by reloc3r/fast3r/mvp3r; dares saved rel pose1to2
                        _ , pose2 = self.models["pose"](view1,view2)
                        # notice we save pose2to1 as usually saved by reloc3r/fast3r/mvp3r; dares saved rel pose1to2
                        outputs[("cam_T_cam", 0, f_i)] = pose2["pose"] # we need pose tgt2src, ie: pose2to1, i.e the pose2 in breif in reloc3r model.

                        # debug with GT pose
                        if self.opt.gt_metric_rel_pose_as_estimates_debug:
                            print('Debug: use GT metric rel pose as estimates!!!')
                            gt_tgt_abs_poses = inputs[("gt_c2w_poses", 0)]  # (B, 4, 4)
                            gt_src_abs_poses = inputs[("gt_c2w_poses", f_i)]  # (B, 4, 4)
                            gt_tgt2src_rel_poses = torch.inverse(gt_src_abs_poses) @ gt_tgt_abs_poses
                            outputs[("cam_T_cam", 0, f_i)] = gt_tgt2src_rel_poses

 
                    else:
                        assert 0, f'{self.opt.pose_model_type} is not implemented'
        return outputs


    def predict_motion_flow_with_MF_net(self, inputs, outputs):
        """Predict poses between input frames for monocular sequences.
        """
        if self.num_pose_frames == 2:
            # only pose_flow corrected: for self.opt.frame_ids[1:]
            # print('Aviable keys in outputs:')
            # print(outputs.keys())
            img_feats = {f_i: outputs[("color", f_i, 0)].detach() for f_i in self.opt.frame_ids[1:]}# only pose_flow corrected 
            img_feats[self.opt.frame_ids[0]] = inputs["color_aug", self.opt.frame_ids[0], 0] # gt tgt img

            assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
            assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":

                    # position - handle both custom networks and RAFT
                    if self.opt.use_raft_flow:
                        # prepare to get motion flow (tgt_motion_status_2_src_motion_status): tgt(at tgt motion status) -> tgt_warped_from_src_with_PF(at src motion status)
                        # as later, we warp src to tgt with PF(s2t) +MF (s2t) 
                        inputs_all = [img_feats[0], img_feats[f_i]]# we need to place gt_tgt_in the front
                        # bug fix: we saved s2t_motion_flow in 'motion_flow'---all the enable mutual is wrong!
                        # inputs_all = [img_feats[f_i], img_feats[0]]# we need to place gt_tgt_in the front
                        if self.opt.enable_mutual_motion:
                            inputs_all_reverse = [img_feats[f_i], img_feats[0]]# src_motion_status_2_tgt_motion_status


                        num_flow_udpates = 12
                        outputs_0_raw = self.models["motion_raft_flow"](img_feats[0], img_feats[f_i])
                        # RAFT expects separate image inputs, not concatenated features
                        # print('inference motion flow...')
                        outputs_0 = self.reformat_raft_output(num_flow_udpates, outputs_0_raw)

                        if self.opt.enable_mutual_motion:
                            outputs_1_raw = self.models["motion_raft_flow"](img_feats[f_i], img_feats[0])
                            outputs_1 = self.reformat_raft_output(num_flow_udpates, outputs_1_raw)
                    else:
                        #we want to obtain t2s motion flow, when reuse author network.
                        # the orignal network is trained to be like, given input (1,2), estimated flow(2to1)
                            
                        inputs_all = [img_feats[f_i], img_feats[0]]
                        if self.opt.enable_mutual_motion:
                            inputs_all_reverse = [img_feats[0], img_feats[f_i]]
                            
                        # Original custom networks
                        position_inputs = self.models["motion_position_encoder"](torch.cat(inputs_all, 1))
                        outputs_0 = self.models["motion_position"](position_inputs)

                        if self.opt.enable_mutual_motion:
                            position_inputs_reverse = self.models["motion_position_encoder"](torch.cat(inputs_all_reverse, 1))
                            outputs_1 = self.models["motion_position"](position_inputs_reverse)

                    solve_MF_issue_from_root_debug = True
                    Solve_MF_issue_from_root_debug = False
                    for scale in self.opt.scales:
                        # check if there is nan or inf in outputs_0[("position", scale)]
                        if torch.isnan(outputs_0[("position", scale)]).any() or torch.isinf(outputs_0[("position", scale)]).any():
                            print('!!!!**************!!!!')
                            # count the number of nan or inf
                            num_nan = torch.isnan(outputs_0[("position", scale)]).sum()
                            num_inf = torch.isinf(outputs_0[("position", scale)]).sum()
                            print(f'Motion_flow outputs_0[("position", scale)] contains nan or inf: {num_nan} nan, {num_inf} inf')
                            print('!!!!**************!!!!')
                            assert 0, f"Motion_flow outputs_0[('position', scale)] contains nan or inf: {num_nan} nan, {num_inf} inf"
                            # outputs_0[("position", scale)] = torch.nan_to_num(outputs_0[("position", scale)], nan=0.0, posinf=0.0, neginf=0.0)
                        outputs[("motion_flow", f_i, scale)] = outputs_0[("position", scale)]# saves t2s flow # no grad anyway due to freeze OF net
                        # solve MF ref issue from source
                        # if solve_MF_issue_from_root_debug:
                        #     mf_ref_tgt = outputs[("motion_flow", f_i, scale)]
                        #     mf_ref_color = self.spatial_transform(
                        #         mf_ref_tgt,
                        #         outputs[("pose_flow", "high", f_i, scale)], #infact: tgt2src
                        #         # outputs[("pose_flow", "high", f_i, scale)].detach(), 
                        #         )
                        #     outputs[("motion_flow", f_i, scale)] = mf_ref_color

                        outputs[("motion_flow", "high", f_i, scale)] = F.interpolate(
                            outputs[("motion_flow", f_i, scale)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        # spatial transform
                        # A: perferred(nosiy depth, otherwise good) --at least known to get pose properly!
                        # infact--no any constraint--below formula always exist no matter whatever pose_flow.
                        # outputs[("color_MotionCorrected", f_i, scale)] = self.spatial_transform(
                        #     inputs[("color", f_i, 0)],
                        #     outputs[("motion_flow", "high", f_i, scale)].detach() + outputs[("pose_flow", "high", f_i, scale)], 
                        #     )
                         # A'': motion_flow: tgt2color    pose_flow: color2src
                         # while we need OF: tgt2src
                        # outputs[("color_MotionCorrected", f_i, scale)] = self.spatial_transform(
                        #     inputs[("color", f_i, 0)],
                        #     outputs[("motion_flow", "high", f_i, scale)] + outputs[("pose_flow", "high", f_i, scale)], 
                        #     )
                        # B'': robust_color_corrected: _no_noisy (wrong but good trend to let pose flow learn properly--seem to bring curve downer)More robust compared to the above: depth get less affected?
                        # it also significantly affect depth:  it will enforce dy area to be extremely deep---reasonable. but not correct
                        # here the major aim is still supervise pose_flow properly in a full frame regime, we already indirectly supervised MF by reused the OF net.
                        outputs[("color_MotionCorrected", f_i, scale)] = self.spatial_transform(
                            outputs[("color", f_i, 0)],
                            outputs[("motion_flow", "high", f_i, scale)].detach(), 
                            )       

                        # A''': perferred(nosiy depth, otherwise good) --at least known to get pose properly!
                        # infact--no any constraint--below formula always exist no matter whatever pose_flow.
                        # mf_ref_color = outputs[("motion_flow", "high", f_i, scale)].detach()
                        # # mf_ref_src = # color + s2c
                        # mf_ref_src = self.spatial_transform(
                        #     mf_ref_color,
                        #     outputs[("pose_flow", "high", f_i, scale)], 
                        #     )
                        # outputs[("color_MotionCorrected", f_i, scale)] = self.spatial_transform(
                        #     inputs[("color", f_i, 0)],
                        #     # below is wrong!: the ref frame changes of motino
                        #     # outputs[("motion_flow", "high", f_i, scale)].detach() + outputs[("pose_flow", "high", f_i, scale)], 
                        #     mf_ref_src + outputs[("pose_flow", "high", f_i, scale)], 
                        #     )

                        # A''''
                        # outputs[("color_MotionCorrected", f_i, scale)] = self.spatial_transform(
                        #     inputs[("color", f_i, 0)],
                        #     # outputs[("motion_flow", "high", f_i, scale)] + outputs[("pose_flow", "high", f_i, scale)].detach(), 
                        #     # quick tmp fix--later deteaoe;d corrected. from define them, usage funtion.
                        #     outputs[("motion_flow", "high", f_i, scale)] - outputs[("pose_flow", "high", f_i, scale)].detach(), 
                        #     )

                        #X: A': not known---affect pose_flow indirecly from motion_flow(computed from color img and real_img)
                        # outputs[("color_MotionCorrected", f_i, scale)] = self.spatial_transform(
                        #     inputs[("color", f_i, 0)],
                        #     outputs[("motion_flow", "high", f_i, scale)] + outputs[("pose_flow", "high", f_i, scale)].detach(), 
                        #     )
 


                        
                 
  

                        if self.opt.enable_mutual_motion:
                            outputs[("motion_flow_s2t", f_i, scale)] = outputs_1[("position", scale)]# no grad anyway due to freeze OF net 
                            if solve_MF_issue_from_root_debug:
                                pass

                            
                            outputs[("motion_flow_s2t", "high", f_i, scale)] = F.interpolate(
                                outputs[("motion_flow_s2t", f_i, scale)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                    
                        # if scale == 0:
                        # extend for muti level despite we only use level 0 for supervision
                        if self.opt.use_soft_motion_mask:
                            #generate from optic_flow and pose_flow
                            outputs[("motion_mask_backward", scale, f_i)] = self.get_motion_mask_soft(
                                optic_flow = outputs[("position", "high", scale, f_i)],
                                pose_flow = outputs[("pose_flow", "high", f_i, scale)],
                                detach=(not self.opt.enable_grad_flow_motion_mask), 
                                thre_px=self.opt.motion_mask_thre_px)
                            if self.opt.enable_mutual_motion:
                                outputs[("motion_mask_s2t_backward", scale, f_i)] = self.get_motion_mask_soft(
                                    optic_flow = outputs[("position_reverse","high", scale, f_i)],
                                    pose_flow = outputs[("pose_flow_s2t", "high", f_i, scale)],
                                    detach=(not self.opt.enable_grad_flow_motion_mask), 
                                    thre_px=self.opt.motion_mask_thre_px)

                        else:
                            # generate directly from thresholding motion flow
                            outputs[("motion_mask_backward", scale, f_i)] = self.get_motion_mask(
                                outputs[("motion_flow", "high", f_i, scale)],
                                detach=(not self.opt.enable_grad_flow_motion_mask), 
                                thre_px=self.opt.motion_mask_thre_px)
                            if self.opt.enable_mutual_motion:
                                outputs[("motion_mask_s2t_backward", scale, f_i)] = self.get_motion_mask(
                                    outputs[("motion_flow_s2t", "high", f_i, scale)],
                                    detach=(not self.opt.enable_grad_flow_motion_mask), 
                                    thre_px=self.opt.motion_mask_thre_px)
 

        return outputs

    def get_motion_mask_soft(self, optic_flow, pose_flow, detach = True, thre_px = 3):
        '''
        static area will be set to 1, motion area will be set to 0.
        Therefor can be directly used for validness when supervise.
        '''
        # Use learnable threshold if enabled, otherwise use the provided threshold
        if self.opt.enable_learned_motion_mask_thre_px and self.learned_motion_mask_thre_px is not None:
            threshold = self.learned_motion_mask_thre_px
        else:
            threshold = thre_px
            
        # obtain motion mask from motion_flow:
        if detach:
            motion_mask = get_texu_mask(optic_flow.detach(), 
                                        pose_flow.detach(),
                                        ret_conf = True)
        else:
            motion_mask = get_texu_mask(optic_flow, 
                                        pose_flow,
                                        ret_conf = True)

        # use hard code thershoulding of motion_flow:
        # l2 norm of flow_vector is longer than 3 px
        # motion_mask_v2 = (outputs[("motion_flow", frame_id, 0)].norm(dim=1, keepdim=True) > 3).detach()
        # print('motion_mask_requires_grad:')
        # print(motion_mask.requires_grad)
        # print('motion_flow.requires_grad:')
        # print(motion_flow.requires_grad)
        
        return motion_mask

    #compute and regisered the masks: can be used to masked out loss 
    def get_motion_mask(self, motion_flow, detach = True, thre_px = 3):
        '''
        static area will be set to 1, motion area will be set to 0.
        Therefor can be directly used for validness when supervise.
        '''
        # Use learnable threshold if enabled, otherwise use the provided threshold
        if self.opt.enable_learned_motion_mask_thre_px and self.learned_motion_mask_thre_px is not None:
            threshold = self.learned_motion_mask_thre_px
        else:
            threshold = thre_px
            
        # obtain motion mask from motion_flow:
        if detach:
            # motion_mask = get_texu_mask(outputs[("position", 0, frame_id)].detach(), 
                                        # outputs[("pose_flow", "high", frame_id, 0)].detach())
            motion_mask = (motion_flow.norm(dim=1, keepdim=True) <= threshold).detach()
            motion_mask = motion_mask.float()            
        else:
            # motion_mask = get_texu_mask(outputs[("position", 0, frame_id)], 
            #                             outputs[("pose_flow", "high", frame_id, 0)])
            mask_norm = motion_flow.norm(dim=1, keepdim=True)
            # mask_hard = (mask_norm > 0.5).float()# no grad

            mask_hard = (mask_norm <= threshold).float()# no grad
            
            motion_mask = mask_hard + mask_norm - mask_norm.detach() #still binary but diffirentiable
        
        return motion_mask

    def get_current_motion_mask_threshold(self):
        """Get the current motion mask threshold value"""
        if self.opt.enable_learned_motion_mask_thre_px and self.learned_motion_mask_thre_px is not None:
            return self.learned_motion_mask_thre_px.item()
        else:
            return self.opt.motion_mask_thre_px

    def gen_sample_and_pose_flow(self, inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, tgt_frame_id = 0, frame_id = None):
        '''
        only udpate the samples and pose_flow:
        extend with samples_s2t pose_flow_s2t: controler by given swappred input id
        '''
        
        if tgt_frame_id == 0:
            assert frame_id in self.opt.frame_ids[1:], f'frame_id {frame_id} is not in self.opt.frame_ids[1:]'
            compute_tgt2src_sampling = True
        else:
            assert frame_id == 0, f'src_frame_id {frame_id} is not 0'
            assert tgt_frame_id in self.opt.frame_ids[1:], f'tgt_frame_id {tgt_frame_id} is not in self.opt.frame_ids[1:]'
            compute_tgt2src_sampling = False

        if frame_id == "s":
            T = inputs["stereo_T"]
        else:
            if compute_tgt2src_sampling:
                # print('/////!compute tgt2src')
                T = outputs[("cam_T_cam", 0, frame_id)]
            else:
                # print('/////!waring..to be optimal later')
                T = outputs[("cam_T_cam", 0, tgt_frame_id)]
                T = torch.inverse(T) 
        



        if self.opt.zero_pose_debug:
            T = torch.eye(4).to(self.device).repeat(T.shape[0], 1, 1)

        # if self.opt.pose_model_type == "posecnn":
        #     assert 0, 'posecnn is not supported for mutual motion'

        #     axisangle = outputs[("axisangle", 0, frame_id)]
        #     translation = outputs[("translation", 0, frame_id)]

        #     inv_depth = 1 / outputs[("depth", tgt_frame_id, scale)]
        #     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

        #     T = transformation_from_parameters(
        #         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
        

        # print('////*****Gen sample and pose flow based in inv_K and depth****/////')
        # print('max depth:', outputs[("depth", tgt_frame_id, scale)].max())
        # print('min depth:', outputs[("depth", tgt_frame_id, scale)].min())
        # print('mean depth:', outputs[("depth", tgt_frame_id, scale)].mean())
        # print('used K:')
        # print(inputs[("K", source_scale)])
        
        # we used the same scale mesh_grid as 'depth' are high res depth across all 'scales'
        if compute_tgt2src_sampling:
            
            cam_points = self.backproject_depth[source_scale](
                outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])# 3D pts  B 4 N
            if outputs.get(tuple(["cam_points", 0, scale])) is None:
                outputs[("cam_points", 0, scale)] = cam_points
            else:
                # print(f'////////////cam points already computed ...pcrnet?{frame_id}, {scale}')
                # print('logged cam points shape:', outputs[("cam_points", 0, scale)].shape)
                # print('recomputed cam points shape:', cam_points.shape)

                pass
                # assert (outputs[("cam_points", 0, scale)] == cam_points).all(), f'cam_points mismatch: {outputs[("cam_points", 0, scale)]} != {cam_points}'
        else:
            cam_points = self.backproject_depth[source_scale](
                outputs[("depth", tgt_frame_id, scale)], inputs[("inv_K", source_scale)])# 3D pts
            if outputs.get(tuple(["cam_points", tgt_frame_id, scale])) is None:
                outputs[("cam_points", tgt_frame_id, scale)] = cam_points
            else:
                pass
                # print(f'////////////cam points already computed ...pcrnet?{frame_id}, {scale}')
                # assert (outputs[("cam_points", tgt_frame_id, scale)] == cam_points).all(), f'cam_points mismatch: {outputs[("cam_points", tgt_frame_id, scale)]} != {cam_points}'


        # print('cam_points.shape:')
        # print(cam_points.shape)
        # Project3D: it saves values in range [-1,1] for direct sampling
        # pix_coords saves values in range [-1,1]
        # print('compute pix_coords')
        debug_only = True
        debug_only = False
        if frame_id == -1 and debug_only:
            print('////////detailed cam points for frame_id', frame_id, 'scale', scale, '////////////////')
            # print('used K:')
            # print(inputs[("K", source_scale)])
            # print('used T:')
            # print(T)
            print('Used cam_points shape:')
            print(cam_points.shape)
            print('Used cam_points max min z:')
            print(cam_points[:,2,:].max())
            print(cam_points[:,2,:].min())
            print('Used cam_points mean x:')
            print(cam_points[:,0,:].mean())
            print('Used cam_points mean y:')
            print(cam_points[:,1,:].mean())
            print('Used cam_points mean z:')
            print(cam_points[:,2,:].mean())
            # print('Used cam_points mean 4:')
            # print(cam_points[:,3,:].mean())
        pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)# 2D pxs; T: f0 -> f1  f0->f-1
        

        debug_only = True
        debug_only = False
        if debug_only:
            print('pix_coords shape:')
            print(pix_coords.shape)
            print('pix_coords max min:')
            print(pix_coords.max())
            # using tanh to clamp pix_coords
            # pix_coords = torch.tanh(pix_coords)
            print('pix_coords max min after tanh:')
            print(pix_coords.max())
            print(pix_coords.min())
            # clamp pix_coords to be in range [-0.1,0.1]
            # pix_coords = torch.clamp(pix_coords, -0.02, 0.02)

            # reg the position wise sampling.

            # loc_per_pixel = torch.meshgrid(torch.linspace(-1, 1, self.opt.width), torch.linspace(-1, 1, self.opt.height), indexing='ij')
            # loc_per_pixel = torch.stack(loc_per_pixel, dim=-1)

            # Create meshgrid with correct coordinate order
            x_coords = torch.linspace(-1, 1, self.opt.width)   # Width dimension
            y_coords = torch.linspace(-1, 1, self.opt.height)  # Height dimension

            # Create meshgrid - this gives (height, width) for each coordinate
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Stack to get (height, width, 2) where [..., 0] = x, [..., 1] = y
            loc_per_pixel = torch.stack([x_grid, y_grid], dim=-1)

            loc_per_pixel = loc_per_pixel.unsqueeze(0)
            loc_per_pixel = loc_per_pixel.repeat(pix_coords.shape[0], 1, 1, 1)#.permute(0, 3, 1, 2)
            loc_per_pixel = loc_per_pixel.to(self.device)

            pix_delta_raw = pix_coords - loc_per_pixel
            # pix_delta = torch.clamp(pix_delta_raw, -0.0, 0.0)
            # pix_delta = torch.clamp(pix_delta_raw, -0.01, 0.01) # a good value
            pix_delta = torch.clamp(pix_delta_raw, -0.05, 0.05) # a good value
            # pix_delta = torch.tanh(pix_delta_raw)*0.05
            
            # pix_delta = torch.clamp(pix_delta_raw, -0.2, 0.2)
            pix_coords = loc_per_pixel + pix_delta

        
        if compute_tgt2src_sampling:
            outputs[("sample", frame_id, scale)] = pix_coords # b h w 2
        else:
            outputs[("sample_s2t", tgt_frame_id, scale)] = pix_coords # b h w 2

        debug_only = True
        debug_only = False
        if debug_only:
            if frame_id == -1:
                print('normed pix_coords l2_norm max min:')
                print(pix_coords.norm(dim=-1, keepdim=True).max())
                print(pix_coords.norm(dim=-1, keepdim=True).min())

        # generate pose_flow from pix_coords
        # norm_width_source_scale = self.project_3d[scale].width
        # norm_height_source_scale = self.project_3d[scale].height

        #fix an issue
        norm_width_source_scale = self.project_3d[source_scale].width
        norm_height_source_scale = self.project_3d[source_scale].height

        # print('norm_width_source_scale:')
        # print(norm_width_source_scale)
        # print('norm_height_source_scale:')
        # print(norm_height_source_scale)
        # compute the raw_unit value pix_coords_raw from pix_coords, leveraging the fact that pix_coords saves values in range [-1,1]
        if compute_tgt2src_sampling:
            pix_coords_raw = outputs[("sample", frame_id, scale)].clone()#.detach()
        else:
            pix_coords_raw = outputs[("sample_s2t", tgt_frame_id, scale)].clone()#.detach()
        pix_coords_raw = pix_coords_raw * 0.5 + 0.5 # convert to range [0,1]
        # at high resolution
        pix_coords_raw[..., 0] = pix_coords_raw[..., 0] * (norm_width_source_scale - 1)
        pix_coords_raw[..., 1] = pix_coords_raw[..., 1] * (norm_height_source_scale - 1)
        # tgt2src pose flow
        if compute_tgt2src_sampling:
            outputs[("pose_flow", "high", frame_id, scale)] = pix_coords_raw.permute(0, 3, 1, 2) - mesh_gird_high_res # there is grad; B 2 H W 
        else:
            outputs[("pose_flow_s2t", "high", tgt_frame_id, scale)] = pix_coords_raw.permute(0, 3, 1, 2) - mesh_gird_high_res # there is grad; B 2 H W 
        
        return outputs

    # @classmethod
    def generate_motion_flow(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                # If you want to warp the source image src to the target view tgt, you need the flow that maps pixels from the target frame to the source frame, i.e., tgt → src.
                # assert 0,f'use color is correct! not need to use color_motion_corrected! considering the pose_flow now is computed from tgt depth+pose'
                # tgt2src motion flow
                # avoid grad in motion_flow! elsewise the sample_motion_corrected will loss grad at all.
                outputs[("motion_flow","high", frame_id, scale)] = - outputs[("pose_flow", "high", frame_id, scale)].detach() + outputs[("position", "high", scale, frame_id)]#.detach() # 
                # we enable grad in motion_flow, can be latered used to reg motion mask
                # there is no grad in OF anyway in for 2nd statge traning

                # if scale == 0:
                if self.opt.use_soft_motion_mask:
                    outputs[("motion_mask_backward", scale, frame_id)] = self.get_motion_mask_soft(
                                                        outputs[("position", "high", scale, frame_id)], 
                                                            outputs[("pose_flow", "high", frame_id, scale)],
                                                            detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                            thre_px=self.opt.motion_mask_thre_px)
                else:
                    outputs[("motion_mask_backward", scale, frame_id)] = self.get_motion_mask(
                                                                                    #  do not use outputs[("motion_flow", frame_id, 0)], depend on what is set to it, the grad might get lost!
                                                                                    # use below to make sure grad exists. 
                                                                                        - outputs[("pose_flow", "high", frame_id, scale)] + outputs[("position", "high", scale, frame_id)], # there is grad_flow here! differ from motion_flow
                                                                                        detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                                        thre_px=self.opt.motion_mask_thre_px)
                
                if self.opt.enable_mutual_motion:
                    # compute s2t motion_flow and s2t_motion_mask
                    # we enable grad in motion_flow, can be latered used to reg motion mask
                    outputs[("motion_flow_s2t", "high",frame_id, scale)] = - outputs[("pose_flow_s2t", "high", frame_id, scale)] + outputs[("position_reverse", "high", scale, frame_id)]#.detach() # 
                    # if scale == 0:
                    if self.opt.use_soft_motion_mask:
                        outputs[("motion_mask_s2t_backward", scale, frame_id)] = self.get_motion_mask_soft(
                                                    optic_flow = outputs[("position_reverse", "high", scale, frame_id)], 
                                                            pose_flow = outputs[("pose_flow_s2t", "high", frame_id, scale)],
                                                            detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                            thre_px=self.opt.motion_mask_thre_px)
                    else:
                        # static area will be set to 1
                        outputs[("motion_mask_s2t_backward", scale, frame_id)] = self.get_motion_mask(
                                                                                            # outputs[("motion_flow_s2t", frame_id, 0)], 
                                                                                            # desipte there is grad in motion_flow, we readd for safety.
                                                                                            motion_flow = - outputs[("pose_flow_s2t", "high", frame_id, scale)] + outputs[("position_reverse", "high", scale, frame_id)], # there is grad_flow here! differ from motion_flow
                                                                                            detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                                            thre_px=self.opt.motion_mask_thre_px)

                # gen sample_motion_corrected from pose_flow and motion_flow, then sample image with sample_motion_corrected
                # sample_motion_corrected = (outputs[("pose_flow", "high", frame_id, scale)] + outputs[("motion_flow", frame_id, scale)])#.permute(0, 2, 3, 1)
                # do not use motion_flow, but use below to make sure grad exists. 
                sample_motion_corrected = outputs[("pose_flow", "high", frame_id, scale)] + \
                    (- outputs[("pose_flow", "high", frame_id, scale)].detach() + outputs[("position", "high", scale, frame_id)])
                #use self.spatial_transform to sample

                outputs[("color_MotionCorrected", frame_id, scale)] = self.spatial_transform(
                    inputs[("color", frame_id, 0)],
                    # inputs[("color", 0, source_scale)],
                    sample_motion_corrected)
        
        return outputs


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            # disp = outputs[("disp", scale)]
            # if self.opt.v1_multiscale:
            #     source_scale = scale
            # else:
            #     disp = F.interpolate(
            #         disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # outputs[("depth", 0, scale)] = depth

            # if self.opt.enable_mutual_motion and self.opt.enable_motion_computation:
            #     # collected depth--prepared for mutual pose flow, finally mutual motion_mask
            #     for frame_id in self.opt.frame_ids[1:]:
            #         assert frame_id != 0,f'frame_id == 0 already computed'
            #         disp_i = outputs[("disp", scale, frame_id)]

            #         assert not self.opt.v1_multiscale,f'v1_multiscale is not supported for mutual motion'
            #         disp_i = F.interpolate(
            #             disp_i, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                    
            #         _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, self.opt.max_depth)
            #         outputs[("depth", frame_id, scale)] = depth_i

            source_scale = 0

            # # Create sampling grid
            # use pose_flow(sample-img_grid) and optic_flow(position)
            # implement mem effecient mesh_gird_high_res 
            x = torch.linspace(0, self.opt.width - 1, self.opt.width, device=self.device)
            y = torch.linspace(0, self.opt.height - 1, self.opt.height, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
            mesh_gird_high_res = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.unsqueeze(0)  # (1, H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.permute(0, 3, 1, 2)  # (B, 2, H, W)            
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                        tgt_frame_id = 0, frame_id = frame_id)

                if (self.opt.enable_motion_computation and self.opt.enable_mutual_motion) or self.opt.enable_all_depth:
                        # it will update: samples_s2t and pose_flow_s2t in outputs
                        outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                            tgt_frame_id = frame_id, frame_id = 0)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)
                
        return outputs

    def generate_images_pred_ori(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            if (self.opt.enable_motion_computation and self.opt.enable_mutual_motion) or self.opt.enable_all_depth:
            # if self.opt.enable_motion_computation:
            #     if self.opt.enable_mutual_motion or self.opt.enable_all_depth:
                    # collected depth--prepared for mutual pose flow, finally mutual motion_mask
                    for frame_id in self.opt.frame_ids[1:]:
                        assert frame_id != 0,f'frame_id == 0 already computed'
                        disp_i = outputs[("disp", scale, frame_id)]

                        assert not self.opt.v1_multiscale,f'v1_multiscale is not supported for mutual motion'
                        disp_i = F.interpolate(
                            disp_i, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        
                        _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, self.opt.max_depth)
                        outputs[("depth", frame_id, scale)] = depth_i

            source_scale = 0

            # # Create sampling grid
            # use pose_flow(sample-img_grid) and optic_flow(position)
            # implement mem effecient mesh_gird_high_res 
            x = torch.linspace(0, self.opt.width - 1, self.opt.width, device=self.device)
            y = torch.linspace(0, self.opt.height - 1, self.opt.height, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
            mesh_gird_high_res = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.unsqueeze(0)  # (1, H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.permute(0, 3, 1, 2)  # (B, 2, H, W)            
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                        tgt_frame_id = 0, frame_id = frame_id)

                if (self.opt.enable_motion_computation and self.opt.enable_mutual_motion) or self.opt.enable_all_depth:
                # if self.opt.enable_motion_computation:
                #    if self.opt.enable_mutual_motion or self.opt.enable_all_depth:
                        # it will update: samples_s2t and pose_flow_s2t in outputs
                        outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                            tgt_frame_id = frame_id, frame_id = 0)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)
                
        return outputs

    # @classmethod
    def generate_depth_pred_from_disp(self, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                assert 0,f'v1_multiscale is not supported for depth prediction'
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth


            if (self.opt.enable_motion_computation and self.opt.enable_mutual_motion) or self.opt.enable_all_depth:
            # if self.opt.enable_motion_computation:
                # if self.opt.enable_mutual_motion or self.opt.enable_all_depth:
                    # collected depth--prepared for mutual pose flow, finally mutual motion_mask
                    for frame_id in self.opt.frame_ids[1:]:
                        assert frame_id != 0,f'frame_id == 0 already computed'
                        disp_i = outputs[("disp", scale, frame_id)]

                        assert not self.opt.v1_multiscale,f'v1_multiscale is not supported for mutual motion'
                        disp_i = F.interpolate(
                            disp_i, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        
                        _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, self.opt.max_depth)
                        outputs[("depth", frame_id, scale)] = depth_i

            #debug
            debug_only = True
            debug_only = False
            if debug_only:
                print('max min disp at scale:', scale, ':', disp.max(), disp.min())
                print('max depth after disp2depth f0 with scale:', scale, ':', outputs[("depth", 0, scale)].max())
                print('min depth after disp2depth f0 with scale:', scale, ':', outputs[("depth", 0, scale)].min())

            # source_scale = 0

            # # # Create sampling grid
            # # use pose_flow(sample-img_grid) and optic_flow(position)
            # # implement mem effecient mesh_gird_high_res 
            # x = torch.linspace(0, self.opt.width - 1, self.opt.width, device=self.device)
            # y = torch.linspace(0, self.opt.height - 1, self.opt.height, device=self.device)
            # grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
            # mesh_gird_high_res = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
            # mesh_gird_high_res = mesh_gird_high_res.unsqueeze(0)  # (1, H, W, 2)
            # mesh_gird_high_res = mesh_gird_high_res.permute(0, 3, 1, 2)  # (B, 2, H, W)            
            
            # for i, frame_id in enumerate(self.opt.frame_ids[1:]):

            #     outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
            #                                             tgt_frame_id = 0, frame_id = frame_id)

            #     if self.opt.enable_mutual_motion and self.opt.enable_motion_computation:
            #         # it will update: samples_s2t and pose_flow_s2t in outputs
            #         outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
            #                                             tgt_frame_id = frame_id, frame_id = 0)

            #     outputs[("color", frame_id, scale)] = F.grid_sample(
            #         inputs[("color", frame_id, source_scale)],
            #         outputs[("sample", frame_id, scale)],
            #         padding_mode="border",
            #         align_corners=True)
                
        return outputs




    def generate_images_pred_v0(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            if (self.opt.enable_motion_computation and self.opt.enable_mutual_motion) or self.opt.enable_all_depth:
            # if self.opt.enable_motion_computation:
            #    if self.opt.enable_mutual_motion or self.opt.enable_all_depth:
                    # collected depth--prepared for mutual pose flow, finally mutual motion_mask
                    for frame_id in self.opt.frame_ids[1:]:
                        assert frame_id != 0,f'frame_id == 0 already computed'
                        disp_i = outputs[("disp", scale, frame_id)]

                        assert not self.opt.v1_multiscale,f'v1_multiscale is not supported for mutual motion'
                        disp_i = F.interpolate(
                            disp_i, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        
                        _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, self.opt.max_depth)
                        outputs[("depth", frame_id, scale)] = depth_i

            source_scale = 0

            # # Create sampling grid
            # use pose_flow(sample-img_grid) and optic_flow(position)
            # implement mem effecient mesh_gird_high_res 
            x = torch.linspace(0, self.opt.width - 1, self.opt.width, device=self.device)
            y = torch.linspace(0, self.opt.height - 1, self.opt.height, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
            mesh_gird_high_res = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.unsqueeze(0)  # (1, H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.permute(0, 3, 1, 2)  # (B, 2, H, W)            
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                        tgt_frame_id = 0, frame_id = frame_id)

                if (self.opt.enable_motion_computation and self.opt.enable_mutual_motion) or self.opt.enable_all_depth:
                # if self.opt.enable_motion_computation:
                #    if self.opt.enable_mutual_motion or self.opt.enable_all_depth:
                        # it will update: samples_s2t and pose_flow_s2t in outputs
                        outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                            tgt_frame_id = frame_id, frame_id = 0)

                if self.opt.enable_motion_computation:
                    # If you want to warp the source image src to the target view tgt, you need the flow that maps pixels from the target frame to the source frame, i.e., tgt → src.
                    # assert 0,f'use color is correct! not need to use color_motion_corrected! considering the pose_flow now is computed from tgt depth+pose'
                    # tgt2src motion flow
                    # avoid grad in motion_flow! elsewise the sample_motion_corrected will loss grad at all.
                    # outputs[("motion_flow", frame_id, scale)] = - outputs[("pose_flow", "high", frame_id, scale)].detach() + outputs[("position", "high", 0, frame_id)]#.detach() # 
                    # we enable grad in motion_flow, can be latered used to reg motion mask
                    # there is no grad in OF anyway in for 2nd statge traning

                    outputs[("motion_flow", frame_id, scale)] = - outputs[("pose_flow", "high", frame_id, scale)] + outputs[("position", "high", 0, frame_id)]#.detach() # 
                    if scale == 0:
                        if self.opt.use_soft_motion_mask:
                            outputs[("motion_mask_backward", 0, frame_id)] = self.get_motion_mask_soft(outputs[("position", 0, frame_id)], 
                                                                  outputs[("pose_flow", "high", frame_id, 0)],
                                                                  detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                  thre_px=self.opt.motion_mask_thre_px)
                        else:
                            outputs[("motion_mask_backward", 0, frame_id)] = self.get_motion_mask(
                                                                                            #  do not use outputs[("motion_flow", frame_id, 0)], depend on what is set to it, the grad might get lost!
                                                                                            # use below to make sure grad exists. 
                                                                                              - outputs[("pose_flow", "high", frame_id, scale)] + outputs[("position", "high", 0, frame_id)], # there is grad_flow here! differ from motion_flow
                                                                                              detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                                              thre_px=self.opt.motion_mask_thre_px)
                    
                    if self.opt.enable_mutual_motion:
                        # compute s2t motion_flow and s2t_motion_mask
                        # we enable grad in motion_flow, can be latered used to reg motion mask
                        outputs[("motion_flow_s2t", frame_id, 0)] = - outputs[("pose_flow_s2t", "high", frame_id, 0)] + outputs[("position_reverse", "high", 0, frame_id)]#.detach() # 
                        if scale == 0:
                            if self.opt.use_soft_motion_mask:
                                outputs[("motion_mask_s2t_backward", 0, frame_id)] = self.get_motion_mask_soft(optic_flow = outputs[("position_reverse", 0, frame_id)], 
                                                                  pose_flow = outputs[("pose_flow_s2t", "high", frame_id, 0)],
                                                                  detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                  thre_px=self.opt.motion_mask_thre_px)
                            else:
                                # static area will be set to 1
                                outputs[("motion_mask_s2t_backward", 0, frame_id)] = self.get_motion_mask(
                                                                                                    # outputs[("motion_flow_s2t", frame_id, 0)], 
                                                                                                    # desipte there is grad in motion_flow, we readd for safety.
                                                                                                    motion_flow = - outputs[("pose_flow_s2t", "high", frame_id, 0)] + outputs[("position_reverse", "high", 0, frame_id)], # there is grad_flow here! differ from motion_flow
                                                                                                    detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                                                    thre_px=self.opt.motion_mask_thre_px)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)
                
                if self.opt.enable_motion_computation:
                    # gen sample_motion_corrected from pose_flow and motion_flow, then sample image with sample_motion_corrected
                    # sample_motion_corrected = (outputs[("pose_flow", "high", frame_id, scale)] + outputs[("motion_flow", frame_id, scale)])#.permute(0, 2, 3, 1)
                    # do not use motion_flow, but use below to make sure grad exists. 
                    sample_motion_corrected = outputs[("pose_flow", "high", frame_id, scale)] + \
                        (- outputs[("pose_flow", "high", frame_id, scale)].detach() + outputs[("position", "high", scale, frame_id)])
                    #use self.spatial_transform to sample

                    outputs[("color_MotionCorrected", frame_id, scale)] = self.spatial_transform(
                        inputs[("color", frame_id, 0)],
                        # inputs[("color", 0, source_scale)],
                        sample_motion_corrected)
        
        return outputs


    def compute_pose_metrics(self, inputs, outputs):
        """
        Compute pose errors between ground truth and predicted relative poses.
        
        Args:
            inputs: Dictionary containing ground truth poses ("gt_c2w_poses", frame_id)
            outputs: Dictionary containing predicted poses ("cam_T_cam", 0, frame_id)
            
        Returns:
            trans_err_mean: Mean translation error across batch and frames (in mm)
            rot_err_mean: Mean rotation error across batch and frames (in degrees)
        """
        metrics_list_dict = {}

        # This line has a bug - frame_id is not defined yet
        # esti_tgt2src_rel_poses = outputs[("cam_T_cam", 0, frame_id)]
        if ('gt_c2w_poses', 0) not in inputs:
            print(f'warning: gt_c2w_poses not in inputs')
            print(f'load_gt is train:{self.train_dataset.load_gt_poses}')
            print(f'load_gt is val:{self.val_dataset.load_gt_poses}')
            return metrics_list_dict
        
        # gt_tgt_abs_poses: (B, 4, 4)
        gt_tgt_abs_poses = inputs[("gt_c2w_poses", 0)]  # (B, 4, 4)
        for frame_id in self.opt.frame_ids[1:]:
            gt_src_abs_poses = inputs[("gt_c2w_poses", frame_id)]  # (B, 4, 4)
            pred_rel_poses_batch = outputs[("cam_T_cam", 0, frame_id)]  # (B, 4, 4)
            gt_tgt2src_rel_poses = torch.inverse(gt_src_abs_poses) @ gt_tgt_abs_poses
            assert gt_tgt2src_rel_poses.shape == pred_rel_poses_batch.shape, f'gt_tgt2src_rel_poses.shape: {gt_tgt2src_rel_poses.shape}, pred_rel_poses_batch.shape: {pred_rel_poses_batch.shape}'
            
            # err_dict = compute_pose_error(gt_tgt2src_rel_poses, pred_rel_poses_batch.detach())
            # trans_err_list.append(err_dict['trans_err'])
            # rot_err_list.append(err_dict['rot_err'])

            err_dict = compute_pose_error_v2(gt_tgt2src_rel_poses, pred_rel_poses_batch.detach())
            for k, v in err_dict.items():
                # print(f'{k}: {v}')
                assert k in ['trans_err_ang', 'trans_err_ang_deg', 'trans_err_scale', 'trans_err_scale_norm', 'rot_err', 'rot_err_deg']
                metrics_list_dict[k] = metrics_list_dict.get(k, []) + [v]

            # log in scale of estimated translation
            pred_rel_trans_scale = outputs[("cam_T_cam", 0, frame_id)][:, :3, 3].norm(dim=1).mean()
            metrics_list_dict['pred_rel_trans_scale'] = metrics_list_dict.get('pred_rel_trans_scale', []) + [pred_rel_trans_scale]
            # log in scale of depth
            pred_f0_depth_scale = outputs[("depth", 0, 0)].mean()
            metrics_list_dict['pred_f0_depth_scale'] = metrics_list_dict.get('pred_f0_depth_scale', []) + [pred_f0_depth_scale]
            if self.opt.enable_mutual_motion:
                pred_fi_depth_scale = outputs[("depth", frame_id, 0)].mean()
                metrics_list_dict['pred_fi_depth_scale'] = metrics_list_dict.get('pred_fi_depth_scale', []) + [pred_fi_depth_scale]

            # log in scale of cam_points
            # cam_points: B 4 N
            def add_cam_points_metrics(frame_id, prefix, metrics_list_dict):
                cam_points = outputs[("cam_points", frame_id, 0)]
                for i, coord in enumerate(['x', 'y', 'z']):
                    mean_val = cam_points[:, i, :].mean()
                    metrics_list_dict[f'{prefix}_cam_points_{coord}_mean'] = metrics_list_dict.get(f'{prefix}_cam_points_{coord}_mean', []) + [mean_val]
            
            add_cam_points_metrics(0, 'pred_f0', metrics_list_dict=metrics_list_dict)
            if self.opt.enable_mutual_motion:
                add_cam_points_metrics(frame_id, 'pred_fi', metrics_list_dict=metrics_list_dict)

            # # convert 
            # print('pred_rel_trans_scale shape:')
            # print(pred_rel_trans_scale)
            # print('pred_rel_trans_scale:')
            # print(pred_rel_trans_scale)
            # print('metrics_list_dict:')
            # print(metrics_list_dict['trans_err_scale'])


            # print('gt_tgt2src_rel_poses shape:')
            # print(gt_tgt2src_rel_poses.shape)
            # print('pred_rel_poses_batch shape:')
            # print(pred_rel_poses_batch.shape)
            # if frame_id == self.opt.frame_ids[1]:
            if frame_id in self.opt.frame_ids[1:]:
                print('////frame_id as src:', frame_id)
                print('gt_tgt2src_rel_poses trans:')
                print(gt_tgt2src_rel_poses[:, :3, 3])
                print('pred_rel_poses_batch trans:')
                print(pred_rel_poses_batch[:, :3, 3])

            # print('gt_tgt2src_rel_poses rot:')
            # print(gt_tgt2src_rel_poses[:, :3, :3])
            # print('pred_rel_poses_batch rot:')
            # print(pred_rel_poses_batch[:, :3, :3])

        # trans_err_list = torch.cat(trans_err_list, 0)
        # rot_err_list = torch.cat(rot_err_list, 0)
        # print esti_rel and gt_rel for debug purpose
        # return trans_err_list.mean(), rot_err_list.mean()

        # print('Report all metrics:')
        # for k, v in metrics_list_dict.items():
        #     print(f'{k}: {v}')

        # return trans_err_list, rot_err_list
        return metrics_list_dict

    def compute_motion_mask_reg_loss(self, reg_tgt_flow, motion_mask, is_soft_mask):
        '''
        we initially tried reg_tgt_flow as optic_flow, but it is not ideal.
        Try using motion flow now: while it is recommended to detach the grad of motion flow--else wise the motion_mask might bailan.
        '''
        assert reg_tgt_flow.dim() == 4, f'reg_tgt_flow.dim() is {reg_tgt_flow.dim()}'
        assert motion_mask.dim() == 3, f'motion_mask.dim() is {motion_mask.dim()}'
        assert reg_tgt_flow.shape[1] == 2, f'reg_tgt_flow.shape[1] is {reg_tgt_flow.shape[1]}'
        # print(f'optic_flow.shape: {optic_flow.shape}')
        # print(f'motion_mask.shape: {motion_mask.shape}')
        
        from mask_utils import structure_loss_soft, structure_loss
        
        if is_soft_mask:
            loss_mag, loss_edge, loss_dice, imgs_debug = structure_loss_soft(reg_tgt_flow, motion_mask,
                                                                             valid_motion_threshold_px=self.opt.valid_motion_threshold_px,
                                                                             contrast_alpha=self.opt.contrast_alpha,
                                                                             static_flow_noise_thre=self.opt.static_flow_noise_thre,
                                                                             )
        else:
            loss_mag, loss_edge, loss_dice, imgs_debug = structure_loss(reg_tgt_flow, motion_mask,
                                                                         valid_motion_threshold_px=self.opt.valid_motion_threshold_px,
                                                                         contrast_alpha=self.opt.contrast_alpha,
                                                                         static_flow_noise_thre=self.opt.static_flow_noise_thre,
                                                                         )
        return loss_mag, loss_edge, loss_dice, imgs_debug

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ms_ssim_loss = 1 - self.ms_ssim(pred, target)
            reprojection_loss = 0.9 * ms_ssim_loss + 0.1 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0
            if self.opt.use_loss_reproj2_nomotion:
                loss2_reprojection = 0
            if self.opt.use_loss_motion_mask_reg:
                assert self.opt.enable_grad_flow_motion_mask, "enable_grad_flow_motion_mask must be True when use_loss_motion_mask_reg is True"

                # loss_motion_mask_reg = 0
                loss_reg_dice = 0
                loss_reg_edge = 0
                loss_reg_mag = 0
            
            debug_only = True
            # debug_only = False
            if debug_only:
                self.opt.use_R_reg = True
                self.opt.use_t_mag_reg = True
                self.opt.use_R_reg = False
                self.opt.use_t_mag_reg = False
                self.opt.R_reg_weight = 10
                self.opt.t_mag_reg_weight = 10
                # use weight annelling
                # use temporal reg: the relative pose of fi and fi' w.r.t to f0 should be inverse.


                loss_R_reg = 0
                loss_t_mag_reg = 0

            if self.opt.reg_mutual_raw_disp_based_OF_for_consistency_and_correctness:
                loss_reg_mutual_raw_disp_based_OF = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]


            for frame_id in self.opt.frame_ids[1:]:
                                # valid_mask = occu_mask_backward * ~outputs[("motion_mask_backward", 0, frame_id)].detach()

                # img warped from pose_flow is saved as "color"; img warped from optic_flow is saved as "registration"
                # register for debug monitoring
                
                # the 1st reporj_loss: main aim: enforce proper AF learning when everything defautl as it is.
                if self.opt.reproj_supervised_which in ["color_MotionCorrected", "color_MotionCorrected_motiononly"] \
                    and self.opt.enable_motion_computation:
                    reproj_loss_supervised_tgt_color = outputs[("color_MotionCorrected", frame_id, scale)]
                elif self.opt.reproj_supervised_which == "color":
                    reproj_loss_supervised_tgt_color = outputs[("color", frame_id, scale)] 
                else:
                    assert self.opt.enable_motion_computation, "enable_motion_computation must be True when reproj_supervised_which is color_MotionCorrected"
                    raise ValueError(f"Invalid reproj_supervised_which: {self.opt.reproj_supervised_which}")

                #phedo gt
                if self.opt.reproj_supervised_with_which == "refined":
                    reproj_loss_supervised_signal_color = outputs[("refined", scale, frame_id)]
                elif self.opt.reproj_supervised_with_which == "raw_tgt_gt":
                    reproj_loss_supervised_signal_color = inputs[("color", 0, 0)]
                else:
                    raise ValueError(f"Invalid reproj_supervised_with_which: {self.opt.reproj_supervised_with_which}")

                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                valid_mask = occu_mask_backward

                # if motion only, only supervise the motion area
                if self.opt.reproj_supervised_which == "color_MotionCorrected_motiononly" and self.opt.enable_motion_computation:
                    assert 0,f'not working...'
                    motion_area_mask = outputs[("motion_mask_backward", 0, frame_id)].detach() < 0.5
                    if self.opt.enable_mutual_motion:
                        motion_area_mask = motion_area_mask | (outputs[("motion_mask_s2t_backward", 0, frame_id)].detach() < 0.5)
                    valid_mask = valid_mask * motion_area_mask.float()
                    # percentage of valid px
                    valid_percentage = valid_mask.sum() / (valid_mask.numel() + 1e-6)
                    print(f'////valid_percentage: {valid_percentage}')


                denom_safe = torch.clamp(valid_mask.sum(), min=1e-6)
                loss_reprojection += (
                    self.compute_reprojection_loss(reproj_loss_supervised_tgt_color, reproj_loss_supervised_signal_color) * valid_mask).sum() / denom_safe
                loss_transform += (
                    torch.abs(outputs[("refined", scale, frame_id)] - outputs[("registration", 0, frame_id)].detach()).mean(1, True) * valid_mask).sum() / denom_safe
                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)], inputs[("color", 0, 0)], outputs[("registration", scale, frame_id)].detach(), valid_mask)

                #register reproj_loss_supervised_tgt_color for debugging
                outputs[("reproj_supervised_tgt_color_debug", scale, frame_id)] = reproj_loss_supervised_tgt_color.detach() #reproj_loss_supervised_tgt_color.detach()
                outputs[("reproj_supervised_signal_color_debug", scale, frame_id)] = reproj_loss_supervised_signal_color.detach() #reproj_loss_supervised_tgt_color.detach()

                if self.opt.use_R_reg:
                    esti_rot = outputs[("cam_T_cam", 0, frame_id)][:, :3, :3]

                    def rotation_angle_from_R(R, eps=1e-6):
                        # R: (..., 3, 3)
                        tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
                        cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0+eps, 1.0-eps)
                        theta = torch.acos(cos_theta)
                        return theta  # shape (...)
                    
                    theta = rotation_angle_from_R(esti_rot)
                    # regularize to be identity
                    loss_R_reg += (theta ** 2).mean()
                
                if self.opt.use_t_mag_reg:
                    esti_trans = outputs[("cam_T_cam", 0, frame_id)][:, :3, 3]
                    esti_trans_mag = esti_trans.norm(dim=1)
                    loss_t_mag_reg += (esti_trans_mag ** 2).mean()



                if self.opt.enable_motion_computation:
                    # the 2st reporj_loss: main aim: supervise PoseNet properly by masking out the mutual motion area
                    if self.opt.use_loss_reproj2_nomotion:
                        if self.opt.reproj2_supervised_which == "color":
                            reproj_loss2_supervised_tgt_color = outputs[("color", frame_id, scale)]
                        else:
                            raise ValueError(f"Invalid reproj2_supervised_which: {self.opt.reproj2_supervised_which}")
                        
                        if self.opt.reproj2_supervised_with_which == "refined":
                            reproj_loss2_supervised_signal_color = outputs[("refined", scale, frame_id)]
                        else:
                            raise ValueError(f"Invalid reproj2_supervised_with_which: {self.opt.reproj2_supervised_with_which}")
                    
                    motion_mask_backward = outputs[("motion_mask_backward", 0, frame_id)].detach()
                    
                    if self.opt.enable_mutual_motion:
                        # computational expensive but safer
                        motion_mask_s2t_backward = outputs[("motion_mask_s2t_backward", 0, frame_id)].detach()
                        # conver to binary if it was soft motion mask: this is critical for soft motion mask regularization loss
                        valid_mask2 = (motion_mask_backward > 0.5).float() * (motion_mask_s2t_backward > 0.5).float()
                    else:
                        valid_mask2 = (motion_mask_backward > 0.5).float()

                    if self.opt.use_loss_reproj2_nomotion:
                        denom_safe2 = torch.clamp(valid_mask2.sum(), min=1e-6)
                        loss2_reprojection += (
                            self.compute_reprojection_loss(reproj_loss2_supervised_tgt_color, reproj_loss2_supervised_signal_color) * valid_mask2).sum() / denom_safe2 
                        
                    #compute motion mask reg loss
                    if self.opt.use_loss_motion_mask_reg and scale == 0:
                        # optic_flow = outputs[("position", "high", 0, frame_id)]
                        # reg_tgt_flow = optic_flow # we use optic_flow as reg_tgt_flow, not good.

                        motion_flow = outputs[("motion_flow", "high", frame_id, scale)]
                        reg_tgt_flow = motion_flow.detach() # we want to make sure there is no grad in motion_flow, elsewise the motion_mask might bailan.
                        motion_mask = outputs[("motion_mask_backward", scale, frame_id)].squeeze(1)

                        # motion mask is already properly supervised in loss_reproj2
                        # here supervise motion_flow(pose_flow)? by detacch motion_mask but remain the grad for motion_flow?

                        
                        loss_mag, loss_edge, loss_dice, imgs_debug = self.compute_motion_mask_reg_loss(
                            reg_tgt_flow = reg_tgt_flow, 
                            motion_mask = 1-motion_mask,# we want to apply strcuture loss of moiton_mask where motion is positive
                            is_soft_mask = self.opt.use_soft_motion_mask)
                        # print(f'loss_mag: {loss_mag}, loss_edge: {loss_edge}, loss_dice: {loss_dice}')
                        

                        if self.step % self.opt.log_frequency == 0:
                            save_root = os.path.join(self.log_path, f'motion_mask_reg_related')
                            os.makedirs(save_root, exist_ok=True)
                            # concat image in imgs_debug
                            # from utils import color_to_cv_img
                            from utils import gray_to_cv_img
                            concat_imgs = []
                            for batch_idx in range(self.opt.batch_size):
                                # concat_img = np.concatenate([color_to_cv_img(img[batch_idx][None]) for k, img in imgs_debug.items()], axis=1)
                                concat_img = np.concatenate([gray_to_cv_img(img[batch_idx][None]) for k, img in imgs_debug.items()], axis=1)
                                concat_imgs.append(concat_img)
                            concat_img = np.concatenate(concat_imgs, axis=0)
                            save_path = os.path.join(save_root, f'{self.step}_concat.png')
                            cv2.imwrite(save_path, concat_img)
                            print(f'saved concat image to {save_path}')
                                # cv2.imwrite(save_path, concat_img)
                                # print(f'saved concat image to {save_path}')

                            # # save each image in imgs_debug
                            # for k, img in imgs_debug.items():
                            #     # print(f'{k}: {img.shape}')
                            #     from utils import color_to_cv_img
                            #     batch_idx = 0
                            #     img_cv = color_to_cv_img(img[batch_idx][None])
                            #     save_path = os.path.join(save_root, f'{self.step}_{k}.png')
                            #     cv2.imwrite(save_path, img_cv)
                            #     # print(f'saved {k} to {save_path}')

                        loss_reg_dice += loss_dice
                        loss_reg_edge += loss_edge
                        loss_reg_mag += loss_mag

                        if self.opt.enable_mutual_motion:
                            # optic_flow = outputs[("position_inverse", "high", 0, frame_id)]
                            # reg_tgt_flow = optic_flow

                            motion_flow = outputs[("motion_flow_s2t", "high", frame_id, scale)]
                            reg_tgt_flow = motion_flow.detach() # we want to make sure there is no grad in motion_flow, elsewise the motion_mask might bailan.

                            motion_mask = outputs[("motion_mask_s2t_backward", scale, frame_id)].squeeze(1) # there  is grad


                            loss_mag, loss_edge, loss_dice, imgs_debug = self.compute_motion_mask_reg_loss(
                                reg_tgt_flow = reg_tgt_flow, 
                                motion_mask = 1-motion_mask,# we want to apply strcuture loss of moiton_mask where motion is positive
                                is_soft_mask = self.opt.use_soft_motion_mask)
                            # loss_motion_mask_reg += (weights[0] * loss_mag + weights[1] * loss_edge + weights[2] * loss_dice) * self.opt.motion_mask_reg_loss_weight
                            loss_reg_dice += loss_dice
                            loss_reg_edge += loss_edge
                            loss_reg_mag += loss_mag

                    # if self.opt.reg_mutual_raw_disp_based_OF_for_consistency_and_correctness:
                    #     def compute_disp_loss(disp, target_disp):
                    #         return torch.abs(disp - target_disp).mean()
                    #     if scale == 0:
                    #         disp_tgt_img = outputs[("disp", scale)]
                    #         disp_src_img = outputs[("disp", scale, frame_id)]
                    #         # warp the src disp based on the optic flow
                    #         of_t2s = outputs[("position", 'high', scale, frame_id)].detach()
                    #         disp_tgt_img_esti = self.spatial_transform(disp_src_img, of_t2s)
                    #         loss_reg_mutual_raw_disp_based_OF += compute_disp_loss(disp_tgt_img_esti, disp_tgt_img)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0) 
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            # if self.opt.reg_mutual_raw_disp_based_OF_for_consistency_and_correctness:
            #     loss += loss_reg_mutual_raw_disp_based_OF / 2.0
            
            if self.opt.use_loss_motion_mask_reg:
                weights = [1.0, 0.2, 0.05]
                # cheat for potentially success full training!
                # remove edge loss--not sig helps? but the new_norm introduce complex grads?
                weights = [1.0, 0.0, 0.05]
                weights = [1.0, 0.0, 0.0]


                # only on scale 0
                loss += self.opt.motion_mask_reg_loss_weight * \
                    ((weights[0] * loss_reg_mag \
                        + weights[1] * loss_reg_edge \
                            + weights[2] * loss_reg_dice) / 2.0 )

            if self.opt.use_loss_reproj2_nomotion:
                loss += self.opt.loss_reproj2_nomotion_weight * loss2_reprojection / 2.0

            if self.opt.use_R_reg:
                loss += self.opt.R_reg_weight * loss_R_reg
            if self.opt.use_t_mag_reg:
                loss += self.opt.t_mag_reg_weight * loss_t_mag_reg

            total_loss += loss
            # total
            losses["loss/{}".format(scale)] = loss

            # log in loss breakdown
            # log in the loss_reproj
            losses['loss/scale_{}_reproj'.format(scale)] = loss_reprojection
            losses['loss/scale_{}_transform'.format(scale)] = loss_transform
            losses['loss/scale_{}_cvt'.format(scale)] = loss_cvt
            if self.opt.use_loss_motion_mask_reg:
                losses['loss/scale_{}_motion_mask_reg_mag'.format(scale)] = loss_reg_mag
                losses['loss/scale_{}_motion_mask_reg_edge'.format(scale)] = loss_reg_edge
                losses['loss/scale_{}_motion_mask_reg_dice'.format(scale)] = loss_reg_dice

            if self.opt.use_loss_reproj2_nomotion:
                losses['loss/scale_{}_reproj2_nomotion'.format(scale)] = loss2_reprojection
            
            if self.opt.reg_mutual_raw_disp_based_OF_for_consistency_and_correctness:
                losses['loss/scale_{}_reg_mutual_raw_disp_based_OF'.format(scale)] = loss_reg_mutual_raw_disp_based_OF


        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def val(self):
        print('Do val....')

        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses, compute_vis=False, online_vis=False)
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            print('CHECK: self.opt.pose_model_type == "shared"')
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            print('CHECK: self.opt.pose_model_type != "shared"')
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        outputs.update(self.generate_depth_pred_from_disp(outputs))

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        # outputs.update(self.generate_images_pred(inputs, outputs))
        outputs.update(self.generate_images_pred(inputs, outputs))

        if self.opt.enable_motion_computation:
            if self.opt.use_MF_network:
                outputs.update(self.predict_motion_flow_with_MF_net(inputs, outputs))
            else:
                outputs.update(self.generate_motion_flow(inputs, outputs))


        losses = self.compute_losses_val(inputs, outputs)
        # compute the pose errors
        metrics_list_dict = self.compute_pose_metrics(inputs, outputs)
        
        # for k, v in metrics_list_dict.items():
            # print(f'{k}: {v}')

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(outputs[("registration", scale, frame_id)].mean(1, True), target.mean(1, True)))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss

        return losses

    def log_time(self, batch_idx, duration, loss, loss_0=None, metric_errs=None):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f}"
        
        if loss_0 is not None:
            print_string += " | loss_0: {:.5f}"
        
        print_string += " | time elapsed: {} | time left: {}"
        
        if loss_0 is not None:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, loss_0,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
        # Add pose metrics if available
        if metric_errs:
            print("  Pose Errors:", end=" ")
            for k, v in metric_errs.items():
                if isinstance(v, torch.Tensor):
                    # v = v.mean().item()
                    v = v.item()
                print(f"{k}: {v:.3f}", end=" | ")
            print()

    def log(self, mode, inputs, outputs, scalers_to_log, compute_vis=True, online_vis=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        for l, v in scalers_to_log.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        # Log learnable motion mask threshold if enabled
        if self.opt.enable_learned_motion_mask_thre_px and self.learned_motion_mask_thre_px is not None:
            writer.add_scalar("motion_mask_threshold", self.learned_motion_mask_thre_px.item(), self.step)
        #

        # src_imgs = []
        # tgt_imgs = []
        # registered_tgt_imgs = []
        # colored_tgt_imgs = []
        # colored_motion_tgt_imgs = []
        # optic_flow_imgs = []
        # pose_flow_imgs = []
        # motion_flow_imgs = []
        # depth_imgs = []
        # brightness_imgs = []
        # refined_tgt_imgs = []

        # occlursion_mask_imgs = []
        # motion_mask_imgs = []
        # motion_mask_s2t_imgs = []

        # reproj_supervised_tgt_color_debug_imgs = []

        concat_img = None
        concat_img_list = []
        img_name_list = []

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        tgt_scale_to_vis = [0]
        tgt_frame_id_to_vis = self.opt.frame_ids[1:2]

        # debug_only = True
        # if debug_only:
        #     #vis more
        #     tgt_scale_to_vis = [0,3]
        #     tgt_frame_id_to_vis = self.opt.frame_ids[1:]

        # motion_flow and motion_mask will have adapted shape at various level while the others are all on the level0 shape
        for j in range(min(1, self.opt.batch_size)):  # write a maxmimum of 2 images
            # for s in self.opt.scales:
            for s in tgt_scale_to_vis:
                # frames_ids = [0,-1,1]
                assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
                assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
                # for frame_id in self.opt.frame_ids[1:]:

                # only predicted depth from the center image (frame_id == 0)
                writer.add_image(
                    "Depth/disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                writer.add_image(
                    'Depth/depth_{}/{}'.format(s, j),
                    normalize_image(outputs[("depth", 0, s)][j]), self.step)                
                # for frame_id in self.opt.frame_ids[1:2]:  # only for one is enough for debug
                for frame_id in tgt_frame_id_to_vis:  # only for one is enough for debug
                    # if s in tgt_scale_to_vis:

                    writer.add_image(
                        # "IMG/tgt_refined_{}_{}/{}".format(frame_id, s, j),
                        "GT/tgt_refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "Other/brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data,
                        self.step)
                    writer.add_image(
                        "IMG/registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)

                # if s == 0:
                # if s in tgt_scale_to_vis:
                    writer.add_image(
                        "Other/occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                        outputs[("occu_mask_backward", s, frame_id)][j].data, self.step)
                    #/////////////EXTEND////////////////////////
                    if self.opt.enable_motion_computation:
                        # if s == s:
                        writer.add_image(
                            "Other/motion_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("motion_mask_backward", s, frame_id)][j].data, self.step)
                    
                        if self.opt.enable_mutual_motion:
                            # if s == s:
                            writer.add_image(
                                "Other/motion_mask_s2t_backward_{}_{}/{}".format(frame_id, s, j),
                                outputs[("motion_mask_s2t_backward", s, frame_id)][j].data, self.step)
                            
                            writer.add_image(
                                "Other/motion_flow_s2t_{}_{}/{}".format(frame_id, s, j),
                                outputs[("motion_flow_s2t","high", frame_id, s)][j].data, self.step)
                    
                    # add src and tgt
                    writer.add_image(
                        "GT/tgt_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", 0, 0)][j].data, self.step) 
                    # add gt_source
                    writer.add_image(
                        "GT/source_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, 0)][j].data, self.step)  
                    # add supervised_img
                    writer.add_image(
                        "GT/reproj_supervised_tgt_color_debug_{}_{}/{}".format(frame_id, s, j),
                        # "IMG/reproj_supervised_tgt_color_debug_{}_{}/{}".format(frame_id, s, j),
                        outputs[("reproj_supervised_tgt_color_debug", 0, frame_id)][j].data, self.step)
                    # add vis of other warped img
                    # add color image as well
                    writer.add_image(
                        "IMG/color_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)
                    
                    if self.opt.enable_motion_computation:
                        writer.add_image(
                            "IMG/color_MotionCorrected_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_MotionCorrected", frame_id, s)][j].data, self.step)

                    # write various flow; import functions from utils
                    from utils import flow_vis, flow_vis_robust
                    vis_flow_func = flow_vis_robust
                    # vis_flow_func = flow_vis
                    # add optic flow
                    writer.add_image(
                        "FLOW/optic_flow_{}_{}/{}".format(frame_id, s, j),
                        vis_flow_func(outputs[("position", "high", s, frame_id)][j].data), self.step)
                    # add pose_flow
                    writer.add_image(
                        "FLOW/pose_flow_{}_{}/{}".format(frame_id, s, j),
                        vis_flow_func(outputs[("pose_flow", "high", frame_id, s)][j].data), self.step)
                    # add motion_flow
                    if self.opt.enable_motion_computation:
                        writer.add_image(
                            "FLOW/motion_flow_{}_{}/{}".format(frame_id, s, j),
                            vis_flow_func(outputs[("motion_flow", "high", frame_id, s)][j].data), self.step)



                    if compute_vis:
                    # if compute_vis and s in tgt_scale_to_vis:
                    # if compute_vis and s == 0:
                        '''
                        only vis scale 0
                        '''


                        src_imgs = []
                        tgt_imgs = []
                        registered_tgt_imgs = []
                        colored_tgt_imgs = []
                        colored_motion_tgt_imgs = []
                        optic_flow_imgs = []
                        pose_flow_imgs = []
                        motion_flow_imgs = []
                        depth_imgs = []
                        brightness_imgs = []
                        refined_tgt_imgs = []

                        occlursion_mask_imgs = []
                        motion_mask_imgs = []
                        motion_mask_s2t_imgs = []

                        reproj_supervised_tgt_color_debug_imgs = []

                        img_order_strs = []
                        concat_imgs_list =[]

                        assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
                        src_imgs.append(inputs[("color", frame_id, 0)][j].data)
                        tgt_imgs.append(inputs[("color", self.opt.frame_ids[0], 0)][j].data)

                        registered_tgt_imgs.append(outputs[("registration", s, frame_id)][j].data)
                        depth_imgs.append(outputs[("depth", self.opt.frame_ids[0], s)][j].data)
                        brightness_imgs.append(outputs[("transform","high", s, frame_id)][j].data)
                        refined_tgt_imgs.append(outputs[("refined", s, frame_id)][j].data)

                        colored_tgt_imgs.append(outputs[("color", frame_id, s)][j].data)
                        optic_flow_imgs.append(outputs[("position", "high", s, frame_id)][j].data)
                        occlursion_mask_imgs.append(outputs[("occu_mask_backward", s, frame_id)][j].data)
                        pose_flow_imgs.append(outputs[("pose_flow", "high", frame_id, s)][j].data)
                        reproj_supervised_tgt_color_debug_imgs.append(outputs[("reproj_supervised_tgt_color_debug", 0, frame_id)][j].data)
                        if self.opt.enable_motion_computation:
                            colored_motion_tgt_imgs.append(outputs[("color_MotionCorrected", frame_id, s)][j].data)
                            # motion_flow_imgs.append(outputs[("motion_flow", frame_id, s)][j].data)
                            motion_flow_imgs.append(outputs[("motion_flow", "high", frame_id, s)][j].data)
                            # if s == s:
                            motion_mask_imgs.append(outputs[("motion_mask_backward", s, frame_id)][j].data)
                            if self.opt.enable_mutual_motion:
                                motion_mask_s2t_imgs.append(outputs[("motion_mask_s2t_backward", s, frame_id)][j].data)


                    # if compute_vis:
                        # Now apply
                        src_imgs = [color_to_cv_img(img) for img in src_imgs]
                        tgt_imgs = [color_to_cv_img(img) for img in tgt_imgs]
                        reproj_supervised_tgt_color_debug_imgs = [color_to_cv_img(img) for img in reproj_supervised_tgt_color_debug_imgs]
                        registered_tgt_imgs = [color_to_cv_img(img) for img in registered_tgt_imgs]
                        depth_imgs = [gray_to_cv_img(normalize_image(img)).astype(np.uint8) for img in depth_imgs]
                        brightness_imgs = [color_to_cv_img(img) for img in brightness_imgs]
                        refined_tgt_imgs = [color_to_cv_img(img) for img in refined_tgt_imgs]
                        colored_tgt_imgs = [color_to_cv_img(img) for img in colored_tgt_imgs]
                        # print('Optic flow:')
                        optic_flow_imgs = [flow_to_cv_img(img) for img in optic_flow_imgs]
                        # print('Pose flow:')
                        pose_flow_imgs = [flow_to_cv_img(img) for img in pose_flow_imgs]
                        if self.opt.enable_motion_computation:
                            colored_motion_tgt_imgs = [color_to_cv_img(img) for img in colored_motion_tgt_imgs]
                            # print('Motion flow:')
                            motion_flow_imgs = [flow_to_cv_img(img) for img in motion_flow_imgs]
                            # do not norm!
                            motion_mask_imgs = [gray_to_cv_img(img) for img in motion_mask_imgs]
                            if self.opt.enable_mutual_motion:
                                motion_mask_s2t_imgs = [gray_to_cv_img(img) for img in motion_mask_s2t_imgs]
                        # do not norm!
                        occlursion_mask_imgs = [gray_to_cv_img(img) for img in occlursion_mask_imgs]


                        # concat src_imgs and tgt_imgs vertically
                        src_concat_img = np.concatenate(src_imgs, axis=0)
                        img_order_strs.append('Src-')
                        concat_imgs_list.append(src_concat_img)
                        colored_tgt_concat_img = np.concatenate(colored_tgt_imgs, axis=0)
                        img_order_strs.append('Colored_Tgt-')
                        concat_imgs_list.append(colored_tgt_concat_img)
                        tgt_concat_img = np.concatenate(tgt_imgs, axis=0)
                        img_order_strs.append('Tgt-')
                        concat_imgs_list.append(tgt_concat_img)
                        # add reproj_supervised_tgt_color_debug
                        # reproj_supervised_tgt_color_debug_concat_img = np.concatenate(reproj_supervised_tgt_color_debug_imgs, axis=0)
                        # img_order_strs.append('Reproj_Sup-')

                        registered_tgt_concat_img = np.concatenate(registered_tgt_imgs, axis=0)
                        img_order_strs.append('Registered_Tgt-')
                        concat_imgs_list.append(registered_tgt_concat_img)
                        refined_tgt_concat_img = np.concatenate(refined_tgt_imgs, axis=0)
                        img_order_strs.append('Refined_Tgt-')
                        concat_imgs_list.append(refined_tgt_concat_img)
                        if self.opt.enable_motion_computation:
                            colored_motion_tgt_concat_img = np.concatenate(colored_motion_tgt_imgs, axis=0)
                            img_order_strs.append('Colored_Motion_Tgt-')
                            concat_imgs_list.append(colored_motion_tgt_concat_img)

                        optic_flow_concat_img = np.concatenate(optic_flow_imgs, axis=0)
                        img_order_strs.append('Optic_Flow-')
                        concat_imgs_list.append(optic_flow_concat_img)
                        pose_flow_concat_img = np.concatenate(pose_flow_imgs, axis=0)
                        img_order_strs.append('Pose_Flow-')
                        concat_imgs_list.append(pose_flow_concat_img)
                        occlursion_mask_concat_img = np.concatenate(occlursion_mask_imgs, axis=0)
                        img_order_strs.append('Occlursion_Mask-')
                        concat_imgs_list.append(occlursion_mask_concat_img)
                        depth_concat_img = np.concatenate(depth_imgs, axis=0)
                        img_order_strs.append('Depth-')
                        concat_imgs_list.append(depth_concat_img)
                        brightness_concat_img = np.concatenate(brightness_imgs, axis=0)
                        img_order_strs.append('Brightness-')
                        concat_imgs_list.append(brightness_concat_img)

                        if self.opt.enable_motion_computation:
                            # colored_motion_tgt_concat_img = np.concatenate(colored_motion_tgt_imgs, axis=0)
                            # img_order_strs.append('Colored_Motion_Tgt-')
                            motion_flow_concat_img = np.concatenate(motion_flow_imgs, axis=0)
                            img_order_strs.append('Motion_Flow-')
                            concat_imgs_list.append(motion_flow_concat_img)
                            motion_mask_concat_img = np.concatenate(motion_mask_imgs, axis=0)
                            img_order_strs.append('Motion_Mask-')
                            concat_imgs_list.append(motion_mask_concat_img)

                            if self.opt.enable_mutual_motion:
                                motion_mask_s2t_concat_img = np.concatenate(motion_mask_s2t_imgs, axis=0)
                                img_order_strs.append('Motion_Mask_S2T-')
                                concat_imgs_list.append(motion_mask_s2t_concat_img)

                        concat_img = np.concatenate(concat_imgs_list, axis=1)

                        joined_img_order_strs = ''.join(img_order_strs)
                        
                        scale_frameid_str = f'{s}_{frame_id}'
                        if online_vis:
                            title = f'{joined_img_order_strs}_{scale_frameid_str}'
                            cv2.imshow(title, concat_img/255)
                            cv2.waitKey(1)
                        else:
                            import os, cv2
                            save_path = os.path.join(self.log_path, f"imgs")
                            os.makedirs(save_path, exist_ok=True)
                            save_path = os.path.join(save_path, f"{self.step}_{joined_img_order_strs}_{scale_frameid_str}.png")
                            cv2.imwrite(save_path, concat_img)
                            print(f"saved {joined_img_order_strs}.png in {save_path}")
                        
                        concat_img_list.append(concat_img)
                        img_name_list.append(f'{joined_img_order_strs}_{scale_frameid_str}')

        return concat_img_list, img_name_list


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)
            print(f"saved {model_name}.pth in {save_path}")

        # Save optimizer states
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        
        # Save learnable motion mask threshold if enabled
        if self.opt.enable_learned_motion_mask_thre_px and self.learned_motion_mask_thre_px is not None:
            save_path = os.path.join(save_folder, "learned_motion_mask_thre_px.pth")
            torch.save(self.learned_motion_mask_thre_px.state_dict(), save_path)
            print(f"saved learned_motion_mask_thre_px.pth in {save_path}")
        
        save_path = os.path.join(save_folder, "{}.pth".format("adam_0"))
        torch.save(self.model_optimizer_0.state_dict(), save_path)
        
        # Save scheduler states
        save_path = os.path.join(save_folder, "{}.pth".format("scheduler"))
        torch.save(self.model_lr_scheduler.state_dict(), save_path)
        
        save_path = os.path.join(save_folder, "{}.pth".format("scheduler_0"))
        torch.save(self.model_lr_scheduler_0.state_dict(), save_path)
        
        # Save training progress
        training_progress = {
            'epoch': self.epoch,
            'step': self.step,
            'start_time': self.start_time
        }
        save_path = os.path.join(save_folder, "{}.pth".format("training_progress"))
        torch.save(training_progress, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")
        
        # Load learnable motion mask threshold if enabled and exists
        if self.opt.enable_learned_motion_mask_thre_px and self.learned_motion_mask_thre_px is not None:
            threshold_path = os.path.join(self.opt.load_weights_folder, "learned_motion_mask_thre_px.pth")
            if os.path.isfile(threshold_path):
                print("Loading learned motion mask threshold")
                threshold_dict = torch.load(threshold_path)
                self.learned_motion_mask_thre_px.load_state_dict(threshold_dict)
                print(f"Loaded learned motion mask threshold: {self.learned_motion_mask_thre_px.item():.4f}")
            else:
                print("No learned motion mask threshold found, using initialized value")

    def resume_training(self):
        """Resume training from a previously trained model directory
        """
        if not self.opt.load_weights_folder:
            raise ValueError("--load_weights_folder must be specified when using --resume_training")
        
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        
        # Find the latest checkpoint or specific epoch
        models_dir = os.path.join(self.opt.load_weights_folder, "models")
        if not os.path.exists(models_dir):
            raise ValueError(f"Models directory not found: {models_dir}")
        
        # Find available checkpoints
        available_epochs = []
        for item in os.listdir(models_dir):
            if item.startswith("weights_") and os.path.isdir(os.path.join(models_dir, item)):
                try:
                    epoch = int(item.split("_")[1])
                    available_epochs.append(epoch)
                except (ValueError, IndexError):
                    continue
        
        if not available_epochs:
            raise ValueError(f"No valid checkpoints found in {models_dir}")
        
        # Select epoch to resume from
        if self.opt.resume_from_epoch is not None:
            if self.opt.resume_from_epoch not in available_epochs:
                raise ValueError(f"Epoch {self.opt.resume_from_epoch} not found. Available epochs: {sorted(available_epochs)}")
            resume_epoch = self.opt.resume_from_epoch
        else:
            resume_epoch = max(available_epochs)
        
        print(f"Resuming training from epoch {resume_epoch}")
        
        # Load checkpoint
        checkpoint_dir = os.path.join(models_dir, f"weights_{resume_epoch}")
        
        # Load all models
        for model_name, model in self.models.items():
            model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
            if os.path.exists(model_path):
                print(f"Loading {model_name} weights...")
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                print(f"Warning: {model_name}.pth not found in checkpoint, skipping...")
        
        # Load optimizer states
        optimizer_path = os.path.join(checkpoint_dir, "adam.pth")
        if os.path.exists(optimizer_path):
            print("Loading main optimizer state...")
            self.model_optimizer.load_state_dict(torch.load(optimizer_path))
        
        optimizer_0_path = os.path.join(checkpoint_dir, "adam_0.pth")
        if os.path.exists(optimizer_0_path):
            print("Loading optimizer_0 state...")
            self.model_optimizer_0.load_state_dict(torch.load(optimizer_0_path))
        
        # Load scheduler states
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pth")
        if os.path.exists(scheduler_path):
            print("Loading main scheduler state...")
            self.model_lr_scheduler.load_state_dict(torch.load(scheduler_path))
        
        scheduler_0_path = os.path.join(checkpoint_dir, "scheduler_0.pth")
        if os.path.exists(scheduler_0_path):
            print("Loading scheduler_0 state...")
            self.model_lr_scheduler_0.load_state_dict(torch.load(scheduler_0_path))
        
        # Load training progress
        progress_path = os.path.join(checkpoint_dir, "training_progress.pth")
        if os.path.exists(progress_path):
            print("Loading training progress...")
            training_progress = torch.load(progress_path)
            self.epoch = training_progress['epoch']
            self.step = training_progress['step']
            self.start_time = training_progress['start_time']
            print(f"Resumed from epoch {self.epoch}, step {self.step}")
        else:
            # Fallback: set epoch and step based on checkpoint
            self.epoch = resume_epoch
            self.step = resume_epoch * (len(self.train_dataset) // self.opt.batch_size)
            print(f"Training progress not found, setting epoch to {self.epoch}, step to {self.step}")
        
        print("Training resumed successfully!")

    def find_latest_checkpoint(self, models_dir):
        """Find the latest checkpoint in the models directory
        """
        available_epochs = []
        for item in os.listdir(models_dir):
            if item.startswith("weights_") and os.path.isdir(os.path.join(models_dir, item)):
                try:
                    epoch = int(item.split("_")[1])
                    available_epochs.append(epoch)
                except (ValueError, IndexError):
                    continue
        
        if not available_epochs:
            return None
        
        return max(available_epochs)



if __name__ == "__main__":
    import torch
    from layers import BackprojectDepth
    batch_size = 1
    height = 480
    width = 640
    backproject_depth = BackprojectDepth(batch_size, height, width)
    # project3d = Project3D(batch_size, height, width)
    project3d = Project3D_Raw(batch_size, height, width)
    depth = torch.randn(batch_size, 1, height, width)
    fx, fy, cx, cy = 1000, 1000, 320, 240
    K = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0)
    inv_K = torch.inverse(K)
    T = torch.eye(4)

    cam_points = backproject_depth(depth, inv_K)
    pix_coords = project3d(cam_points, K, T)
    print(pix_coords.shape)
    pix_coords_dbg = pix_coords.view(batch_size, -1, 2).permute(0, 2, 1)
    print(pix_coords_dbg[0,:,::640])


