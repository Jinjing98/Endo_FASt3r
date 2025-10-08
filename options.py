from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # eval only
        self.parser.add_argument("--skip_inference",
                                 help="if set skips inference",
                                 action="store_true")
        
        # dataset options
        self.parser.add_argument("--of_samples",
                                 help="alwasy choose seveal samples for used for of",
                                 action="store_true")
        self.parser.add_argument("--of_samples_num",                                 
                                 help="alwasy choose seveal samples for used for of",
                                 type=int,
                                 default=10)
        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "endovis_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # Arch/Loss options
        self.parser.add_argument("--enable_motion_computation",
                                 help="it will compute: motion_flow, pose_flow, motion_mask, color_motion_warped",
                                 action="store_true")
        
        self.parser.add_argument("--use_MF_network",
                                 help="if set, uses MF network",
                                 action="store_true")
        self.parser.add_argument("--shared_MF_OF_network",
                                 help="if set, uses MF network",
                                 action="store_true")

        self.parser.add_argument("--reproj_supervised_with_which",
                                 type=str,
                                 help="which GT image to supervise with",
                                 default="refined",
                                 choices=["refined", "raw_tgt_gt"])
        self.parser.add_argument("--reproj_supervised_which",
                                 type=str,
                                 help="which image to supervise",
                                 default="color",
                                 choices=["color",'color_MotionCorrected','motion_masked_color', 'gt_motion_masked_color_debug'])
        #/////MOTION HANDLING WHEN CALIB/////
        self.parser.add_argument("--ignore_motion_area_at_calib",
                                 help="if set, ignore the motion area at calib",
                                 action="store_true")

        #/////REGULARIZATION LOSS/////
        self.parser.add_argument("--use_loss_motion_mask_reg",
                                 help="maybe we can unfreeze depth with this regularization,used to force the structure of motion_mask is similar to optic_flow, implementaion varied for soft_mask and hard_mask, make sure gard_is enable for the computed motion_mask",
                                 action="store_true")
        self.parser.add_argument("--enable_grad_flow_motion_mask",
                                 help="This is critical to enabel if use_motion_mask_reg_loss is on.if set, uses grad flow motion mask; we implement such a version also for binary mask",
                                 action="store_true")        
        self.parser.add_argument("--motion_mask_reg_loss_weight",
                                 type=float,
                                 help="weight for the motion mask regularization loss",
                                 default=0.01)
        self.parser.add_argument("--use_soft_motion_mask",
                                 help="if set, uses soft motion mask; we notice we can still get some confidence despite for SCARED dataset---can be potentially useful for other datasets",
                                 action="store_true")
        # other reg related hyper: valid_motion_threshold_px

        self.parser.add_argument("--static_flow_noise_thre",
                                 type=float,
                                 help="threshold for filter noise in flow_mag",
                                 default=5e-2)
        self.parser.add_argument("--valid_motion_threshold_px",
                                 type=float,
                                 help="can be critical--define how to norm the flow_mag when supervise the motion_mask in the reg",
                                 default=0.5)
        self.parser.add_argument("--contrast_alpha",
                                 type=float,
                                 help="contrast_alpha when norm the flow_mag",
                                 default=10.0)
        self.parser.add_argument("--motion_mask_thre_px",
                                 type=float,
                                 help="only used when non_soft, by far only used when compute binary mask, smaller, more aggressive(safe), threshold for motion mask, if flow norm is less than this, set motion mask to 1",
                                 default=3,
                                 choices=[3, 1,])


        self.parser.add_argument("--use_loss_reproj2_nomotion",
                                 help="used to addtionally supervise the pose flow to be effective---imporant to get good pose when eval!",
                                 action="store_true")
        
        self.parser.add_argument("--loss_reproj2_nomotion_weight",
                                 type=float,
                                 help="weight for the reprojection loss",
                                 default=1.0)

        self.parser.add_argument("--reproj2_supervised_with_which",
                                 type=str,
                                 help="which GT image to supervise with",
                                 default="refined",
                                 choices=["refined", "raw_tgt_gt"])
        self.parser.add_argument("--reproj2_supervised_which",
                                 type=str,
                                 help="which image to supervise",
                                 default="color",
                                 choices=['color'])
        # self.parser.add_argument("--reproj2_turnoff_motion_based mask",
        #                          help="it will compute: motion_flow, pose_flow, motion_mask, color_motion_warped",
        #                          action="store_true")
        self.parser.add_argument("--enable_mutual_motion",
                                 help="can be expensive as evertyhign including depth need to computer s2t version, it will compute: motion_flow, pose_flow, motion_mask, color_motion_warped",
                                 action="store_true")
        self.parser.add_argument("--enable_all_depth",
                                 help="have to be on when pose_model_type is pcrnet",
                                 action="store_true")

        self.parser.add_argument("--reg_mutual_raw_disp_based_OF_for_consistency_and_correctness",
                                 help="inspired from VDA; hope to address noisy depth problem",
                                 action="store_true")        

        #/////////

        
        self.parser.add_argument("--flow_reproj_supervised_with_which",
                                 type=str,
                                 help="which image to supervise",
                                 default="detached_refined",
                                 choices=["detached_refined",'raw_tgt_gt'])
        # self.parser.add_argument("--disable_refine",
        #                          help="it will compute: motion_flow, pose_flow, motion_mask, color_motion_warped",
        #                          action="store_true")
        self.parser.add_argument("--zero_pose_flow_debug",
                                 help="it will compute: motion_flow, pose_flow, motion_mask, color_motion_warped",
                                 action="store_true")
        self.parser.add_argument("--gt_metric_rel_pose_as_estimates_debug",
                                 help="it will compute: motion_flow, pose_flow, motion_mask, color_motion_warped",
                                 action="store_true")        
        self.parser.add_argument("--zero_pose_debug",
                                 help="it will set T as eye",
                                 action="store_true")
        self.parser.add_argument("--freeze_depth_debug",
                                 help="it will freeze depth_model, solution space too big for pose model if dpeth is also trainiable",
                                 action="store_true")


        #uni_reloc3r: PoseNet
        # --pretrained "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512/Reloc3r-512.pth" \
        # --pretrained "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/crocoflow.pth" \
        # --pretrained "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
        # --pretrained "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/monst3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \ 

        self.parser.add_argument("--backbone_pretrain_ckpt_path",
                                 type=str,
                                 help="inti dust3r backbone from the pretrained_root",
                                 default='/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512/Reloc3r-512.pth')
        
        # geoaware net
        self.parser.add_argument("--geoaware_cfg_path",
                                 type=str,
                                 help='path to the geoaware config json',
                                 default='/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/geoaware_pnet/transformer/config/nerf_focal_12T1R_256_homo_c2f_geoaware.json'
                                 )
        self.parser.add_argument("--load_geoaware_pretrain_model",
                                 help='if set will load marepo_9D.pt/marepo.pt depending on the config choice',
                                 action="store_true")
        
        
        #model arch
        self.parser.add_argument("--init_3d_scene_flow",
                                 help="if set, initializes the 3d scene flow",
                                 action="store_true")
        self.parser.add_argument("--scene_flow_estimator_type",
                                 type=str,
                                 help="scene flow estimator type",
                                 default="dpt",
                                 choices=["dpt", "linear"])
        self.parser.add_argument("--init_2d_optic_flow",
                                 help="if set, initializes the 2d optic flow",
                                 action="store_true")
        self.parser.add_argument("--optic_flow_estimator_type",
                                 type=str,
                                 help="optic flow estimator type",
                                 default="linear",
                                 choices=["dpt", "linear"])
        
        self.parser.add_argument("--unireloc3r_pose_estimation_mode",
                                 type=str,
                                 help="pose estimation mode",
                                 default="vanilla_pose_head_regression",
                                 choices=["vanilla_pose_head_regression", "epropnp", "geoaware_pnet"])
        self.parser.add_argument("--pose_regression_with_mask",
                                 help="if set, uses mask to regress pose",
                                 action="store_true")
        self.parser.add_argument("--pose_regression_which_mask",
                                 type=str,
                                 help="which mask to use for pose regression",
                                 default="gt",
                                 choices=["gt", "esti", "detached_esti"])
        self.parser.add_argument("--pose_regression_head_input",
                                 type=str,
                                 help="which input to use for pose regression",
                                 default="default",
                                 choices=["default", "mapero", "optic_flow", "corr", "cat_feats", "add_feats", "optic_flow"])
        self.parser.add_argument("--mapero_pixel_pe_scheme",
                                 type=str,
                                 help="which pixel pe scheme to use for pose regression",
                                 default="focal_norm",
                                 choices=["focal_norm", "focal_norm_OF_warped"])
        # overwrite with endofast3r pose_head format
        self.parser.add_argument("--disable_pose_head_overwrite",
                                 help="if set, update the pose head with the endofast3r format",
                                 action="store_true")
        



        # TRAINING options
        self.parser.add_argument("--debug",
                                 help="1.adjust num_epochs, model_name, batch_size, log_frequency, log_dir for debug purposes",
                                 action="store_true")
        self.parser.add_argument("--freeze_as_much_debug",
                                 help="0. frezze a lot param to save mem, local debug only to save mem",
                                 action="store_true")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        # replce with --dataset
        # self.parser.add_argument("--split",
        #                          type=str,
        #                          help="which training split to use: various datasets",
        #                          choices=["endovis", "eigen_zhou", "eigen_full", "odom", "benchmark", "DynaSCARED"],
        #                          default="endovis")
        self.parser.add_argument("--split_appendix",
                                 type=str,
                                 help="appendix to the split: DynaSCARED",
                                 default="",
                                 choices=["", "_CaToTi000", 
                                          "_CaToTi011",
                                          "_CaToTi010", #debug if tool move too fast break flow
                                          ])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="endovis",
                                 choices=["endovis","DynaSCARED","Hamlyn","StereoMIS", "kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-4)
        self.parser.add_argument("--position_smoothness",
                                 type=float,
                                 help="registration smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--consistency_constraint",
                                 type=float,
                                 help="consistency constraint weight",
                                 default=0.01)
        self.parser.add_argument("--epipolar_constraint",
                                 type=float,
                                 help="epipolar constraint weight",
                                 default=0.01)
        self.parser.add_argument("--geometry_constraint",
                                 type=float,
                                 help="geometry constraint weight",
                                 default=0.01)
        self.parser.add_argument("--transform_constraint",
                                 type=float,
                                 help="transform constraint weight",
                                 default=0.01)
        self.parser.add_argument("--transform_smoothness",
                                 type=float,
                                 help="transform smoothness weight",
                                 default=0.01)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth mm",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth mm",
                                 default=150.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--accumulate_steps",
                                 type=int,
                                 help="number of steps to accumulate gradients (effective batch size = batch_size * accumulate_steps)",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--seed",
                                 type=int,
                                 help="random seed",
                                 default=1234)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="endofast3r",
                                 choices=["posecnn", 
                                          "shared",
                                          "geoaware_pnet",
                                          "diffposer_epropnp",
                                          "separate_resnet", 
                                          "endofast3r",
                                          "endofast3r_pose_trained_dbg",
                                          "uni_reloc3r",
                                          "pcrnet",
                                          "posetr_net"
                                          ])
        
        self.parser.add_argument("--depth_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="dam",
                                 choices=["dam", 
                                          "endofast3r_depth_trained_dbg",
                                          ])        
         #pcrnet arch:
        self.parser.add_argument("--pcrnet_max_iteration",
                                 type=int,
                                 help="max iteration for pcrnet",
                                 default=3)
        self.parser.add_argument("--pcrnet_enable_flow_based_matching",
                                 help="if set, enables flow based matching for pcrnet as input",
                                 action="store_true")
         # geoaware_pnet arch:
        self.parser.add_argument("--use_pyramid",
                                 help="if set disables CUDA",
                                 action="store_true")         #         
        self.parser.add_argument("--2d_pe_type",
                                 type=str,
                                 help="2d pe type, lofter is the original version, 2d_bv_nerf is the nerf version",
                                 default="lofter",
                                 choices=["lofter", '2d_bv_nerf'])
        self.parser.add_argument("--3d_pe_type",
                                 type=str,
                                 help="nerf version use 3 chanel while lofer only use the 2 channels of bv",
                                 default="3d_bv_lofter",
                                 choices=["3d_bv_lofter", "nerf"])


        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["position_encoder", "position"])
        
        # OPTICAL FLOW options
        self.parser.add_argument("--use_raft_flow",
                                 help="use RAFT optical flow/motion_flow estimator instead of custom position_encoder/position networks",
                                 action="store_true")
 

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=200)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        # replce with --dataset
        # self.parser.add_argument("--eval_split",
        #                          type=str,
        #                          default="endovis",
        #                          choices=[
        #                             "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "endovis", "DynaSCARED"],
        #                          help="which split to run eval on")
        self.parser.add_argument(
            "--eval_split_appendix",
            type=str,
            default="",
            choices=["", "_CaToTi000", "_CaToTi011", "1", "2", "000_00597", '3'],
            help=(
                "Appendix to the eval split. Options:\n"
                "  Endovis pose: 1, 2, (scared_pose_seq)\n"
                "  StereoMIS pose: 1\n"
                "  StereoMIS pose: 3\n"
                "  Dynascared pose: 000_00597\n"
                "  Endovis depth: ''\n"
                "  Hamlyn_depth: ''\n"
                "  DynaSCARED_depth: '', '_CaToTi000', '_CaToTi011'\n"
            )
        )

        self.parser.add_argument('--eval_model_appendix', type=str, default='')

        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        # self.parser.add_argument("--scared_pose_seq",
        #                          type=str,
        #                          help="pose sequence in scared",
        #                          default=1)


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
    
    def parse_notebook(self, args):
        self.options = self.parser.parse_args(args)
        return self.options

