from __future__ import absolute_import, division, print_function

import os
import torch
import networks
import numpy as np

from torch.utils.data import DataLoader
from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import SCAREDRAWDataset
from datasets import DynaSCAREDRAWDataset
from datasets import StereoMISDataset
import warnings

from torch.cuda.amp import autocast, GradScaler
import PIL
import PIL.Image
from torchvision.transforms import ToPILImage
import torchvision.transforms as tvf
from PIL import Image
from networks.utils.endofas3r_data_utils import prepare_images, resize_pil_image

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def prepare_images(x, device, size, square_ok=False):
#   to_pil = ToPILImage()
#   ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#   # ImgNorm = tvf.Compose([tvf.ToTensor()])
#   imgs = []
#   for idx in range(x.size(0)):
#       tensor = x[idx].cpu()  # Shape [3, 256, 320]
#       img = to_pil(tensor).convert("RGB")
#       W1, H1 = img.size
#       if size == 224:
#           # resize short side to 224 (then crop)
#           img = resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
#       else:
#           # resize long side to 512
#           img = resize_pil_image(img, size)
#       W, H = img.size
#       cx, cy = W//2, H//2
#       if size == 224:
#           half = min(cx, cy)
#           img = img.crop((cx-half, cy-half, cx+half, cy+half))
#       else:
#           halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
#           if not (square_ok) and W == H:
#               halfh = 3*halfw/4
#           img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
#       imgs.append(ImgNorm(img)[None].to(device))
#   return torch.stack(imgs, dim=0).squeeze()

# def resize_pil_image(img, long_edge_size):
#     S = max(img.size)
#     if S > long_edge_size:
#         interp = PIL.Image.LANCZOS
#     elif S <= long_edge_size:
#         interp = PIL.Image.BICUBIC
#     new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
#     return img.resize(new_size, interp)
    
# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]


def create_pose_model(opt, device, model_root=None):
    """Create pose model based on pose_model_type"""
    pose_model_type = getattr(opt, 'pose_model_type', 'endofast3r')
    
    if model_root is None:
        model_root = opt.load_weights_folder
    
    if pose_model_type == "endofast3r":
        return create_endofast3r_model(opt, device, model_root)
    elif pose_model_type == "endofast3r_pose_trained_dbg":
        return create_endofast3r_trained_model(opt, device, model_root)
    elif pose_model_type == "uni_reloc3r":
        return create_uni_reloc3r_model(opt, device, model_root)
    elif pose_model_type == "separate_resnet":
        return create_separate_resnet_model(opt, device, model_root)
    elif pose_model_type == "posetr_net":
        return create_posetr_net_model(opt, device, model_root)
    else:
        raise ValueError(f"Unsupported pose_model_type: {pose_model_type}")

def create_endofast3r_model(opt, device, model_root):
    """Create EndoFASt3r model"""
    # Default backbone path
    reloc3r_ckpt_path = os.path.join(model_root, "Reloc3r-512.pth")
    if not os.path.exists(reloc3r_ckpt_path):
        # Fallback to default path
        reloc3r_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512/Reloc3r-512.pth"
    
    pose_model = networks.Reloc3rX(reloc3r_ckpt_path)
    
    # Load trained weights if available
    pose_model_path = os.path.join(model_root, "pose.pth")
    if os.path.exists(pose_model_path):
        pose_model_dict = torch.load(pose_model_path)
        model_dict = pose_model.state_dict()
        
        # Log overwritten/remaining layers
        overwritten_layers = [k for k in pose_model_dict.keys() if k in model_dict]
        remain_layers = [k for k in pose_model_dict.keys() if k not in model_dict]
        print("Overwritten layers:", set([k.split(".")[0] for k in overwritten_layers]))
        print("Remaining layers:", set([k.split(".")[0] for k in remain_layers]))
        
        pose_model.load_state_dict({k: v for k, v in pose_model_dict.items() if k in model_dict})
        print(f"Loaded trained EndoFASt3r weights from {pose_model_path}")
    else:
        print(f"Using pretrained EndoFASt3r from {reloc3r_ckpt_path}")
    
    return pose_model

def create_endofast3r_trained_model(opt, device, model_root):
    """Create trained EndoFASt3r model"""
    pose_model_path = os.path.join(model_root, "pose.pth")
    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(f"Trained pose model not found at {pose_model_path}")
    
    pose_model_dict = torch.load(pose_model_path)
    
    # Try to find backbone in model_root first
    reloc3r_ckpt_path = os.path.join(model_root, "Reloc3r-512.pth")
    if not os.path.exists(reloc3r_ckpt_path):
        # Fallback to default path
        reloc3r_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512/Reloc3r-512.pth"
    
    pose_model = networks.Reloc3rX(reloc3r_ckpt_path)
    model_dict = pose_model.state_dict()
    
    # Log overwritten/remaining layers
    overwritten_layers = [k for k in pose_model_dict.keys() if k in model_dict]
    remain_layers = [k for k in pose_model_dict.keys() if k not in model_dict]
    print("Overwritten layers:", set([k.split(".")[0] for k in overwritten_layers]))
    print("Remaining layers:", set([k.split(".")[0] for k in remain_layers]))
    
    pose_model.load_state_dict({k: v for k, v in pose_model_dict.items() if k in model_dict})
    print(f"Loaded endofast3r_pose_trained_dbg pose model from {pose_model_path}")
    return pose_model

def create_uni_reloc3r_model(opt, device, model_root):
    """Create UniReloc3r model"""
    # Try to find backbone in model_root first
    reloc3r_ckpt_path = os.path.join(model_root, "Reloc3r-512.pth")
    if not os.path.exists(reloc3r_ckpt_path):
        # Fallback to default path
        reloc3r_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512/Reloc3r-512.pth"
    
    pose_model = networks.UniReloc3r(reloc3r_ckpt_path, opt)
    
    # Load trained weights if available
    pose_model_path = os.path.join(model_root, "pose.pth")
    if os.path.exists(pose_model_path):
        pose_model_dict = torch.load(pose_model_path)
        pose_model.load_state_dict(pose_model_dict)
        print(f"Loaded trained UniReloc3r weights from {pose_model_path}")
    else:
        print(f"Using pretrained UniReloc3r from {reloc3r_ckpt_path}")
    
    return pose_model
 

def create_posetr_net_model(opt, device, model_root):
    """Create PoseTrNet model"""

    from functools import partial
    from posetr.posetr_model_v2 import PoseTransformerV2
    pose_model = PoseTransformerV2(
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

    pose_model_path = os.path.join(model_root, "pose.pth")
    if os.path.exists(pose_model_path):
        pose_model_dict = torch.load(pose_model_path)
        pose_model.load_state_dict(pose_model_dict, strict=True)
        print(f"Loaded trained PoseTrNet weights from {pose_model_path}")
    else:
        assert False, "PoseTrNet weights not found in model_root"
        print(f"Using pretrained PoseTrNet from {model_root}")

 
    return pose_model
 

def create_separate_resnet_model(opt, device, model_root):
    """Create separate ResNet model"""
    pose_encoder_path = os.path.join(model_root, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_root, "pose.pth")
    
    # Fallback to default paths if not found in model_root
    if not os.path.exists(pose_encoder_path):
        assert False, "pose_encoder.pth not found in model_root"
    if not os.path.exists(pose_decoder_path):
        assert False, "pose.pth not found in model_root"
    
    pose_encoder = networks.ResnetEncoder(
        getattr(opt, 'num_layers', 18),
        getattr(opt, 'weights_init', 'pretrained') == "pretrained",
        num_input_images=2
    )
    pose_decoder = networks.PoseDecoder(
        pose_encoder.num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2
    )
    
    # Load weights if available
    if os.path.exists(pose_encoder_path):
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))
        print(f"Loaded pose encoder from {pose_encoder_path}")
    if os.path.exists(pose_decoder_path):
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))
        print(f"Loaded pose decoder from {pose_decoder_path}")
    
    # Combine encoder and decoder
    pose_model = torch.nn.ModuleDict({
        'encoder': pose_encoder,
        'decoder': pose_decoder
    })
    
    print("Loaded separate ResNet pose model...")
    return pose_model

 

def predict_pose_for_model(pose_model, pose_model_type, inputs, device, opt=None):
    """Predict pose based on different pose_model_type implementations"""
    
    if pose_model_type in ["endofast3r", "endofast3r_pose_trained_dbg", "uni_reloc3r"]:
        # These models use the same interface: view1, view2 -> _, pose2
        view1 = {'img': prepare_images(inputs[("color", 1, 0)], device, size=512, Ks=None)}
        view2 = {'img': prepare_images(inputs[("color", 0, 0)], device, size=512, Ks=None)}
        _, pose2 = pose_model(view1, view2)
        return pose2["pose"]
    elif pose_model_type in ["posetr_net"]:
        # These models use the same interface: view1, view2 -> _, pose2
        view1 = {'img': inputs[("color", 1, 0)] }
        view2 = {'img': inputs[("color", 0, 0)] }
        _, pose2 = pose_model(view1, view2)
        return pose2["pose"]

    # elif pose_model_type == "pcrnet":
    #     # PCRNet requires depth and camera points
    #     # For evaluation, we need to provide depth - this is a limitation
    #     # In practice, you'd need a depth model or GT depth
    #     raise NotImplementedError("PCRNet evaluation requires depth input. Please provide depth model or GT depth.")
    
    # elif pose_model_type == "geoaware_pnet":
    #     # GeoAware PNet requires depth and specific processing
    #     # Similar to PCRNet, needs depth input
    #     raise NotImplementedError("GeoAware PNet evaluation requires depth input. Please provide depth model or GT depth.")
    
    # elif pose_model_type == "diffposer_epropnp":
    #     # DiffPoseR epropnp requires specific setup
    #     raise NotImplementedError("DiffPoseR epropnp evaluation not implemented yet.")
    
    elif pose_model_type == "separate_resnet":
        # Separate ResNet uses encoder + decoder
        pose_inputs = [pose_model['encoder'](torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1))]
        axisangle, translation = pose_model['decoder'](pose_inputs)
        
        # Convert to transformation matrix
        from layers import transformation_from_parameters
        pose_matrix = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
        return pose_matrix
    
    # elif pose_model_type == "shared":
    #     # Shared model uses encoder from depth model
    #     raise NotImplementedError("Shared pose model evaluation requires depth model integration.")
    
    # elif pose_model_type == "posecnn":
    #     # PoseCNN takes concatenated images
    #     concat_input = torch.cat([inputs[("color", 0, 0)], inputs[("color", 1, 0)]], 1)
    #     pose_output = pose_model(concat_input)
        
    #     # PoseCNN typically outputs axis-angle and translation
    #     if hasattr(pose_output, 'axisangle') and hasattr(pose_output, 'translation'):
    #         from layers import transformation_from_parameters
    #         pose_matrix = transformation_from_parameters(pose_output.axisangle, pose_output.translation)
    #     else:
    #         # Assume direct pose matrix output
    #         pose_matrix = pose_output
    #     return pose_matrix
    
    else:
        raise ValueError(f"Unsupported pose_model_type for evaluation: {pose_model_type}")

# def predict_pose_with_depth(pose_model, pose_model_type, inputs, device, depth_model=None, opt=None):
#     """Predict pose for models that require depth input (PCRNet, GeoAware PNet)"""
    
#     if pose_model_type == "pcrnet":
#         # Get depth predictions
#         if depth_model is not None:
#             with torch.no_grad():
#                 depth_f0 = depth_model(inputs[("color", 0, 0)])
#                 depth_f1 = depth_model(inputs[("color", 1, 0)])
#         else:
#             # Use GT depth if available
#             if ("depth", 0, 0) in inputs and ("depth", 1, 0) in inputs:
#                 depth_f0 = inputs[("depth", 0, 0)]
#                 depth_f1 = inputs[("depth", 1, 0)]
#             else:
#                 raise ValueError("PCRNet requires depth input. Provide depth_model or GT depth.")
        
#         # Get camera intrinsics
#         K = inputs[("K", 0)][:, :3, :3]  # Assuming K is available
#         inv_K = torch.inverse(K)
        
#         # Create backproject_depth function (simplified version)
#         def backproject_depth(depth, inv_K):
#             B, H, W = depth.shape
#             i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), 
#                                  torch.linspace(0, H-1, H, device=device), indexing='xy')
#             i = i.t().float()
#             j = j.t().float()
            
#             pts = torch.stack([i, j, torch.ones_like(i)], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
#             pts = pts.view(B, 3, -1)
#             pts = inv_K.unsqueeze(-1) @ pts
#             pts = pts * depth.view(B, 1, -1)
#             pts = torch.cat([pts, torch.ones(B, 1, H*W, device=device)], dim=1)
#             return pts
        
#         # Get camera points
#         cam_points_f0 = backproject_depth(depth_f0, inv_K)
#         cam_points_f1 = backproject_depth(depth_f1, inv_K)
        
#         # Convert to point clouds
#         pcd_f0 = cam_points_f0[:, :3, :].permute(0, 2, 1)
#         pcd_f1 = cam_points_f1[:, :3, :].permute(0, 2, 1)
        
#         # Center the point clouds
#         pcd_f0 = pcd_f0 - torch.mean(pcd_f0, dim=1, keepdim=True).detach()
#         pcd_f1 = pcd_f1 - torch.mean(pcd_f1, dim=1, keepdim=True).detach()
        
#         # Get pose estimation
#         pose_result = pose_model(pcd_f1, pcd_f0, max_iteration=getattr(opt, 'pcrnet_max_iteration', 10))
#         return pose_result['est_T']
    
#     elif pose_model_type == "geoaware_pnet":
#         # Similar to PCRNet but with different processing
#         raise NotImplementedError("GeoAware PNet depth-based evaluation not fully implemented yet.")
    
#     else:
#         raise ValueError(f"pose_model_type {pose_model_type} does not require depth input.")

def compute_rpe_translation(gt_poses, pred_poses, delta=1):
    """Compute Relative Pose Error for translation"""
    rpe_trans = []
    
    for i in range(len(gt_poses) - delta):
        # Ground truth relative pose
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        gt_trans = gt_rel[:3, 3]
        
        # Predicted relative pose
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + delta]
        pred_trans = pred_rel[:3, 3]
        
        # Translation error
        trans_error = np.linalg.norm(gt_trans - pred_trans)
        rpe_trans.append(trans_error)
    
    return np.array(rpe_trans)

def compute_rpe_rotation(gt_poses, pred_poses, delta=1):
    """Compute Relative Pose Error for rotation"""
    rpe_rot = []
    
    for i in range(len(gt_poses) - delta):
        # Ground truth relative pose
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        gt_rot = gt_rel[:3, :3]
        
        # Predicted relative pose
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + delta]
        pred_rot = pred_rel[:3, :3]
        
        # Rotation error (angle between rotation matrices)
        R = gt_rot @ np.linalg.inv(pred_rot)
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rpe_rot.append(angle)
    
    return np.array(rpe_rot)

def evaluate(opt, model_root=None, depth_model=None):
    """Evaluate odometry on the SCARED dataset
    """
    if model_root is None:
        model_root = opt.load_weights_folder
    
    assert os.path.isdir(model_root), \
        "Cannot find a folder at {}".format(model_root)

    if opt.dataset == 'endovis':
        assert opt.eval_split_appendix in ['1','2'], "eval_split_appendix should be empty for endovis"
        assert opt.data_path == '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/', f"data_path {opt.data_path} is not correct"
    elif opt.dataset == 'StereoMIS':
        # /mnt/nct-zfs/TCO-All/SharedDatasets/StereoMIS_DARES_test/
        # assert opt.eval_split_appendix == '', "eval_split_appendix should be empty for StereoMIS"
        assert opt.data_path == '/mnt/nct-zfs/TCO-All/SharedDatasets/StereoMIS_DARES_test/', f"data_path {opt.data_path} is not correct"
        assert NotImplementedError("StereoMIS is not supported for pose evaluation yet")
    elif opt.dataset == 'DynaSCARED':
        # use the 100_300 test traj!
        assert opt.eval_split_appendix in ['000_00597'], "eval_split_appendix can be case_scene format when eval pose for DynaSCARED"
        assert NotImplementedError("DynaSCARED is not supported for pose evaluation yet")
        assert opt.data_path == '/mnt/cluster/datasets/Surg_oclr_stereo/', f"data_path {opt.data_path} is not correct"
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")

    fpath = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "test_files_sequence{}.txt".format(opt.eval_split_appendix))
    filenames = readlines(fpath)

    # Check if skip_inference is enabled
    skip_inference = opt.skip_inference#getattr(opt, 'skip_inference', False)
    
    if skip_inference:
        print("-> Skipping inference, loading pre-computed trajectory estimates")
        
        # Load pre-computed predictions
        pred_poses_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, 
                                       "pred_pose_sq{}{}.npz".format(opt.eval_split_appendix, opt.eval_model_appendix))
        
        if not os.path.exists(pred_poses_path):
            raise FileNotFoundError(f"Pre-computed predictions not found at {pred_poses_path}. Please run inference first or disable skip_inference.")
        
        pred_poses = np.load(pred_poses_path)["data"]
        print(f"Loaded {len(pred_poses)} pre-computed pose predictions from {pred_poses_path}")
        
    else:
        # Original inference pipeline
        # data
        if opt.dataset == 'DynaSCARED':
            dataset = DynaSCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                                    [0, 1], 4, is_train=False)
        elif opt.dataset == 'endovis':
            dataset = SCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                                    [0, 1], 4, is_train=False)
        elif opt.dataset == 'StereoMIS':
            dataset = StereoMISDataset(opt.data_path, filenames, opt.height, opt.width,
                                    [0, 1], 4, is_train=False)
        else:
            assert False, "Unknown dataset: {opt.dataset}"

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        
        # Create pose model based on pose_model_type
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pose_model = create_pose_model(opt, device, model_root)
        pose_model.cuda()
        pose_model.eval()

        pred_poses = []
        pose_model_type = getattr(opt, 'pose_model_type', 'endofast3r')

        print("-> Computing pose predictions")
        print(f"Using pose_model_type: {pose_model_type}")
        print(f"Model root: {model_root}")

        opt.frame_ids = [0, 1]  # pose network only takes two frames as input

        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(device)

                try:
                    pose_matrix = predict_pose_for_model(pose_model, pose_model_type, inputs, device, opt)
                    pred_poses.append(pose_matrix.cpu().numpy())
                    
                except NotImplementedError as e:
                    print(f"Error: {e}")
                    print(f"Skipping evaluation for pose_model_type: {pose_model_type}")
                    return
                except Exception as e:
                    print(f"Error during pose prediction: {e}")
                    print(f"pose_model_type: {pose_model_type}")
                    raise

        pred_poses = np.concatenate(pred_poses)
        
        # Save predictions for future use
        np.savez_compressed(os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "pred_pose_sq{}.npz".format(opt.eval_split_appendix)), data=np.array(pred_poses))
        print(f"Saved predictions to splits/{opt.dataset}/pred_pose_sq{opt.eval_split_appendix}.npz")

    # Load ground truth poses (same for both inference and skip_inference)
    if opt.dataset == 'endovis':
        gt_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "gt_poses_sq{}.npz".format(opt.eval_split_appendix))
        gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    elif opt.dataset == 'StereoMIS':
        gt_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "gt_poses_sq{}.npz".format(opt.eval_split_appendix))
        gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        # scale the translation with 1000
        gt_local_poses[:, :3, 3] = gt_local_poses[:, :3, 3] * 1000
    elif opt.dataset == 'DynaSCARED':
        # For DynaSCARED, we need to load the dataset to get ground truth poses
        dataset = DynaSCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                                [0, 1], 4, is_train=False)
        assert len(dataset.trajs_dict) == 1, f"Number of trajectories {len(dataset.trajs_dict)} does not match number of predictions {len(pred_poses)}"
        gt_local_poses = dataset.trajs_dict[list(dataset.trajs_dict.keys())[0]]
        print(f"Loaded {len(gt_local_poses)} gt poses")

    else:
        assert False, "Unknown dataset: {opt.dataset}"

    # Compute evaluation metrics
    print("\n-> Computing evaluation metrics...")
    
    # ATE and RE (existing metrics)
    def compute_metrics(gt_local_poses, pred_poses, track_length):
        ates = []
        res = []
        num_frames = gt_local_poses.shape[0]
        # Use consistent values
        # track_length = 5
        delta = track_length - 1  # delta = 4

        for i in range(0, num_frames - 1):
            local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
            gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
            local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
            gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))

            ates.append(compute_ate(gt_local_xyzs, local_xyzs))
            res.append(compute_re(local_rs, gt_rs))

        # RPE metrics (new)
        # TODO: SCALE the esti
        # rpe_trans = compute_rpe_translation(gt_local_poses, pred_poses, delta=delta)
        # rpe_rot = compute_rpe_rotation(gt_local_poses, pred_poses, delta=delta)
        

        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS GIVEN TRACK LENGTH = {}".format(track_length))
        print("="*60)
        
        print(f"\nAbsolute Trajectory Error (ATE):")
        print(f"   Mean: {np.mean(ates):.8f}, Std: {np.std(ates):.8f}")
        print(f"\nRotation Error (RE):")
        print(f"   Mean: {np.mean(res):.8f}, Std: {np.std(res):.8f}")
        
        # print(f"\nRelative Pose Error - Translation (RPE-T):")
        # print(f"   Delta={delta}: Mean: {np.mean(rpe_trans):.8f}, Std: {np.std(rpe_trans):.8f}")
        # print(f"\nRelative Pose Error - Rotation (RPE-R):")
        # print(f"   Delta={delta}: Mean: {np.mean(rpe_rot):.8f}, Std: {np.std(rpe_rot):.8f}")
        
        print("="*60)


    compute_metrics(gt_local_poses, pred_poses, track_length=len(pred_poses))
    compute_metrics(gt_local_poses, pred_poses, track_length=5)
    compute_metrics(gt_local_poses, pred_poses, track_length=2)

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
