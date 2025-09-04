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


def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    if opt.dataset == 'endovis':
        assert opt.eval_split_appendix in ['1','2'], "eval_split_appendix should be empty for endovis"
        assert opt.data_path == '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/', f"data_path {opt.data_path} is not correct"
    elif opt.dataset == 'StereoMIS':
        # /mnt/nct-zfs/TCO-All/SharedDatasets/StereoMIS_DARES_test/
        assert opt.eval_split_appendix == '', "eval_split_appendix should be empty for StereoMIS"
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

    dataset = SCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    
    # load reloc3r pose model, then overwrite if saved in pose.pth
    pose_model_path = os.path.join(opt.load_weights_folder, "pose.pth")
    pose_model_dict = torch.load(pose_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    reloc3r_ckpt_path = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512/Reloc3r-512.pth"
    pose_model = networks.Reloc3rX(reloc3r_ckpt_path)
    # we tested the framework should be correct as above when eval on pose_eval task.
    # pose_model = networks.UniReloc3r(reloc3r_ckpt_path, 
    #                                  opt,
    #                                  )
    model_dict = pose_model.state_dict()
    
    # log in the layers that are overwritten or remain
    # only the 1st level name should be fine
    overwritten_layers = [k for k in pose_model_dict.keys() if k in model_dict]
    remain_layers = [k for k in pose_model_dict.keys() if k not in model_dict]
    print("Overwritten layers:", set([k.split(".")[0] for k in overwritten_layers]))
    print("Remaining layers:", set([k.split(".")[0] for k in remain_layers]))

    pose_model.load_state_dict({k: v for k, v in pose_model_dict.items() if k in model_dict})
    pose_model.cuda()
    pose_model.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)

            # view1 = {'img':prepare_images(inputs[("color", 1, 0)] , device,  size = 512)}
            # view0 = {'img':prepare_images(inputs[("color", 0, 0)], device, size = 512)}
            # pose2,_  = pose_model(view0,view1)

            view1 = {'img':prepare_images(inputs[("color", 1, 0)], device, size = 512, Ks = None)}
            view2 = {'img':prepare_images(inputs[("color", 0, 0)], device, size = 512, Ks = None)}
            _ , pose2 = pose_model(view1,view2)# 

            pred_poses.append(pose2["pose"].cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    np.savez_compressed(os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "pred_pose_sq{}.npz".format(opt.eval_split_appendix)), data=np.array(pred_poses))

    gt_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "gt_poses_sq{}.npz".format(opt.eval_split_appendix))
    gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    ates = []
    res = []
    num_frames = gt_local_poses.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
        gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        res.append(compute_re(local_rs, gt_rs))

    print("\n   Trajectory error: {:0.4f}, std: {:0.4f}\n".format(np.mean(ates), np.std(ates)))
    print("\n   Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res), np.std(res)))


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
