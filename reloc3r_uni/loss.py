# references: DUSt3R: https://github.com/naver/dust3r


from copy import copy, deepcopy
import math
import torch
import torch.nn as nn
from mvp3r.loss import LLoss
from mvp3r.loss import L21, BCE, LRT
from mvp3r.loss import Recon, L21PoseLoss, ConfLoss, Regr3D_Sym, RelativeCameraPoseRegression, MotionConfRegression,Regr3D_Sym_ScaleShiftInv
from mvp3r.loss import SSIM, Pose_LRT, MM_BCE, Pose_LRT_MM_BCE, Pose_LRT_MM_BCE_Flow_l21 # internall constructed for short
from mvp3r.loss import LaplacianLossBounded_OF
from mvp3r.loss import Criterion, MultiLoss


def loss_of_one_batch(batch, model, criterion, device, use_amp=False, ret=None):
    try:
        # the dataset is in reloc3r/dust3r format
        view1, view2 = batch
        # print('processing view1:',view1['label'], view1['instance'], view1['camera_pose'])
        # print('processing view2:',view2['label'], view2['instance'], view2['camera_pose'])
    except ValueError:
        # the dataset is in croco_flow format
        # reconstruct the OF batch as reloc3r format
        image1, image2, fwd_flow1to2, pairname = batch
        # prepare as view1, view2 format
        view1 = dict(img=image1, gt_flow=fwd_flow1to2, )#camera_intrinsics=None, camera_pose=None, dynamic_mask=None, valid_mask=None, pts3d=None)
        print('TODO we do not have backward GT flow for sintel yet')
        view2 = dict(img=image2, gt_flow=-fwd_flow1to2, )#camera_intrinsics=None, camera_pose=None, dynamic_mask=None, valid_mask=None, pts3d=None)
        batch = [view1, view2]

    for view in batch:
        # for name in 'img camera_intrinsics camera_pose'.split(): 
        # put_in_device_list = ['img', 'camera_intrinsics', 'camera_pose', 'dynamic_mask', 'valid_mask','pts3d']
        put_in_device_list = ['img', 'camera_intrinsics', 'camera_pose', 'dynamic_mask', 'valid_mask', 'pts3d', 'gt_flow']
        # for name in 'img camera_intrinsics camera_pose dynamic_mask valid_mask'.split(): 
        for name in put_in_device_list: 
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):

        pose1, pose2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            # if isinstance(criterion,LaplacianLossBounded_OF):
            #     loss = criterion(view1, view2, pose1, pose2) 
            # else:
            # loss = criterion(view1, view2, pose1, pose2) if criterion is not None else None
            loss = criterion(view1, view2, pose1, pose2, exp_id = model.exp_id, output_dir = model.output_dir) if criterion is not None else None

    # Extract and clear clamping statistics if available
    clamp_stats = {}
    if hasattr(model, 'clamp_stats') and model.clamp_stats is not None:
        clamp_stats = model.clamp_stats.copy()
        model.clamp_stats = {}  # Clear to avoid accumulation

    result = dict(view1=view1, view2=view2, pose1=pose1, pose2=pose2, loss=loss, clamp_stats=clamp_stats)
    return result[ret] if ret else result

