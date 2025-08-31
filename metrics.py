# gt_c2w_poses
# gt_rel_poses
import torch 

# the metrics are extendted from relo3r


def transl_ang_loss(t, tgt, eps=1e-6):
    """
    extend  for metric pose
    Args: 
        t: estimated translation vector [B, 3]
        tgt: ground-truth translation vector [B, 3]
    Returns: 
        T_err: translation direction angular error 
    """
    assert t.dim() == 2, f't: {t.shape}'
    assert tgt.dim() == 2, f'tgt: {tgt.shape}'
    assert t.shape[1] == 3, f't: {t.shape}'
    assert tgt.shape[1] == 3, f'tgt: {tgt.shape}'
    
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t_normed = t / (t_norm + eps)
    tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
    tgt_normed = tgt / (tgt_norm + eps)
    cosine = torch.sum(t_normed * tgt_normed, dim=1)
    T_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
    return T_err.mean(), T_err

def transl_scale_loss(t, tgt, eps=1e-6, norm_gt=False, norm_esti=False):
    """
    Args: 
        t: estimated translation vector [B, 3]
        tgt: ground-truth translation vector [B, 3]
        norm: whether to normalize vectors before computing loss
        eps: small value to prevent division by zero
    Returns: 
        T_err: translation error
    """
    if norm_esti:
        t_norm = torch.norm(t, dim=1, keepdim=True)
        t_normed = t / (t_norm + eps)
    else:
        t_normed = t
    if norm_gt:
        tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
        tgt_normed = tgt / (tgt_norm + eps)
    else:
        tgt_normed = tgt

    transl_loss = torch.norm(t_normed - tgt_normed, dim=1)
    
    return transl_loss.mean(), transl_loss


def rot_ang_loss(R, Rgt, eps=1e-6):
    """
    Args:
        R: estimated rotation matrix [B, 3, 3]
        Rgt: ground-truth rotation matrix [B, 3, 3]
    Returns:  average_over_the_batch
        R_err: rotation angular error 
    """
    residual = torch.matmul(R.transpose(1, 2), Rgt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
    return R_err.mean(), R_err

def compute_pose_error_v2(gt_rel_poses, pred_rel_poses):
    """
    Compute pose errors between ground truth and predicted relative poses.
    Args:
        gt_rel_poses: (B, 4, 4) Ground truth relative poses
        pred_rel_poses: (B, 4, 4) Predicted relative poses
    Returns:
        err_dict: Dictionary containing translation and rotation errors
        support multiple metrics: 
            trans_err_ang, 
            trans_err_ang_deg, 
            trans_err_scale, 
            trans_err_scale_norm, 
            rot_err, 
            rot_err_deg
    """

    # Extract translation and rotation components
    t = pred_rel_poses[:, 0:3, -1]  # [B, 3]
    tgt = gt_rel_poses[:, 0:3, -1]  # [B, 3]
    R = pred_rel_poses[:, :3, :3]   # [B, 3, 3]
    Rgt = gt_rel_poses[:, :3, :3]   # [B, 3, 3]

    # compute translation error with angular version and scalr version
    # already mean over batches internally; the raw_saved non meaned version
    trans_err_ang, trans_err_ang_raw = transl_ang_loss(t, tgt)
    trans_err_scale, trans_err_scale_raw  = transl_scale_loss(t, tgt, norm_gt=False, norm_esti=False)
    trans_err_scale_norm, trans_err_scale_norm_raw  = transl_scale_loss(t, tgt, norm_gt=True, norm_esti=True)

    # compute rotation error
    rot_err, rot_err_raw = rot_ang_loss(R, Rgt)
    
    # comment depending on the needs:
    # the raw version is all errors per batch without mean averaging
    err_dict = {
        # 'trans_err_ang': trans_err_ang,
        'trans_err_ang_deg': trans_err_ang * 180 / torch.pi,
        'trans_err_scale': trans_err_scale,
        # 'trans_err_scale_norm': trans_err_scale_norm,
        # 'rot_err': rot_err,
        'rot_err_deg': rot_err * 180 / torch.pi
    }
    
    # print('rot_err_deg_raw:')
    # print(rot_err_raw * 180 / torch.pi)
    # print('rot_err_deg:')
    # print(rot_err * 180 / torch.pi)

    return err_dict





# def compute_pose_error(gt_rel_poses, pred_rel_poses):
#     """
#     Compute pose errors between ground truth and predicted relative poses.
    
#     Args:
#         gt_rel_poses: (N, 4, 4) Ground truth relative poses
#         pred_rel_poses: (N, 4, 4) Predicted relative poses
        
#     Returns:
#         trans_err: (N,) Translation error in mm
#         rot_err: (N,) Rotation error in degrees
#     """
#     # Compute relative pose errors
#     pose_errors = gt_rel_poses @ torch.inverse(pred_rel_poses)  # (N, 4, 4)
    
#     # Translation error (Euclidean distance in mm)
#     trans_err = torch.norm(pose_errors[:, :3, 3], dim=1)  # (N,)
    
#     # Rotation error (angle in degrees)
#     rot_matrices = pose_errors[:, :3, :3]  # (N, 3, 3)
#     # Convert rotation matrix to angle using trace
#     # trace(R) = 1 + 2*cos(theta) => theta = arccos((trace(R) - 1) / 2)
#     trace = torch.diagonal(rot_matrices, dim1=1, dim2=2).sum(dim=1)  # (N,)
#     cos_theta = (trace - 1) / 2
#     # Clamp to valid range [-1, 1] to avoid numerical issues
#     cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
#     rot_err = torch.acos(cos_theta) #* 180 / torch.pi  # Convert to degrees
    
#     err_dict = {
#         'trans_err': trans_err,
#         'rot_err': rot_err
#     }
#     return err_dict
#     # return trans_err, rot_err

