# gt_c2w_poses
# gt_rel_poses
import torch 

def compute_pose_error(gt_rel_poses, pred_rel_poses):
    """
    Compute pose errors between ground truth and predicted relative poses.
    
    Args:
        gt_rel_poses: (N, 4, 4) Ground truth relative poses
        pred_rel_poses: (N, 4, 4) Predicted relative poses
        
    Returns:
        trans_err: (N,) Translation error in mm
        rot_err: (N,) Rotation error in degrees
    """
    # Compute relative pose errors
    pose_errors = gt_rel_poses @ torch.inverse(pred_rel_poses)  # (N, 4, 4)
    
    # Translation error (Euclidean distance in mm)
    trans_err = torch.norm(pose_errors[:, :3, 3], dim=1)  # (N,)
    
    # Rotation error (angle in degrees)
    rot_matrices = pose_errors[:, :3, :3]  # (N, 3, 3)
    # Convert rotation matrix to angle using trace
    # trace(R) = 1 + 2*cos(theta) => theta = arccos((trace(R) - 1) / 2)
    trace = torch.diagonal(rot_matrices, dim1=1, dim2=2).sum(dim=1)  # (N,)
    cos_theta = (trace - 1) / 2
    # Clamp to valid range [-1, 1] to avoid numerical issues
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    rot_err = torch.acos(cos_theta) * 180 / torch.pi  # Convert to degrees
    
    return trans_err, rot_err

