import torch

def quaternion_wxyz_to_matrix_pytorch3d(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions(wxyz): quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def xyz_quat_to_matrix_kornia(
    xyz_quat: torch.Tensor, 
    quat_format: str = 'wxyz',
    soft_clamp_quat: bool = False,
    max_angle_rad: float = 0.1,
) -> torch.Tensor:
    """
    Convert [B, 7] tensor (tx,ty,tz,qx,qy,qz,qw or wxyz) to [B,4,4] homogeneous matrices.
    
    Args:
        xyz_quat: [B, 7] tensor, where the last 4 are quaternion components.
        quat_format: 'xyzw' if input quaternions are (x,y,z,w), 
                    'wxyz' if input is (w,x,y,z)
                    
    Returns:
        T: [B,4,4] homogeneous transformation matrices
    """
    assert xyz_quat.dim() == 2, f'xyz_quat.dim() should be 3, but got {xyz_quat.dim()}'
    assert xyz_quat.shape[1] == 7, f'xyz_quat.shape[2] should be 7, but got {xyz_quat.shape[2]}'
    assert quat_format in ('xyzw', 'wxyz'), "quat_format must be 'xyzw' or 'wxyz'"
    B = xyz_quat.shape[0]
    t = xyz_quat[:, :3]   # [B,3]
    quat = xyz_quat[:, 3:]  # [B,4]
    # Reorder quaternion for Kornia (expects w,x,y,z)
    if quat_format == 'xyzw':
        quat_wxyz = torch.cat([quat[:, 3:], quat[:, :3]], dim=-1)  # x,y,z,w -> w,x,y,z
    elif quat_format == 'wxyz':  # already wxyz
        quat_wxyz = quat
    else:
        assert 0, f'Unknown quat_format {quat_format}, should be one of [xyzw, wxyz]'

    def clamp_quaternion_angle(q: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
        """
        Clamp quaternion rotations by maximum angle.
        
        Args:
            q: (B, 4) tensor of quaternions (w, x, y, z), not necessarily normalized
            max_angle_rad: float, maximum allowed rotation angle in radians

        Returns:
            (B, 4) tensor of clamped, normalized quaternions
        """
        # normalize in case of drift
        q = q / q.norm(dim=-1, keepdim=True)

        w, xyz = q[:, 0], q[:, 1:]
        theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # (B,)

        # scale factor = sin(new_theta/2) / sin(theta/2)
        scale = torch.ones_like(theta)
        mask = theta > max_angle_rad
        if mask.any():
            new_theta = torch.full_like(theta[mask], max_angle_rad)
            scale_val = torch.sin(new_theta / 2) / torch.sin(theta[mask] / 2)
            scale[mask] = scale_val

        # apply scaling to xyz
        xyz = xyz * scale.unsqueeze(-1)

        # recompute w for clamped ones
        w = torch.where(mask, torch.cos(max_angle_rad / 2).expand_as(w), w)

        q_clamped = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
        # normalize again for safety
        return q_clamped / q_clamped.norm(dim=-1, keepdim=True)

    def soft_clamp_quaternion_angle(q: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
        """
        Softly clamp quaternion rotations by max_angle_rad using a smooth squash.
        
        Args:
            q: (B, 4) quaternions (w, x, y, z), not necessarily normalized
            max_angle_rad: float, maximum allowed rotation angle in radians
        
        Returns:
            (B, 4) softly clamped, normalized quaternions
        """
        # normalize in case of drift
        q = q / q.norm(dim=-1, keepdim=True)

        w, xyz = q[:, 0], q[:, 1:]
        theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # (B,)

        # avoid div by zero: if theta ~ 0, just keep axis as xyz
        axis = torch.zeros_like(xyz)
        mask = theta > 1e-8
        axis[mask] = xyz[mask] / torch.sin(theta[mask] / 2).unsqueeze(-1)

        # squash angle smoothly
        theta_clamped = max_angle_rad * torch.tanh(theta / max_angle_rad)

        # rebuild quaternion
        w_new = torch.cos(theta_clamped / 2)
        xyz_new = axis * torch.sin(theta_clamped / 2).unsqueeze(-1)

        q_new = torch.cat([w_new.unsqueeze(-1), xyz_new], dim=-1)
        return q_new / q_new.norm(dim=-1, keepdim=True)


    if soft_clamp_quat:
        quat_wxyz = soft_clamp_quaternion_angle(quat_wxyz, max_angle_rad=max_angle_rad)

    # Convert quaternion to rotation matrix: [B,3,3]
    # R = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat_wxyz) # seems have issue for kornia implementation
    R = quaternion_wxyz_to_matrix_pytorch3d(quat_wxyz)

    # Build homogeneous matrices
    T = torch.eye(4, device=xyz_quat.device, dtype=xyz_quat.dtype).unsqueeze(0).repeat(B,1,1)

    T[:, :3, :3] = R
    T[:, :3, 3] = t

    return T

