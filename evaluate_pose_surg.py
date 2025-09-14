from __future__ import absolute_import, division, print_function
"""
Surg_oclr_stereo pose evaluation and visualization

Usage
- Requirements:
  - A weights folder that contains: pose_encoder.pth and pose.pth. Download from https://drive.google.com/file/d/11C0sw396TcH2hMM7u6uMr-uBsCP4l2Kd/view
  - Python packages used by this repo (torch, numpy, matplotlib, PIL, etc.)
- Optional environment variables:
  - SURG_IMG_DIR: directory containing images (default set inside this file)
    Expected image naming: final_*l.png (left camera). If you want to use right images,
    update the call to list_image_pairs(..., left_suffix='r').
  - SURG_GT_PATH: path to the ground-truth pose .txt (xyz in meters, quaternion)
  - SURG_MAX_OFFSET: integer, max temporal offset searched for alignment (default 10)
- Run:
  python /mnt/cluster/workspaces/students/lisuqi/DARES/evaluate_pose_surg.py \
    --load_weights_folder /path/to/weights_folder
- Output:
  - Figure saved to results/ as trajectory_pose_surg_###.png (auto-incremented index)
  - Console prints:
    - Trajectory error (ATE, m): mean ± std (official compute_ate from evaluate_pose.py)
    - Rotation error (rad): mean ± std (official compute_re from evaluate_pose.py)
    - Overlay RMSE (m): scale-only overlay error for a quick visual sanity check
- Notes:
  - The script uses natural-sorted frame order and small temporal offset search.
  - Metrics are computed on short tracks (track_length=5) to match the official approach.
"""

import os
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import networks
from layers import transformation_from_parameters
from options import MonodepthOptions
from evaluate_pose import compute_ate as eval_compute_ate, compute_re as eval_compute_re, dump_xyz as eval_dump_xyz, dump_r as eval_dump_r

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def to_tensor(img: Image.Image):
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def resize(img: Image.Image, height: int, width: int) -> Image.Image:
    return img.resize((width, height), resample=Image.LANCZOS)


def _natural_key(path: str):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]

def list_image_pairs(root_dir: str, left_suffix: str = 'l'):
    paths = glob.glob(os.path.join(root_dir, f"final_*{left_suffix}.png"))
    paths.sort(key=_natural_key)
    return paths


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    # Normalize quaternion
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-9
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    # Rotation matrix
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
    ], dtype=np.float64)
    return R


def load_gt_poses_txt(gt_path: str):
    # Format per line: idx x y z qx qy qz qw (as given: xyz in meter) x y z Quaternion
    # Provided file seems: index, x, y, z, qx, qy, qz, qw or maybe qw last.
    # Inspect first line pattern; we assume: idx x y z qx qy qz qw
    data = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            idx = int(float(parts[0]))
            x, y, z = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            T = np.eye(4, dtype=np.float64)
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            T[:3, :3] = R
            T[:3, 3] = np.array([x, y, z], dtype=np.float64)
            data.append((idx, T))
    # Sort by idx
    data.sort(key=lambda t: t[0])
    poses = [T for _, T in data]
    return np.array(poses)


def absolute_to_local_transforms(abs_poses: np.ndarray):
    # abs_poses: [N,4,4]; return relative transforms between consecutive frames (source->target)
    rels = []
    for i in range(1, len(abs_poses)):
        rel = np.linalg.inv(abs_poses[i - 1]) @ abs_poses[i]
        rels.append(rel)
    return np.array(rels)


def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return np.array(xyzs)


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        rs.append(cam_to_world[:3, :3])
    return np.array(rs)


# --- Alignment & Visualization helpers ---



def set_equal_aspect_3d(ax, pts_a: np.ndarray, pts_b: np.ndarray):
    pts = np.vstack([pts_a, pts_b])
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_mid, y_mid, z_mid = (x.min()+x.max())/2.0, (y.min()+y.max())/2.0, (z.min()+z.max())/2.0
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
    if max_range == 0:
        max_range = 1.0
    r = max_range / 2.0
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)


def compute_ate(gtruth_xyz, pred_xyz_o):
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]
    scale = np.sum(gtruth_xyz * pred_xyz) / (np.sum(pred_xyz ** 2) + 1e-9)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    for gt_pose, pred_pose in zip(gtruth_r, pred_r):
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0], R[1, 2] - R[2, 1], R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        RE += np.arctan2(s, c)
    return RE / gtruth_r.shape[0]


def run_pose_network_on_images(image_paths, height, width, pose_encoder, pose_decoder):
    pred_rels = []
    with torch.no_grad():
        for i in range(1, len(image_paths)):
            img_t = pil_loader(image_paths[i - 1])
            img_t1 = pil_loader(image_paths[i])
            img_t = resize(img_t, height, width)
            img_t1 = resize(img_t1, height, width)
            t0 = to_tensor(img_t)
            t1 = to_tensor(img_t1)
            # Network expects [B, 6, H, W] stacked as (t1, t0) like evaluate_pose.py
            all_color_aug = torch.cat([t1.unsqueeze(0), t0.unsqueeze(0)], dim=1).to(device)
            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)
            T_rel = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()[0]
            pred_rels.append(T_rel)
    return np.array(pred_rels)


def visualize_trajectories(gt_rels: np.ndarray, pred_rels: np.ndarray, out_path: str):
    gt_xyz = dump_xyz(gt_rels)
    pred_xyz = dump_xyz(pred_rels)

    # Scale-only overlay
    scale = np.sum(gt_xyz * pred_xyz) / (np.sum(pred_xyz ** 2) + 1e-9)
    pred_xyz_scaled = pred_xyz * scale

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='GT', c='b', linewidth=1.6)
    ax.plot(pred_xyz_scaled[:, 0], pred_xyz_scaled[:, 1], pred_xyz_scaled[:, 2], label='Pred (scale-only)', c='g', linewidth=1.2)

    set_equal_aspect_3d(ax, gt_xyz, pred_xyz_scaled)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    try:
        plt.show()
    except Exception:
        pass


def _trajectory_step_dists(T_rels: np.ndarray):
    xyz = dump_xyz(T_rels)
    diffs = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    return diffs


def _best_offset_by_step_dist(gt_rels: np.ndarray, pred_rels: np.ndarray, max_offset: int = 10):
    gt_d = _trajectory_step_dists(gt_rels)
    pr_d = _trajectory_step_dists(pred_rels)
    best_off, best_score = 0, -1e9
    max_offset = max(0, int(max_offset))
    for off in range(-max_offset, max_offset + 1):
        if off >= 0:
            a = gt_d[:min(len(gt_d), len(pr_d) - off)]
            b = pr_d[off:off + len(a)]
        else:
            a = gt_d[-off:-off + min(len(gt_d) + off, len(pr_d))]
            b = pr_d[:len(a)]
        if len(a) < 5:
            continue
        # normalized correlation
        aa = (a - a.mean()) / (a.std() + 1e-9)
        bb = (b - b.mean()) / (b.std() + 1e-9)
        score = float(np.dot(aa, bb) / len(aa))
        if score > best_score:
            best_score = score
            best_off = off
    return best_off, best_score


def next_indexed_filepath(results_dir: str, base: str = 'trajectory_pose_surg', ext: str = '.png'):
    os.makedirs(results_dir, exist_ok=True)
    existing = glob.glob(os.path.join(results_dir, f"{base}_*{ext}"))
    idx = 1
    if existing:
        def _extract(e):
            name = os.path.basename(e)
            m = re.search(rf"{re.escape(base)}_(\d+){re.escape(ext)}$", name)
            return int(m.group(1)) if m else 0
        idx = max(_extract(e) for e in existing) + 1
    return os.path.join(results_dir, f"{base}_{idx:03d}{ext}")


def main():
    opt = MonodepthOptions().parse()

    # Required inputs via CLI or environment
    data_root = os.environ.get('SURG_IMG_DIR', None)
    gt_pose_path = os.environ.get('SURG_GT_PATH', None)

    if data_root is None:
        # Default to your provided path pattern's directory
        data_root = "/mnt/cluster/datasets/Surg_oclr_stereo/test_100_300/fixedCam0_fixedTool1_fixedTissue0/00070"
    if gt_pose_path is None:
        gt_pose_path = "/mnt/cluster/datasets/Surg_oclr_stereo/test_100_300/fixedCam0_fixedTool1_fixedTissue0/00070/vid/SCARED_8_3_000107_toolf1depth40tissuef1depth62_pose.txt"

    image_dir = os.path.join(data_root)
    image_paths = list_image_pairs(image_dir, left_suffix='l')
    assert len(image_paths) >= 2, f"No images found at {image_dir}/final_*l.png"

    # Load model weights
    assert os.path.isdir(opt.load_weights_folder), f"Cannot find model folder: {opt.load_weights_folder}"
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=device.type))
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location=device.type))
    pose_encoder.to(device).eval()
    pose_decoder.to(device).eval()

    # Predict
    pred_rels = run_pose_network_on_images(image_paths, opt.height, opt.width, pose_encoder, pose_decoder)

    # GT
    gt_abs = load_gt_poses_txt(gt_pose_path)
    gt_rels = absolute_to_local_transforms(gt_abs)

    # Optional: temporal offset search via env var
    try:
        max_off = int(os.environ.get('SURG_MAX_OFFSET', '10'))
    except Exception:
        max_off = 10
    best_off, best_score = _best_offset_by_step_dist(gt_rels, pred_rels, max_offset=max_off)
    if best_off != 0:
        print(f"   Temporal offset applied: {best_off} (score={best_score:.3f})")
    if best_off >= 0:
        pred_rels = pred_rels[best_off:]
    else:
        gt_rels = gt_rels[-best_off:]

    # Align lengths (use min sequence length)
    L = min(len(pred_rels), len(gt_rels))
    pred_rels = pred_rels[:L]
    gt_rels = gt_rels[:L]

    # Metrics (ATE + RE)
    ates = []
    res = []
    track_length = 5
    for i in range(0, L - 1):
        local_xyzs = np.array(eval_dump_xyz(pred_rels[i:i + track_length - 1]))
        gt_local_xyzs = np.array(eval_dump_xyz(gt_rels[i:i + track_length - 1]))
        ates.append(eval_compute_ate(gt_local_xyzs, local_xyzs))
        local_rs = np.array(eval_dump_r(pred_rels[i:i + track_length - 1]))
        gt_rs = np.array(eval_dump_r(gt_rels[i:i + track_length - 1]))
        res.append(eval_compute_re(gt_rs, local_rs))

    print("-> Num frames:", L + 1)
    if len(ates) > 0:
        print("   Trajectory error (ATE, m): {:0.4f} ± {:0.4f}".format(np.mean(ates), np.std(ates)))
    if len(res) > 0:
        print("   Rotation error (rad): {:0.4f} ± {:0.4f}".format(np.mean(res), np.std(res)))
        # Global diagnostics
        gt_xyz = np.array(eval_dump_xyz(gt_rels))
        pred_xyz = np.array(eval_dump_xyz(pred_rels))
        scale_vis = float((gt_xyz * pred_xyz).sum() / (pred_xyz**2).sum())
        pred_xyz_scaled = pred_xyz * scale_vis
        overlay_err = np.linalg.norm(pred_xyz_scaled - gt_xyz, axis=1)
        print("   Overlay RMSE (m): {:0.4f}, max: {:0.4f}".format(float(np.sqrt((overlay_err**2).mean())), float(overlay_err.max())))

    # Visualization -> save to results/ with incremental index
    results_dir = os.path.join(os.getcwd(), 'results')
    out_png = next_indexed_filepath(results_dir, base='trajectory_pose_surg', ext='.png')
    visualize_trajectories(gt_rels, pred_rels, out_png)
    print(f"Saved trajectory figure to {out_png}")


if __name__ == "__main__":
    main() 