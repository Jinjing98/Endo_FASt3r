# import necessary module
# from mpl_toolkits.mplot3d import axes3d
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from options import MonodepthOptions
opt = MonodepthOptions().parse()

# load data from file
# you replace this using with open
gt_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "gt_poses_sq{}.npz".format(opt.eval_split_appendix))
gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

# our_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "pred_pose_sq{}.npz".format(opt.eval_split_appendix))
our_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "pred_pose_sq{}{}.npz".format(opt.eval_split_appendix, 
                                                                                                        opt.eval_model_appendix))
our_local_poses = np.load(our_path, fix_imports=True, encoding='latin1')["data"]

if opt.dataset == 'StereoMIS':
    # conduct inverse for the gt_poses
    # the issue of fast3r provided npz format?
    gt_local_poses = np.linalg.inv(gt_local_poses)
    gt_local_poses[:, :3, 3] *= 1000



def dump(source_to_target_transformations):
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms


def compute_scale(gtruth, pred):

    # Optimize the scaling factor
    scale = np.sum(gtruth[:, :3, 3] * pred[:, :3, 3]) / np.sum(pred[:, :3, 3] ** 2)

    print('scale: ', scale)
    # hard_code_scale = -0.04#36
    # print('hard_code_scale: ', hard_code_scale)
    # scale = hard_code_scale

    return scale


def extract_xyz_rpy(transformation_matrices):
    """
    Extract x, y, z, roll, pitch, yaw from transformation matrices.
    
    Args:
        transformation_matrices: numpy array of shape (N, 4, 4)
        
    Returns:
        xyz_rpy: numpy array of shape (N, 6) where columns are [x, y, z, roll, pitch, yaw]
    """
    xyz_rpy = np.zeros((len(transformation_matrices), 6))
    
    for i, T in enumerate(transformation_matrices):
        # Extract translation (x, y, z)
        xyz_rpy[i, :3] = T[:3, 3]
        
        # Extract rotation matrix
        R_matrix = T[:3, :3]
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        # Using ZYX convention (yaw-pitch-roll)
        r = R.from_matrix(R_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        
        xyz_rpy[i, 3:] = [roll, pitch, yaw]
    
    return xyz_rpy


debug_only = True
debug_only = False
plot_num = 150
plot_num = 100
if debug_only:
    gt_local_poses = gt_local_poses[:plot_num]
    our_local_poses = our_local_poses[:plot_num]

dump_gt = np.array(dump(gt_local_poses))
dump_our = np.array(dump(our_local_poses))

# scale_our = dump_our * compute_scale(dump_gt, dump_our)
scale_our = dump_our.copy()
scale_our[:, :3, 3] *= compute_scale(dump_gt, dump_our) #* compute_scale(dump_gt, dump_our)

num = gt_local_poses.shape[0]
points_our = []
points_gt = []
origin = np.array([[0], [0], [0], [1]])

for i in range(0, num):
    point_our = np.dot(scale_our[i], origin)
    point_gt = np.dot(dump_gt[i], origin)

    points_our.append(point_our)
    points_gt.append(point_gt)

points_our = np.array(points_our)
points_gt = np.array(points_gt)

# Check if xyz_rpy plotting is enabled
if hasattr(opt, 'plot_xyz_rpy') and opt.plot_xyz_rpy:
    print("Generating combined 3D trajectory and XYZ-RPY plots...")
    
    # Extract xyz_rpy from transformation matrices
    gt_xyz_rpy = extract_xyz_rpy(dump_gt)
    pred_xyz_rpy = extract_xyz_rpy(scale_our)
    
    # Create a large figure with subplots for both 3D trajectory and xyz_rpy
    fig = plt.figure(figsize=(20, 12))
    
    # 3D trajectory subplot (top-left, spans 2 rows)
    ax_3d = fig.add_subplot(2, 4, (1, 5), projection='3d')
    ax_3d.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], label='GT', linestyle='-', c='blue', linewidth=1.6)
    ax_3d.plot(points_our[:, 0, 0], points_our[:, 1, 0], points_our[:, 2, 0], label='Prediction', linestyle='-', c='red', linewidth=1.6)
    ax_3d.set_xlabel("x [mm]")
    ax_3d.set_ylabel("y [mm]")
    ax_3d.set_zlabel("z [mm]")
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend()
    
    # XYZ-RPY subplots (remaining 6 subplots)
    labels = ['x [mm]', 'y [mm]', 'z [mm]', 'roll [deg]', 'pitch [deg]', 'yaw [deg]']
    gt_color = 'blue'
    pred_color = 'red'
    time_steps = np.arange(len(gt_xyz_rpy))
    
    subplot_positions = [2, 3, 4, 6, 7, 8]  # Positions for the 6 xyz_rpy subplots
    
    for i in range(6):
        ax = fig.add_subplot(2, 4, subplot_positions[i])
        ax.plot(time_steps, gt_xyz_rpy[:, i], label='GT', color=gt_color, linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, pred_xyz_rpy[:, i], label='Prediction', color=pred_color, linewidth=1.5, linestyle='--', alpha=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i].split()[0].upper()} Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(f'Pose Analysis - Seq {opt.eval_split_appendix} - Model {opt.eval_model_appendix}', fontsize=16)
    
    plt.tight_layout()
    
    # Save the combined plot
    filename = f'combined_pose_analysis_seq{opt.eval_split_appendix}{opt.eval_model_appendix}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f'Combined pose analysis plot saved as: {filename}')
    
    plt.show()
    
else:
    # Original 3D trajectory only
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # set figure information
    # ax.set_title("3D_Curve")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    
    # draw the figure, the color is r = read
    figure1, = ax.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], label = 'GT', linestyle = '-', c='b', linewidth=1.6)
    figure2, = ax.plot(points_our[:, 0, 0], points_our[:, 1, 0], points_our[:, 2, 0], label = 'Prediction', linestyle = '-', c='g', linewidth=1.6)
    
    plt.legend()
    plt.savefig('trajectory_pose_seq{}.png'.format(opt.eval_split_appendix),dpi=600)
    plt.show()
