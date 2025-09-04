import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def flow3DToColor(pts3d, maxflow=None):
    """Convert 3D points/flow to RGB color visualization.
    Args:
        pts3d: numpy array of shape (H, W, 3) containing 3D coordinates
        maxflow: maximum flow magnitude for normalization
    Returns:
        RGB visualization of shape (H, W, 3)
    """
    # Calculate magnitude of 3D flow
    magnitude = np.sqrt(np.sum(pts3d**2, axis=2))
    if maxflow is None:
        maxflow = np.percentile(magnitude, 95)  # Use 95th percentile as max
    # Normalize magnitude to [0,1]
    magnitude = np.clip(magnitude / maxflow, 0, 1)
    # Normalize 3D coordinates to unit vectors for direction
    direction = pts3d / (np.linalg.norm(pts3d, axis=2, keepdims=True) + 1e-6)
    # Convert direction to RGB using spherical coordinates
    # Map x,y,z to r,g,b with proper scaling
    rgb = (direction + 1) / 2  # Map from [-1,1] to [0,1]
    # Apply magnitude as intensity
    rgb = rgb * magnitude[..., np.newaxis]
    return (rgb * 255).astype(np.uint8)

def plot(cams_T_world,fig = None, ax= None,label='Camera Trajectory', color='b', is_extrinsics=True, step = 10):
    '''
    plot the pose traj
    '''
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot first trajectory
    traj1_xyz = []
    for i, extrinsics in enumerate(cams_T_world):
        if i% step != 0:
            continue
        R = extrinsics[:3,:3]
        t = extrinsics[:3,3]
        if is_extrinsics:
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t
        else:
            camera_pose = extrinsics
        traj1_xyz.append(camera_pose[:3,3])

        # if i == 0:
            # ax.scatter(*camera_pose[:3,3], color='r', label=f'{label} Start')
        # else:
        ax.scatter(*camera_pose[:3,3], color=f'{color}')
    traj1_xyz = np.array(traj1_xyz)
    return traj1_xyz, ax, fig

def plot_camera_trajectories(cams_T_world_1, cams_T_world_2=None, 
                             label1='Trajectory 1', label2='Trajectory 2'):
    # plot the pose traj
    
    traj1_xyz, ax, fig = plot(cams_T_world_1, label=label1, color='b')
    # Plot second trajectory if provided
    if cams_T_world_2 is not None:
        traj2_xyz, ax, fig = plot(cams_T_world_2, fig=fig, ax=ax, label=label2, color='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectories')
    ax.legend()
    plt.show()