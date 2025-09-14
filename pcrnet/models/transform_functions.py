import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .quaternion_utils import soft_clamp_quaternion_angle, soft_clamp_translation_magnitude
from .quaternion_utils import euler_to_quaternion, qrot
from .quaternion_utils import quaternion_to_axis_angle, axis_angle_to_quaternion

  
class PCRNetTransform:
    def __init__(self, data_size, angle_range=45, translation_range=1):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.index = 0

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg

    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)
        rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
        trans = np.random.uniform(-max_translation, max_translation, [1, 3])
        quat = euler_to_quaternion(rot, "xyz")

        vec = np.concatenate([quat, trans], axis=1)
        vec = torch.tensor(vec, dtype=dtype)
        return vec

    @staticmethod
    def create_pose_7d_downscale(vector: torch.Tensor, trans_downscale_factor = 0.001, rot_downscale_factor = 0.001):
        # quan: wxyz
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)
        # conver to angle axis representation, then downscale the angle
        angle_axis = quaternion_to_axis_angle(normalized_quaternion)
        angle_axis = angle_axis * rot_downscale_factor
        scaled_quaternion = axis_angle_to_quaternion(angle_axis)
        normalized_quaternion = scaled_quaternion 
        # normalized_quaternion = F.normalize(scaled_quaternion, dim=1) #redundant

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        translation = translation * trans_downscale_factor
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def create_pose_7d_modest(vector: torch.Tensor, max_angle_rad=0.01, max_magnitude=0.001):
        # can you create a function which sooth the estimated pose
        # soft clamp the rotatio part
        # soft clamp the translation magnitude
        # return the vector
        quaternion = vector[:, 0:4]
        translation = vector[:, 4:]
        quaternion_smooth = soft_clamp_quaternion_angle(quaternion, 
                                                        max_angle_rad= max_angle_rad)
        translation_smooth = soft_clamp_translation_magnitude(translation, 
                                                              max_magnitude = max_magnitude)
        # print('quaternion_smooth:', quaternion_smooth)
        # print('translation_smooth:', translation_smooth)

        quaternion_normalized = F.normalize(quaternion_smooth, dim=1)
        vector = torch.cat([quaternion_normalized, translation_smooth], dim=1)
        return vector.view([-1, 7])


    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
        return pose_7d[:, 4:]

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = PCRNetTransform.get_quaternion(pose_7d).expand([N, -1])
            rotated_point_cloud = qrot(quat, point_cloud)# wxyz

        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = PCRNetTransform.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()
            rotated_point_cloud = qrot(quat, point_cloud)# wxyz

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = PCRNetTransform.quaternion_rotate(point_cloud, pose_7d) + PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)                        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix

    def __call__(self, template):
        assert 0, f'temporally disabled'
        self.igt = self.transformations[self.index]
        igt = self.create_pose_7d(self.igt)
        self.igt_rotation = self.quaternion_rotate(torch.eye(3), igt).permute(1, 0)        # [3x3]
        self.igt_translation = self.get_translation(igt)                                   # [1x3]
        source = self.quaternion_rotate(template, igt) + self.get_translation(igt)
        return source


