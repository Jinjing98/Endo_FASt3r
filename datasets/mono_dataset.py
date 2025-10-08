from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile
from scipy.spatial.transform import Rotation as R

import torch
import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_ids
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_ids,
                 num_scales,
                 is_train=False,
                 img_ext='.png',
                 of_samples=False,
                 of_samples_num=10,):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        if of_samples:
            print('use only of_samples_num samples for OF, for OF training...')
            self.filenames = self.filenames[:of_samples_num]
            print(f"using {len(self.filenames)} samples for OF, for speed up training..")
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.ANTIALIAS
        self.interp = Image.LANCZOS

        self.frame_ids = frame_ids
        assert self.frame_ids[0] == 0, f"frame_ids[0] must be 0, but got {self.frame_ids[0]}"
        # not assert during eval.
        # assert len(self.frame_ids) == 3, f"frame_ids must be of length 3, but got {self.frame_ids}"

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.load_depth = self.check_depth()

        self.trajs_dict = {}
        self.trans_scale_gt_traj = 1000  # m to mm
        # self.trans_scale_gt_traj = 1 #m


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        if self.dataset_name == 'DynaSCARED':
            folder_components = folder.split('/')
            sequence = folder_components[-2]
            sequence = ''.join(filter(str.isdigit, sequence))
            keyframe = folder_components[-1]
        elif self.dataset_name == 'SCARED':
            sequence = folder[7]
            keyframe = folder[-1]
        elif self.dataset_name == 'StereoMIS':
            folder_components = folder.split('/')
            sequence = folder.split('/')[0]
            keyframe = folder.split('/')[1]
        else:
            raise ValueError(f'Unknown dataset name: {self.dataset_name}')
        # print('===============================================')
        # print('read_folder:', folder)
        # print('dataset_name:', self.dataset_name)
        # print('components:', folder_components)
        # print(f'sequence: {sequence}, keyframe: {keyframe}')
        if self.dataset_name == 'StereoMIS':
            inputs["sequence"] = torch.from_numpy(np.array(int(sequence[-1])))
            inputs["keyframe"] = torch.from_numpy(np.array(int(keyframe[-1])))
        else:
            inputs["sequence"] = torch.from_numpy(np.array(int(sequence)))
            inputs["keyframe"] = torch.from_numpy(np.array(int(keyframe)))  


        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
            
        inputs["frame_id"] = torch.from_numpy(np.array(frame_index))
        for i in self.frame_ids:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            if self.dataset_name == 'SCARED':
                K = self.K.copy()
            elif self.dataset_name == 'StereoMIS':
                K = self.K.copy()
            elif self.dataset_name == 'DynaSCARED':
                K = self.K_dict_registered[folder].copy()
            else:
                raise ValueError(f'Unknown dataset name: {self.dataset_name}')

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            # print(f'K for scale {scale} with img size : {self.width // (2 ** scale)}x{self.height // (2 ** scale)}:')
            # print(K)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_ids:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        if self.load_gt_poses:
            if self.dataset_name == 'SCARED':
                offset = -1
            elif self.dataset_name == 'DynaSCARED':
                offset = 0
            elif self.dataset_name == 'StereoMIS':
                offset = -1
            else:
                raise ValueError(f'Unknown dataset name: {self.dataset_name}')
            
            for i in self.frame_ids:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("gt_c2w_poses", i)] = torch.from_numpy(self.get_poses_for_frames(folder, [frame_index], offset=offset))
                else:
                    inputs[("gt_c2w_poses", i)] = torch.from_numpy(self.get_poses_for_frames(folder, [frame_index + i], offset=offset))
            # print('===============================================')
            # print('read_folder:', folder)
            # print('sequence:', inputs["sequence"])
            # print('keyframe:', inputs["keyframe"])
            # print('Loaded gt poses this item index:', index)
            # print('frame_index:', frame_index)
            # print('frame_ids:', self.frame_ids)
            # print('tgt frame id:', inputs["frame_id"])
            # for i in self.frame_ids:
            #     print('gt_c2w_poses:', inputs[("gt_c2w_poses", i)])

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def pose_vec_to_mat(self, pose_vec):
        """
        Convert a 7D vector (xyz + quaternion) to a 4x4 transformation matrix.
        
        pose_vec: array-like, shape (7,)
                  Format: [x, y, z, qx, qy, qz, qw]
        
        Returns: np.ndarray, shape (4, 4)
                 The SE(3) transformation matrix.
        """
        pose_vec = np.asarray(pose_vec)
        assert pose_vec.shape[-1] == 7, f"{pose_vec} Expected input shape (..., 7)"
        
        trans = pose_vec[:3]
        quat = pose_vec[3:]
        
        rot = R.from_quat(quat).as_matrix()  # 3x3 rotation matrix

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T

    def read_freiburg_scipy(self, path: str, ret_stamps=False, no_stamp=False):
        '''
        Read trajectory file in Freiburg format.
        '''
        with open(path, 'r') as f:
            data = f.read()
            lines = data.replace(",", " ").replace("\t", " ").split("\n")
            list_data = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                        len(line) > 0 and line[0] != "#"]
        #close file
        f.close()

        # self.trans_scale_gt_traj = 1000  # m to mm
        trans_scale = self.trans_scale_gt_traj  # m to mm
        if no_stamp:       
            trans_np = np.asarray([l[0:3] for l in list_data if len(l) > 0], dtype=float)
            quat_np = np.asarray([l[3:] for l in list_data if len(l) > 0], dtype=float)
            trans_quat_np = np.hstack((trans_np, quat_np)) 
            trans_quat_np[:,:3] *= trans_scale 
            pose_matrix = np.asarray([self.pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float)
        else:
            time_stamp = [l[0] for l in list_data if len(l) > 0]
            try:
                time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp])*100
            except IndexError:
                time_stamp = np.asarray([int(l) for l in time_stamp])
            trans_np = np.asarray([l[1:4] for l in list_data if len(l) > 0], dtype=float)
            quat_np = np.asarray([l[4:] for l in list_data if len(l) > 0], dtype=float)
            trans_quat_np = np.hstack((trans_np, quat_np))
            trans_quat_np[:,:3] *= trans_scale 
            pose_matrix = np.asarray([self.pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float)
            if ret_stamps:
                return pose_matrix, time_stamp
        return pose_matrix

    def map_traj_search(self, folder):
        """
        Map folder path to trajectory search path.
        """
        # Extract sequence and keyframe from folder name
        # Expected format: "dataset_X/keyframe_Y"
        parts = folder.split('/')
        if len(parts) == 2:
            if self.dataset_name == 'SCARED':
                sequence = int(parts[0][-1])  # "datasetX"
                keyframe = int(parts[1][-1])  # "keyframeY"
            elif self.dataset_name == 'DynaSCARED':
                assert 0, f'not tested'
            else:
                raise ValueError(f"Unknown dataset name: {self.dataset_name}")
            
            assert isinstance(sequence, int) and isinstance(keyframe, int), f'sequence and keyframe must be int, but got {sequence} and {keyframe}'
            split_dir_ori = 'testing' if sequence in range(8, 10) else 'training' 
            sequence_ori = sequence 
            keyframe_ori = keyframe - 1 if sequence in range(8, 10) else keyframe
            
            trajfolder = f'{self.traj_data_root}/{split_dir_ori}/dataset_{sequence_ori}/keyframe_{keyframe_ori}/'
            # print('traj folder is:')
            # print(trajfolder)
            return trajfolder
        else:
            raise ValueError(f"Invalid folder format: {folder}")

    def get_gt_poses(self):
        """
        Load ground truth poses for all folders in filenames.
        Returns a dictionary mapping folder to trajectory data.
        """
        trajs_dict = {}
        # Get unique folders from filenames
        unique_folders = set()
        for filename in self.filenames:
            line = filename.split()
            if len(line) >= 1:
                folder = line[0]
                unique_folders.add(folder)
            else:
                raise ValueError(f"Invalid filename format: {filename}")
        
        print(f"Loading trajectories for {len(unique_folders)} unique folders...")
        # print('Load from folder:', unique_folders)
        for folder in unique_folders:
            # print('**********dataset name:', self.dataset_name)
            # print('**********folder:', folder)
            if self.dataset_name == 'DynaSCARED':
                import glob
                traj_full_folder = os.path.join(self.traj_data_root, folder, 'vid')
                traj_path = glob.glob(f'{traj_full_folder}/*.txt')
                assert len(traj_path) == 1, f'Expected 1 trajectory file, but got {len(traj_path)} for {traj_full_folder}'
                traj_path = traj_path[0]
            elif self.dataset_name == 'StereoMIS':
                traj_full_folder = os.path.join(self.traj_data_root, folder.split('/')[0])
                traj_path = f'{traj_full_folder}/groundtruth.txt'
            elif self.dataset_name == 'SCARED':
                    traj_full_folder = self.map_traj_search(folder)
                    traj_path = f'{traj_full_folder}/groundtruth.txt'
            else:
                raise ValueError(f'Unknown dataset name: {self.dataset_name}')
            
            if os.path.exists(traj_path):
                traj = self.read_freiburg_scipy(traj_path, ret_stamps=False, no_stamp=False)
                trajs_dict[folder] = traj
                # print(f"Loaded trajectory for {folder}: {len(traj)} poses")
            else:
                assert 0, f'Trajectory file {traj_path} does not exist for {folder}'
        
        print(f"Successfully loaded {len([v for v in trajs_dict.values() if v is not None])} trajectories")
        return trajs_dict

    def get_poses_for_frames(self, folder, frame_indices, offset):
        """
        Get ground truth poses for specific frame indices in a folder.
        
        Args:
            offset (int): should be -1 for scared. Offset to add to frame indices to get trajectory indices
            folder (str): Folder path in format "dataset_X/keyframe_Y"
            frame_indices (list): List of frame indices to get poses for
            
        Returns:
            list: List of 4x4 transformation matrices (cam2world)
        """
        poses = []
        for frame_idx in frame_indices:
            # Convert to 0-indexed for trajectory lookup
            # traj_idx = frame_idx - 1
            traj_idx = frame_idx + offset
            
            if 0 <= traj_idx < len(self.trajs_dict[folder]):
                pose = self.trajs_dict[folder][traj_idx].astype(np.float32)
                poses.append(pose)
            else:
                raise ValueError(f"Frame index {frame_idx} out of range for trajectory in {folder}. Using identity pose.")
        # print(f'poses shape: {np.array(poses).shape}')
        return np.array(poses).squeeze()  