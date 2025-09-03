import numpy as np
from baselines.reloc3r.reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from baselines.reloc3r.reloc3r.utils.image import imread_cv2, cv2
import os
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from baselines.reloc3r.reloc3r.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape, depthmap_to_absolute_camera_coordinates
# from pdb import set_trace as bb


SPLITS_ROOT = '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/'
DATA_ROOT = '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/'
RECT_DATA_ROOT = '/mnt/ceph/tco/TCO-Staff/Homes/jinjing/Datasets/SCARED_rectified/'
TRAJ_DATA_ROOT = '/mnt/cluster/datasets/SCARED/'


def map_resize_frames_info_to_rect_format(pair_names_info):
    pair_names_info_rectified = []
    for line in pair_names_info:
        line = line.strip()
        assert len(line.split()) == 3, "Each line in the pairs file should contain a folder and two frame names."
        line = line.split()
        folder = line[0]
        sequence = int(folder[7])
        keyframe = int(folder[-1])
        name1 = int(line[1])
        # construct new line
        name1_rectified = name1 - 1  # convert to 0-indexed
        keyframe_rectified = keyframe - 1 if sequence in range(8, 10) else keyframe
        new_line = f'dataset{sequence}/keyframe{keyframe_rectified} {name1_rectified} l\n'
        pair_names_info_rectified.append(new_line)
    return pair_names_info_rectified


def label_to_str(label):
    return '_'.join(label)

def pose_vec_to_mat(pose_vec):
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

# pose_vec = list(map(float, line.strip().split()))
# T = pose_vec_to_mat(pose_vec)
def read_freiburg_scipy(path: str, ret_stamps=False, no_stamp=False):
    '''
    src: hayoz_robust.
    '''
    with open(path, 'r') as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]

    trans_scale = 1000 #m2mm
    if no_stamp:       
        trans_np = np.asarray([l[0:3] for l in list if len(l) > 0], dtype=float)
        quat_np = np.asarray([l[3:] for l in list if len(l) > 0], dtype=float)
        trans_quat_np = np.hstack((trans,quat)) 
        trans_quat_np[:3] *= trans_scale 
        # pose_matrix = torch.from_numpy(np.asarray([pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float))
        pose_matrix = np.asarray([pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float)
    else:
        time_stamp = [l[0] for l in list if len(l) > 0]
        try:
            time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp])*100
        except IndexError:
            time_stamp = np.asarray([int(l) for l in time_stamp])
        # trans = torch.from_numpy(np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float))
        # trans *= 1000.0  # m to mm
        # quat = torch.from_numpy(np.asarray([l[4:] for l in list if len(l) > 0], dtype=float))
        trans_np = np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float)
        quat_np = np.asarray([l[4:] for l in list if len(l) > 0], dtype=float)
        trans_quat_np = np.hstack((trans_np,quat_np))

        trans_quat_np[:,:3] *= trans_scale 
        # pose_matrix = torch.from_numpy(np.asarray([pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float))
        pose_matrix = np.asarray([pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float)
        if ret_stamps:
            return pose_matrix, time_stamp
    return pose_matrix

def read_freiburg_lietorch(path: str, ret_stamps=False, no_stamp=False):
    '''
    src: hayoz_robust.
    '''
    with open(path, 'r') as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
    if no_stamp:       
        trans = torch.from_numpy(np.asarray([l[0:3] for l in list if len(l) > 0], dtype=float))
        trans *= 1000.0  #m to mm
        quat = torch.from_numpy(np.asarray([l[3:] for l in list if len(l) > 0], dtype=float))
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))
    else:
        time_stamp = [l[0] for l in list if len(l) > 0]
        try:
            time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp])*100
        except IndexError:
            time_stamp = np.asarray([int(l) for l in time_stamp])
        trans = torch.from_numpy(np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float))
        trans *= 1000.0  # m to mm
        quat = torch.from_numpy(np.asarray([l[4:] for l in list if len(l) > 0], dtype=float))
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))
        if ret_stamps:
            return pose_se3, time_stamp
    return pose_se3

class SCAREDReloc3R(BaseStereoViewDataset):
    '''
    update from scannet1500 of reloc3r'''

    def __init__(self, 
                 split='train',
                 quick=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.data_root = DATA_ROOT
        self.traj_data_root = TRAJ_DATA_ROOT
        self.split = split
        assert split in ['train', 'val', 'test'], "Split must be one of ['train', 'val', 'test']"
        self.pairs_path = '{}/{}_files.txt'.format(self.data_root, split)

        # read all the pairs (1st) from the txt
        with open(osp.join(self.pairs_path), 'r') as f:
            self.pair_names_info = f.readlines()
        self.trajs_dict = {}
        self.pair_names = []
        for line in self.pair_names_info:
            line = line.strip()
            assert len(line.split()) == 3, "Each line in the pairs file should contain a folder and two frame names."
            line = line.split()
            folder = line[0]
            sequence = int(folder[7])
            keyframe = int(folder[-1])
            name1 = int(line[1])
            name2 = name1 + 1
            #//////
            # prepare traj per seq_KF
            if folder not in self.trajs_dict:
                traj_full_folder = self.map_traj_search(self.split, sequence, keyframe)
                # read the traj file for the 1st time
                traj_path = f'{traj_full_folder}/groundtruth.txt'
                assert os.path.exists(traj_path), f"Trajectory file {traj_path} does not exist."
                traj = np.loadtxt(traj_path, dtype=np.float32)
                traj = read_freiburg_scipy(traj_path, ret_stamps=False, no_stamp=False)  # cam2worlds
                self.trajs_dict[folder] = traj
            self.pair_names.append((folder, 
                                    sequence, 
                                    keyframe, 
                                    name1, 
                                    name2))
            if quick:
                if len(self.pair_names) >= 10:
                    break
    def __len__(self):
        return len(self.pair_names)

    def map_traj_search(self, split, sequence, keyframe):
        split_dir_ori = 'testing' if sequence in range(8, 10) else 'training' 
        sequence_ori = sequence 
        keyframe_ori = keyframe-1 if sequence in range(8, 10) else keyframe
        trajfolder = f'{self.traj_data_root}/{split_dir_ori}/dataset_{sequence_ori}/keyframe_{keyframe_ori}/'
        return trajfolder

    def get_depth(self, depth_folder, frame_index, do_flip = False):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            # self.data_path,
            depth_folder,
            # "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)
        assert os.path.exists(depth_path), f"Depth file {depth_path} does not exist."
        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def _get_views(self, idx, resolution, rng):
        folder, scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]
        views = []

        img_full_folder = f'{self.data_root}/{folder}/image_02/data/'# only read the processed left ones with dim 320 256
        depth_full_folder = f'{self.map_traj_search(self.split, scene_name, scene_sub_name)}/data/scene_points/'
        for name in [name1, name2]: 
            color_path = f'{img_full_folder}/{name:010d}.png'
            color_image = imread_cv2(color_path)

            # color_image = cv2.resize(color_image, (640, 480))
            # depthmap = None
            depthmap = self.get_depth(depth_full_folder, name, do_flip=False)
            depthmap = np.expand_dims(depthmap, 0)
            # inputs["depth_gt"] = np.expand_dims(depthmap, 0)
            # print('////depth range',torch.from_numpy(depthmap.astype(np.float32)).min(),
                #   torch.from_numpy(depthmap.astype(np.float32)).max())

            K = self.K.copy()
            height, width = color_image.shape[:2]
            K[0, :] *= width 
            K[1, :] *= height 
            intrinsics = K[0:3,0:3]

            camera_pose = self.trajs_dict[folder][name-1, :].astype(np.float32)  # cam2world
            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     intrinsics, 
                                                                     resolution, 
                                                                     rng=rng)
            # color_image, depthmap, intrinsics = self._crop_resize_if_necessary_with_depth(
            #     color_image, depthmap, intrinsics, resolution, rng=rng)
            # , info=view_idx)

            view_idx_splits = color_path.split('/')

            views.append(dict(
                img = color_image,
                # depthmap=depthmap.astype(np.float32),
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'SCARED',
                label = label_to_str(view_idx_splits[:-3]),
                instance = view_idx_splits[-1],
                ))
        return views

class RectSCAREDReloc3R(BaseStereoViewDataset):
    '''
    update from scannet1500 of reloc3r'''

    def __init__(self, 
                 split='train',
                #  split_version=0,
                 split_version=1,
                 quick=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.data_root = RECT_DATA_ROOT
        self.splits_root = SPLITS_ROOT
        self.traj_data_root = TRAJ_DATA_ROOT
        self.split = split
        self.split_version = split_version
        
        # Validate split_version and split combinations
        assert split_version in [0, 1], "split_version must be either 0 or 1"
        if split_version == 0:
            assert split in ['train', 'test'], "Split must be one of ['train', 'test'] for split_version 0"
        else:
            assert split in ['train', 'val', 'test', 'sequence1','sequence2','sequence1_short'], "Split must be one of ['train', 'val', 'test','sequence1','sequence2','sequence1_short'] for split_version 1"

        # Load sequences based on split_version
        if self.split_version == 1:
            if 'sequence' in split:
                assert split in ['sequence1','sequence2','sequence1_short'],f'split {split} should be one of ["sequence1","sequence2","sequence1_short"]'
                self.pairs_path = '{}/test_files_{}.txt'.format(self.splits_root, split)
            else:
                self.pairs_path = '{}/{}_files.txt'.format(self.splits_root, split)
            assert os.path.exists(self.pairs_path), f"Pairs file {self.pairs_path} does not exist."
            
            # read all the pairs from the txt
            with open(osp.join(self.pairs_path), 'r') as f:
                self.pair_names_info = f.readlines()
            # convert to rectified format
            self.pair_names_info = map_resize_frames_info_to_rect_format(self.pair_names_info)
     
        elif self.split_version == 0:
            '''
            construct the pair_names_info based on auto search results
            '''
            self.pair_names_info = []
            # search all the keyframes in the data_path
            for sequence in range(1, 10):
                if split == 'train' and sequence in range(8, 10):
                    continue
                if split == 'test' and sequence not in range(8, 10):
                    continue
                import glob
                for i in range(0, 5):  
                    keyframes_i = sorted(glob.glob(f'{self.data_root}/dataset_{sequence}/keyframe_{i}/data/left_rectified/*.png'))
                    if len(keyframes_i) <= 1:
                        print(f"No keyframes found for dataset_{sequence}_keyframe_{i} in split {split}.")
                        continue
                    print(f"Found {len(keyframes_i)} keyframes in dataset_{sequence}_keyframe_{i} for split {split}.")
                    # construct the pair_names_info
                    for j in range(len(keyframes_i)-1):# we need to construct pair, so remain one buffer
                        frame_j = keyframes_i[j]
                        img_name = os.path.basename(frame_j).split('.')[0]
                        self.pair_names_info.append(f'dataset{sequence}/keyframe{i} {img_name} {"l"}\n')
        else:
            raise ValueError("split_version should be either 0 or 1.")

        # construct trajs
        self.trajs_dict = {}
        self.K_dict = {}
        self.bf_dict = {}
        self.pair_names = []
        for line in self.pair_names_info:
            line = line.strip()
            assert len(line.split()) == 3, "Each line in the pairs file should contain a folder and two frame names."
            line = line.split()
            folder = line[0]  
            sequence = int(folder[7])
            keyframe = int(folder[-1])
            name1 = int(line[1])
            name2 = name1 + 1
            self.pair_names.append((folder, 
                                    sequence, 
                                    keyframe, 
                                    name1, 
                                    name2))
            #//////
            # prepare traj per seq_KF
            if folder not in self.trajs_dict:
                split_dir_ori = 'testing' if sequence in range(8, 10) else 'training' 
                traj_full_folder = f'{self.traj_data_root}/{split_dir_ori}/dataset_{sequence}/keyframe_{keyframe}/'
                # traj_full_folder = self.map_traj_search(self.split, sequence, keyframe)
                # read the traj file for the 1st time
                traj_path = f'{traj_full_folder}/groundtruth.txt'
                assert os.path.exists(traj_path), f"Trajectory file {traj_path} does not exist."
                traj = np.loadtxt(traj_path, dtype=np.float32)
                traj = read_freiburg_scipy(traj_path, ret_stamps=False, no_stamp=False)  # cam2worlds
                self.trajs_dict[folder] = traj
            # prepare bf and K per seq_KF-used for depth computation
            if folder not in self.bf_dict or folder not in self.K_dict:
                calib_full_folder = f'{self.data_root}/dataset_{sequence}/keyframe_{keyframe}/'
                calib_file = f'{calib_full_folder}/stereo_calib.json'
                assert os.path.exists(calib_file), f"Calibration file {calib_file} does not exist."
                with open(calib_file, 'r') as f:
                    import json
                    calib = json.load(f)
                P2 = np.array(calib['P2']['data']).astype(np.float32).reshape(3, 4)
                bf = P2[0,-1].astype(np.float32)
                K = P2[:3,:3].astype(np.float32)
                self.bf_dict[folder] = bf
                self.K_dict[folder] = K


            if quick:
                if len(self.pair_names) >= 10:
                    break
    def __len__(self):
        return len(self.pair_names)

    # def map_traj_search(self, split, sequence, keyframe):
    #     split_dir_ori = 'testing' if sequence in range(8, 10) else 'training' 
    #     # sequence_ori = sequence 
    #     trajfolder = f'{self.traj_data_root}/{split_dir_ori}/dataset_{sequence_ori}/keyframe_{keyframe_ori}/'
    #     return trajfolder

    def get_depth(self, depth_folder, frame_index, do_flip = False):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            # self.data_path,
            depth_folder,
            # "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)
        assert os.path.exists(depth_path), f"Depth file {depth_path} does not exist."
        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def _get_views(self, idx, resolution, rng):
        folder, scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]
        views = []

        img_full_folder = f'{self.data_root}/dataset_{scene_name}/keyframe_{scene_sub_name}/data/left_rectified'# only read the processed left ones with dim 320 256
        depth_full_folder = f'{self.data_root}/dataset_{scene_name}/keyframe_{scene_sub_name}/data/depthmap_rectified/'
        depth_full_folder = f'{self.data_root}/dataset_{scene_name}/keyframe_{scene_sub_name}/data/depth_raft/'

        for name in [name1, name2]: 
            color_path = f'{img_full_folder}/{name:06d}.png'
            assert os.path.exists(color_path), f"Color image {color_path} does not exist."
            color_image = imread_cv2(color_path)

            # # Load: both scale is 256.0
            scale_depth = 256.0  # this value is used when gen the depth
            depth_path = f'{depth_full_folder}/{name:06d}.png'
            assert os.path.exists(depth_path), f"Depth image {depth_path} does not exist."
            depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / scale_depth
            # print('////depth range',torch.from_numpy(depthmap.astype(np.float32)).min(),
                #   torch.from_numpy(depthmap.astype(np.float32)).max())
            intrinsics = self.K_dict[folder]
            bf = self.bf_dict[folder]
            camera_pose = self.trajs_dict[folder][name, :].astype(np.float32)  # cam2world
            # we need world to cam?
            
            # print('img HW dim and K before crop',color_image.shape, intrinsics)
            color_image, depthmap, intrinsics = self._crop_resize_if_necessary_with_depth(
                color_image, depthmap, intrinsics, resolution, rng=rng)
            # print('pillow img WH dim and K after crop',
                #   color_image.size, intrinsics)
            view_idx_splits = color_path.split('/')

            views.append(dict(
                img = color_image,
                depthmap=depthmap.astype(np.float32),
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'SCARED',
                label = label_to_str(view_idx_splits[:-3]),
                instance = view_idx_splits[-1],
                ))
        return views
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            if 'depthmap' in view:
                assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
                view['pts3d'] = pts3d
                view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            
            #hard set as 0
            view['dynamic_mask'] = np.zeros_like(valid_mask, dtype=np.float32)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            # print('view info:')
            # print(view['label'],view['instance'],view['camera_pose'])
            # print(view['camera_intrinsics'])
        return views
    
   

            
if __name__ == "__main__":
    # # Example usage
    # dataset = SCAREDReloc3R(split='train', quick=True)
    # print(f"Dataset size: {len(dataset)}")
    # for i in range(len(dataset)):
    #     views = dataset._get_views(i, resolution=(320, 256), rng=None)
    #     for view in views:
    #         print(f"View {i}: {view['instance']}, shape: {view['img'].shape}, depth shape: {view['depthmap'].shape}")
    
    
    # Test RectSCAREDReloc3R
    # rect_dataset = RectSCAREDReloc3R(split='train', split_version=0, quick=True)
    rect_dataset = RectSCAREDReloc3R(split='sequence1_short', split_version=0, quick=True)
    print(f"Rectified Dataset size: {len(rect_dataset)}")
    for i in range(len(rect_dataset)):
        views = rect_dataset._get_views(i, resolution=(320, 256), rng=None)
        for view in views:
            print(f"Rectified View {i}: {view['instance']}, shape: {view['img'].shape}, depth shape: {view['depthmap'].shape}")
            # read the image and depth then save 
            img_save_path = 'img.png'
            # cv2.imwrite(img_save_path, view['img'])
            depth_save_path = 'depth.png'
            # cv2.imwrite(depth_save_path, view['depthmap'][0] * 256.0)
            break
        break



