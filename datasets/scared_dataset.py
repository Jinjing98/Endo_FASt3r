from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.dataset_name = 'SCARED'

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

        self.load_gt_poses = True # will enable register_gt_traj_dict and load gt_abs_poses when get_item
        # self.load_gt_poses = False

        # register gt_traj for get_gt_poses
        self.traj_data_root = '/mnt/cluster/datasets/SCARED/'
        if self.load_gt_poses:
            self.trajs_dict = self.get_gt_poses()
            print(f"Loaded {len(self.trajs_dict)} trajectories")


    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

# run unit test
if __name__ == "__main__":

    fpath = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/splits/endovis/train_files.txt'
    # fpath = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/splits/endovis/val_files.txt'
    from utils import readlines
    filenames = readlines(fpath)

    dataset = SCAREDRAWDataset(
        data_path="/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/",
        filenames=filenames,
        height=256,
        width=320,
        frame_ids=[0, -4, 4],# control which nbr frames to load: [0, -1, 1] / [0, 1]
        num_scales=4,
        # is_train=True, # control aug or not
        is_train=False, # control aug or not
        img_ext=".png",
    )
    
    dataset[5]
    # print(dataset[0])