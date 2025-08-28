from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class DynaSCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(DynaSCAREDDataset, self).__init__(*args, **kwargs)

        # self.K = np.array([[0.82, 0, 0.5, 0],
        #                    [0, 1.02, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        
        # norm x_in_K: 0.5
        # norm y_in_K: 0.5

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        #
        self.dataset_name = 'DynaSCARED'

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        # print('Folder: ', folder)
        # print('Frame index: ', frame_index)
        
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class DynaSCAREDRAWDataset(DynaSCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(DynaSCAREDRAWDataset, self).__init__(*args, **kwargs)

        self.K_dict_registered = {} # key: folder
    def read_yaml_calib(self, yaml_file):
        import cv2
        import numpy as np
        fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
        # Read the matrix M1
        M1_node = fs.getNode("M1")
        M1 = M1_node.mat()  # Returns a numpy array
        fs.release()
        # print("M1 shape:", M1.shape)
        # print(M1)
        return M1


    def get_image_path(self, folder, frame_index, side, log_K = True):
        # load left frames
        f_str = "final_{}l{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)

        if log_K:
            if folder not in self.K_dict_registered:
                import glob
                K_path = glob.glob(os.path.join(self.data_path, folder, "vid",  "*.yaml"))
                assert len(K_path) == 1, f'K_path {K_path} not found: {os.path.join(self.data_path, folder, "vid",  "*.yaml")}'
                assert os.path.exists(K_path[0]), f'K_path {K_path[0]} not found'
                K_raw = self.read_yaml_calib(K_path[0]) 
                # adjust K
                # consider later resize_img from 640,512 to (self.width, self.height), adjust the intrisics K accordingly
                K_resize = K_raw.copy()
                K_resize[0,0] = K_resize[0,0] * self.width / 640
                K_resize[1,1] = K_resize[1,1] * self.height / 512
                K_resize[0,2] = K_resize[0,2] * self.width / 640
                K_resize[1,2] = K_resize[1,2] * self.height / 512
                # print('Resized K: ')
                # print(K_resize)

                # norm_k: to be consisten with scared impelemtnation
                # norm_x and norm_y so that cx cy be 0.5
                norm_scale_fx = K_resize[0,0] / K_resize[0,2]
                norm_scale_fy = K_resize[1,1] / K_resize[1,2]
                K = np.eye(4, dtype=np.float32)
                K[0,2] = 0.5
                K[1,2] = 0.5
                K[0,0] = 0.5*norm_scale_fx
                K[1,1] = 0.5*norm_scale_fy
                # print('Normed K: ')
                # print(K)

                self.K_dict_registered[folder] = K
                
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

    data_path="/mnt/cluster/datasets/Surg_oclr_stereo/"
    fpath = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/splits/DynaSCARED/val_CaToTi011.txt'
    from utils import readlines
    filenames = readlines(fpath)

    dataset = DynaSCAREDRAWDataset(
        data_path=data_path,
        filenames=filenames,
        height=256,
        width=320,
        frame_ids=[0, -1, 1],# control which nbr frames to load: [0, -1, 1] / [0, 1]
        num_scales=4,
        is_train=True, # control aug or not
        img_ext=".png",
    )
    
    dataset[0]
    # print(dataset[0])

