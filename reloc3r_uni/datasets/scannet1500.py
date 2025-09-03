import numpy as np
from reloc3r_uni.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r_uni.utils.image import imread_cv2, cv2
# from pdb import set_trace as bb


DATA_ROOT = './data/scannet1500' 
DATA_ROOT = '/mnt/cluster/datasets/scannet1500'


def label_to_str(label):
    return '_'.join(label)


class ScanNet1500(BaseStereoViewDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT
        self.pairs_path = '{}/test.npz'.format(self.data_root)
        self.subfolder_mask = 'scannet_test_1500/scene{:04d}_{:02d}'
        with np.load(self.pairs_path) as data:
            self.pair_names = data['name']
            pair_names_hard_code = [
                # [806, 0  ,765, 960]  #better:960 bigger in overlap scene
                # [806, 0  ,960, 765]
                # [806, 0  ,630, 705]
                # [806, 0 , 705, 630]
                # [806, 0 , 960, 1020]
                # [806, 0 , 1020, 960]
                # [764, 0 , 2370, 675]
                [764, 0 , 675, 2370]
                ]
            # self.pair_names = pair_names_hard_code

 #better:960 bigger in overlap scene
# [806, 0  ,765, 960]  #better! {'auc@5': np.float64(0.7975706100463867), 'auc@10': np.float64(0.8987853050231933), 'auc@20': np.float64(0.9493926525115967)}
# [806, 0  ,960, 765] {'auc@5': np.float64(0.8849240303039551), 'auc@10': np.float64(0.9424620151519776), 'auc@20': np.float64(0.9712310075759888)}
 #better:overlap scene is equally small
# [806, 0  ,630, 705] #{'auc@5': np.float64(0.8651038408279419), 'auc@10': np.float64(0.932551920413971), 'auc@20': np.float64(0.9662759602069855)}
# [806, 0 , 705, 630] #{'auc@5': np.float64(0.8508729219436646), 'auc@10': np.float64(0.9254364609718323), 'auc@20': np.float64(0.9627182304859161)}
 #better:overlap scene is equally big
# [806, 0 , 960, 1020] #{'auc@5': np.float64(0.6916237115859986), 'auc@10': np.float64(0.8458118557929992), 'auc@20': np.float64(0.9229059278964996)}
# [806, 0 , 1020, 960] #{'auc@5': np.float64(0.6720271825790405), 'auc@10': np.float64(0.8360135912895202), 'auc@20': np.float64(0.9180067956447602)}
    
    def __len__(self):
        return len(self.pair_names)

    def _get_views(self, idx, resolution, rng):
        scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]

        views = []

        for name in [name1, name2]: 
            intrinsics_path = '{}/{}/intrinsic/intrinsic_depth.txt'.format(self.data_root, self.subfolder_mask).format(scene_name, scene_sub_name)
            intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[0:3,0:3]

            pose_path = '{}/{}/pose/{}.txt'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            camera_pose = np.loadtxt(pose_path).astype(np.float32)# cam2world

            color_path = '{}/{}/color/{}.jpg'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            color_image = imread_cv2(color_path)  
            color_image = cv2.resize(color_image, (640, 480))

            # Load depthmap
            depth_path = '{}/{}/depth/{}.png'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            # depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            # color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                    #  intrinsics, 
                                                                    #  resolution, 
                                                                    #  rng=rng)
            color_image, depthmap, intrinsics = self._crop_resize_if_necessary_with_depth(
                color_image, depthmap, intrinsics, resolution, rng=rng)
            # , info=view_idx)

            view_idx_splits = color_path.split('/')

            views.append(dict(
                img = color_image,
                depthmap=depthmap.astype(np.float32),
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'ScanNet1500',
                label = label_to_str(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                ))
        return views

