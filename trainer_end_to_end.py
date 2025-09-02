from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
from networks import Customised_DAM
import numpy as np
import torch.optim as optim
import torch.nn as nn

from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from metrics import compute_pose_error_v2

# from networks import DINOEncoder
import random
# import ipdb
# from networks import RelocerX
# from torch.cuda.amp import autocast, GradScaler
import PIL
import PIL.Image
from torchvision.transforms import ToPILImage
import torchvision.transforms as tvf
from PIL import Image

AF_PRETRAINED_ROOT = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/DARES/af_sfmlearner_weights"
RELOC3R_PRETRAINED_ROOT = "/mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/reloc3r/checkpoints/reloc3r-512"


def clamp_pose(pose, min_value=-1.0, max_value=1.0):
    return torch.clamp(pose, min=min_value, max=max_value)

def prepare_images(x, device, size, square_ok=False):
  to_pil = ToPILImage()
  ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  # ImgNorm = tvf.Compose([tvf.ToTensor()])
  imgs = []
  for idx in range(x.size(0)):
      tensor = x[idx].cpu()  # Shape [3, 256, 320]
      img = to_pil(tensor).convert("RGB")
      W1, H1 = img.size
      if size == 224:
          # resize short side to 224 (then crop)
          img = resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
      else:
          # resize long side to 512
          img = resize_pil_image(img, size)
      W, H = img.size
      cx, cy = W//2, H//2
      if size == 224:
          half = min(cx, cy)
          img = img.crop((cx-half, cy-half, cx+half, cy+half))
      else:
          halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
          if not (square_ok) and W == H:
              halfh = 3*halfw/4
          img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
      imgs.append(ImgNorm(img)[None].to(device))
#   print('img dim',imgs[0].shape)
#   print('stack dim',torch.stack(imgs, dim=0).squeeze(1).shape)
#   print('cat dim',torch.cat(imgs, dim=0).shape)
  return torch.cat(imgs, dim=0)
#   return torch.stack(imgs, dim=0).squeeze(1)# redundant: .squeeze(1) safer when batch_size = 1

def resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed for a single GPU
    torch.cuda.manual_seed_all(seed)  # Seed for all GPUs (if you are using multi-GPU)
    
    # Set deterministic option
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, options):

        # update options for debug purposes
        if options.debug:
            print("DEBUG MODE")
            print('update options for debug purposes...')
            options.num_epochs = 50000
            options.batch_size = 1
            options.batch_size = 2
            options.accumulate_steps = 1  # Effective batch size = 1 * 12 = 12
            options.log_frequency = 10
            options.save_frequency = 100000# no save
            options.log_dir = "/mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm_debug"

            options.enable_motion_computation = True
            options.use_loss_reproj2_nomotion = True
            options.use_soft_motion_mask = True


            options.enable_grad_flow_motion_mask = True
            options.use_loss_motion_mask_reg = True

            # options.enable_mutual_motion = True


            # options.zero_pose_debug = True

            options.freeze_depth_debug = True
            options.model_name = "debug"


            options.use_raft_flow = True

            # options.zero_pose_flow_debug = True
            # options.reproj_supervised_with_which = "raw_tgt_gt"
            # options.reproj_supervised_which = "color_MotionCorrected"

            # options.flow_reproj_supervised_with_which = "raw_tgt_gt"

            # options.transform_constraint = 0.0
            # options.transform_smoothness = 0.0
            # options.disparity_smoothness = 0.0

            # options.freeze_as_much_debug = True #save mem # need to be on for OF exp

            options.of_samples = True
            options.of_samples_num = 100
            # options.of_samples_num = 1
            options.is_train = True
            options.is_train = False # no augmentation

            options.frame_ids = [0, -1, 1]
            # options.frame_ids = [0, -2, 2]
            # options.frame_ids = [0, -14, 14]

            # not okay to use: we did not adjust the init_K accordingly yet
            options.height = 192
            options.width = 224

            options.dataset = "endovis"
            options.data_path = "/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/"
            options.split_appendix = ""

            options.dataset = "DynaSCARED"
            options.data_path = "/mnt/cluster/datasets/Surg_oclr_stereo/"
            options.split_appendix = "_CaToTi000"
            options.split_appendix = "_CaToTi001"
            # options.split_appendix = "_CaToTi110"
            options.split_appendix = "_CaToTi101"

        self.opt = options
        
        # sanity check some params early
        if self.opt.use_loss_motion_mask_reg:
            assert self.opt.enable_grad_flow_motion_mask, "enable_grad_flow_motion_mask must be True when use_loss_motion_mask_reg is True"




        from datetime import datetime
        # Current timestamp without year: month-day-hour-minute-second
        timestamp = datetime.now().strftime("%m%d-%H%M%S")
        
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name, timestamp)
        os.makedirs(self.log_path, exist_ok=True)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        set_seed(self.opt.seed)
        # self.scaler = GradScaler()

        print(self.opt.seed, "is the seed!")

        print("learning rate:", self.opt.learning_rate)

        # print("learning rate is",self.opt.learning_rate)
        print("batch size is:",self.opt.batch_size)
        print("accumulate steps is:",self.opt.accumulate_steps)
        print("effective batch size is:",self.opt.batch_size * self.opt.accumulate_steps)

        self.models = {}  
        self.parameters_to_train = []
        self.parameters_to_train_0 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        print("Using DoMoRA")
        self.models["depth_model"] = networks.Endo_FASt3r_depth()
        


        self.models["depth_model"].to(self.device)

        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))

        # Initialize optical flow networks (either custom or RAFT)
        if self.opt.use_raft_flow:
            print("Using RAFT optical flow estimator instead of custom position networks")
            # Use RAFT as a direct replacement for both position_encoder and position
            from networks import RAFT
            self.models["raft_flow"] = RAFT(self.device).model
            # self.models["raft_flow"].to(self.device)
            # RAFT is trainable, so add its parameters to training
            self.parameters_to_train_0 += list(self.models["raft_flow"].parameters())
            print("RAFT flow estimator initialized (trainable)")
        else:
            # Use original custom networks
            self.models["position_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
            
            self.models["position_encoder"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/position_encoder.pth"))
            self.models["position_encoder"].to(self.device)
            self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())

            self.models["position"] = networks.PositionDecoder(
                self.models["position_encoder"].num_ch_enc, self.opt.scales)
            self.models["position"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/position.pth"))
            
            self.models["position"].to(self.device)
            self.parameters_to_train_0 += list(self.models["position"].parameters())

        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.models["transform_encoder"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/transform_encoder.pth"))
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())


        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.models["transform"].load_state_dict(torch.load(f"{AF_PRETRAINED_ROOT}/transform.pth"))
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                reloc3r_ckpt_path = f"{RELOC3R_PRETRAINED_ROOT}/Reloc3r-512.pth"
                from networks import Reloc3rX
                self.models["pose"] = Reloc3rX(reloc3r_ckpt_path)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)
                self.models["pose"].load_state_dict(torch.load(pose_decoder_path))

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                self.models["pose"].load_state_dict(torch.load(pose_decoder_path))

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["pose"].parameters()))

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            print('CHECK: self.opt.predictive_mask')
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        else:
            print('CHECK: NO self.opt.predictive_mask')

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        if self.opt.dataset == 'DynaSCARED':
            assert self.opt.data_path == '/mnt/cluster/datasets/Surg_oclr_stereo/', f"data_path {self.opt.data_path} is not correct"
            datasets_dict = {self.opt.dataset: datasets.DynaSCAREDRAWDataset}
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.dataset, "{}.txt")
            fpath_train = fpath.format(f"train{self.opt.split_appendix}")
            fpath_val = fpath.format(f"val{self.opt.split_appendix}")
            assert self.opt.split_appendix in ['',] or '_CaToTi' in self.opt.split_appendix, f"split_appendix {self.opt.split_appendix} is not correct"
        elif self.opt.dataset == 'endovis':
            assert self.opt.data_path == '/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/', f"data_path {self.opt.data_path} is not correct"
            datasets_dict = {self.opt.dataset: datasets.SCAREDRAWDataset}
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.dataset, "{}_files.txt")
            assert self.opt.split_appendix == '', "split_appendix should be empty for endovis"
            fpath_train = fpath.format(f"train{self.opt.split_appendix}")
            fpath_val = fpath.format(f"val{self.opt.split_appendix}")
        else:
            raise ValueError(f"Unknown dataset: {self.opt.dataset} {self.opt.data_path}")
        self.dataset = datasets_dict[self.opt.dataset]

        train_filenames = readlines(fpath_train)
        val_filenames = readlines(fpath_val)
        img_ext = '.png'  

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            # self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, of_samples=self.opt.of_samples, of_samples_num=self.opt.of_samples_num)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            # self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, of_samples=self.opt.of_samples, of_samples_num=self.opt.of_samples_num)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            self.ms_ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        # self.project_3d_raw = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
            # self.project_3d_raw[scale] = Project3D_Raw(self.opt.batch_size, h, w)
            # self.project_3d_raw[scale].to(self.device)

            # not used?
            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using dataset:\n  ", self.opt.dataset)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        # ipdb.set_trace()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def set_train_0(self):
        """Convert all models to training mode
        """
        if self.opt.use_raft_flow:
            # RAFT models are trainable
            for param in self.models["raft_flow"].parameters():
                param.requires_grad = True
            self.models["raft_flow"].train()
        else:
            # Custom position networks are trainable
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = True
            for param in self.models["position"].parameters():
                param.requires_grad = True
            self.models["position_encoder"].train()
            self.models["position"].train()

        for param in self.models["depth_model"].parameters():
            param.requires_grad = False
        # for param in self.models["pose_encoder"].parameters():
        #     param.requires_grad = False
        for param in self.models["pose"].parameters():
            param.requires_grad = False
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["transform"].parameters():
            param.requires_grad = False

        self.models["depth_model"].eval()
        # self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()

    def freeze_params(self,keys = []):
        # no grad compuation: debug only
        # for all keys in self.models, set requires_grad to False
        for key in keys:
            if key not in self.models:
                continue
            for param in self.models[key].parameters():
                param.requires_grad = False

    def set_train(self):
        """Convert all models to training mode
        """
        if self.opt.use_raft_flow:
            # RAFT models are frozen during main training
            for param in self.models["raft_flow"].parameters():
                param.requires_grad = False
            self.models["raft_flow"].eval()
        else:
            # Custom position networks are frozen during main training
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = False
            for param in self.models["position"].parameters():
                param.requires_grad = False
            self.models["position_encoder"].eval()
            self.models["position"].eval()

        # for param in self.models["encoder"].parameters():
            # param.requires_grad = True
        for param in self.models["depth_model"].parameters():
            param.requires_grad = True
            #debug: freeze depth_model
            if self.opt.freeze_depth_debug:
                param.requires_grad = False

        # for param in self.models["pose_encoder"].parameters():
        #     param.requires_grad = True
        for param in self.models["pose"].parameters():
            param.requires_grad = True
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["transform"].parameters():
            param.requires_grad = True

        self.models["depth_model"].train()
        # self.models["pose_encoder"].train()
        self.models["pose"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
    
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if self.opt.use_raft_flow:
            self.models["raft_flow"].eval()
        else:
            self.models["position_encoder"].eval()
            self.models["position"].eval()
            
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        # self.models["pose_encoder"].eval()
        self.models["pose"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):

            self.run_epoch(self.epoch)
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self, epoch):
        """Run a single epoch of training and validation
        """

        print("Training")

        # Initialize gradient accumulation counters
        accumulate_step = 0
        
        for batch_idx, inputs in enumerate(self.train_loader):
            # print('step:', self.step)

            before_op_time = time.time()

            # position
            self.set_train_0()
            if self.opt.freeze_as_much_debug:
                if self.opt.use_raft_flow:
                    pass
                    # self.freeze_params(keys = ['raft_flow',])#debug only for save mem
                else:
                    self.freeze_params(keys = ['position_encoder',])#debug only for save mem
            _, losses_0 = self.process_batch_0(inputs)
            
            # Scale loss by accumulate_steps for gradient accumulation
            scaled_loss_0 = losses_0["loss"] / self.opt.accumulate_steps
            scaled_loss_0.backward()
            
            accumulate_step += 1
            
            # Only step optimizer when accumulation is complete
            if accumulate_step % self.opt.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train_0, max_norm=1.0)
                self.model_optimizer_0.step()
                self.model_optimizer_0.zero_grad()

            self.set_train()
            if self.opt.freeze_as_much_debug:
                self.freeze_params(keys = ['depth_model', 'pose', 'transform_encoder'])#debug only
            outputs, losses = self.process_batch(inputs) # img_warped_from_pose_flow saved as "color"; img_warped_from_optic_flow saved as "registration"

            # Scale loss by accumulate_steps for gradient accumulation
            scaled_loss = losses["loss"] / self.opt.accumulate_steps
            scaled_loss.backward()
            
            # Only step optimizer when accumulation is complete
            if accumulate_step % self.opt.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                # Use the accumulated loss for logging (multiply back to show effective loss)
                # effective_loss = losses["loss"] * (self.opt.accumulate_steps / max(1, accumulate_step % self.opt.accumulate_steps))
                # self.log_time(batch_idx, duration, effective_loss.cpu().data)

                # losses_0
                losses_to_log = {
                    "loss_0": losses_0["loss"],
                }
                # catains the breakdown of losses
                losses_to_log = {**losses_to_log, **losses}

                errs_to_log = {}
                metric_errs = {}
                if self.train_dataset.load_gt_poses:
                    # compute the pose errors
                    metric_errs = self.compute_pose_metrics(inputs, outputs) # dict
                    # print('trans_err:', trans_err)
                    # print('rot_err:', rot_err)
                    # metric_errs = {
                    #     'trans_err': trans_err,
                    #     'rot_err': rot_err
                    # }

                for k, v in metric_errs.items():
                    assert len(v) == len(self.opt.frame_ids)-1, f'{k}: {v}'
                    # mean over frames_ids (already mean over batches internally)
                    errs_to_log[f"{k}"] = torch.mean(torch.stack(v)).item()

                self.log_time(batch_idx, duration, 
                              scaled_loss.cpu().data, 
                              scaled_loss_0.cpu().data, 
                              errs_to_log)

                scalers_to_log = {**losses_to_log, **errs_to_log}
                # add 'scalar/' prefix to the keys
                scalers_to_log = {f"scalar/{k}": v for k, v in scalers_to_log.items()}

                self.log("train", inputs, outputs, 
                         scalers_to_log, 
                         compute_vis=True)
                # self.log("train", inputs, outputs, losses, compute_vis=True, online_vis=True)

            self.step += 1
            
        # Step schedulers at the end of epoch
        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    def process_batch_0(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            # print('process_batch_0, key:', key)
            inputs[key] = ipt.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses_0(inputs))
        losses = self.compute_losses_0(inputs, outputs)

        return outputs, losses

    def reformat_raft_output(self, num_flow_udpates, outputs_raw):
        outputs = {}
        assert len(outputs_raw) == num_flow_udpates, f"outputs_raw length {len(outputs_raw)} != num_flow_udpates {num_flow_udpates}"

        for scale, scale_raw in enumerate([11,8,5,2]):
            #high to low
            # resize the resolution according to the scale
            # pyramid resolution
            pyramid_resolution_height = self.opt.height // (2 ** scale)
            pyramid_resolution_width = self.opt.width // (2 ** scale)
            if scale!=0:
                outputs[("position", scale)] = F.interpolate(
                        outputs_raw[scale_raw], [pyramid_resolution_height, pyramid_resolution_width], mode="bilinear",
                        align_corners=True)
            else:
                outputs[("position", scale)] = outputs_raw[scale_raw]


        return outputs

    def predict_poses_0(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
            assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # ipdb.set_trace()

                    inputs_all = [pose_feats[f_i], pose_feats[0]]# tgt to src flow
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # position - handle both custom networks and RAFT
                    if self.opt.use_raft_flow:
                        # RAFT expects separate image inputs, not concatenated features
                        # we reformat raft 12 resolution output to 4 resolution output
                        num_flow_udpates = 12
                        outputs_0_raw = self.models["raft_flow"](pose_feats[f_i], pose_feats[0])
                        outputs_1_raw = self.models["raft_flow"](pose_feats[0], pose_feats[f_i])
                        outputs_0 = self.reformat_raft_output(num_flow_udpates, outputs_0_raw)
                        outputs_1 = self.reformat_raft_output(num_flow_udpates, outputs_1_raw)
                    else:
                        # Original custom networks
                        position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                        outputs_0 = self.models["position"](position_inputs)
                        outputs_1 = self.models["position"](position_inputs_reverse)

                    # for k, v in outputs_0.items():
                    #     print(f"{k}: {v.shape}")

                    for scale in self.opt.scales:
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)

                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)],
                                                                                       outputs[(
                                                                                       "position", "high", scale, f_i)])

                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                            outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)])

                    # transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)
                        outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * outputs[
                            ("occu_mask_backward", 0, f_i)].detach() + inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0,
                                                                       max=1.0)
        return outputs

    def compute_losses_0(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += (get_smooth_loss(outputs[("position", scale, frame_id)], color))

                if self.opt.flow_reproj_supervised_with_which == "raw_tgt_gt":
                    reproj_loss_supervised_signal_color = inputs[("color", 0, 0)].detach()
                elif self.opt.flow_reproj_supervised_with_which == "detached_refined":
                    reproj_loss_supervised_signal_color = outputs[("refined", scale, frame_id)].detach()
                else:
                    raise ValueError(f"Invalid flow_reproj_supervised_with_which: {self.opt.flow_reproj_supervised_with_which}")

                loss_registration += (self.compute_reprojection_loss(outputs[("registration", scale, frame_id)], 
                                                                     reproj_loss_supervised_signal_color) * occu_mask_backward).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        if self.opt.enable_mutual_motion:
            # extend the depth prediction for srcs
            for frame_id in self.opt.frame_ids[1:]:
                outputs_i = self.models["depth_model"](inputs["color_aug", frame_id, 0])
                for scale in self.opt.scales:
                    outputs["disp", scale, frame_id] = outputs_i["disp", scale]


        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))# img is warp from optic_flow, save as "registration" 

        outputs = self.generate_images_pred(inputs, outputs)# img is warp from pose_flow('sample'), save as "color"
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses


    def predict_poses(self, inputs, disps):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
            assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":
                    
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # position - handle both custom networks and RAFT
                    if self.opt.use_raft_flow:
                        num_flow_udpates = 12
                        outputs_0_raw = self.models["raft_flow"](pose_feats[f_i], pose_feats[0])
                        outputs_1_raw = self.models["raft_flow"](pose_feats[0], pose_feats[f_i])
                        outputs_0 = self.reformat_raft_output(num_flow_udpates, outputs_0_raw)
                        outputs_1 = self.reformat_raft_output(num_flow_udpates, outputs_1_raw)
                        # RAFT expects separate image inputs, not concatenated features
                    else:
                        # Original custom networks
                        position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                        outputs_0 = self.models["position"](position_inputs)
                        outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:

                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)])
                    
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)],  outputs[("occu_map_backward", scale, f_i)]= self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(outputs[("position", "high", scale, f_i)],
                                                                                                          outputs[("position_reverse", "high", scale, f_i)])

                    # transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:

                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()  + inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0, max=1.0)



                    # view0 = {'img':prepare_images(pose_feats[f_i],self.device, size = 512)}
                    # view1 = {'img':prepare_images(pose_feats[0], self.device, size = 512)}
                    # # pose2 = self.models["pose"](view0,view1)
                    # pose2, _ = self.models["pose"](view0,view1)# notice we save pose2to1 as usually saved by reloc3r/fast3r/mvp3r; dares saved rel pose1to2
                    # outputs[("cam_T_cam", 0, f_i)] = pose2["pose"] # it shoudl save transformation: tgt to src

                    # udpate reloc3r_relpose forward returning and below to be consistent with original reloc3r
                    view1 = {'img':prepare_images(pose_feats[f_i],self.device, size = 512)}
                    view2 = {'img':prepare_images(pose_feats[0], self.device, size = 512)}
                    # pose2 = self.models["pose"](view0,view1)
                    # pose2, _ = self.models["pose"](view0,view1)# notice we save pose2to1 as usually saved by reloc3r/fast3r/mvp3r; dares saved rel pose1to2
                    _ , pose2 = self.models["pose"](view1,view2)# 
                    # notice we save pose2to1 as usually saved by reloc3r/fast3r/mvp3r; dares saved rel pose1to2
                    outputs[("cam_T_cam", 0, f_i)] = pose2["pose"] # we need pose tgt2src, ie: pose2to1, i.e the pose2 in breif in reloc3r model.

                    
        return outputs
    def get_motion_mask_soft(self, optic_flow, pose_flow, detach = True, thre_px = 3):
        '''
        static area will be set to 1, motion area will be set to 0.
        Therefor can be directly used for validness when supervise.
        '''
        # obtain motion mask from motion_flow:
        if detach:
            motion_mask = get_texu_mask(optic_flow.detach(), 
                                        pose_flow.detach(),
                                        ret_conf = True)
        else:
            motion_mask = get_texu_mask(optic_flow, 
                                        pose_flow,
                                        ret_conf = True)

        # use hard code thershoulding of motion_flow:
        # l2 norm of flow_vector is longer than 3 px
        # motion_mask_v2 = (outputs[("motion_flow", frame_id, 0)].norm(dim=1, keepdim=True) > 3).detach()
        # print('motion_mask_requires_grad:')
        # print(motion_mask.requires_grad)
        # print('motion_flow.requires_grad:')
        # print(motion_flow.requires_grad)
        
        return motion_mask

    #compute and regisered the masks: can be used to masked out loss 
    def get_motion_mask(self, motion_flow, frame_id, detach = True, thre_px = 3):
        '''
        static area will be set to 1, motion area will be set to 0.
        Therefor can be directly used for validness when supervise.
        '''
        # obtain motion mask from motion_flow:
        if detach:
            # motion_mask = get_texu_mask(outputs[("position", 0, frame_id)].detach(), 
                                        # outputs[("pose_flow", frame_id, 0)].detach())
            motion_mask = (motion_flow.norm(dim=1, keepdim=True) <= thre_px).detach()
            motion_mask = motion_mask.float()            
        else:
            # motion_mask = get_texu_mask(outputs[("position", 0, frame_id)], 
            #                             outputs[("pose_flow", frame_id, 0)])
            mask_norm = motion_flow.norm(dim=1, keepdim=True)
            mask_hard = (mask_norm > 0.5).float()# no grad
            motion_mask = mask_hard + mask_norm - mask_norm.detach() #still binary but diffirentiable


        # use hard code thershoulding of motion_flow:
        # l2 norm of flow_vector is longer than 3 px
        # motion_mask_v2 = (outputs[("motion_flow", frame_id, 0)].norm(dim=1, keepdim=True) > 3).detach()
        # print('motion_mask_requires_grad:')
        # print(motion_mask.requires_grad)
        # print('motion_flow.requires_grad:')
        # print(motion_flow.requires_grad)
        
        return motion_mask

    def gen_sample_and_pose_flow(self, inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, tgt_frame_id = 0, frame_id = None):
        '''
        only udpate the samples and pose_flow:
        extend with samples_s2t pose_flow_s2t: controler by given swappred input id
        '''
        
        if tgt_frame_id == 0:
            assert frame_id in self.opt.frame_ids[1:], f'frame_id {frame_id} is not in self.opt.frame_ids[1:]'
            compute_tgt2src = True
        else:
            assert frame_id == 0, f'src_frame_id {frame_id} is not 0'
            assert tgt_frame_id in self.opt.frame_ids[1:], f'tgt_frame_id {tgt_frame_id} is not in self.opt.frame_ids[1:]'
            compute_tgt2src = False

        if frame_id == "s":
            T = inputs["stereo_T"]
        else:
            if compute_tgt2src:
                # print('/////!compute tgt2src')
                T = outputs[("cam_T_cam", 0, frame_id)]
            else:
                # print('/////!waring..to be optimal later')
                T = outputs[("cam_T_cam", 0, tgt_frame_id)]
                T = torch.inverse(T) 

        if self.opt.zero_pose_debug:
            T = torch.eye(4).to(self.device).repeat(T.shape[0], 1, 1)

        # if self.opt.pose_model_type == "posecnn":
        #     assert 0, 'posecnn is not supported for mutual motion'

        #     axisangle = outputs[("axisangle", 0, frame_id)]
        #     translation = outputs[("translation", 0, frame_id)]

        #     inv_depth = 1 / outputs[("depth", tgt_frame_id, scale)]
        #     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

        #     T = transformation_from_parameters(
        #         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
        

        # print('tgt frame id:')
        # print(tgt_frame_id)
        # print('frame id:')
        # print(frame_id)
        # print('Compute cam_points')
        cam_points = self.backproject_depth[source_scale](
            outputs[("depth", tgt_frame_id, scale)], inputs[("inv_K", source_scale)])# 3D pts
        # print('cam_points.shape:')
        # print(cam_points.shape)
        # Project3D: it saves values in range [-1,1] for direct sampling
        # pix_coords saves values in range [-1,1]
        # print('compute pix_coords')
        pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)# 2D pxs; T: f0 -> f1  f0->f-1
        # print('pix_coords.shape:')
        # print(pix_coords.shape)
        if compute_tgt2src:
            outputs[("sample", frame_id, scale)] = pix_coords # b h w 2
        else:
            outputs[("sample_s2t", tgt_frame_id, scale)] = pix_coords # b h w 2
        # generate pose_flow from pix_coords
        norm_width_source_scale = self.project_3d[scale].width
        norm_height_source_scale = self.project_3d[scale].height
        # compute the raw_unit value pix_coords_raw from pix_coords, leveraging the fact that pix_coords saves values in range [-1,1]
        if compute_tgt2src:
            pix_coords_raw = outputs[("sample", frame_id, scale)].clone()#.detach()
        else:
            pix_coords_raw = outputs[("sample_s2t", tgt_frame_id, scale)].clone()#.detach()
        pix_coords_raw = pix_coords_raw * 0.5 + 0.5 # convert to range [0,1]
        # at high resolution
        pix_coords_raw[..., 0] = pix_coords_raw[..., 0] * (norm_width_source_scale - 1)
        pix_coords_raw[..., 1] = pix_coords_raw[..., 1] * (norm_height_source_scale - 1)
        # tgt2src pose flow
        if compute_tgt2src:
            outputs[("pose_flow", frame_id, scale)] = pix_coords_raw.permute(0, 3, 1, 2) - mesh_gird_high_res # there is grad; B 2 H W 
        else:
            outputs[("pose_flow_s2t", tgt_frame_id, scale)] = pix_coords_raw.permute(0, 3, 1, 2) - mesh_gird_high_res # there is grad; B 2 H W 
        
        return outputs



    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            if self.opt.enable_mutual_motion and self.opt.enable_motion_computation:
                # collected depth--prepared for mutual pose flow, finally mutual motion_mask
                for frame_id in self.opt.frame_ids[1:]:
                    assert frame_id != 0,f'frame_id == 0 already computed'
                    disp_i = outputs[("disp", scale, frame_id)]

                    assert not self.opt.v1_multiscale,f'v1_multiscale is not supported for mutual motion'
                    disp_i = F.interpolate(
                        disp_i, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                    
                    _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, self.opt.max_depth)
                    outputs[("depth", frame_id, scale)] = depth_i

            source_scale = 0

            # # Create sampling grid
            # use pose_flow(sample-img_grid) and optic_flow(position)
            # implement mem effecient mesh_gird_high_res 
            x = torch.linspace(0, self.opt.width - 1, self.opt.width, device=self.device)
            y = torch.linspace(0, self.opt.height - 1, self.opt.height, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
            mesh_gird_high_res = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.unsqueeze(0)  # (1, H, W, 2)
            mesh_gird_high_res = mesh_gird_high_res.permute(0, 3, 1, 2)  # (B, 2, H, W)            
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                        tgt_frame_id = 0, frame_id = frame_id)
                
                # outputs_dbg = outputs.copy()
                # outputs_dbg = self.gen_sample_and_pose_flow(inputs, outputs_dbg, mesh_gird_high_res, scale, source_scale = 0, 
                #                                         tgt_frame_id = 0, frame_id = frame_id)
                # check the each ekement in outputs and outputs_dbg should be exactly the same
                # for k, v in outputs.items():
                #     if k in outputs_dbg:
                #         print(f'check {k} in outputs and outputs_dbg should be exactly the same')
                #         assert torch.all(v == outputs_dbg[k]), f'{k} in outputs and outputs_dbg should be exactly the same'
                #     else:
                #         print(f'{k} in outputs_dbg not in outputs')
                #         assert 0,f'{k} in outputs_dbg not in outputs'

                
                if self.opt.enable_mutual_motion and self.opt.enable_motion_computation:
                    # it will update: samples_s2t and pose_flow_s2t in outputs
                    outputs = self.gen_sample_and_pose_flow(inputs, outputs, mesh_gird_high_res, scale, source_scale = 0, 
                                                        tgt_frame_id = frame_id, frame_id = 0)

                if self.opt.enable_motion_computation:
                    # If you want to warp the source image src to the target view tgt, you need the flow that maps pixels from the target frame to the source frame, i.e., tgt → src.
                    # assert 0,f'use color is correct! not need to use color_motion_corrected! considering the pose_flow now is computed from tgt depth+pose'
                    # tgt2src motion flow
                    # avoid grad in motion_flow! elsewise the sample_motion_corrected will loss grad at all.
                    outputs[("motion_flow", frame_id, scale)] = - outputs[("pose_flow", frame_id, scale)].detach() + outputs[("position", "high", 0, frame_id)]#.detach() # 
                    

                    if scale == 0:
                        if self.opt.use_soft_motion_mask:
                            outputs[("motion_mask_backward", 0, frame_id)] = self.get_motion_mask_soft(outputs[("position", 0, frame_id)], 
                                                                  outputs[("pose_flow", frame_id, 0)],
                                                                  detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                  thre_px=self.opt.motion_mask_thre_px)
                        else:
                            outputs[("motion_mask_backward", 0, frame_id)] = self.get_motion_mask(
                                                                                            #   outputs[("motion_flow", frame_id, 0)], 
                                                                                              - outputs[("pose_flow", frame_id, scale)] + outputs[("position", "high", 0, frame_id)], # there is grad_flow here! differ from motion_flow
                                                                                              frame_id, 
                                                                                              detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                                              thre_px=self.opt.motion_mask_thre_px)
                    
                    if self.opt.enable_mutual_motion:
                        # compute s2t motion_flow and s2t_motion_mask
                        outputs[("motion_flow_s2t", frame_id, 0)] = - outputs[("pose_flow_s2t", frame_id, 0)].detach() + outputs[("position_reverse", "high", 0, frame_id)]#.detach() # 
                        if scale == 0:
                            if self.opt.use_soft_motion_mask:
                                outputs[("motion_mask_s2t_backward", 0, frame_id)] = self.get_motion_mask_soft(outputs[("position_reverse", 0, frame_id)], 
                                                                  outputs[("pose_flow_s2t", frame_id, 0)],
                                                                  detach=(not self.opt.enable_grad_flow_motion_mask), 
                                                                  thre_px=self.opt.motion_mask_thre_px)
                            else:
                                outputs[("motion_mask_s2t_backward", 0, frame_id)] = self.get_motion_mask(
                                                                                                    # outputs[("motion_flow_s2t", frame_id, 0)], 
                                                                                                    - outputs[("pose_flow_s2t", frame_id, 0)] + outputs[("position_reverse", "high", 0, frame_id)], # there is grad_flow here! differ from motion_flow
                                                                                                    frame_id, 
                                                                                                      detach=(not self.opt.enable_grad_flow_motion_mask), thre_px=self.opt.motion_mask_thre_px)


                # pose_flow
                # assert outputs[("sample", frame_id, scale)].max() <= 1.0 and outputs[("sample", frame_id, scale)].min() >= -1.0
                # print('max min outputs[("sample", frame_id, scale)]:')
                # print(outputs[("sample", frame_id, scale)].max())
                # print(outputs[("sample", frame_id, scale)].min())
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)
                
                if self.opt.enable_motion_computation:
                    # gen sample_motion_corrected from pose_flow and motion_flow, then sample image with sample_motion_corrected
                    # sample_motion_corrected = (outputs[("sample", frame_id, scale)] + outputs[("motion_flow", frame_id, scale)].permute(0, 2, 3, 1))
                    sample_motion_corrected = (outputs[("pose_flow", frame_id, scale)] + outputs[("motion_flow", frame_id, scale)])#.permute(0, 2, 3, 1)
                    #use self.spatial_transform to sample

                    outputs[("color_MotionCorrected", frame_id, scale)] = self.spatial_transform(
                        inputs[("color", frame_id, 0)],
                        # inputs[("color", 0, source_scale)],
                        sample_motion_corrected)

                # # not used
                # outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                #         cam_points, inputs[("K", source_scale)], T)
        
        return outputs


    def compute_pose_metrics(self, inputs, outputs):
        """
        Compute pose errors between ground truth and predicted relative poses.
        
        Args:
            inputs: Dictionary containing ground truth poses ("gt_c2w_poses", frame_id)
            outputs: Dictionary containing predicted poses ("cam_T_cam", 0, frame_id)
            
        Returns:
            trans_err_mean: Mean translation error across batch and frames (in mm)
            rot_err_mean: Mean rotation error across batch and frames (in degrees)
        """
        metrics_list_dict = {}

        # This line has a bug - frame_id is not defined yet
        # esti_tgt2src_rel_poses = outputs[("cam_T_cam", 0, frame_id)]
        if ('gt_c2w_poses', 0) not in inputs:
            print(f'warning: gt_c2w_poses not in inputs')
            print(f'load_gt is train:{self.train_dataset.load_gt_poses}')
            print(f'load_gt is val:{self.val_dataset.load_gt_poses}')
            return metrics_list_dict
        
        # gt_tgt_abs_poses: (B, 4, 4)
        gt_tgt_abs_poses = inputs[("gt_c2w_poses", 0)]  # (B, 4, 4)
        for frame_id in self.opt.frame_ids[1:]:
            gt_src_abs_poses = inputs[("gt_c2w_poses", frame_id)]  # (B, 4, 4)
            pred_rel_poses_batch = outputs[("cam_T_cam", 0, frame_id)]  # (B, 4, 4)
            gt_tgt2src_rel_poses = torch.inverse(gt_src_abs_poses) @ gt_tgt_abs_poses
            assert gt_tgt2src_rel_poses.shape == pred_rel_poses_batch.shape, f'gt_tgt2src_rel_poses.shape: {gt_tgt2src_rel_poses.shape}, pred_rel_poses_batch.shape: {pred_rel_poses_batch.shape}'
            
            # err_dict = compute_pose_error(gt_tgt2src_rel_poses, pred_rel_poses_batch.detach())
            # trans_err_list.append(err_dict['trans_err'])
            # rot_err_list.append(err_dict['rot_err'])

            err_dict = compute_pose_error_v2(gt_tgt2src_rel_poses, pred_rel_poses_batch.detach())
            for k, v in err_dict.items():
                # print(f'{k}: {v}')
                assert k in ['trans_err_ang', 'trans_err_ang_deg', 'trans_err_scale', 'trans_err_scale_norm', 'rot_err', 'rot_err_deg']
                metrics_list_dict[k] = metrics_list_dict.get(k, []) + [v]


            # print('gt_tgt2src_rel_poses shape:')
            # print(gt_tgt2src_rel_poses.shape)
            # print('pred_rel_poses_batch shape:')
            # print(pred_rel_poses_batch.shape)
            if frame_id == self.opt.frame_ids[1]:
                print('gt_tgt2src_rel_poses trans:')
                print(gt_tgt2src_rel_poses[:, :3, 3])
                print('pred_rel_poses_batch trans:')
                print(pred_rel_poses_batch[:, :3, 3])

            # print('gt_tgt2src_rel_poses rot:')
            # print(gt_tgt2src_rel_poses[:, :3, :3])
            # print('pred_rel_poses_batch rot:')
            # print(pred_rel_poses_batch[:, :3, :3])

        # trans_err_list = torch.cat(trans_err_list, 0)
        # rot_err_list = torch.cat(rot_err_list, 0)
        # print esti_rel and gt_rel for debug purpose
        # return trans_err_list.mean(), rot_err_list.mean()

        # print('Report all metrics:')
        # for k, v in metrics_list_dict.items():
        #     print(f'{k}: {v}')

        # return trans_err_list, rot_err_list
        return metrics_list_dict

    def compute_motion_mask_reg_loss(self, optic_flow, motion_mask, is_soft_mask):
        assert optic_flow.dim() == 4, f'optic_flow.dim() is {optic_flow.dim()}'
        assert motion_mask.dim() == 3, f'motion_mask.dim() is {motion_mask.dim()}'
        assert optic_flow.shape[1] == 2, f'optic_flow.shape[1] is {optic_flow.shape[1]}'
        # print(f'optic_flow.shape: {optic_flow.shape}')
        # print(f'motion_mask.shape: {motion_mask.shape}')
        
        from mask_utils import structure_loss_soft, structure_loss
        
        if is_soft_mask:
            loss_mag, loss_edge, loss_dice, imgs_debug = structure_loss_soft(optic_flow, motion_mask)
        else:
            # check the motion mask is binary
            assert motion_mask.max() in [0,1], f'motion_mask.max() is {motion_mask.max()}'
            assert motion_mask.min() in [0,1], f'motion_mask.min() is {motion_mask.min()}'
            loss_mag, loss_edge, loss_dice, imgs_debug = structure_loss(optic_flow, motion_mask)
        
        return loss_mag, loss_edge, loss_dice, imgs_debug

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ms_ssim_loss = 1 - self.ms_ssim(pred, target)
            reprojection_loss = 0.9 * ms_ssim_loss + 0.1 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0
            if self.opt.use_loss_reproj2_nomotion:
                loss2_reprojection = 0
            if self.opt.use_loss_motion_mask_reg:
                assert self.opt.enable_grad_flow_motion_mask, "enable_grad_flow_motion_mask must be True when use_loss_motion_mask_reg is True"

                # loss_motion_mask_reg = 0
                loss_reg_dice = 0
                loss_reg_edge = 0
                loss_reg_mag = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]


            for frame_id in self.opt.frame_ids[1:]:
                                # valid_mask = occu_mask_backward * ~outputs[("motion_mask_backward", 0, frame_id)].detach()

                # img warped from pose_flow is saved as "color"; img warped from optic_flow is saved as "registration"
                # register for debug monitoring
                
                # the 1st reporj_loss: main aim: enforce proper AF learning when everything defautl as it is.
                if self.opt.reproj_supervised_which == "color_MotionCorrected" and self.opt.enable_motion_computation:
                    reproj_loss_supervised_tgt_color = outputs[("color_MotionCorrected", frame_id, scale)]
                elif self.opt.reproj_supervised_which == "color":
                    reproj_loss_supervised_tgt_color = outputs[("color", frame_id, scale)] 
                else:
                    raise ValueError(f"Invalid reproj_supervised_which: {self.opt.reproj_supervised_which}")

                #phedo gt
                if self.opt.reproj_supervised_with_which == "refined":
                    reproj_loss_supervised_signal_color = outputs[("refined", scale, frame_id)]
                elif self.opt.reproj_supervised_with_which == "raw_tgt_gt":
                    reproj_loss_supervised_signal_color = inputs[("color", 0, 0)]
                else:
                    raise ValueError(f"Invalid reproj_supervised_with_which: {self.opt.reproj_supervised_with_which}")

                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                valid_mask = occu_mask_backward

                loss_reprojection += (
                    self.compute_reprojection_loss(reproj_loss_supervised_tgt_color, reproj_loss_supervised_signal_color) * valid_mask).sum() / valid_mask.sum()  
                loss_transform += (
                    torch.abs(outputs[("refined", scale, frame_id)] - outputs[("registration", 0, frame_id)].detach()).mean(1, True) * valid_mask).sum() / valid_mask.sum()
                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)], inputs[("color", 0, 0)], outputs[("registration", scale, frame_id)].detach(), valid_mask)

                #register reproj_loss_supervised_tgt_color for debugging
                outputs[("reproj_supervised_tgt_color_debug", scale, frame_id)] = reproj_loss_supervised_tgt_color.detach() #reproj_loss_supervised_tgt_color.detach()
                outputs[("reproj_supervised_signal_color_debug", scale, frame_id)] = reproj_loss_supervised_signal_color.detach() #reproj_loss_supervised_tgt_color.detach()


                # the 2st reporj_loss: main aim: supervise PoseNet properly by masking out the mutual motion area
                if self.opt.use_loss_reproj2_nomotion:
                    if self.opt.reproj2_supervised_which == "color":
                        reproj_loss2_supervised_tgt_color = outputs[("color", frame_id, scale)]
                    else:
                        raise ValueError(f"Invalid reproj2_supervised_which: {self.opt.reproj2_supervised_which}")
                    
                    if self.opt.reproj2_supervised_with_which == "refined":
                        reproj_loss2_supervised_signal_color = outputs[("refined", scale, frame_id)]
                    else:
                        raise ValueError(f"Invalid reproj2_supervised_with_which: {self.opt.reproj2_supervised_with_which}")
                    
                motion_mask_backward = outputs[("motion_mask_backward", 0, frame_id)].detach()
                if self.opt.enable_mutual_motion:
                    # computational expensive but safer
                    motion_mask_s2t_backward = outputs[("motion_mask_s2t_backward", 0, frame_id)].detach()
                    # conver to binary if it was soft motion mask: this is critical for soft motion mask regularization loss
                    valid_mask2 = (motion_mask_backward > 0.5).float() * (motion_mask_s2t_backward > 0.5).float()
                else:
                    valid_mask2 = (motion_mask_backward > 0.5).float()

                if self.opt.use_loss_reproj2_nomotion:
                    loss2_reprojection += (
                        self.compute_reprojection_loss(reproj_loss2_supervised_tgt_color, reproj_loss2_supervised_signal_color) * valid_mask2).sum() / valid_mask2.sum()  
                    
                #compute motion mask reg loss
                if self.opt.use_loss_motion_mask_reg and scale == 0:
                    optic_flow = outputs[("position", "high", 0, frame_id)]
                    motion_mask = outputs[("motion_mask_backward", 0, frame_id)].squeeze(1)

                    loss_mag, loss_edge, loss_dice, imgs_debug = self.compute_motion_mask_reg_loss(
                        optic_flow, 
                        1-motion_mask,# we want to apply strcuture loss of moiton_mask where motion is positive
                        is_soft_mask=self.opt.use_soft_motion_mask)
                    # print(f'loss_mag: {loss_mag}, loss_edge: {loss_edge}, loss_dice: {loss_dice}')
                    

                    if self.step % self.opt.log_frequency == 0:
                        save_root = os.path.join(self.log_path, f'motion_mask_reg_related')
                        os.makedirs(save_root, exist_ok=True)
                        # concat image in imgs_debug
                        from utils import color_to_cv_img
                        concat_imgs = []
                        for batch_idx in range(self.opt.batch_size):
                            concat_img = np.concatenate([color_to_cv_img(img[batch_idx][None]) for k, img in imgs_debug.items()], axis=1)
                            concat_imgs.append(concat_img)
                        concat_img = np.concatenate(concat_imgs, axis=0)
                        save_path = os.path.join(save_root, f'{self.step}_concat.png')
                        cv2.imwrite(save_path, concat_img)
                        print(f'saved concat image to {save_path}')
                            # cv2.imwrite(save_path, concat_img)
                            # print(f'saved concat image to {save_path}')

                        # # save each image in imgs_debug
                        # for k, img in imgs_debug.items():
                        #     # print(f'{k}: {img.shape}')
                        #     from utils import color_to_cv_img
                        #     batch_idx = 0
                        #     img_cv = color_to_cv_img(img[batch_idx][None])
                        #     save_path = os.path.join(save_root, f'{self.step}_{k}.png')
                        #     cv2.imwrite(save_path, img_cv)
                        #     # print(f'saved {k} to {save_path}')

                    loss_reg_dice += loss_dice
                    loss_reg_edge += loss_edge
                    loss_reg_mag += loss_mag

                    if self.opt.enable_mutual_motion:
                        optic_flow = outputs[("position_inverse", "high", 0, frame_id)]
                        motion_mask = outputs[("motion_mask_s2t_backward", 0, frame_id)].squeeze(1)

                        loss_mag, loss_edge, loss_dice, imgs_debug = self.compute_motion_mask_reg_loss(
                            optic_flow, 
                            1-motion_mask,# we want to apply strcuture loss of moiton_mask where motion is positive
                            is_soft_mask=self.opt.use_soft_motion_mask)
                        # loss_motion_mask_reg += (weights[0] * loss_mag + weights[1] * loss_edge + weights[2] * loss_dice) * self.opt.motion_mask_reg_loss_weight
                        loss_reg_dice += loss_dice
                        loss_reg_edge += loss_edge
                        loss_reg_mag += loss_mag

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0) 
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            if self.opt.use_loss_motion_mask_reg:
                weights = [1.0, 0.2, 0.05]
                # only on scale 0
                loss += self.opt.motion_mask_reg_loss_weight * \
                    ((weights[0] * loss_reg_mag \
                        + weights[1] * loss_reg_edge \
                            + weights[2] * loss_reg_dice) / 2.0 )

            if self.opt.use_loss_reproj2_nomotion:
                loss += loss2_reprojection / 2.0

            total_loss += loss
            # total
            losses["loss/{}".format(scale)] = loss

            # log in loss breakdown
            # log in the loss_reproj
            losses['loss/scale_{}_reproj'.format(scale)] = loss_reprojection
            losses['loss/scale_{}_transform'.format(scale)] = loss_transform
            losses['loss/scale_{}_cvt'.format(scale)] = loss_cvt
            if self.opt.use_loss_motion_mask_reg:
                losses['loss/scale_{}_motion_mask_reg_mag'.format(scale)] = loss_reg_mag
                losses['loss/scale_{}_motion_mask_reg_edge'.format(scale)] = loss_reg_edge
                losses['loss/scale_{}_motion_mask_reg_dice'.format(scale)] = loss_reg_dice

            if self.opt.use_loss_reproj2_nomotion:
                losses['loss/scale_{}_reproj2_nomotion'.format(scale)] = loss2_reprojection


        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def val(self):
        print('Do val....')

        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses, compute_vis=False, online_vis=False)
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            print('CHECK: self.opt.pose_model_type == "shared"')
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            print('CHECK: self.opt.pose_model_type != "shared"')
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        outputs = self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)
        # compute the pose errors
        metrics_list_dict = self.compute_pose_metrics(inputs, outputs)
        
        # for k, v in metrics_list_dict.items():
            # print(f'{k}: {v}')

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(outputs[("registration", scale, frame_id)].mean(1, True), target.mean(1, True)))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss

        return losses

    def log_time(self, batch_idx, duration, loss, loss_0=None, metric_errs=None):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f}"
        
        if loss_0 is not None:
            print_string += " | loss_0: {:.5f}"
        
        print_string += " | time elapsed: {} | time left: {}"
        
        if loss_0 is not None:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, loss_0,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
        # Add pose metrics if available
        if metric_errs:
            print("  Pose Errors:", end=" ")
            for k, v in metric_errs.items():
                if isinstance(v, torch.Tensor):
                    # v = v.mean().item()
                    v = v.item()
                print(f"{k}: {v:.3f}", end=" | ")
            print()

    def log(self, mode, inputs, outputs, scalers_to_log, compute_vis=True, online_vis=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        for l, v in scalers_to_log.items():
            writer.add_scalar("{}".format(l), v, self.step)
        #

        src_imgs = []
        tgt_imgs = []
        registered_tgt_imgs = []
        colored_tgt_imgs = []
        colored_motion_tgt_imgs = []
        optic_flow_imgs = []
        pose_flow_imgs = []
        motion_flow_imgs = []
        depth_imgs = []

        occlursion_mask_imgs = []
        motion_mask_imgs = []
        motion_mask_s2t_imgs = []

        reproj_supervised_tgt_color_debug_imgs = []
        img_order_strs = []
        concat_img = None

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        for j in range(min(1, self.opt.batch_size)):  # write a maxmimum of 2 images
            # for s in self.opt.scales:
            for s in [0]:
                # frames_ids = [0,-1,1]
                assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
                assert len(self.opt.frame_ids) == 3, "frame_ids must be have 3 frames"
                # for frame_id in self.opt.frame_ids[1:]:
                for frame_id in self.opt.frame_ids[1:2]:  # only for one is enough for debug

                    writer.add_image(
                        "IMG/tgt_refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "Other/brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "IMG/registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)

                    if s == 0:
                        writer.add_image(
                            "Other/occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id)][j].data, self.step)
                        #/////////////EXTEND////////////////////////
                        if self.opt.enable_motion_computation:
                            writer.add_image(
                                "Other/motion_mask_backward_{}_{}/{}".format(frame_id, s, j),
                                outputs[("motion_mask_backward", s, frame_id)][j].data, self.step)
                            if self.opt.enable_mutual_motion:
                                writer.add_image(
                                    "Other/motion_mask_s2t_backward_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("motion_mask_s2t_backward", s, frame_id)][j].data, self.step)
                                writer.add_image(
                                    "Other/motion_flow_s2t_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("motion_flow_s2t", frame_id, s)][j].data, self.step)
                        
                        # add src and tgt
                        writer.add_image(
                            "GT/tgt_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step) 
                       # add gt_source
                        writer.add_image(
                            "GT/source_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", 0, s)][j].data, self.step)  
                        # add supervised_img
                        writer.add_image(
                            "GT/reproj_supervised_tgt_color_debug_{}_{}/{}".format(frame_id, s, j),
                            # "IMG/reproj_supervised_tgt_color_debug_{}_{}/{}".format(frame_id, s, j),
                            outputs[("reproj_supervised_tgt_color_debug", 0, frame_id)][j].data, self.step)
                        # add vis of other warped img
                        # add color image as well
                        writer.add_image(
                            "IMG/color_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                        
                        if self.opt.enable_motion_computation:
                            writer.add_image(
                                "IMG/color_MotionCorrected_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color_MotionCorrected", frame_id, s)][j].data, self.step)

                        # write various flow; import functions from utils
                        from utils import flow_vis, flow_vis_robust
                        vis_flow_func = flow_vis_robust
                        # vis_flow_func = flow_vis
                        # add optic flow
                        writer.add_image(
                            "FLOW/optic_flow_{}_{}/{}".format(frame_id, s, j),
                            vis_flow_func(outputs[("position", s, frame_id)][j].data), self.step)
                        # add pose_flow
                        writer.add_image(
                            "FLOW/pose_flow_{}_{}/{}".format(frame_id, s, j),
                            vis_flow_func(outputs[("pose_flow", frame_id, s)][j].data), self.step)
                        # add motion_flow
                        if self.opt.enable_motion_computation:
                            writer.add_image(
                                "FLOW/motion_flow_{}_{}/{}".format(frame_id, s, j),
                                vis_flow_func(outputs[("motion_flow", frame_id, s)][j].data), self.step)
                    
                    if compute_vis and s == 0:
                        '''
                        only vis scale 0
                        '''
                        assert self.opt.frame_ids[0] == 0, "frame_id 0 must be the first frame"
                        src_imgs.append(inputs[("color", frame_id, s)][j].data)
                        tgt_imgs.append(inputs[("color", self.opt.frame_ids[0], s)][j].data)
                        registered_tgt_imgs.append(outputs[("registration", s, frame_id)][j].data)
                        depth_imgs.append(outputs[("depth", self.opt.frame_ids[0], s)][j].data)

                        colored_tgt_imgs.append(outputs[("color", frame_id, s)][j].data)
                        optic_flow_imgs.append(outputs[("position", "high", s, frame_id)][j].data)
                        occlursion_mask_imgs.append(outputs[("occu_mask_backward", s, frame_id)][j].data)
                        pose_flow_imgs.append(outputs[("pose_flow", frame_id, s)][j].data)
                        reproj_supervised_tgt_color_debug_imgs.append(outputs[("reproj_supervised_tgt_color_debug", 0, frame_id)][j].data)
                        if self.opt.enable_motion_computation:
                            colored_motion_tgt_imgs.append(outputs[("color_MotionCorrected", frame_id, s)][j].data)
                            motion_flow_imgs.append(outputs[("motion_flow", frame_id, s)][j].data)
                            motion_mask_imgs.append(outputs[("motion_mask_backward", s, frame_id)][j].data)
                            if self.opt.enable_mutual_motion:
                                motion_mask_s2t_imgs.append(outputs[("motion_mask_s2t_backward", s, frame_id)][j].data)

                # only predicted depth from the center image (frame_id == 0)
                writer.add_image(
                    "Depth/disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                writer.add_image(
                    'Depth/depth_{}/{}'.format(s, j),
                    normalize_image(outputs[("depth", 0, s)][j]), self.step)

        if compute_vis:
            from utils import color_to_cv_img, gray_to_cv_img, flow_to_cv_img
            # Now apply
            src_imgs = [color_to_cv_img(img) for img in src_imgs]
            tgt_imgs = [color_to_cv_img(img) for img in tgt_imgs]
            reproj_supervised_tgt_color_debug_imgs = [color_to_cv_img(img) for img in reproj_supervised_tgt_color_debug_imgs]
            registered_tgt_imgs = [color_to_cv_img(img) for img in registered_tgt_imgs]
            depth_imgs = [gray_to_cv_img(normalize_image(img)).astype(np.uint8) for img in depth_imgs]
            colored_tgt_imgs = [color_to_cv_img(img) for img in colored_tgt_imgs]
            # print('Optic flow:')
            optic_flow_imgs = [flow_to_cv_img(img) for img in optic_flow_imgs]
            # print('Pose flow:')
            pose_flow_imgs = [flow_to_cv_img(img) for img in pose_flow_imgs]
            if self.opt.enable_motion_computation:
                colored_motion_tgt_imgs = [color_to_cv_img(img) for img in colored_motion_tgt_imgs]
                # print('Motion flow:')
                motion_flow_imgs = [flow_to_cv_img(img) for img in motion_flow_imgs]
                motion_mask_imgs = [gray_to_cv_img(img) for img in motion_mask_imgs]
                if self.opt.enable_mutual_motion:
                    motion_mask_s2t_imgs = [gray_to_cv_img(img) for img in motion_mask_s2t_imgs]

            occlursion_mask_imgs = [gray_to_cv_img(img) for img in occlursion_mask_imgs]


            # concat src_imgs and tgt_imgs vertically
            src_concat_img = np.concatenate(src_imgs, axis=0)
            img_order_strs.append('Src-')
            tgt_concat_img = np.concatenate(tgt_imgs, axis=0)
            img_order_strs.append('Tgt-')

            # add reproj_supervised_tgt_color_debug
            reproj_supervised_tgt_color_debug_concat_img = np.concatenate(reproj_supervised_tgt_color_debug_imgs, axis=0)
            img_order_strs.append('Reproj_Sup-')
            
            registered_tgt_concat_img = np.concatenate(registered_tgt_imgs, axis=0)
            img_order_strs.append('Registered_Tgt-')
            colored_tgt_concat_img = np.concatenate(colored_tgt_imgs, axis=0)
            img_order_strs.append('Colored_Tgt-')
            optic_flow_concat_img = np.concatenate(optic_flow_imgs, axis=0)
            img_order_strs.append('Optic_Flow-')
            pose_flow_concat_img = np.concatenate(pose_flow_imgs, axis=0)
            img_order_strs.append('Pose_Flow-')
            occlursion_mask_concat_img = np.concatenate(occlursion_mask_imgs, axis=0)
            img_order_strs.append('Occlursion_Mask-')
            depth_concat_img = np.concatenate(depth_imgs, axis=0)
            img_order_strs.append('Depth-')
            concat_img = np.concatenate([src_concat_img, tgt_concat_img, reproj_supervised_tgt_color_debug_concat_img, \
                                         registered_tgt_concat_img, colored_tgt_concat_img, optic_flow_concat_img, \
                                            pose_flow_concat_img, occlursion_mask_concat_img, depth_concat_img], axis=1)
            if self.opt.enable_motion_computation:
                colored_motion_tgt_concat_img = np.concatenate(colored_motion_tgt_imgs, axis=0)
                img_order_strs.append('Colored_Motion_Tgt-')
                motion_flow_concat_img = np.concatenate(motion_flow_imgs, axis=0)
                img_order_strs.append('Motion_Flow-')
                motion_mask_concat_img = np.concatenate(motion_mask_imgs, axis=0)
                img_order_strs.append('Motion_Mask-')
                if self.opt.enable_mutual_motion:
                    motion_mask_s2t_concat_img = np.concatenate(motion_mask_s2t_imgs, axis=0)
                    img_order_strs.append('Motion_Mask_S2T-')
                    concat_img = np.concatenate([concat_img, colored_motion_tgt_concat_img, motion_flow_concat_img, motion_mask_concat_img, motion_mask_s2t_concat_img], axis=1)
                else:
                    concat_img = np.concatenate([concat_img, colored_motion_tgt_concat_img, motion_flow_concat_img, motion_mask_concat_img], axis=1)

            img_order_strs = ''.join(img_order_strs)
            
            if online_vis:
                title = f'{img_order_strs}'
                cv2.imshow(title, concat_img/255)
                cv2.waitKey(1)
            else:
                import os, cv2
                save_path = os.path.join(self.log_path, f"imgs")
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f"{self.step}_{img_order_strs}.png")
                cv2.imwrite(save_path, concat_img)
                print(f"saved {img_order_strs}.png in {save_path}")

        return concat_img, img_order_strs


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")



if __name__ == "__main__":
    import torch
    from layers import BackprojectDepth
    batch_size = 1
    height = 480
    width = 640
    backproject_depth = BackprojectDepth(batch_size, height, width)
    # project3d = Project3D(batch_size, height, width)
    project3d = Project3D_Raw(batch_size, height, width)
    depth = torch.randn(batch_size, 1, height, width)
    fx, fy, cx, cy = 1000, 1000, 320, 240
    K = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0)
    inv_K = torch.inverse(K)
    T = torch.eye(4)

    cam_points = backproject_depth(depth, inv_K)
    pix_coords = project3d(cam_points, K, T)
    print(pix_coords.shape)
    pix_coords_dbg = pix_coords.view(batch_size, -1, 2).permute(0, 2, 1)
    print(pix_coords_dbg[0,:,:600])
    print(pix_coords_dbg[0,:,::640])


