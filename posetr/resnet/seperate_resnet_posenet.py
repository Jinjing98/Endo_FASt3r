from resnet_encoder import ResnetEncoder
from pose_decoder import PoseDecoder
import torch.nn as nn
import os
import torch

 

if __name__ == "__main__":
        pose_encoder_path = os.path.join('/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/', "af_sfmlearner_weights", "pose_encoder.pth")
        pose_decoder_path = os.path.join('/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/', "af_sfmlearner_weights", "pose.pth")
        
        num_layers = 18
        num_pose_frames = 2
        weights_init = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pose_encoder = ResnetEncoder(
            num_layers,
            weights_init == "pretrained",
            num_input_images=num_pose_frames).to(device)
        assert os.path.exists(pose_encoder_path), f"pose_encoder_path {pose_encoder_path} does not exist"
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))
        pose_encoder.to(device)

        pose_head = PoseDecoder(
            pose_encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2).to(device)
        assert os.path.exists(pose_decoder_path), f"pose_decoder_path {pose_decoder_path} does not exist"
        pose_head.load_state_dict(torch.load(pose_decoder_path))
        print('loaded separate_resnet pose model...')



        img_i = torch.randn(1, 3, 256, 320).to(device)
        img_0 = torch.randn(1, 3, 256, 320).to(device)

        # put i in the front
        pose_inputs = [pose_encoder(torch.cat([img_i, img_0], 1))]
        for i in range(len(pose_inputs[0])):
            print('pose_inputs[0][{}].shape'.format(i), pose_inputs[0][i].shape)


        axisangle, translation = pose_head(pose_inputs)

        print(axisangle.shape)
        print(translation.shape)

        axisangle_02i = axisangle[:, 0]
        translation_02i = translation[:, 0]

        # axisangle_i20 = axisangle[:, 1]
        # translation_i20 = translation[:, 1]

        print(axisangle_02i.shape)
        print(translation_02i.shape)

