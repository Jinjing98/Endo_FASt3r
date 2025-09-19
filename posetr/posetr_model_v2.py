import torch
import torch.nn as nn
from transformers import ViTModel
import os
from functools import partial
import torchvision.models as models


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded = torch.hub.load_state_dict_from_url(models.ResNet18_Weights.IMAGENET1K_V1.url)
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


import reloc3r.utils.path_to_croco
from reloc3r.patch_embed import ManyAR_PatchEmbed
from models.pos_embed import RoPE2D 
from models.blocks import Block, DecoderBlock
from models.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
from models.pos_embed import interpolate_pos_embed
from models.head_downstream import PixelwiseTaskWithDPT

class CroCoV2FeatureExtractor(nn.Module):
    def __init__(self,
                 img_size=320,#512,          # input image size---resize the raw img accordingly
                 patch_size=16,         # patch_size 
                 enc_embed_dim=768,#1024,    # encoder feature dimension
                 enc_depth=12, #24,          # encoder depth 
                 enc_num_heads=12, #16,      # encoder number of heads in the transformer block 
                 dec_embed_dim=512, #768,     # decoder feature dimension 
                 dec_depth=8,#12,          # decoder depth 
                 dec_num_heads=16,#12,      # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',   # positional embedding (either cosine or RoPE100)
                ):   
        super(CroCoV2FeatureExtractor, self).__init__()

        # patchify and positional embedding
        self.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)

        # ViT encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        # ViT decoder
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from encoder to decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)

        # # pose regression head
        # self.pose_head = PoseHead(net=self)
        # self.head = transpose_to_landscape(self.pose_head, activate=True)

        # self.initialize_weights() 

    # def initialize_weights(self):
    #     # patch embed 
    #     self.patch_embed._init_weights()
    #     # linears and layer norms
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # def freeze_encoder(self):
    #     freeze_all_params([self.patch_embed, self.enc_blocks])

    def load_state_dict(self, ckpt_path, **kw):
        """Load state dict from checkpoint with key mapping."""
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path)
        
        # Extract state dict
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # Map dec_blocks2 -> dec_blocks keys
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('dec_blocks2'):
                # croco model does not have this....it is already the cross view decoder...
                new_key = key.replace('dec_blocks2', 'dec_blocks')
                mapped_state_dict[new_key] = value
                print(f"Mapped {key} -> {new_key}")
            else:
                mapped_state_dict[key] = value
        
        # Load with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = super().load_state_dict(mapped_state_dict, strict=False)
        
        # Report missing/unexpected keys
        if missing_keys:
            assert 0, f'{missing_keys}'
            missing_keys_sim = {k.split('.')[0].split('[')[0] for k in missing_keys}
            print(f'Warning: Missing keys: {missing_keys_sim}')
        if unexpected_keys:
            unexpected_keys_sim = {k.split('.')[0].split('[')[0] for k in unexpected_keys}
            print(f'Warning: Unexpected keys: {unexpected_keys_sim}')
        
        return missing_keys, unexpected_keys

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encoder(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk in self.dec_blocks:
            # img1 side
            f1, _ = blk(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    # def _downstream_head(self, decout, img_shape):
    #     B, S, D = decout[-1].shape
    #     return self.head(decout, img_shape)

    def forward(self, view1, view2):
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encoder(view1, view2)  # B,S,D
        # print(f"feat1.shape: {feat1.shape}")
        # print(f"feat2.shape: {feat2.shape}")
        # print(f"pos1.shape: {pos1.shape}")
        # print(f"pos2.shape: {pos2.shape}")


        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        return feat1, feat2, pos1, pos2, dec1, dec2


        with torch.cuda.amp.autocast(enabled=False):

            pose2 = self._downstream_head([tok.float() for tok in dec2], shape2)  # relative camera pose from 2 to 1. 
            
        return None, pose2 # try to be consistent with original reloc3r
        # return pose2, dec2[-1]


class CroCoV2FeatureExtractor_old(nn.Module):
    """
    CroCo v2 encoder using ViT-Base architecture with CroCo pretrained weights.
    """
    def __init__(self, img_size=(256, 320), patch_size=16, embed_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Load ViT-Base as base architecture
        self.vit_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Load CroCo encoder weights
        checkpoint_path = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/croco_ckpts'
        self._load_croco_encoder_weights(checkpoint_path)
        
        # Freeze encoder parameters
        self._freeze_encoder()
        
        # Projection layer to match desired embed_dim
        if embed_dim != 768:  # ViT-Base has 768 dimensions
            assert 0, f'embed_dim != 768: {embed_dim}'
            self.projection = nn.Linear(768, embed_dim)
        else:
            self.projection = nn.Identity()
    
    def _load_croco_encoder_weights(self, checkpoint_path):
        """Load CroCo encoder weights into ViT-Base"""
        # Find checkpoint file
        checkpoint_files = []
        for file in os.listdir(checkpoint_path):
            if file.endswith(('.pth', '.pt', '.ckpt')):
                checkpoint_files.append(file)
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
        
        checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # # DEBUG: Print CroCo keys
        # print("CroCo keys containing 'enc_norm':")
        # for key in state_dict.keys():
        #     if 'enc_norm' in key:
        #         print(f"  {key}: {state_dict[key].shape}")
        
        # Map CroCo encoder weights to ViT-Base weights
        vit_state_dict = self._map_croco_to_vit_weights(state_dict)
        
        # # DEBUG: Print mapped keys
        # print("Mapped ViT keys:")
        # for key in vit_state_dict.keys():
        #     if 'layernorm' in key:
        #         print(f"  {key}: {vit_state_dict[key].shape}")
        
        # Load weights into ViT-Base
        missing_keys, unexpected_keys = self.vit_encoder.load_state_dict(vit_state_dict, strict=False)
        
        print(f"Loaded CroCo encoder weights from {checkpoint_file}")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    def _map_croco_to_vit_weights(self, croco_state_dict):
        """Map CroCo encoder weights to ViT-Base weight names"""
        vit_state_dict = {}
        
        # Map patch embedding
        if 'patch_embed.proj.weight' in croco_state_dict:
            vit_state_dict['embeddings.patch_embeddings.projection.weight'] = croco_state_dict['patch_embed.proj.weight']
        if 'patch_embed.proj.bias' in croco_state_dict:
            vit_state_dict['embeddings.patch_embeddings.projection.bias'] = croco_state_dict['patch_embed.proj.bias']
        
        # Map positional embedding - FIXED
        if 'enc_pos_embed' in croco_state_dict:
            # CroCo: (196, 768) -> ViT: (1, 197, 768)
            pos_embed = croco_state_dict['enc_pos_embed']  # Shape: (196, 768)
            print(f"croco pos_embed.shape: {pos_embed.shape}")
            
            # Add CLS token positional embedding (zeros)
            cls_pos_embed = torch.zeros(1, 768, device=pos_embed.device)  # CLS token position
            pos_embed_with_cls = torch.cat([cls_pos_embed, pos_embed], dim=0)  # Shape: (197, 768)
            pos_embed_with_cls = pos_embed_with_cls.unsqueeze(0)  # Shape: (1, 197, 768)
            
            vit_state_dict['embeddings.position_embeddings'] = pos_embed_with_cls
        
        # Map encoder blocks (rest remains the same)
        for i in range(12):  # ViT-Base has 12 layers
            croco_prefix = f'enc_blocks.{i}'
            vit_prefix = f'encoder.layer.{i}'
            
            # Attention weights
            if f'{croco_prefix}.attn.qkv.weight' in croco_state_dict:
                qkv_weight = croco_state_dict[f'{croco_prefix}.attn.qkv.weight']  # Shape: (768*3, 768)
                q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
                vit_state_dict[f'{vit_prefix}.attention.attention.query.weight'] = q_weight
                vit_state_dict[f'{vit_prefix}.attention.attention.key.weight'] = k_weight
                vit_state_dict[f'{vit_prefix}.attention.attention.value.weight'] = v_weight
            
            if f'{croco_prefix}.attn.qkv.bias' in croco_state_dict:
                qkv_bias = croco_state_dict[f'{croco_prefix}.attn.qkv.bias']  # Shape: (768*3,)
                q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
                vit_state_dict[f'{vit_prefix}.attention.attention.query.bias'] = q_bias
                vit_state_dict[f'{vit_prefix}.attention.attention.key.bias'] = k_bias
                vit_state_dict[f'{vit_prefix}.attention.attention.value.bias'] = v_bias
            
            # Attention output projection
            if f'{croco_prefix}.attn.proj.weight' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.attention.output.dense.weight'] = croco_state_dict[f'{croco_prefix}.attn.proj.weight']
            if f'{croco_prefix}.attn.proj.bias' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.attention.output.dense.bias'] = croco_state_dict[f'{croco_prefix}.attn.proj.bias']
            
            # Layer norm 1
            if f'{croco_prefix}.norm1.weight' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.layernorm_before.weight'] = croco_state_dict[f'{croco_prefix}.norm1.weight']
            if f'{croco_prefix}.norm1.bias' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.layernorm_before.bias'] = croco_state_dict[f'{croco_prefix}.norm1.bias']
            
            # MLP weights
            if f'{croco_prefix}.mlp.fc1.weight' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.intermediate.dense.weight'] = croco_state_dict[f'{croco_prefix}.mlp.fc1.weight']
            if f'{croco_prefix}.mlp.fc1.bias' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.intermediate.dense.bias'] = croco_state_dict[f'{croco_prefix}.mlp.fc1.bias']
            
            if f'{croco_prefix}.mlp.fc2.weight' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.output.dense.weight'] = croco_state_dict[f'{croco_prefix}.mlp.fc2.weight']
            if f'{croco_prefix}.mlp.fc2.bias' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.output.dense.bias'] = croco_state_dict[f'{croco_prefix}.mlp.fc2.bias']
            
            # Layer norm 2
            if f'{croco_prefix}.norm2.weight' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.layernorm_after.weight'] = croco_state_dict[f'{croco_prefix}.norm2.weight']
            if f'{croco_prefix}.norm2.bias' in croco_state_dict:
                vit_state_dict[f'{vit_prefix}.layernorm_after.bias'] = croco_state_dict[f'{croco_prefix}.norm2.bias']
        
        # Skip encoder layer norm mapping - let ViT use its own
        # if 'enc_norm.weight' in croco_state_dict:
        #     vit_state_dict['encoder.layernorm.weight'] = croco_state_dict['enc_norm.weight']
        # if 'enc_norm.bias' in croco_state_dict:
        #     vit_state_dict['encoder.layernorm.bias'] = croco_state_dict['enc_norm.bias']
        
        return vit_state_dict
    
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        self.vit_encoder.eval()
    
    def forward(self, x, true_shape=None):
        """
        Args:
            x: Input image (B, 3, H, W)
            true_shape: True shape tensor (B, 2) - not used
        Returns:
            features: Encoder features (B, N+1, embed_dim)
        """
        # Resize to 224x224
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Forward pass through ViT encoder
        with torch.no_grad():
            outputs = self.vit_encoder(x)
            features = outputs.last_hidden_state  # (B, N+1, 768) - includes CLS token
        
        # Project to desired embed_dim
        features = self.projection(features)  # (B, N+1, embed_dim)
        
        return features

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet18 feature extractor for visual embedding extraction.
    """
    def __init__(self, img_size=(256, 320), embed_dim=384):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # # Load pretrained ResNet18
        # self.resnet = models.resnet18(pretrained=True)
        # # Remove the final classification layers
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc
        
        # # Load pretrained ResNet18--support pair input!
        self.resnet = resnet_multiimage_input(num_layers=18, 
                                              pretrained=True, 
                                              num_input_images=2)


        # Get ResNet output dimension (512 for ResNet18)
        resnet_dim = 512
        
        self.projection = nn.Identity()
        
        # Calculate spatial dimensions after ResNet
        # ResNet18 reduces spatial size by 32x
        self.spatial_h = img_size[0] // 32
        self.spatial_w = img_size[1] // 32
        self.num_patches = self.spatial_h * self.spatial_w
    
    def forward(self, x, true_shape=None):
        """
        Args:
            x: Input image (B, 3, H, W)
            true_shape: True shape tensor (B, 2) - not used with ResNet
        """
        # Resize image to expected size if needed
        if x.shape[-2:] != self.img_size:
            assert 0
            x = torch.nn.functional.interpolate(
                x, size=self.img_size, mode='bilinear', align_corners=False
            )

        # Forward pass through ResNet
        # features = self.resnet(x)  # (B, 512, H/32, W/32)

        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        self.features.append(self.resnet.relu(x))
        self.features.append(self.resnet.layer1(self.resnet.maxpool(self.features[-1])))
        self.features.append(self.resnet.layer2(self.features[-1]))
        self.features.append(self.resnet.layer3(self.features[-1]))
        self.features.append(self.resnet.layer4(self.features[-1]))
        features = self.features[-1]

        print('features.shape:')
        print(features.shape)
        
        return features  # (B, num_patches+1, embed_dim)

class ViTFeatureExtractor(nn.Module):
    """
    Hugging Face pretrained ViT-Small for visual embedding extraction.
    """
    def __init__(self, img_size=(256, 320), patch_size=16, embed_dim=384, 
                 depth=6, num_heads=6, mlp_ratio=4, norm_layer=None):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Load pretrained ViT-Small (correct model name)
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
        
        # Projection layer to match desired embed_dim
        if embed_dim != 384:  # ViT-Small has 384 dimensions
            assert 0
            self.projection = nn.Linear(384, embed_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x, true_shape=None):
        """
        Args:
            x: Input image (B, 3, H, W)
            true_shape: True shape tensor (B, 2) - not used with HF ViT
        """
        # Resize image to 224x224 (ViT input size)
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Forward pass through pretrained ViT
        outputs = self.vit(x)
        
        # Get the last hidden states (B, num_patches+1, 384)
        hidden_states = outputs.last_hidden_state
        
        # Project to desired embed_dim
        hidden_states = self.projection(hidden_states)

        # print(f"hidden_states.shape: {hidden_states.shape}")
        
        return hidden_states  # (B, num_patches+1, embed_dim)

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for communication between two images.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Features from image 1 (B, N, D)
            x2: Features from image 2 (B, N, D)
        Returns:
            Enhanced features for both images
        """
        # Cross-attention: x1 attends to x2
        x1_attended, _ = self.cross_attn(x1, x2, x2)
        x1 = self.norm1(x1 + x1_attended)
        
        # Cross-attention: x2 attends to x1
        x2_attended, _ = self.cross_attn(x2, x1, x1)
        x2 = self.norm2(x2 + x2_attended)
        
        # MLP for both
        x1 = x1 + self.mlp(x1)
        x2 = x2 + self.mlp(x2)
        
        return x1, x2

class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for enhancing features within each image.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, N, D)
        Returns:
            Enhanced features
        """
        # Self-attention
        x_attended, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x_attended)
        
        # MLP
        x = x + self.mlp(x)
        
        return x

class RelativePoseHead(nn.Module):
    """
    Pose regression head for relative pose estimation.
    Outputs 3D translation and 3D angle-axis rotation.
    """
    def __init__(self, embed_dim, num_patches, patch_size=16, img_size_H_W = (256, 320), process_with_2D_conv = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.img_size_H_W = img_size_H_W
        
        # Global average pooling + MLP
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.process_with_2D_conv = process_with_2D_conv
        self.num_frames_to_predict_for = 2
        
        if self.process_with_2D_conv:

            assert 0

            self.conv_layers = nn.Sequential(
                nn.Conv2d(embed_dim, 256, 1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(256, 6*self.num_frames_to_predict_for, 1)  # 6 for pose (3 trans + 3 rot)
            )
        else:
            # assert 0
            # MLP layers
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, 6*self.num_frames_to_predict_for)  # 3 for translation + 3 for angle-axis
            )
        
        # Scaling factors (similar to pose_decoder.py)
        self.trans_scale = 0.001
        self.rot_scale = 0.001

    
    def forward(self, x):
        """
        Args:
            x: Input features (B, N, D) where N = num_patches + 1 (including cls token)
        Returns:
            pose: Relative pose (B, 4, 4)
        """
 
        B = x.shape[0]
        
        # use feat for pose regression
        feat = x[:, 1:, :] # B N 512  # rm the cls_token from ViT

        if self.process_with_2D_conv:
            # Reshape to spatial h w format
            H = self.img_size_H_W[0] // self.patch_size
            W = self.img_size_H_W[1] // self.patch_size
            feat = feat.transpose(1, 2).view(B, self.embed_dim, H, W)
            pose_6d = self.conv_layers(feat)
            pose_6d = pose_6d.mean(dim=3).mean(dim=2)
        else:
            # process with mlp
            # mean
            feat = feat.mean(dim=1)

            pose_6d = self.mlp(feat)  # (B, N, 6)


        # Global average pooling
        # out = out.mean(3).mean(2)  # (B, 6)
        out = 0.001*pose_6d.view(-1, self.num_frames_to_predict_for, 1, 6)
        # Split into translation and rotation
        translation = out[:, 0, :, :3]  # (B, 1, 3)
        angle_axis = out[:, 0, :, 3:]  # (B, 1, 3)

        # this may be critical due to grad propergation!!!
        from layers import transformation_from_parameters
        pose = transformation_from_parameters(angle_axis, translation)

        pose_dict = {}
        pose_dict['pose'] = pose



        # # Split into translation and rotation
        # translation = pose_6d[:, :3] * self.trans_scale  # (B, 3)
        # angle_axis = pose_6d[:, 3:] * self.rot_scale     # (B, 3)
        
        # # Convert angle-axis to rotation matrix
        # rotation_matrix = self.angle_axis_to_rotation_matrix(angle_axis)
        # # Create 4x4 transformation matrix
        # pose = torch.eye(4, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        # pose[:, :3, :3] = rotation_matrix
        # pose[:, :3, 3] = translation

        # pose_dict = {}
        # pose_dict['pose'] = pose
        
        return pose_dict
    
    def angle_axis_to_rotation_matrix(self, angle_axis):
        """
        Convert angle-axis representation to rotation matrix.
        Args:
            angle_axis: (B, 3) angle-axis representation
        Returns:
            rotation_matrix: (B, 3, 3) rotation matrix
        """
        B = angle_axis.shape[0]
        
        # Compute angle
        angle = torch.norm(angle_axis, dim=1, keepdim=True)  # (B, 1)
        
        # Avoid division by zero
        angle = torch.clamp(angle, min=1e-8)
        
        # Normalize axis
        axis = angle_axis / angle  # (B, 3)
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Cross product matrix
        K = torch.zeros(B, 3, 3, device=angle_axis.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=angle_axis.device).unsqueeze(0).repeat(B, 1, 1)
        K_squared = torch.bmm(K, K)
        
        rotation_matrix = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * K_squared
        
        return rotation_matrix

class ConcatenatedPoseHead(nn.Module):
    """
    Pose regression head that concatenates features from both images.
    Similar to ResNet pose decoder implementation.
    """
    def __init__(self, embed_dim, num_patches, is_vit_feat = True, patch_size=32, img_size=(256, 320), res_feat_concat_later = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.is_vit_feat = is_vit_feat
        self.patch_size_res18 = patch_size
        self.img_size = img_size
        
        self.res_feat_concat_later = res_feat_concat_later
        if self.res_feat_concat_later:
            assert 0
            self.conv_layers = nn.Sequential(
                nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1),  # 2*256 for concatenated features
                nn.ReLU(),
                nn.Conv2d(embed_dim, 256, 3, 1, 1),  # 2*256 for concatenated features
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(256, 6, 1)  # 6 for pose (3 trans + 3 rot)
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(embed_dim, 256, 1),  # 2*256 for concatenated features
                nn.ReLU(),
                # nn.Conv2d(256, 256, 3, 1, 1),
                # nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(256, 6*2, 1)  # 6 for pose (3 trans + 3 rot)
            )

        # Scaling factors
        # self.trans_scale = 0.001
        self.pose_scale = 0.001
        self.num_frames_to_predict_for = 2
    
    def forward(self, feat1, feat2, cat_features = None):
        """
        Args:
            feat1: Features from image 1 (B, N, D)
            feat2: Features from image 2 (B, N, D)
        Returns:
            pose: Relative pose (B, 4, 4)
        """
        
        if self.res_feat_concat_later:
            assert 0
            B = feat1.shape[0]
            assert cat_features is None, "cat_features is required for res_feat_concat_later"

            # Remove cls token and reshape to spatial format
            feat1_spatial = feat1[:, 1:, :]  # Remove cls token (B, N, D)
            feat2_spatial = feat2[:, 1:, :]  # Remove cls token (B, N, D)
            
            # Reshape to spatial format
            if self.is_vit_feat:
                H = W = int((feat1_spatial.shape[1]) ** 0.5)
            else:
                H = self.img_size[0] // self.patch_size_res18
                W = self.img_size[1] // self.patch_size_res18
            print(f"H: {H}, W: {W}")
            print(f"feat1_spatial.shape: {feat1_spatial.shape}")

            feat1_spatial = feat1_spatial.transpose(1, 2).view(B, self.embed_dim, H, W)
            feat2_spatial = feat2_spatial.transpose(1, 2).view(B, self.embed_dim, H, W)
            
            # Concatenate features
            cat_features = torch.cat([feat1_spatial, feat2_spatial], dim=1)  # (B, 2*256, H, W)
            
            # Apply remaining layers
            out = self.conv_layers(cat_features)  # Rest of the layers
        else:
            B = cat_features.shape[0]
            assert feat1 is None, "feat1_spatial is not allowed"
            assert feat2 is None, "feat2_spatial is not allowed"
            assert cat_features is not None, "cat_features is required"
            out = self.conv_layers(cat_features)  # Rest of the layers

        # Global average pooling
        out = out.mean(3).mean(2)  # (B, 6)
        
        out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)


        # Split into translation and rotation
        translation = out[:, 0, :, :3]  # (B, 1, 3)
        angle_axis = out[:, 0, :, 3:]  # (B, 1, 3)
        
        # # Convert angle-axis to rotation matrix
        # rotation_matrix = self.angle_axis_to_rotation_matrix(angle_axis)
        
        # # Create 4x4 transformation matrix
        # pose = torch.eye(4, device=rotation_matrix.device).unsqueeze(0).repeat(B, 1, 1)
        # pose[:, :3, :3] = rotation_matrix
        # pose[:, :3, 3] = translation

        # print('angle_axis.shape:')
        # print(angle_axis.shape)
        # print('translation.shape:')
        # print(translation.shape)

        # this may be critical due to grad propergation!!!
        pose = self.transformation_from_parameters_v2(angle_axis, translation)
        
        pose_dict = {}
        pose_dict['pose'] = pose
        
        return pose_dict

    def transformation_from_parameters_v2(self, axisangle, translation, invert=False):
        """Convert the network's (axisangle, translation) output into a 4x4 matrix
        """
        assert axisangle.dim() == 3 and translation.dim() == 3, "axisangle and translation must be 2D"
        assert axisangle.shape[1] == 1 and translation.shape[1] == 1, "axisangle and translation must have 1 frame"
        from layers import transformation_from_parameters
        return transformation_from_parameters(axisangle, translation, invert)

    def angle_axis_to_rotation_matrix(self, angle_axis):
        """
        Convert angle-axis representation to rotation matrix.
        """
        B = angle_axis.shape[0]
        
        # Compute angle
        angle = torch.norm(angle_axis, dim=1, keepdim=True)  # (B, 1)
        
        # Avoid division by zero
        angle = torch.clamp(angle, min=1e-8)
        
        # Normalize axis
        axis = angle_axis / angle  # (B, 3)
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Cross product matrix
        K = torch.zeros(B, 3, 3, device=angle_axis.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=angle_axis.device).unsqueeze(0).repeat(B, 1, 1)
        K_squared = torch.bmm(K, K)
        
        rotation_matrix = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * K_squared
        
        return rotation_matrix

class PoseTransformerV2(nn.Module):
    """
    Relative pose regression network with pretrained ViT-Small, CroCo v2 Small, or ResNet18.
    """
    def __init__(self,
                 img_size=(256, 320),
                 patch_size=16,
                 embed_dim=384,  # Match ViT-Small dimension
                 attention_depth=4,
                 attention_num_heads=6,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 skip_sa_ca=False,  # Skip self and cross attention
                 use_vit=True,      # Use ViT-based encoder (ViT or CroCo)
                 croco_vit=False):  # If use_vit=True, choose CroCo v2 Small over ViT Small
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.attention_depth = attention_depth
        self.skip_sa_ca = skip_sa_ca
        self.use_vit = use_vit
        self.croco_vit = croco_vit
        
        # Feature extractor selection
        if use_vit:
            if croco_vit:
                # Use CroCo v2 encoder (ViT-Base + CroCo weights)
                # self.feature_extractor = CroCoV2FeatureExtractor(
                #     img_size=img_size,
                #     patch_size=patch_size,
                #     embed_dim=768, #vit base
                # )
                self.feature_extractor = CroCoV2FeatureExtractor()
                ckpt_path = '/mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/croco_ckpts/CroCo_V2_ViTBase_SmallDecoder.pth'
                self.feature_extractor.load_state_dict(ckpt_path)
                print('loaded pretrained croco v2 weights...')

                num_patches = int(256/16) * int(320/16)

                self.proj_dec_to_head = nn.Linear(512, embed_dim)
            
            else:
                # Use ViT Small feature extractor
                self.feature_extractor = ViTFeatureExtractor(
                    img_size=img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim
                )
                num_patches = (224 // patch_size) ** 2  # ViT uses 224x224 input
        else:
            # Use ResNet18 feature extractor---pair image as input
            self.feature_extractor = ResNetFeatureExtractor(
                img_size=img_size,
                embed_dim=embed_dim
            )
            # self.feature_extractor = resnet_multiimage_input(num_layers=18, 
            #                                                        pretrained=True, 
            #                                                        num_input_images=2)
            num_patches = self.feature_extractor.num_patches
        
        if not skip_sa_ca:
            if not croco_vit:
                # Alternating self and cross attention blocks
                self.attention_blocks = nn.ModuleList()
                for i in range(attention_depth):
                    if i % 2 == 0:
                        # Self-attention blocks
                        self.attention_blocks.append(
                            SelfAttentionBlock(embed_dim, attention_num_heads, mlp_ratio, norm_layer)
                        )
                    else:
                        # Cross-attention blocks
                        self.attention_blocks.append(
                            CrossAttentionBlock(embed_dim, attention_num_heads, mlp_ratio, norm_layer)
                        )
                
            # Pose regression heads (separate for each image)
            # self.pose_head_1to2 = RelativePoseHead(embed_dim, num_patches, 
            self.pose_head = RelativePoseHead(embed_dim, num_patches, 
                                                   patch_size = patch_size if self.use_vit else 32, 
                                                   img_size_H_W = (224, 224) if self.use_vit else img_size, 
                                                   process_with_2D_conv = False,
                                                   )
 
        else:
            # Concatenated pose head (single relative pose)
            if self.use_vit:

                res_feat_concat_later=False
                
                self.pose_head = ConcatenatedPoseHead(embed_dim, num_patches, 
                                                      is_vit_feat=True,
                                                      patch_size = 16,
                                                      img_size = (224, 224),
                                                      res_feat_concat_later=res_feat_concat_later,
                                                      )
                if not res_feat_concat_later:
                    self.proj_back_to_dim = nn.Linear(2*embed_dim, embed_dim)
                
            else:
                # take the fused res feat as input(already commnicated)
                self.pose_head = ConcatenatedPoseHead(embed_dim, num_patches, 
                                                      is_vit_feat=False, 
                                                      patch_size=32, 
                                                      img_size=img_size,
                                                      res_feat_concat_later=False,
                                                      )
        
        # # Initialize weights
        # self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights for non-pretrained parts"""
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, view1, view2):
        """
        Forward pass for relative pose regression.
        """
        # Extract images
        img1 = view1['img']
        img2 = view2['img']
 
        # Feature extraction
        if self.use_vit:
            if self.croco_vit:
                # resize the image from 256,320 to croco format with img_size=320

                # encoder of croco v2
                feat1, feat2, pos1, pos2, dec1, dec2 = self.feature_extractor(view1, view2)
                # print(f"feat1.shape: {feat1.shape}")
                # print(f"feat2.shape: {feat2.shape}")
                # print(f"pos1.shape: {pos1.shape}")
                # print(f"pos2.shape: {pos2.shape}")
                
                feat = dec1[-1]#2 320 512
                # print(f"feat.shape: {feat.shape}")
                feat = self.proj_dec_to_head(feat)
                pose = self.pose_head(feat)  # (B, 4, 4)


            else:
                feat1 = self.feature_extractor(img1)  # (B, N+1, D)
                feat2 = self.feature_extractor(img2)  # (B, N+1, D)
                print(f"feat1.shape: {feat1.shape}")
                print(f"feat2.shape: {feat2.shape}")

                if not self.skip_sa_ca:
                    # Alternating self and cross attention
                    for i, block in enumerate(self.attention_blocks):
                        if i % 2 == 0:
                            # Self-attention: enhance features within each image
                            feat1 = block(feat1)
                            feat2 = block(feat2)
                        else:
                            # Cross-attention: communication between images
                            feat1, feat2 = block(feat1, feat2)
                        print('////attn layer: %d' % i)
                        print(f"feat1.shape: {feat1.shape}")
                        print(f"feat2.shape: {feat2.shape}")
            
                # # Concatenate features and regress single relative pose
                # pose = self.pose_head(feat1, feat2, feat = None)  # (B, 4, 4)


                # also try to use the concatenated features
                feat = torch.cat([feat1, feat2], dim= -1) # B N+1 2*D
                # project back to D, apply gelu activation
                feat = nn.GELU()(self.proj_back_to_dim(feat))
                # remove the cls token and recover the spatial shape
                
                use_glo_token = True
                if use_glo_token:
                    feat = feat[:, :1, :]
                    B, N, D = feat.shape
                    feat = feat.transpose(1, 2)
                    feat = feat.view(B, D, 1, 1)
                else:

                    feat = feat[:, 1:, :]
                    B, N, D = feat.shape
                    H = W = int((N) ** 0.5)
                    feat = feat.transpose(1, 2).view(B, D, H, W)
                print(f"feat.shape: {feat.shape}")

                pose = self.pose_head(feat1 = None, feat2 = None, cat_features = feat)  # (B, 4, 4)

        else:
            # use resnet
            # early fusion!---quick debug
            imgs = torch.cat([img1, img2], dim=1)  # (B, 6, 256, 320)
            feat = self.feature_extractor(imgs)  # (B, 512, H/32, W/32)

            # Concatenate features and regress single relative pose
            pose = self.pose_head(feat1 = None, feat2 = None, cat_features = feat)  # (B, 4, 4)                


        
        # Return same pose for both directions (or inverse)
        return pose, pose

if __name__ == "__main__":
    # Test different configurations
    
    # # # 1. ViT Small + Attention
    # model_vit_attention = PoseTransformerV2(use_vit=True, croco_vit=False, skip_sa_ca=False)
    
    # # # 2. ViT Small + No Attention
    # model_vit_no_attention = PoseTransformerV2(use_vit=True, 
    #                                            croco_vit=False, 
    #                                            skip_sa_ca=True)
    
    # # 3. CroCo v2 Small + Attention
    model_croco_attention = PoseTransformerV2(use_vit=True, 
                                              croco_vit=True, 
                                              skip_sa_ca=False,
                                              embed_dim=512,
                                            #   embed_dim=1024,
                                              )
    
    # # 4. CroCo v2 Small + No Attention
    # model_croco_no_attention = PoseTransformerV2(use_vit=True, croco_vit=True, skip_sa_ca=True)
    
    # 5. ResNet18 + Attention
    # model_resnet_attention = PoseTransformerV2(use_vit=False, 
    #                                            skip_sa_ca=False,
    #                                            embed_dim=512,
    #                                            attention_num_heads=8, # we want to avoid any leanred projection layer by far....
    #                                            )
    
    # # # 6. ResNet18 + No Attention
    # model_resnet_no_attention = PoseTransformerV2(use_vit=False, 
    #                                               embed_dim=512, # we want to avoid any leanred projection layer by far....
    #                                               skip_sa_ca=True)
    
    # Test forward pass
    img1 = torch.randn(2, 3, 256, 320)
    img2 = torch.randn(2, 3, 256, 320)
    
    view1 = {'img': img1, 'true_shape': torch.tensor([[256, 320], [256, 320]])}
    view2 = {'img': img2, 'true_shape': torch.tensor([[256, 320], [256, 320]])}
    
    # Test all configurations
    configs = [
        # ("ViT Small + Attention", model_vit_attention),
        # ("ViT Small + No Attention", model_vit_no_attention),
        ("CroCo v2 Small + Attention", model_croco_attention),
        # ("CroCo v2 Small + No Attention", model_croco_no_attention),
        # ("ResNet18 + Attention", model_resnet_attention),
        # ("ResNet18 + No Attention", model_resnet_no_attention)
    ]
    
    for name, model in configs:
        if "CroCo v2 Small" in name:
            # not necessary to use: as 256,320 is already a good fit for croco v2, no need for resize
            from networks.utils.endofas3r_data_utils import prepare_images
            resized_img1= prepare_images(view1['img'], img1.device, size = 320)
            resized_img2= prepare_images(view2['img'], img2.device, size = 320)
            view1 = {'img':resized_img1}
            view2 = {'img':resized_img2}


        pose1, pose2 = model(view1, view2)
        params = sum(p.numel() for p in model.parameters()) / 1024 / 1024
        print(f"{name}: Pose shape {pose1['pose'].shape}, Parameters: {params:.2f}M")