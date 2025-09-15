from resnet.resnet_encoder import ResnetEncoder
from resnet.pose_decoder import PoseDecoder

import torch.nn as nn
import os
import torch

 
from copy import deepcopy
import os
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial
import reloc3r.utils.path_to_croco
from reloc3r.patch_embed import ManyAR_PatchEmbed
from models.pos_embed import RoPE2D 
from models.blocks import Block, DecoderBlock
from reloc3r.pose_head import PoseHead as PoseHead_reloc3r
from networks.models.endofast3r_posehead import PoseHead as PoseHead_endofast3r
from reloc3r.utils.misc import freeze_all_params, transpose_to_landscape
from pdb import set_trace as bb


from resnet.resnet_encoder import ResnetEncoder


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class ResNetFeatureEmbed(nn.Module):
    """
    ResNet-based feature extractor that replaces patch embedding.
    Extracts features at H/32, W/32 resolution with 512 dimensions.
    """
    
    def __init__(self, img_size=(256, 320), enc_embed_dim=512, norm_layer=None):
        super().__init__()
        
        self.img_size = img_size
        self.enc_embed_dim = enc_embed_dim
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # ResNet16 backbone (ResNet18 with modified first layer)
        self.resnet = self._make_resnet16()
        
        # Project ResNet features to encoder dimension
        self.feature_proj = nn.Conv2d(512, enc_embed_dim, kernel_size=1)
        
        # Calculate the output feature map size
        # For input (256, 320), ResNet with stride 32 gives (8, 10)
        self.feature_h = img_size[0] // 32  # 8
        self.feature_w = img_size[1] // 32  # 10
        
    def _make_resnet16(self):
        """Create ResNet16 (modified ResNet18)"""
        import torchvision.models as models
        
        # Load ResNet18 and modify the first layer for our input size
        resnet = models.resnet18(pretrained=True)
        
        # Modify the first conv layer to handle our input size better
        # Original: conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # We keep the same structure but ensure it works with our input
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final classification layers
        resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        return resnet
    
    def forward(self, img, true_shape):
        """
        Args:
            img: Input image tensor (B, C, H, W)
            true_shape: True shape tensor (B, 2) - not used in this implementation
        Returns:
            x: Feature tokens (B, N, C) where N = H*W
            pos: Position tokens (B, N, 2)
        """
        B, C, H, W = img.shape
        
        # Ensure input is the expected size
        if (H, W) != self.img_size:
            img = F.interpolate(img, size=self.img_size, mode='bilinear', align_corners=False)
            H, W = self.img_size
        
        # Extract features using ResNet
        features = self.resnet(img)  # (B, 512, H/32, W/32)
        
        # Project to encoder dimension
        features = self.feature_proj(features)  # (B, enc_embed_dim, H/32, W/32)
        
        # Reshape to tokens: (B, C, H, W) -> (B, H*W, C)
        B, C, feat_h, feat_w = features.shape
        x = features.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, feat_h * feat_w, C)  # (B, H*W, C)
        
        # Generate position tokens - CRITICAL: must be (B, N, 2) for RoPE
        y_coords = torch.arange(feat_h, device=x.device).unsqueeze(0).repeat(feat_w, 1).T  # (H, W)
        x_coords = torch.arange(feat_w, device=x.device).unsqueeze(0).repeat(feat_h, 1)   # (H, W)
        
        # Stack to get (H, W, 2) then flatten to (H*W, 2)
        pos = torch.stack([y_coords, x_coords], dim=-1)  # (H, W, 2)
        pos = pos.view(-1, 2)  # (H*W, 2)
        
        # Repeat for batch: (B, H*W, 2)
        pos = pos.unsqueeze(0).repeat(B, 1, 1)
        
        return x, pos

class ResNetReloc3rRelpose(nn.Module):
    """
    Modified Reloc3rRelpose that uses ResNet feature extraction instead of patch embedding.
    """
    
    def __init__(self,
                 img_size=(256, 320),      # input image size (H, W)
                 enc_embed_dim=512,        # encoder feature dimension
                 enc_depth=24,             # encoder depth 
                 enc_num_heads=16,         # encoder number of heads in the transformer block 
                 dec_embed_dim=768,        # decoder feature dimension 
                 dec_depth=12,             # decoder depth 
                 dec_num_heads=12,         # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,     # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',      # positional embedding (either cosine or RoPE100)
                 **kwargs):
        super().__init__()
        
        # ResNet feature extractor instead of patch embedding
        self.feature_embed = ResNetFeatureEmbed(img_size, enc_embed_dim, norm_layer)
        
        # Positional embedding setup
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        
        # Import RoPE2D
        try:
            from models.pos_embed import RoPE2D
        except ImportError:
            from croco.models.pos_embed import RoPE2D
        
        if RoPE2D is None: 
            raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        
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
        
        # Pose regression head
        # self.pose_head = PoseHead_reloc3r(net=self)
        print('pose_head param:')
        print(dec_embed_dim)
        self.pose_head = PoseHead_endofast3r(net=None, 
                                             patch_size=32, #due to res18 
                                             dec_embed_dim=dec_embed_dim)    
        self.head = transpose_to_landscape(self.pose_head, activate=True)
        
        self.initialize_weights()
        
    
    def _encode_image(self, image, true_shape):
        """Encode image using ResNet feature extractor"""
        x, pos = self.feature_embed(image, true_shape)
        
        # Apply transformer encoder
        for blk in self.enc_blocks:
            x = blk(x, pos)
        
        x = self.enc_norm(x)
        return x, pos, None
    
    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        """Encode image pairs"""
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
        """Encoder for image pairs"""
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)
        
        return (shape1, shape2), (feat1, feat2), (pos1, pos2)
    
    def _decoder(self, f1, pos1, f2, pos2, mask1=None, mask2=None):
        """Decoder with cross-attention - FIXED VERSION"""
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk in self.dec_blocks:
            # img1 side: blk(x, y, xpos, ypos) where x=f1, y=f2, xpos=pos1, ypos=pos2
            f1, _ = blk(*final_output[-1][::+1], pos1, pos2)
            # img2 side: blk(x, y, xpos, ypos) where x=f2, y=f1, xpos=pos2, ypos=pos1
            f2, _ = blk(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        # Convert zip result to list
        dec1, dec2 = zip(*final_output)
        return list(dec1), list(dec2)
    
    # def _downstream_head(self, decout, img_shape):
    #     """Downstream head wrapper"""
    #     B, S, D = decout[-1].shape
    #     return self.head(decout, img_shape)
    
    def forward(self, view1, view2):
        """Forward pass"""
        # Encode
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encoder(view1, view2)
        
        # Decode
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        
        # Pose regression
        pose1 = self.head(dec1, shape1)
        pose2 = self.head(dec2, shape2)
        
        return pose1, pose2
    
    def initialize_weights(self):
        """Initialize weights"""
        # Initialize ResNet feature projection
        nn.init.xavier_uniform_(self.feature_embed.feature_proj.weight)
        if self.feature_embed.feature_proj.bias is not None:
            nn.init.constant_(self.feature_embed.feature_proj.bias, 0)
        
        # Initialize other layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 

if __name__ == "__main__":
    # init smaller transformer
    model = ResNetReloc3rRelpose(enc_embed_dim=512,
                                 enc_depth=6,
                                 enc_num_heads=8,
                                 dec_embed_dim=384,
                                 dec_depth=4,
                                 dec_num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 norm_im2_in_dec=True,
                                 pos_embed='RoPE100')
                                 
    img = torch.randn(1, 3, 256, 320)
    view1 = {'img': img}
    view2 = {'img': img}
    pose1, pose2 = model(view1, view2)
    print(pose1['pose'].shape)
    print(pose2['pose'].shape)
    print(pose1['pose'])
    print(pose2['pose'])
    # print the storage in mb of the model
    print(sum(p.numel() for p in model.parameters()) / 1024 / 1024)