# PCRNet Integration for Endo_FASt3r

import torch
import torch.nn as nn
import torch.nn.functional as F
from pcrnet import iPCRNet
from pcrnet import PointNet
 

def load_pcrnet_pose_head(emb_dims=1024):
    """
    Load PCRNet pose head
    
    Args:
        emb_dims: Embedding dimensions for PointNet
        max_iterations: Maximum iterations for iterative PCRNet
        
    Returns:
        PCRNetPoseEstimator instance
    """

 
    
    # Initialize PCRNet
    pointnet = PointNet(emb_dims=emb_dims, input_shape="bnc")
    return iPCRNet(feature_model=pointnet, droput=0.0, pooling='max')    

if __name__ == "__main__":
    # Test the implementation
    model = load_pcrnet_pose_head()
    
    # Test with dummy data
    B, H, W = 2, 32, 40
    sc_3d_matched = torch.randn(B, 3, H, W)
    intrinsics = torch.randn(B, 3, 3)
    
    result = model(sc_3d_matched, intrinsics, sample_level=3)
    print(f"Input shape: {sc_3d_matched.shape}")
    print(f"Output shape: {result[0].shape}")
    print(f"Pose matrix:\n{result[0][0]}")
