import torch
import torch.nn.functional as F
import kornia  # optional, for Sobel and Gaussian

eps = 1e-6

# def flow_to_magnitude(flow):            # flow: (B,2,H,W)
#     u = flow[:,0]
#     v = flow[:,1]
#     mag = torch.sqrt(u*u + v*v + 1e-12)
#     return mag  # (B,H,W)

def flow_to_magnitude_robust_simple(flow, noise_threshold_px=5e-2):
    """
    Ultra-simple robust flow magnitude computation.
    If max magnitude is below noise threshold, return zeros.
    """
    u = flow[:,0]
    v = flow[:,1]
    
    # Compute raw magnitude
    raw_mag = torch.sqrt(u*u + v*v + eps)
    
    # If max magnitude is below noise threshold, return zeros
    if raw_mag.max() < noise_threshold_px:
        # print('///////////////zet to zero of the flow//////////////////////')
        assert raw_mag.requires_grad == False, 'below implementation will drop grads if exists, but in current implementation we use motion_flow/optic_flow without grads as supervision for motion mask learning'
        raw_mag = torch.zeros_like(raw_mag)

    return raw_mag


import torch

# def normalize_map_push_contrasive_flow(x, valid_motion_threshold_px = 0.25, alpha=10.0, relu_bias_for_negative_px = 0.5):
def normalize_map(x, valid_motion_threshold_px_confident_lower_bound = 0.05, valid_motion_threshold_px = 0.5, alpha=10.0, relu_bias_for_negative_px = 0.5):
    """
    use to gen soft motion mask from raw flow magnitude map
    aim: for mag > valid_motion_threshold_px, the mask should be approching 1 --while the motion flow mag indeed give some intuation regarding motion status
    for mag < valid_motion_threshold_px, the mask should be approching 0 --while we achieve via relu+bias


    flow_mag: (H, W) flow magnitude map, tensor
    threshold: scalar, flow magnitude where mask ~0.5
    alpha: controls sharpness of transition, bigger sharper
    """
    # return torch.sigmoid(alpha * (x - valid_motion_threshold_px))


    # def soft_mask_from_magnitude(flow_mag, threshold=0.5, alpha=10.0):
    bias = alpha * relu_bias_for_negative_px
    return torch.sigmoid(alpha * torch.relu(x - valid_motion_threshold_px) -bias)


def normalize_map_ori(x, p=99.0):
# def normalize_map(x, p=99.0):
    # robust normalization: divide by p-th percentile per-sample to avoid outliers
    B = x.shape[0]
    out = torch.empty_like(x)
    for i in range(B):
        px = torch.quantile(x[i].flatten(), p/100.0)
        denom = px if px>0 else x[i].max().clamp_min(eps)
        out[i] = (x[i] / (denom + eps)).clamp(0.0,1.0)
    return out
 

 

def gaussian_blur(x, kernel_size=11, sigma=1.5):
    # use kornia if available for GPU Gaussian
    return kornia.filters.gaussian_blur2d(x.unsqueeze(1), (kernel_size,kernel_size), (sigma,sigma)).squeeze(1)

def edge_map(x):
    '''
    maybe improper for soft masks?
    '''

    # x: (B,H,W) float
    # if x.dim() == 3:
    assert x.dim() == 3
    gx = x.unsqueeze(1)  # (B,1,H,W)
    # print(f'gx.shape: {gx.shape}')
    # else:
    #     assert x.dim() == 4
    #     assert x.shape[1] == 1
    #     gx = x

    # spatial_gradient returns (B,1,2,H,W) where 2 is [grad_x, grad_y]
    grad = kornia.filters.spatial_gradient(gx)  # (B,1,2,H,W)
    # Extract x and y gradients correctly
    grad_x = grad[:, 0, 0]  # (B,H,W) - x direction gradient
    grad_y = grad[:, 0, 1]  # (B,H,W) - y direction gradient
    # Compute gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12)  # (B,H,W)
    return grad_mag

# Example loss builder
def structure_loss(flow, mask, weights=(1.0,0.2,0.5),static_flow_noise_thre = 1e-3):
    # flow: (B,2,H,W), mask: (B,1,H,W) or (B,H,W) with {0,1}
    if mask.dim()==4:
        mask = mask.squeeze(1)
    # mag = flow_to_magnitude(flow)    
    mag = flow_to_magnitude_robust_simple(flow, noise_threshold_px=static_flow_noise_thre)
    #         # (B,H,W)
    mag_n = normalize_map(mag)               # (B,H,W), in [0,1]
    mask_f = mask.float()

    # optionally blur to capture coarse shape
    # mag_blur = gaussian_blur(mag_n)
    # mask_blur = gaussian_blur(mask_f)
    mag_blur = mag_n
    mask_blur = mask_f

    L_mag = F.l1_loss(mag_blur, mask_blur)
    # enforce the num of positive px and negative px are similar among the two
    #  L_mag alone might cause the model to be lazy/cheat                # L1 on blurred maps
    L_dice = 1.0 - (2*(mag_n*mask_f).sum(dim=[1,2]) + eps) / (mag_n.sum(dim=[1,2]) + mask_f.sum(dim=[1,2]) + eps)
    L_dice = L_dice.mean()

    # edge loss
    e1 = edge_map(mag_n)
    e2 = edge_map(mask_f)
    L_edge = F.l1_loss(e1, e2)

    # lam_mag, lam_edge, lam_dice = weights
    # return lam_mag*L_mag + lam_edge*L_edge + lam_dice*L_dice
    imgs_debug = {
        'mag_n': mag_n.detach(),
        'mask_f': mask_f.detach(),
        'e1': e1.detach(),
        'e2': e2.detach(),
    }
    return L_mag, L_edge, L_dice, imgs_debug

    # return L_mag, L_edge, L_dice



def structure_loss_soft(flow, mask, weights=(1.0,0.2,0.5),static_flow_noise_thre = 1e-3):
    # flow: (B,2,H,W), mask: (B,H,W) in [0,1]
    # mag = flow_to_magnitude(flow)           
    mag = flow_to_magnitude_robust_simple(flow, noise_threshold_px=static_flow_noise_thre)
    mag_n = normalize_map(mag)
    # print(f'flow.shape: {flow.shape}')               # [0,1]
    # print(f'mask.shape: {mask.shape}')
    # print(f'mag.shape: {mag.shape}')
    # print(f'mag_n.shape: {mag_n.shape}')
             # [0,1]
    
    # blur for coarse structure
    # mag_blur = gaussian_blur(mag_n)
    # mask_blur = gaussian_blur(mask)
    mag_blur = mag_n
    mask_blur = mask

    L_mag = F.l1_loss(mag_blur, mask_blur)

    # soft dice
    L_dice = 1.0 - (2*(mag_n*mask).sum(dim=[1,2]) + eps) / \
                    (mag_n.sum(dim=[1,2]) + mask.sum(dim=[1,2]) + eps)
    L_dice = L_dice.mean()

    # edge loss
    e1 = edge_map(mag_n)
    e2 = edge_map(mask)
    # print(f'e1.shape: {e1.shape}')
    # print(f'e2.shape: {e2.shape}')
    L_edge = F.l1_loss(e1, e2)

    # lam_mag, lam_edge, lam_dice = weights
    # return lam_mag*L_mag + lam_edge*L_edge + lam_dice*L_dice

    imgs_debug = {
        'mag_n': mag_n.detach(),
        'mask': mask.detach(),
        'e1': e1.detach(),
        'e2': e2.detach(),
    }
    return L_mag, L_edge, L_dice, imgs_debug


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic test data
    B, H, W = 2, 64, 64  # batch size, height, width
    
    # Create synthetic optical flow (B, 2, H, W)
    # Create a circular motion pattern
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_x, center_y = W // 2, H // 2
    dx = (x - center_x).float() / (W // 4)  # normalize
    dy = (y - center_y).float() / (H // 4)  # normalize
    
    # Create circular flow pattern
    flow_u = -dy  # tangential velocity
    flow_v = dx   # tangential velocity
    
    # # Add some noise and scale
    # flow_u = flow_u + 0.01 * torch.randn(H, W)
    # flow_v = flow_v + 0.01 * torch.randn(H, W)
    # flow_u *= 5.0  # scale up
    # flow_v *= 5.0  # scale up
    
    # Stack into flow tensor (B, 2, H, W)
    flow = torch.stack([flow_u, flow_v], dim=0).unsqueeze(0).repeat(B, 1, 1, 1).to(device)
    

    static_flow_noise_thre = 0.01#1e-3

    #debug flow_mag_robust
    # flow = flow*(static_flow_noise_thre*0.01)

    # debug later mag_n processing
    # set flow vector to be static_flow_noise_thre if the its magnitude is below 0.1 px
    flow_mag = torch.sqrt(flow_u**2 + flow_v**2)
    flow_u = flow_u.clone()
    flow_v = flow_v.clone()
    flow_u[flow_mag < 10] = 0
    flow_v[flow_mag < 10] = 0
    # flow = torch.stack([flow_u, flow_v], dim=0).unsqueeze(0).repeat(B, 1, 1, 1).to(device)

    # Create binary mask (B, H, W) with {0, 1}
    binary_mask = torch.zeros(B, H, W, device=device)
    # Create circular mask
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    circle_mask = (dist < 15).float()
    binary_mask[0] = circle_mask
    binary_mask[1] = 1.0 - circle_mask  # inverted for second sample
    
    # Create soft gray mask (B, H, W) in [0, 1]
    soft_mask = torch.zeros(B, H, W, device=device)
    # Create Gaussian-like soft mask
    gaussian_mask = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 8**2))
    soft_mask[0] = gaussian_mask
    soft_mask[1] = 1.0 - gaussian_mask  # inverted for second sample
    
    print("Testing structure_loss (binary mask)...")
    L_mag_bin, L_edge_bin, L_dice_bin, imgs_debug_bin = structure_loss(flow, binary_mask,static_flow_noise_thre=static_flow_noise_thre)
    print(f"Binary mask losses - L_mag: {L_mag_bin:.4f}, L_edge: {L_edge_bin:.4f}, L_dice: {L_dice_bin:.4f}")
    
    print("\nTesting structure_loss_soft (soft gray mask)...")
    L_mag_soft, L_edge_soft, L_dice_soft, imgs_debug_soft = structure_loss_soft(flow, soft_mask,static_flow_noise_thre=static_flow_noise_thre)
    print(f"Soft mask losses - L_mag: {L_mag_soft:.4f}, L_edge: {L_edge_soft:.4f}, L_dice: {L_dice_soft:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Binary mask results
    axes[0, 0].imshow(imgs_debug_bin['mag_n'][0].cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title('Flow Magnitude (Binary)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(imgs_debug_bin['mask_f'][0].cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(imgs_debug_bin['e1'][0].cpu().numpy(), cmap='hot')
    axes[0, 2].set_title('Flow Edges (Binary)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(imgs_debug_bin['e2'][0].cpu().numpy(), cmap='hot')
    axes[0, 3].set_title('Mask Edges (Binary)')
    axes[0, 3].axis('off')
    
    # Soft mask results
    axes[1, 0].imshow(imgs_debug_soft['mag_n'][0].cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title('Flow Magnitude (Soft)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(imgs_debug_soft['mask'][0].cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Soft Gray Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(imgs_debug_soft['e1'][0].cpu().numpy(), cmap='hot')
    axes[1, 2].set_title('Flow Edges (Soft)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(imgs_debug_soft['e2'][0].cpu().numpy(), cmap='hot')
    axes[1, 3].set_title('Mask Edges (Soft)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('submodule/Endo_FASt3r/mask_utils_test_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'mask_utils_test_results.png'")
    
    # Test with different flow patterns
    print("\nTesting with different flow patterns...")
    
    # Linear flow pattern
    linear_flow = torch.zeros(B, 2, H, W, device=device)
    linear_flow[:, 0] = torch.linspace(-10, 10, W).unsqueeze(0).repeat(H, 1)  # horizontal flow
    linear_flow[:, 1] = torch.linspace(-5, 5, H).unsqueeze(1).repeat(1, W)    # vertical flow
    
    # Test with linear flow
    L_mag_lin, L_edge_lin, L_dice_lin, _ = structure_loss(linear_flow, binary_mask,static_flow_noise_thre=static_flow_noise_thre)
    print(f"Linear flow with binary mask - L_mag: {L_mag_lin:.4f}, L_edge: {L_edge_lin:.4f}, L_dice: {L_dice_lin:.4f}")
    
    L_mag_lin_soft, L_edge_lin_soft, L_dice_lin_soft, _ = structure_loss_soft(linear_flow, soft_mask,static_flow_noise_thre=static_flow_noise_thre)
    print(f"Linear flow with soft mask - L_mag: {L_mag_lin_soft:.4f}, L_edge: {L_edge_lin_soft:.4f}, L_dice: {L_dice_lin_soft:.4f}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Zero flow
    zero_flow = torch.zeros(B, 2, H, W, device=device)
    L_mag_zero, L_edge_zero, L_dice_zero, _ = structure_loss(zero_flow, binary_mask,static_flow_noise_thre=static_flow_noise_thre)
    print(f"Zero flow - L_mag: {L_mag_zero:.4f}, L_edge: {L_edge_zero:.4f}, L_dice: {L_dice_zero:.4f}")
    
    # Perfect match (flow magnitude matches mask exactly)
    perfect_flow = torch.zeros(B, 2, H, W, device=device)
    perfect_mag = binary_mask * 10.0  # scale to make magnitude significant
    # Create flow that produces this magnitude
    perfect_flow[:, 0] = perfect_mag * 0.7  # u component
    perfect_flow[:, 1] = perfect_mag * 0.7  # v component
    L_mag_perf, L_edge_perf, L_dice_perf, _ = structure_loss(perfect_flow, binary_mask,static_flow_noise_thre=static_flow_noise_thre)
    print(f"Perfect match - L_mag: {L_mag_perf:.4f}, L_edge: {L_edge_perf:.4f}, L_dice: {L_dice_perf:.4f}")
    
    print("\nTest completed successfully!")

