import torch
import torch.nn.functional as F
import kornia  # optional, for Sobel and Gaussian

eps = 1e-6

def flow_to_magnitude(flow):            # flow: (B,2,H,W)
    u = flow[:,0]
    v = flow[:,1]
    mag = torch.sqrt(u*u + v*v + 1e-12)
    return mag  # (B,H,W)

def normalize_map(x, p=99.0):
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
    # x: (B,H,W) float
    gx = x.unsqueeze(1)  # (B,1,H,W)
    edges = kornia.filters.sobel(gx)    # returns (B,1,2,H,W)? or use kornia.filters.spatial_gradient
    # simpler: spatial gradient magnitude
    grad = kornia.filters.spatial_gradient(gx)  # (B,1,2,H,W)
    grad = torch.sqrt(grad[:,0,0]**2 + grad[:,0,1]**2 + 1e-12)  # (B,H,W)
    return grad

# Example loss builder
def structure_loss(flow, mask, weights=(1.0,0.2,0.5)):
    # flow: (B,2,H,W), mask: (B,1,H,W) or (B,H,W) with {0,1}
    if mask.dim()==4:
        mask = mask.squeeze(1)
    mag = flow_to_magnitude(flow)            # (B,H,W)
    mag_n = normalize_map(mag)               # (B,H,W), in [0,1]
    mask_f = mask.float()

    # optionally blur to capture coarse shape
    mag_blur = gaussian_blur(mag_n)
    mask_blur = gaussian_blur(mask_f)

    L_mag = F.l1_loss(mag_blur, mask_blur)                # L1 on blurred maps
    L_dice = 1.0 - (2*(mag_n*mask_f).sum(dim=[1,2]) + eps) / (mag_n.sum(dim=[1,2]) + mask_f.sum(dim=[1,2]) + eps)
    L_dice = L_dice.mean()

    # edge loss
    e1 = edge_map(mag_n)
    e2 = edge_map(mask_f)
    L_edge = F.l1_loss(e1, e2)

    # lam_mag, lam_edge, lam_dice = weights
    # return lam_mag*L_mag + lam_edge*L_edge + lam_dice*L_dice

    return L_mag, L_edge, L_dice



def structure_loss_soft(flow, mask, weights=(1.0,0.2,0.5)):
    # flow: (B,2,H,W), mask: (B,H,W) in [0,1]
    mag = flow_to_magnitude(flow)           
    mag_n = normalize_map(mag)               # [0,1]
    
    # blur for coarse structure
    mag_blur = gaussian_blur(mag_n)
    mask_blur = gaussian_blur(mask)

    L_mag = F.l1_loss(mag_blur, mask_blur)

    # soft dice
    L_dice = 1.0 - (2*(mag_n*mask).sum(dim=[1,2]) + eps) / \
                    (mag_n.sum(dim=[1,2]) + mask.sum(dim=[1,2]) + eps)
    L_dice = L_dice.mean()

    # edge loss
    e1 = edge_map(mag_n)
    e2 = edge_map(mask)
    L_edge = F.l1_loss(e1, e2)

    # lam_mag, lam_edge, lam_dice = weights
    # return lam_mag*L_mag + lam_edge*L_edge + lam_dice*L_dice

    return L_mag, L_edge, L_dice

