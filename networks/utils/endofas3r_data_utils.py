import torch
import torchvision.transforms as tvf
from torchvision.transforms import ToPILImage
import PIL

def prepare_images(x, device, size, square_ok=False, Ks=None):
    to_pil = ToPILImage()
    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imgs = []
    adapted_Ks = []
    
    for idx in range(x.size(0)):
        tensor = x[idx].cpu()  # Shape [3, 256, 320]
        img = to_pil(tensor).convert("RGB")
        W1, H1 = img.size
        
        # Store original dimensions for K adaptation
        orig_w, orig_h = W1, H1
        
        if size == 224:
            assert 0, "not used?"
            # resize short side to 224 (then crop)
            img = resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to size
            img = resize_pil_image(img, size)
        
        W, H = img.size
        cx, cy = W//2, H//2
        
        # Calculate crop parameters
        if size == 224:
            half = min(cx, cy)
            crop_x1, crop_y1 = cx-half, cy-half
            crop_x2, crop_y2 = cx+half, cy+half
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            crop_x1, crop_y1 = cx-halfw, cy-halfh
            crop_x2, crop_y2 = cx+halfw, cy+halfh
        
        # Crop the image
        img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Adapt K if provided
        if Ks is not None:
            assert Ks.dim() == 3, f'{Ks.shape}' 
            K = Ks[idx]
            # Calculate scaling factors
            scale_x = W / orig_w
            scale_y = H / orig_h
            
            # Calculate crop offsets
            offset_x = crop_x1
            offset_y = crop_y1
            
            # Create transformation matrix for K
            # K_new = [fx*scale_x, 0, cx*scale_x - offset_x]
            #         [0, fy*scale_y, cy*scale_y - offset_y]
            #         [0, 0, 1]
            K_adapted = K.clone()
            K_adapted[0, 0] *= scale_x  # fx
            K_adapted[1, 1] *= scale_y  # fy
            K_adapted[0, 2] = K[0, 2] * scale_x - offset_x  # cx
            K_adapted[1, 2] = K[1, 2] * scale_y - offset_y  # cy
            
            adapted_Ks.append(K_adapted)
        
        imgs.append(ImgNorm(img)[None].to(device))
    
    if Ks is not None:
        return torch.cat(imgs, dim=0), torch.stack(adapted_Ks, dim=0)
    else:
        return torch.cat(imgs, dim=0)
    
def prepare_images_v0(x, device, size, square_ok=False, K=None):
  to_pil = ToPILImage()
  ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  # ImgNorm = tvf.Compose([tvf.ToTensor()])
  imgs = []
  for idx in range(x.size(0)):
      tensor = x[idx].cpu()  # Shape [3, 256, 320]
      img = to_pil(tensor).convert("RGB")
      W1, H1 = img.size
      if size == 224:
          assert 0, "not used?"
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

