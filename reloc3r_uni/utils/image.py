# utilitary functions adapted from DUSt3R
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import math


try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
Img2Tensor = tvf.Compose([tvf.ToTensor()])


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    # assert 0, f'{img.size} {new_size}'
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img_path = os.path.join(root, path)
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        elif size == 512:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        else:
            raise ValueError(f'Unsupported size: {size}')
        
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

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(
            img=ImgNorm(img)[None], 
            true_shape=np.int32(
            [img.size[::-1]]), 
            idx=len(imgs), 
            img_path=img_path,
            instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

#/////follow traning img processing
def _crop_resize_if_necessary(image, intrinsics, resolution=(512,384), rng=None, info=None):
    """ 
    siyan: this function can change the camera center, but the corresponding pose does not transform accordingly...
    """
    import PIL
    # from reloc3r.reloc3r_visloc import Reloc3rVisloc
    import reloc3r_uni.datasets.utils.cropping as cropping

    if not isinstance(image, PIL.Image.Image): 
        image = PIL.Image.fromarray(image)

    # downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    print('/////current img size_wh and resolution',image.size, resolution, cx, cy)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    # assert min_margin_x > W/5, f'Bad principal point in view={info} {min_margin_x} {W/5} {cx} {W}'
    # assert min_margin_y > H/5, f'Bad principal point in view={info} {min_margin_y} {H/5} {cy} {H}'
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    # image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
    image, intrinsics = cropping.crop_image(image, intrinsics, crop_bbox)

    # transpose the resolution if necessary
    W, H = image.size  # new size
    assert resolution[0] >= resolution[1]

    if H > 1.1*W:
        # image is portrait mode
        resolution = resolution[::-1]
    elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        # image is square, so we chose (portrait, landscape) randomly
        if rng.integers(2):
            resolution = resolution[::-1]
    # high-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    # image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
    image, intrinsics = cropping.rescale_image(image, intrinsics, target_resolution)

    # actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    # image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
    image, intrinsics2 = cropping.crop_image(image, intrinsics, crop_bbox)

    return image, intrinsics2

def load_images_trn(folder_or_list, resolution, square_ok=False, verbose=True, intrinsics=None, device=None):
    '''
    implement as load_images, but use _crop_resize_if_necessary for resizing
    '''
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img_path = os.path.join(root, path)
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        
        assert len(resolution) == 2, f'resolution must be a tuple of two elements, but got {resolution}'
        
        # Apply training-style crop and resize
        img, intrinsics2 = _crop_resize_if_necessary(img, intrinsics, resolution, rng=None, info=path)
        
        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(
            img=ImgNorm(img)[None], 
            true_shape=np.int32([img.size[::-1]]), 
            idx=len(imgs), 
            img_path=img_path,
            instance=str(len(imgs)),
            camera_intrinsics=intrinsics2[None],
            ))

    assert imgs, 'no images found at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def parse_video(video_path, output_folder, max_frames=24): 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file')
        exit()
    
    # images from the video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    step = math.ceil(len(frames)/max_frames)
    frames = frames[0:-1:step]

    # save the images 
    for fid in range(len(frames)):
        image_path = output_folder + '/frame_{:04d}.jpg'.format(fid)
        cv2.imwrite(image_path, frames[fid])

    cap.release()


def check_images_shape_format(images, device):
    for i in range(len(images)):
        images[i]['img'] = images[i]['img'].to(device)
        images[i]['true_shape'] = torch.Tensor(images[i]['true_shape']).to(device)
        if 'camera_intrinsics' in images[i]:
            images[i]['camera_intrinsics'] = torch.Tensor(images[i]['camera_intrinsics']).to(device)#.unsqueeze(0)

    return images

