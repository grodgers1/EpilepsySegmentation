import os
import numpy as np
import torch
from PIL import Image

from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.
    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, crop=(256, 256), p_hflip=0.5, p_vflip=0.5, p_noise=0.0, noise_sigma=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, random_affine_params=(-20, 20, 0.25, 0.25, 0.75, 1.25, -20, 20), long_mask=False):
        self.crop = crop
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_noise = p_noise
        self.noise_sigma = noise_sigma
        self.p_random_affine = p_random_affine
        self.random_affine_params = random_affine_params
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        if np.random.rand() < self.p_noise:
            noise = np.random.normal(0.0,self.noise_sigma,image.shape)
            tmpim = np.array(image) + noise
            image = Image.fromarray(tmpim)

        if np.random.rand() < self.p_hflip:
            image, mask = F.hflip(image), F.hflip(mask)

        if np.random.rand() < self.p_vflip:
            image, mask = F.vflip(image), F.vflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        random_affine_params = self.random_affine_params
        if random_affine_params:
            affine_params = T.RandomAffine.get_params(random_affine_params[0:2],
                                                  random_affine_params[2:4],
                                                  random_affine_params[4:6],
                                                  random_affine_params[6:8],
                                                  image.size)
        else:
            affine_params = [0,0,0,0,1,1,0,0]

        if np.random.rand() < self.p_random_affine:
            doRA = True
        else:
            doRA = False

        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
        else:
            i, j, h, w = 0, 0, image.size[0], image.size[1]

        dummy = Image.fromarray(np.zeros((image.size[0], image.size[1])).transpose())
        if doRA:
            dummy = F.affine(dummy, *affine_params, T.InterpolationMode.NEAREST, 255)

        # random crop
        if self.crop:
            dummy = F.crop(dummy, i, j, h, w)

        dummy = np.any(np.array(dummy) == 255)
        if not dummy and doRA:
            image = F.affine(image, *affine_params, T.InterpolationMode.NEAREST, 0)
            mask = F.affine(mask, *affine_params, T.InterpolationMode.NEAREST, 0)

        if self.crop:
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        # transforming to tensor
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask

class ImageAndMask2D(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.mask_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)
        self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.input_path, image_filename))
        image = correct_dims(image)
        image = self.transform(image)
        mask = io.imread(os.path.join(self.mask_path, image_filename))
        mask = torch.as_tensor(mask,dtype=torch.int64)

        return image, mask, image_filename

class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.
    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...
        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))


    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        # read image
        image = io.imread(os.path.join(self.input_path, image_filename))
        # read mask image
        mask = io.imread(os.path.join(self.output_path, image_filename))

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.
    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, transform: Callable = None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.input_path, image_filename))

        # correct dimensions if needed
        image = correct_dims(image)

        image = self.transform(image)

        return image, image_filename