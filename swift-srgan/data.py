import os
import random
from PIL import Image
from PIL.Image import Resampling
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, pad, rotate, center_crop
from torch.utils.data import Dataset

class RandomReflectiveRotation:
    def __init__(self, degrees, p=0.2, padding=95, output_size=512):
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (-degrees, degrees)
        self.p = p
        self.padding = padding
        self.output_size = output_size

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(*self.degrees)

            # Reflect padding
            img = pad(img, padding=self.padding, padding_mode='reflect')

            # Rotate image
            img = rotate(img, angle, interpolation=transforms.InterpolationMode.BICUBIC)

            # Center crop back to original size
            img = center_crop(img, output_size=[self.output_size, self.output_size])

        return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size=512):
    return transforms.Compose([
        transforms.RandomCrop(crop_size, pad_if_needed=True),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomHorizontalFlip(p=0.3),
        RandomReflectiveRotation(degrees=30, p=0.15, padding=95, output_size=crop_size),
        transforms.ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size // upscale_factor, interpolation=Resampling.BICUBIC),
        transforms.ToTensor()
    ])


display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor()
])


class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ValDataset, self).__init__()
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        
        lr_scale = transforms.Resize(self.crop_size // self.upscale_factor, interpolation=Resampling.BICUBIC)
        hr_scale = transforms.Resize(self.crop_size, interpolation=Resampling.BICUBIC)
        hr_image = transforms.CenterCrop(self.crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return to_tensor(lr_image), to_tensor(hr_restore_img), to_tensor(hr_image)

    def __len__(self):
        return len(self.image_filenames)