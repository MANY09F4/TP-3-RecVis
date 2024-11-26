import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as F


data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_224 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_224_gray_only = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #transforms.Normalize(mean=[0.854, 0.854, 0.854], std=[0.191, 0.191, 0.191])
    ]
)



data_transforms_224_DA = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),

        transforms.ToTensor(),

        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),

        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #transforms.Normalize(mean=[0.854, 0.854, 0.854], std=[0.191, 0.191, 0.191])
    ]
)
