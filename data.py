import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as F

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
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

data_transforms_400 = transforms.Compose(
    [
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_512 = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_224_da_old = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_transforms_224_DA = transforms.Compose(
    [
        # Augmentation
        transforms.Resize((224, 224)),
        #transforms.Grayscale(num_output_channels=3), # Convertir certaines images en niveaux de gris
        transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal aléatoire
        transforms.RandomRotation(15),          # Rotation aléatoire jusqu'à ±15°
        #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Recadrage aléatoire (zoom-in)
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variations de couleurs

        # Prétraitement
        transforms.ToTensor(),                  # Convertir en tenseur
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisation pour niveaux de gris
    ]
)

data_transforms_384 = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EdgeDetectionTransform:
    def __call__(self, img):
        img = np.array(img)
        edges = cv2.Canny(img, threshold1=100, threshold2=200)
        edges = np.stack([edges]*3, axis=-1)  # Convertit en 3 canaux
        return Image.fromarray(edges)


data_transforms_edge = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        EdgeDetectionTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class RandomSketchTransform:
    def __call__(self, img):
        # Applique aléatoirement une inversion de couleurs pour simuler différents styles de sketch
        if random.random() > 0.5:
            img = F.invert(img)
        return img

data_transforms_224_da = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convertit en niveaux de gris tout en conservant 3 canaux
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5)
        ], p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        RandomSketchTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
