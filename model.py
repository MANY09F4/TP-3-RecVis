import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from torchvision import models
from timm import create_model

nclasses = 500

class ResNet18(nn.Module):
    def __init__(self, num_classes=500):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=500):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classe
        # s
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet101(nn.Module):
    def __init__(self, num_classes=500):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=500):
        super(EfficientNetB4, self).__init__()
        self.model = models.efficientnet_b4(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB5(nn.Module):
    def __init__(self, num_classes=500):
        super(EfficientNetB5, self).__init__()
        self.model = models.efficientnet_b5(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB6(nn.Module):
    def __init__(self, num_classes=500):
        super(EfficientNetB6, self).__init__()
        self.model = models.efficientnet_b6(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=500):
        super(EfficientNetB7, self).__init__()
        self.model = models.efficientnet_b7(pretrained=True)
        # Remplace le dernier layer pour correspondre aux 500 classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class VitBase16(nn.Module):
    def __init__(self, num_classes=500):
        super(VitBase16, self).__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # Ajouter un classificateur pour les 500 classes
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.model(x).last_hidden_state[:, 0, :]
        return self.classifier(outputs)

class ConvNextBase(nn.Module):
    def __init__(self, num_classes=500):
        super(ConvNextBase, self).__init__()
        # Charger le modèle pré-entraîné ConvNeXt
        self.model = create_model('convnext_base', pretrained=True)
        # Remplacer la dernière couche (classificateur) pour s'adapter aux 500 classes
        self.model.head = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
