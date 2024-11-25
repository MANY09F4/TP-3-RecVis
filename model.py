import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, AutoModel
from torchvision import models
from timm import create_model
from transformers import AutoModelForImageClassification

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
        in_features = self.model.head.in_features  # Accéder au nombre de features d'entrée de la dernière couche
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetV2M(nn.Module):
    def __init__(self, num_classes=500):
        super(EfficientNetV2M, self).__init__()
        # Charger EfficientNetV2-M depuis timm avec poids pré-entraînés
        self.model = create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)
        # Remplacer la dernière couche pour correspondre au nombre de classes
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DinoV2(nn.Module):
    def __init__(self, num_classes=500, freeze_backbone=True):
        super(DinoV2, self).__init__()
        # Charger le modèle pré-entraîné
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # Optionnel : Geler les couches de la backbone
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        # Ajouter une couche de classification pour 500 classes
        hidden_size = self.backbone.config.hidden_size  # Taille des embeddings de sortie (ex. 768)
        self.classifier = nn.Linear(hidden_size, num_classes)  # Couche dense pour la classification


    def forward(self, x):
        # Passer l'entrée à travers la backbone
        outputs = self.backbone(x)
        # Extraire le token CLS (premier vecteur de la séquence)
        cls_token = outputs.last_hidden_state[:, 0, :]
        # Passer le token CLS dans le classificateur
        logits = self.classifier(cls_token)
        return logits

class DinoV2_perso(nn.Module):
    def __init__(self, num_classes=500, freeze_backbone=True):
        super(DinoV2_perso, self).__init__()
        # Charger le modèle pré-entraîné
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        for name, param in self.backbone.named_parameters():
            if "encoder.layer.11" in name or "layer_norm" in name or "encoder.layer.10" in name or "encoder.layer.9" in name:  # Dernière couche
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Ajouter une couche de classification pour 500 classes
        hidden_size = self.backbone.config.hidden_size  # Taille des embeddings de sortie (ex. 768)
        self.classifier = nn.Sequential(nn.Dropout(0.3),
                                        nn.Linear(hidden_size, num_classes))  # Couche dense pour la classification


    def forward(self, x):
        # Passer l'entrée à travers la backbone
        outputs = self.backbone(x)
        # Extraire le token CLS (premier vecteur de la séquence)
        cls_token = outputs.last_hidden_state[:, 0, :]
        # Passer le token CLS dans le classificateur
        logits = self.classifier(cls_token)
        return logits

class ConvNeXt_perso(nn.Module):
    def __init__(self, num_classes=500, freeze_backbone=True):
        super(ConvNeXt_perso, self).__init__()
        # Charger le modèle pré-entraîné ConvNeXt
        self.backbone = create_model("convnext_base", pretrained=True)

        # Geler ou dégeler des parties spécifiques du backbone
        for name, param in self.backbone.named_parameters():
            if "stages.3" in name or "norm" in name:  # Dernière étape et LayerNorm
                param.requires_grad = True
            else:
                param.requires_grad = not freeze_backbone

        # Ajouter une couche de classification pour 500 classes
        in_features = self.backbone.head.in_features  # Taille des embeddings de sortie
        self.backbone.head = nn.Identity()  # Supprimez l'ancienne tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Dropout pour régularisation
            nn.Linear(in_features, num_classes)  # Couche dense pour la classification
        )

    def forward(self, x):
        # Passer l'entrée dans le backbone
        features = self.backbone(x)
        # Passer les caractéristiques dans le classificateur
        logits = self.classifier(features)
        return logits

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
