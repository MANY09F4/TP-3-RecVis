"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms_224_DA, data_transforms, data_transforms_224, data_transforms_224_da, data_transforms_384, data_transforms_edge, data_transforms_224_da, data_transforms_400, data_transforms_512
from model import ConvNextBase, Net, ResNet18, ResNet50, ResNet101, EfficientNetB4, VitBase16, EfficientNetB5, EfficientNetB6, EfficientNetB7


class ModelFactory:
    def __init__(self, model_name: str, test_mode: bool = False):
        self.test_mode = test_mode
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        if self.model_name == "resnet18":
            return ResNet18()
        if self.model_name == "resnet50":
            return ResNet50()
        if self.model_name == "resnet101":
            return ResNet101()
        if self.model_name == "efficientnet_b4":
            return EfficientNetB4()
        if self.model_name == "efficientnet_b5":
            return EfficientNetB5()
        if self.model_name == "efficientnet_b6":
            return EfficientNetB6()
        if self.model_name == "efficientnet_b7":
            return EfficientNetB7()
        if self.model_name == "vit_base16":
            return VitBase16()
        if self.model_name == "convnext_base":
            return ConvNextBase()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms

        if self.model_name == "resnet18":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224_da

        if self.model_name == "resnet50":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224

        if self.model_name == "resnet101":
            if self.test_mode:
                return data_transforms_384
            return data_transforms_384

        if self.model_name.startswith("efficientnet_b"):
            if self.test_mode:
                return data_transforms_384
            return data_transforms_384

        if self.model_name == "vit_base16":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224_DA

        if self.model_name == "convnext_base":
            if self.test_mode:
                return data_transforms_224_DA
            return data_transforms_224_DA

        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
