import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class ModelSelector(object):
    def __init__(self):
        self.model = None

    def select_model(self, name):
        method = getattr(self, name, lambda: 'Invalid model')
        return method()

    def resnet_50(self):
        self.model = models.resnet50(pretrained=True)
        print("Loaded Pretrained ResNet50 ")
        return self.model

    def resnet_101(self):
        self.model = models.resnet101(pretrained=True)
        print("Loaded Pretrained ResNet101 ")
        return self.model

    def inceptionnetv3(self):
        self.model = models.inception_v3(pretrained=True)
        print("Loaded Pretrained InceptionV3")
        return self.model

    def mobilenetv2(self):
        self.model = models.mobilenet_v2(pretrained=True)
        print("Loaded Pretrained MobileNetV2 ")
        return self.model

    def mobilenetv3(self):
        self.model = models.mobilenet_v3_large(pretrained=True)
        print("Loaded Pretrained MobileNetV3 ")
        return self.model

    def efficientnetb0(self):
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        print("Loaded Pretrained EfficientNet-B0 ")
        return self.model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):

        return x


class Model(nn.Module):
    def __init__(self, model_cfg):
        super(Model, self).__init__()
        self.cfg = model_cfg
        self.extractor = self.get_extractor()
        self.classifier = self.get_classifier()


    def get_extractor(self):
        
        switch = ModelSelector()
        extractor = switch.select_model(self.cfg['name'])
        for name, layer in extractor.named_children():
            if isinstance(layer, nn.Linear):
                extractor.add_module(name, Identity())

        return extractor

    def get_classifier(self):

        return nn.Linear(self.cfg['feat'], 17, bias=True)

    def forward(self, x):
        if self.train:
            if self.cfg['name'] == 'inceptionnetv3':
                if self.train():
                    feat = self.extractor(x)[0]
            else:
                feat = self.extractor(x)
        else:
            feat = self.extractor(x)

        out = self.classifier(feat)

        return feat, out
