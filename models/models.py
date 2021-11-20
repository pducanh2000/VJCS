import torch
import torch.nn as nn
import torchvision.models as models


class Switcher_model(object):
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
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=17)
        print("Loaded Pretrained EfficientNet-B0 ")
        return self.model


class Model(nn.Module):
    def __init__(self, model_cfg):
        super(Model, self).__init__()
        self.cfg = model_cfg
        self.switch = Switcher_model()
        self.pretrain = self.switch.select_model(self.cfg['name'])

        if self.cfg['name'] in ['resnet_50', 'resnet_101', 'inceptionnetv3']:
            self.pretrain.fc = nn.Linear(self.cfg['feat'], 17, bias=True)
        elif self.cfg['name'] in ['mobilenetv2', 'mobilenetv3']:
            self.pretrain.classifier[-1] = nn.Linear(self.cfg['feat'], 17, bias=True)
        else:
            self.pretrain._fc = nn.Linear(self.cfg['feat'], 17, bias=True)

    def forward(self, x):
        if self.cfg['name'] == 'inceptionnetv3':
            if self.train():
                x = self.pretrain(x)[0]
        else:
            x = self.pretrain(x)
        return x
