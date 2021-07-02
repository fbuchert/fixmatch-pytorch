import torchvision.models as models
from torch import nn

import models.preact_resnet as preact_resnet
from models.wideresnet import WideResNet


def get_resnet18(num_classes=10, pretrained=False):
    resnet = models.resnet18(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


def get_resnet34(num_classes=10, pretrained=False):
    resnet = models.resnet34(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


def get_resnet50(num_classes=10, pretrained=False):
    resnet = models.resnet50(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


def get_preact_resnet18(num_classes=10):
    return preact_resnet.preact_resnet18(num_classes)


def get_preact_resnet34(num_classes=10):
    return preact_resnet.preact_resnet34(num_classes)


def get_preact_resnet50(num_classes=10):
    return preact_resnet.preact_resnet50(num_classes)


def get_vgg11(num_classes=10):
    return models.vgg11(num_classes=num_classes)


def get_vgg13(num_classes=10):
    return models.vgg13(num_classes=num_classes)


def get_vgg16(num_classes=10):
    return models.vgg16_bn(num_classes=num_classes)


def get_vgg19(num_classes=10):
    return models.vgg19(num_classes=num_classes)


def get_wide_resnet28_2(pretrained=False, **kwargs):
    return WideResNet(depth=28, widen_factor=2, dropout_rate=0, **kwargs)


def get_wide_resnet28_10(pretrained=False, **kwargs):
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0, **kwargs)


def get_wide_resnet50_2(pretrained=False, **kwargs):
    return models.wide_resnet50_2(pretrained=pretrained, **kwargs)


def get_wide_resnet101_2(pretrained=False, **kwargs):
    return models.wide_resnet101_2(pretrained=pretrained, **kwargs)


def get_densenet121(pretrained=False, **kwargs):
    return models.densenet121(pretrained, **kwargs)


def get_densenet161(pretrained=False, **kwargs):
    return models.densenet161(pretrained, **kwargs)


def get_densenet201(pretrained=False, **kwargs):
    return models.densenet201(pretrained, **kwargs)


MODEL_GETTERS = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "preact_resnet18": get_preact_resnet18,
    "preact_resnet34": get_preact_resnet34,
    "preact_resnet50": get_preact_resnet50,
    "vgg11": get_vgg11,
    "vgg13": get_vgg13,
    "vgg16": get_vgg16,
    "vgg19": get_vgg19,
    "wide_resnet28_2": get_wide_resnet28_2,
    "wide_resnet28_10": get_wide_resnet28_10,
    "wide_resnet50_10": get_wide_resnet50_2,
    "wide_resnet101_2": get_wide_resnet101_2,
    "densenet121": get_densenet121,
    "densenet161": get_densenet161,
    "densenet201": get_densenet201,
}
