import math

import torch
from torch import nn

from . import spatial_weight
from .res_blocks import BasicBlock, Bottleneck


class ConvNet(nn.Module):
    def __init__(self, conv_type, conv_cfg,
                 classifier_cfg,
                 dropout=None):
        """
        :param conv_type: str, either 'vgg' or 'resnet'
        :param conv_cfg: tuple, contain the num of units at each layer for geo-net
        :param classifier_cfg: iterable, the architecture of classifier
        """
        super(ConvNet, self).__init__()

        if conv_type == 'vgg':
            self.conv = make_vgg_layers(conv_cfg)
        elif conv_type == 'resnet':
            self.conv = make_res_layers(conv_cfg)
        else:
            raise Exception('Invalid convnet type')

        self.geo = nn.Sequential(nn.Linear(12, 12),
                                 nn.BatchNorm1d(12),
                                 nn.Tanh())

        self.classifier = make_classifier(classifier_cfg, dropout)

        self._initialize_weights()

    def forward(self, x_geo, x_sat):
        x_sat = self.conv(x_sat)
        x_sat = x_sat.view(x_sat.size(0), -1)
        if x_geo is not None:
            x_geo = self.geo(x_geo)
            x = torch.cat((x_geo, x_sat), dim=1)
        else:
            x = x_sat
        outp = self.classifier(x)
        return outp

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
            elif isinstance(m, spatial_weight.SpatialWeight):
                m.exp_a.requires_grad = False
                m.exp_b.requires_grad = False


def make_vgg_layers(conv_cfg, in_channel=7):
    dist_map = None
    layers_ = []
    for l in conv_cfg:
        if l == 'M3':
            layer = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
            layers_.append(layer)
        elif l == 'M1':
            layer = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            layers_.append(layer)
        elif l == 'A3':
            layer = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
            layers_.append(layer)
        elif l == 'A1':
            layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            layers_.append(layer)
        elif l == 'W':
            if dist_map is None:
                dist_map = spatial_weight.dist_func()
            layer = spatial_weight.SpatialWeight(dist_map=dist_map, plane=in_channel)
            layers_.append(layer)
        elif l == 'E':
            layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
            layers_.append(layer)
        elif l == 'C':
            layers_ += [nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(in_channel),
                        nn.ReLU(inplace=True)]
        elif isinstance(l, int):
            layers_ += [nn.Conv2d(in_channels=in_channel, out_channels=l, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(l),
                        nn.ReLU(inplace=True)]
            in_channel = l
        else:
            raise Exception('Invalid layer type: %s' % l)
    layers_ = nn.Sequential(*layers_)
    return layers_


def make_res_layers(conv_cfg, in_channel=7):
    dist_map = None
    layers_ = []
    layers_ += [nn.Conv2d(in_channels=in_channel, out_channels=64,
                          kernel_size=7, stride=1, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
                ]
    in_channel = 64
    for l in conv_cfg:
        layer_type = l[:1]
        if layer_type == 'B':
            _, planes, stride = l.split('/')
            planes, stride = int(planes), int(stride)
            layer = BasicBlock(inplanes=in_channel, planes=planes, stride=stride)
            in_channel = planes
        elif layer_type == 'N':
            _, planes, stride = l.split('/')
            planes, stride = int(planes), int(stride)
            layer = Bottleneck(inplanes=in_channel, planes=planes, stride=stride)
            in_channel = planes * 4
        elif layer_type == 'W':
            if dist_map is None:
                dist_map = spatial_weight.dist_func()
            layer = spatial_weight.SpatialWeight(dist_map=dist_map, plane=in_channel)
        elif layer_type == 'E':
            layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        else:
            raise Exception('Invalid layer type: %s' % layer_type)
        layers_.append(layer)
    layers_ = nn.Sequential(*layers_)
    return layers_


def make_classifier(classifier_cfg, dropout):
    layers_ = []
    for c1, c2 in zip(classifier_cfg, classifier_cfg[1:]):
        layers_ += [nn.Linear(c1, c2),
                    nn.BatchNorm1d(c2),
                    nn.Tanh()]
    layers_.append(nn.Linear(classifier_cfg[-1], 1))
    layers_ = nn.Sequential(*layers_)
    return layers_
