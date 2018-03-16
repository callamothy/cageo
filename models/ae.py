import math

import torch
from torch import nn
from torch.autograd import Variable as V


def _corrupt(x, corruption_level, mode):
    corrupt_coef = torch.Tensor(x.size())
    if mode == 'bernoulli':
        assert 0. < corruption_level < 1.
        corrupt_coef = corrupt_coef.bernoulli_(1 - corruption_level)
        if x.is_cuda:
            corrupt_coef = corrupt_coef.cuda()
        x.mul_(corrupt_coef)
        return x
    elif mode == 'gaussian':
        assert corruption_level > 0.
        corrupt_coef = corrupt_coef.normal_(0, corruption_level)
        if x.is_cuda:
            corrupt_coef = corrupt_coef.cuda()
        return x.mul_(corrupt_coef)
    else:
        raise Exception('Invalid distribution')


class CDAE(nn.Module):
    def __init__(self, conv_cfg, bottle_neck,
                 corrupt_func='bernoulli', corrupt_level=.2):
        """
        :param conv_cfg: list, architecture of conv layers
        :param bottle_neck: list, three-layer bottleneck architecture
        :param corrupt_func: str, corruption function, either bernoulli or gaussian
        :param corrupt_level: float, depending on the corruption function,
         1 - p if bernoulli, sigma if gaussian
        """
        super(CDAE, self).__init__()
        self.encode_net = CDAE._make_encode_layers(conv_cfg)
        self.decode_net = CDAE._make_decode_layers(conv_cfg)
        self.bottleneck = nn.Sequential(nn.Linear(bottle_neck[0], bottle_neck[1]),
                                        nn.Linear(bottle_neck[1], bottle_neck[2])
                                        )

        self._initialize_weights()

        self.crpt_func = corrupt_func
        self.crpt_lv = corrupt_level

    def forward(self, x, corrupt=True):
        if self.crpt_lv:
            x = _corrupt(x, self.crpt_lv, self.crpt_func)
        x = V(x)

        sizes = []
        idx_ = []
        for _, l in enumerate(self.encode_net):
            sizes.append(x.size())
            if isinstance(l, nn.MaxPool2d):
                x, idx = l(x)
            else:
                x = l(x)
                idx = None
            idx_.append(idx)

        conv_shape = x.size()
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = x.view(conv_shape)

        sizes = sizes[::-1]
        idx_ = idx_[::-1]
        for _, l in enumerate(self.decode_net):
            if isinstance(l, nn.ConvTranspose2d):
                x = l(x, output_size=sizes[_])
            elif isinstance(l, nn.MaxUnpool2d):
                x = l(x, output_size=sizes[_], indices=idx_[_])
            else:
                x = l(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    @staticmethod
    def _make_encode_layers(encode_cfg, in_channel=7):
        layers_ = []
        for l in encode_cfg:
            if l == 'M3':
                layer = nn.MaxPool2d(kernel_size=3, stride=3, padding=1, return_indices=True)
                layers_.append(layer)
            elif l == 'M1':
                layer = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, return_indices=True)
                layers_.append(layer)
            else:
                assert '/' in l
                out_channel, kernel = l.split('/')
                out_channel, kernel = int(out_channel), int(kernel)
                layers_ += [nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                      kernel_size=kernel, stride=1, padding=kernel // 2),
                            nn.BatchNorm2d(out_channel),
                            nn.ReLU()
                            ]
                in_channel = out_channel
        layers_ = nn.Sequential(*layers_)
        return layers_

    @staticmethod
    def _make_decode_layers(decode_cfg, in_channel=7):
        layers_ = []
        for l in decode_cfg:
            if l == 'M3':
                layer = nn.MaxUnpool2d(kernel_size=3, stride=3, padding=1)
                layers_.append(layer)
            elif l == 'M1':
                layer = nn.MaxUnpool2d(kernel_size=3, stride=1, padding=1)
                layers_.append(layer)
            else:
                assert '/' in l
                out_channel, kernel = l.split('/')
                out_channel, kernel = int(out_channel), int(kernel)
                layers_ += [nn.ConvTranspose2d(in_channels=out_channel, out_channels=in_channel,
                                               kernel_size=kernel, stride=1, padding=kernel // 2),
                            nn.BatchNorm2d(out_channel),
                            nn.ReLU()
                            ]
                in_channel = out_channel
        layers_ = layers_[::-1]
        layers_ = nn.Sequential(*layers_)
        return layers_


class Classifier(nn.Module):
    def __init__(self,
                 encoder_net,
                 clf_cfg):
        """
        :param encoder_net: object, trained encoder net
        :param clf_cfg: the architecture of classifier
        """
        super(Classifier, self).__init__()

        assert isinstance(encoder_net, CDAE)
        self.encoder = encoder_net
        self.encoder.freeze()
        self.geo = nn.Sequential(nn.Linear(12, 12),
                                 nn.BatchNorm1d(12),
                                 nn.ReLU())

        self.classifier = self.make_classifier(clf_cfg)

    def forward(self, x_geo, x_sat):
        if self.geo is not None:
            x_geo = self.geo(x_geo)
        for _, l in enumerate(self.encoder.encode_net):
            if isinstance(l, nn.MaxPool2d):
                x_sat, _ = l(x_sat)
            else:
                x_sat = l(x_sat)
        x_sat.detach_()
        x_sat = x_sat.view(x_sat.size(0), -1)
        x = torch.cat((x_geo, x_sat), dim=1)
        outp = self.classifier(x)
        return outp

    @staticmethod
    def make_classifier(classifier_cfg):
        layers_ = []
        for c1, c2 in zip(classifier_cfg, classifier_cfg[1:]):
            layers_ += [nn.Linear(c1, c2),
                        nn.BatchNorm1d(c2),
                        nn.ReLU()]
        layers_.append(nn.Linear(classifier_cfg[-1], 1))
        layers_ = nn.Sequential(*layers_)
        return layers_
