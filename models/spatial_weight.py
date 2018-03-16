import math
from itertools import product
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SpatialWeight(nn.Module):
    def __init__(self, dist_map, plane):
        """
        :param dist_map: 2d tensor, represent the distance-weighted membership function
        :param plane: int, the number of plane(channels), for initializing parameters
        """
        super(SpatialWeight, self).__init__()

        self.exp_a = Parameter(torch.Tensor(plane, 1, 1))
        self.exp_b = Parameter(torch.Tensor(plane, 1, 1))
        self._init_parameter()

        self.dm = dist_map

    def _init_parameter(self):
        self.exp_a.data.fill_(.85)
        self.exp_b.data.fill_(.1)

    def forward(self, inp):
        if inp.size()[-2:] != self.dm.size():
            # feat_map_radius
            r = inp.size(-1) // 2
            # dist_map_central
            c = self.dm.size(0) // 2
            weight = self.dm[(c - r):(c + r + 1), (c - r):(c + r + 1)]
        else:
            weight = self.dm

        weight = weight.expand(inp.size()[1:])
        weight = torch.autograd.Variable(weight)
        if inp.is_cuda:
            weight = weight.cuda()

        out = inp * torch.exp(torch.log(self.exp_a) * self.exp_b * weight)

        return out

    @staticmethod
    def _exp(x, a, b):
        if isinstance(a, float):
            return torch.exp(math.log(a) * b * x)
        else:
            return torch.exp(torch.log(a) * b * x)


def dist_func(image_shape=(27, 27)):
    assert image_shape[0] % 2 == 1 and image_shape[1] % 2 == 1
    diameter = image_shape[0]
    longitude = torch.arange(0, diameter).expand(image_shape).contiguous()
    latitude = longitude.t().contiguous()
    coord = torch.cat((longitude.view(-1, 1), latitude.view(-1, 1)), 1)
    central = torch.Tensor([diameter // 2, diameter // 2]).expand_as(coord)
    dist = F.pairwise_distance(coord, central, p=2)
    dist = dist.view(image_shape)
    return dist
