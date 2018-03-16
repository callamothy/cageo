from torch import nn


class GeoNets(nn.Module):
    def __init__(self, n_hidden, n_layers, n_inp=12, n_outp=1,
                 dropout=.5, nl_func='ReLU'):
        super(GeoNets, self).__init__()

        nets = list()

        # head
        nets.append(nn.Linear(n_inp, n_hidden))
        nets.append(getattr(nn, nl_func)())
        if dropout:
            nets.append(nn.Dropout(dropout))

        # body
        for n in range(n_layers):
            nets.append(nn.Linear(n_hidden, n_hidden))
            nets.append(nn.BatchNorm1d(n_hidden))
            nets.append(getattr(nn, nl_func)())
            if dropout:
                nets.append(nn.Dropout(dropout))

        # tail
        nets.append(nn.Linear(n_hidden, n_outp))

        # assemble
        self.nets = nn.Sequential(*nets)

        # initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inp):
        y = self.nets(inp)
        return y

