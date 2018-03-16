import logging
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable as V


class Run(object):
    def __init__(self, cuda, model):
        """
        :param cuda: boolean, enable cuda support or not
        :param model: torch.nn instance, the model that will be trianed
        """
        self.cuda = cuda
        self.model = model.cuda() if cuda else model
        self.optimizer = None

    def train(self, trail, seed,
              data_train,
              lr, step,
              lr_decay_freq=None,
              optimizer_name='RMSprop', **kwargs):
        loss_freq = 10

        # manually set seed
        if self.cuda:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        # optimizer
        self.optimizer = getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr, **kwargs)

        loss_train = 0.
        i = 0

        start = time.time()

        try:
            for x, y in data_train:
                self.model.zero_grad()

                loss = self.feed_forward(x, y,
                                         model=self.model,
                                         cuda=self.cuda)

                loss.backward()
                self.optimizer.step()

                loss_train += float(loss)

                # online evaluation with plots
                if i % loss_freq == 0 and i != 0:
                    print('\n\tStep: %s, time elapsed: %.2f min.' %
                                (i, (time.time() - start) / 60.))

                    # plot training loss
                    loss_train /= loss_freq
                    self.plot.update_metrics(i, indicator=loss_train)
                    # re-initialize training loss
                    loss_train = 0.

                if lr_decay_freq and i % lr_decay_freq == 0:
                    Run.adjust_lr(init_lr=lr, optimizer=self.optimizer, step=i, decay_freq=lr_decay_freq)

                if i < step:
                    i += 1
                else:
                    break
        except KeyboardInterrupt:
            print('\nStop by the modeler at step: %s' % i)

    def perf_eval(self, data_generator):
        """Perform prediction by stacking predictions of data slices"""
        assert hasattr(data_generator, '__iter__')

        outp_train_ = []
        outp_test_ = []

        i = 0
        print('\nStart Predicting')
        for x_train, x_test in data_generator:
            if i % 500 == 0:
                print('\tNo.', i)
            i += 1

            x_train = V(x_train)
            x_test = V(x_test)
            if self.cuda:
                x_train = x_train.cuda()
                x_test = x_test.cuda()

            for _, l in enumerate(self.model.encode_net):
                if isinstance(l, nn.MaxPool2d):
                    x_train, _ = l(x_train)
                    x_test, _ = l(x_test)
                else:
                    x_train = l(x_train)
                    x_test = l(x_test)

            outp_train_.append(x_train.data.cpu().numpy())
            outp_test_.append(x_test.data.cpu().numpy())

        outp_train_ = np.concatenate(outp_train_, axis=0)
        outp_test_ = np.concatenate(outp_test_, axis=0)

        mse = np.mean((outp_train_ - outp_test_) ** 2)

        return mse

    @staticmethod
    def feed_forward(x, y, model, cuda):
        y = V(y)

        model.train()
        if cuda:
            y, x = y.cuda(), x.cuda()
        outp = model(x)

        criterion = torch.nn.MSELoss()
        loss = criterion(outp, y)

        return loss

    @staticmethod
    def adjust_lr(init_lr, optimizer, step, decay_freq):
        """sets the lr the initial lr decayed by 10 every certain steps"""
        lr = init_lr * (.1 ** (step // decay_freq))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
