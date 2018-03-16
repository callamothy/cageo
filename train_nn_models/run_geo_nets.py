import logging
import time
import random

import numpy as np
import torch
from torch.autograd import Variable as V
from torch.nn import functional as F
from sklearn import metrics


class Run(object):
    def __init__(self, model, cuda):
        self.cuda = cuda
        self.model = model.cuda() if cuda else model
        self.optimizer = None

    def train(self, trail, seed,
              data_train, data_cv,
              lr, step,
              gaussian_noise=None, lr_decay_freq=None,
              optimizer_name='RMSprop', **kwargs):
        loss_freq = 10
        eval_freq = loss_freq * 10

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
        pr = 0.
        i = 0

        start = time.time()

        try:
            for y, x_geo in data_train:
                self.model.zero_grad()

                _, loss = self.feed_forward(x_geo, y,
                                            model=self.model,
                                            cuda=self.cuda,
                                            mode='train')

                if gaussian_noise and gaussian_noise != 0:
                    for p in self.model.parameters():
                        noise = random.gauss(0, gaussian_noise / (1 + i) ** .55)
                        p.register_hook(lambda grad: grad + noise)
                loss.backward()

                self.optimizer.step()

                loss_train += float(loss)

                # online evaluation with plots
                if i % loss_freq == 0 and i != 0:
                    print('\n\tStep: %s, time elapsed: %.2f min.' %
                                (i, (time.time() - start) / 60.))

                    # plot training loss
                    loss_train /= loss_freq
                    # re-initialize training loss
                    loss_train = 0.

                    # evaluate cv loss
                    y_cv, x_geo_cv = data_cv.__next__()
                    outp_t, _ = self.feed_forward(x_geo_cv, y_cv,
                                                  model=self.model,
                                                  cuda=self.cuda,
                                                  mode='eval')
                    outp_t.detach_()
                    precision, recall, _ = metrics.precision_recall_curve(y_cv.view(-1), outp_t.data.view(-1))
                    auc_pr = metrics.auc(recall, precision)
                    pr += auc_pr

                if i % eval_freq == 0 and i != 0:
                    pr_t = pr / (eval_freq / loss_freq)
                    pr = 0

                if lr_decay_freq and i % lr_decay_freq == 0:
                    Run.adjust_lr(init_lr=lr, optimizer=self.optimizer, step=i, decay_freq=lr_decay_freq)

                if i < step:
                    i += 1
                else:
                    break
        except KeyboardInterrupt:
            print('\nStop by the modeler at step: %s' % i)

    def predict(self, data_generator):
        """Perform prediction by stacking predictions of data pieces"""
        assert hasattr(data_generator, '__iter__')

        # true label, end-lu
        y_true = []
        # prediction, simulated-lu
        prob = []

        i = 0
        print('\nStart Predicting')
        for y, x_geo in data_generator:
            if i % 500 == 0:
                print('\tNo.', i)
            i += 1

            outp, _ = self.feed_forward(x_geo, y,
                                        model=self.model, cuda=self.cuda,
                                        mode='eval')
            y_true.append(y.numpy())
            outp.detach_()
            outp = F.sigmoid(outp)
            prob.append(outp.data.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        prob = np.concatenate(prob, axis=0)

        return prob, y_true

    def perf_eval(self, data_generator, mode):
        """Evaluate model performance on a prepared data set"""
        assert mode in ['train', 'test']

        prob, y_true = self.predict(data_generator)

        return prob, y_true

    @staticmethod
    def feed_forward(x1, y, model, cuda, mode=None):
        if mode == 'train':
            model.train()
            y, x1 = V(y), V(x1)
            if cuda:
                y, x1 = y.cuda(), x1.cuda()
            outp = model(x1)
        elif mode == 'eval':
            model.eval()
            # warning: only version 0.4 has no_grad() to replace volatile
            with torch.no_grad():
                y, x1 = V(y), V(x1)
                if cuda:
                    y, x1 = y.cuda(), x1.cuda()
                outp = model(x1)
        else:
            raise Exception('Incorrect mode.')

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(outp, y)

        return outp, loss

    @staticmethod
    def adjust_lr(init_lr, optimizer, step, decay_freq):
        """sets the lr the initial lr decayed by 10 every certain steps"""
        lr = init_lr * (.9 ** (step // decay_freq))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
