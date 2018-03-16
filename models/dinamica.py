from random import gauss, lognormvariate
import time
import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


logger = logging.getLogger('dinamica')
logging.basicConfig(level=logging.INFO)


class Dinamica(object):
    def __init__(self, prob, lu, tran_quantity, patch_expand_ratio, isometry, duration, map_shape):
        """
        :param prob: pd.Series with 1-d index, the transition probability
        :param lu: pd.Series with 1-d index, the actual LU category
        :param tran_quantity: int, the pre-set transition quantity
        :param patch_expand_ratio: float, the respective transition quantities by patcher and expander functions
        :param isometry: float, controls the pattern of simulation
        :param duration: int, the number of simulation years
        :param map_shape: 2-d tuple, the map shape
        """
        # environment attribute
        self.quantity = tran_quantity
        self.ratio = patch_expand_ratio
        self.isometry = isometry
        self.duration = duration
        self.map_shape = map_shape

        self.epoch = None

        # ! INITIALIZATION
        # base, update each duration in patcher/expander function
        assert isinstance(prob, pd.core.series.Series)
        assert isinstance(lu, pd.core.series.Series)
        self.prob = prob
        self.lu = lu
        self.raw_idx = prob.index.values

        self.patch = None
        self.expand = None

        self._update()

        # results
        self.tran_expand = None
        self.tran_patch = None
        self.pred = pd.Series(dtype=str)

        logger.info('Patcher has %s. Expander has %s.' % (self.patch.shape[0], self.expand.shape[0]))

    def run(self, patcher_args, expander_args):
        patch_amount = int(self.quantity * self.ratio / self.duration)
        expand_amount = int(self.quantity * (1. - self.ratio) / self.duration)

        p_mu, p_sigma = patcher_args
        threshold, e_mu, e_sigma = expander_args

        for self.epoch in range(self.duration):
            self._patcher(patch_amount, p_mu, p_sigma)
            self._expander(expand_amount, threshold, e_mu, e_sigma)
            self._update()

        lu_expand = self.pred[self.pred == 'expand']
        lu_patch = self.pred[self.pred == 'patch']

        self.tran_expand = lu_expand.shape[0]
        self.tran_patch = lu_patch.shape[0]

        transition = pd.Series(0, index=self.raw_idx)
        transition[lu_expand.index] = 1
        transition[lu_patch.index] = 1

        transition.sort_index(inplace=True)

        logger.info('Number of patch and expand, %s and %s' % (self.tran_patch, self.tran_expand))

        return transition

    def _update(self):
        if self.patch is not None and self.expand is not None:
            self.prob = pd.concat((self.patch, self.expand))

        # # based on base attributes, update each duration in _classify function
        prob_coord = np.array(np.unravel_index(self.prob.index.values, self.map_shape)).T
        lu_coord = np.array(np.unravel_index(self.lu.index.values, self.map_shape)).T

        lu_tree = cKDTree(lu_coord)

        # obtain the filter
        try:
            is_patch = map(lambda x: len(x) == 0, lu_tree.query_ball_point(prob_coord, r=1, p=np.inf))
            # filtering
            is_patch = np.array(is_patch)
            self.patch = self.prob[is_patch]
            self.expand = self.prob[~is_patch]
        except IndexError:
            self.patch = pd.Series()
            self.expand = self.prob

    def _patcher(self, quantity, mu, sigma):
        assert isinstance(self.patch, pd.core.series.Series)

        if self.patch.shape[0] > 0:

            patch_coord = np.array(np.unravel_index(self.patch.index.values, self.map_shape)).T

            patch = self.patch.copy()
            patch = patch.to_frame('prob')
            patch['x'] = patch_coord[:, 0]
            patch['y'] = patch_coord[:, 1]
            patch.sort_values('prob', ascending=False, inplace=True)

            cum_num = 0
            for x, y in patch[['x', 'y']].values:
                patch_tree = cKDTree(patch[['x', 'y']].values)

                r = lognormvariate(mu, sigma)

                new_index = patch_tree.query_ball_point([x, y], r=r, p=2)
                new_points = patch.iloc[new_index, 0]
                new_points_idx = self._nbr_sim(new_points, np.array([x, y]), r)

                # update
                patch.drop(new_points_idx, inplace=True)
                self.patch.drop(new_points_idx, inplace=True)
                self.pred = self.pred.append(new_points)

                cum_num += len(new_index)

                if cum_num >= quantity:
                    break
                elif patch.shape[0] == 0:
                    break
        else:
            logger.warning('Epoch %s, zero patch' % self.epoch)

    def _expander(self, quantity, threshold, mu, sigma):
        assert isinstance(self.expand, pd.core.series.Series)

        if self.expand.shape[0] > 0:
            expand_coord = np.array(np.unravel_index(self.expand.index.values, self.map_shape)).T
            lu_coord = np.array(np.unravel_index(self.lu.index.values, self.map_shape)).T

            lu_tree = cKDTree(lu_coord)
            is_core = map(lambda x: len(x), lu_tree.query_ball_point(expand_coord, r=1, p=np.inf))
            is_core = np.array(is_core)

            expand = self.expand.copy()
            f = (is_core <= 3) & (expand.values < threshold)
            expand[f] *= np.sqrt(is_core[f]) / np.sqrt(4.)

            expand_coord = np.array(np.unravel_index(expand.index.values, self.map_shape)).T

            expand = expand.to_frame('prob')
            expand['x'] = expand_coord[:, 0]
            expand['y'] = expand_coord[:, 1]
            expand.sort_values('prob', ascending=False, inplace=True)

            cum_num = 0
            for p, x, y in expand[['prob', 'x', 'y']].values:
                expand_tree = cKDTree(expand[['x', 'y']].values)

                r = lognormvariate(mu, sigma)

                if p > threshold:
                    new_index = expand_tree.query_ball_point([x, y], r=r, p=np.inf)
                else:
                    new_index = expand_tree.query_ball_point([x, y], r=0, p=np.inf)

                new_points = expand.iloc[new_index, 0]
                new_points_idx = self._nbr_sim(new_points, np.array([x, y]), r)

                # update
                expand.drop(new_points_idx, inplace=True)
                self.expand.drop(new_points_idx, inplace=True)
                self.pred = self.pred.append(new_points)

                cum_num += len(new_index)

                if cum_num >= quantity:
                    break
                elif expand.shape[0] == 0:
                    break
        else:
            logger.warning('Epoch %s, zero patch' % self.epoch)

    def _nbr_sim(self, pool, center, r):
        select = []
        central_cell = center
        quant = 0
        x_y = [tuple(_) for _ in pool[['x', 'y']].values]
        while True:
            dist = self._dist(pool[['x', 'y']].values, central_cell)
            points = pool[dist < 2]
            idx = np.random.choice(points.index.values, 1, p=points['prob'].values / points['prob'].sum())
            point = tuple(*points.loc[idx, ['x', 'y']].values)
            select.append(point)
            pool.loc[pool.index[[_ == point for _ in x_y]], 'prob'] *= self.isometry

            central_cell = np.array(point)
            quant += 1
            if self._dist(central_cell, center) == r or quant == (r + 1) ** 2:
                break

        select = np.asarray(list(set(select)))

        return select

    @staticmethod
    def _dist(point_1, point_2):
        if point_1.ndim > 1 or point_2.ndim > 2:
            return np.sum((point_1 - point_2) ** 2, 1) ** .5
        else:
            return np.sum((point_1 - point_2) ** 2) ** .5
