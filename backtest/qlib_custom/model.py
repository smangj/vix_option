#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 16:51
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.contrib.model import LinearModel
import statsmodels.api as sm


class SmLinearModel(LinearModel):
    def __init__(self, estimator="ols", alpha=0.0, fit_intercept=False):
        super().__init__(estimator, alpha, fit_intercept)
        self.summary = None

    def _fit(self, X, y, w):
        assert w is None
        if self.fit_intercept:
            X = sm.add_constant(X)
        if self.estimator == self.OLS:
            model = sm.OLS(y, X)
        else:
            raise NotImplementedError
        result = model.fit()

        self.coef_ = result.params[1:]
        self.intercept_ = result.params[0]
        self.summary = str(result.summary())
