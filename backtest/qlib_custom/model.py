#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 16:51
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.contrib.model import LinearModel, XGBModel
import statsmodels.api as sm
import pandas as pd
import xgboost as xgb
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from typing import Text, Union


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


class XgbFix(XGBModel):
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(
            self.model.predict(xgb.DMatrix(x_test.values)), index=x_test.index
        )
