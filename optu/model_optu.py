#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/8/14 14:56
# @Author   : wsy
# @email    : 631535207@qq.com
import abc
import qlib
from qlib.utils import init_instance_by_config, auto_filter_kwargs
from qlib.data.dataset import Dataset
from qlib.model.base import Model
import ruamel.yaml as yaml
import os
import optuna

CONFIG_PATH = os.path.join("optu", "config.yaml")


class ModelOptu(abc.ABC):
    def __init__(self):
        with open(CONFIG_PATH) as fp:
            self.config = yaml.safe_load(fp)

        qlib.init(**self.config.get("qlib_init"))
        self.dataset: Dataset = init_instance_by_config(
            self.config.get("dataset"), accept_types=Dataset
        )

    def prepare_model(self, trial):
        raise NotImplementedError

    def objective(self, trial):

        model = self.prepare_model(trial)
        auto_filter_kwargs(model.fit)(self.dataset)

        preds = model.predict(self.dataset)
        y_test = self.dataset.prepare(segments="test", col_set="label").iloc[:, 0]
        # loss = (((preds.T - test_y.T) ** 4).sum(axis=1)) ** 0.25
        loss = self.loss(preds, y_test)
        # kurtosis_pred=kurtosis(preds)
        # rmse = mean_squared_error(test_y, preds, squared=False)
        # rmse = sum(-1 * (preds > 0).astype('float64') * test_y)  # pred to better than index but worse than index
        return loss

    def loss(self, pred, label):
        raise NotImplementedError


class XGBOptu(ModelOptu):
    def prepare_model(self, trial):
        param = self.config["model"]
        for k, v in param["kwargs"].items():
            if isinstance(v, list):
                if len(v) == 2:
                    param["kwargs"][k] = trial.suggest_float(k, v[0], v[1])
                else:
                    param["kwargs"][k] = trial.suggest_categorical(k, v)
        model: Model = init_instance_by_config(param, accept_types=Model)
        return model

    def loss(self, pred, label):
        return (((pred - label) ** 4).sum()) ** 0.25


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    optu = XGBOptu()
    study.optimize(optu.objective, n_trials=100, timeout=600)
    # plot_optimization_history(study).show()
    # plot_intermediate_values(study).show()
    # plot_param_importances(study).show()
    # plot_parallel_coordinate(study).show()
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
