#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/8/14 14:56
# @Author   : wsy
# @email    : 631535207@qq.com
import abc

import pandas as pd
import qlib
from qlib.utils import init_instance_by_config, auto_filter_kwargs
from qlib.data.dataset import Dataset
from qlib.model.base import Model
from datetime import datetime
import ruamel.yaml as yaml
import os
import copy
import optuna
from optuna.visualization import (
    plot_intermediate_values,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

from utils.path import check_and_mkdirs

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
        param = copy.deepcopy(self.config["model"])
        for k, v in param["kwargs"].items():
            if isinstance(v, list):
                if len(v) == 2:
                    param["kwargs"][k] = trial.suggest_float(k, v[0], v[1])
                    # a = "float"
                    # getattr(trial, 'trial.suggest_' + a)(k, v[0], v[1])
                else:
                    param["kwargs"][k] = trial.suggest_categorical(k, v)
        model: Model = init_instance_by_config(param, accept_types=Model)
        return model

    def loss(self, pred, label):
        return (((pred - label) ** 2).mean()) ** 0.5


if __name__ == "__main__":
    storage_name = "sqlite:///data/optuna.db"
    outputs_path = "outputs/optuna"
    study = optuna.create_study(direction="minimize", storage=storage_name)
    optu = XGBOptu()
    study.optimize(optu.objective, n_trials=100, timeout=600)
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    study_path = os.path.join(outputs_path, time + optu.__class__.__name__)
    check_and_mkdirs(study_path)
    # plot_optimization_history(study).show()
    # plot_intermediate_values(study).show()

    # plot_parallel_coordinate(study).show()
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    pd.Series(trial.params).to_csv(os.path.join(study_path, "best_trial.csv"))

    plot_param_importances(study).write_image(
        os.path.join(study_path, "importances.jpeg")
    )
    plot_optimization_history(study).write_image(
        os.path.join(study_path, "history.jpeg")
    )
    plot_intermediate_values(study).write_image(
        os.path.join(study_path, "intermediate_values.jpeg")
    )
    plot_parallel_coordinate(study).write_image(
        os.path.join(study_path, "parallel_coordinate.jpeg")
    )
