import os
from pathlib import Path

import pandas as pd

import optuna
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import qlib
import yaml
from qlib.config import C
from qlib.data.dataset import Dataset
from qlib.utils import init_instance_by_config
from qlib.workflow.cli import sys_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
from matplotlib import pyplot as plt
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import scienceplots


def op_data(name: str = "linear_xt_one_for_test.yaml", uri_folder="mlruns"):
    file_path = os.path.join("../yaml_config", name)
    with open(file_path) as fp:
        config = yaml.safe_load(fp)
        # config the `sys` section
    sys_config(config, name)
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)
    dataset: Dataset = init_instance_by_config(config.get("task")["dataset"], accept_types=Dataset)
    optuna_data = pd.DataFrame(dataset.prepare(segments=['2000-01-01', '2023-01-01']))
    return optuna_data


#def objective_GRU(trial):


def objective_xgboost_reg(trial):
    optuna_data = op_data().dropna()
    data = optuna_data.iloc[:, :-1]
    target = optuna_data.iloc[:, -1:]
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)
    param = {
        'tree_method': 'gpu_hist',
        # this parameter means using the GPU when training our model to speed up the training process
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': 4000,
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBRegressor(**param)
    # model = xgb.XGBRegressor(tree_method='gpu_hist')
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=30)
    preds = model.predict(test_x)
    rmse = mean_squared_error(test_y, preds, squared=False)
    # rmse = sum(-1 * (preds > 0).astype('float64') * test_y)  # pred to better than index but worse than index
    return rmse


def xgb_features_importance():
    optuna_data = op_data().dropna()
    data = optuna_data.iloc[:, :-1]
    target = optuna_data.iloc[:, -1:]
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)
    model = xgb.XGBRegressor(reg_lambda=0.006079078255144172, reg_alpha=0.059870850402821384,
                             colsample_bytree=0.8, subsample=0.4, learning_rate=0.018,
                             max_depth=17, random_state=2020, min_child_weight=2)
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=30)
    plt.figure(dpi=1000, layout='tight')
    plt.style.use('science')
    result = plot_importance(model)
    plt.show()

    return result, model.feature_importances_


if __name__ == "__main__":
    # feature_sel = xgb_features_importance()
    # print(feature_sel)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_xgboost_reg, n_trials=100, timeout=600)
    plot_optimization_history(study).show()
    plot_intermediate_values(study).show()
    plot_param_importances(study).show()
    plot_parallel_coordinate(study).show()
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

## Number of finished trials:  86
# Best trial:
# Value: 0.04630284383893013
# Params:
# lambda: 0.006079078255144172
# alpha: 0.059870850402821384
# colsample_bytree: 0.8
# subsample: 0.4
# learning_rate: 0.018
# max_depth: 17
# random_state: 2020
# min_child_weight: 2
