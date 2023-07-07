#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/7/7 17:00
# @Author   : wsy
# @email    : 631535207@qq.com

import optuna


def objective(trial):
    sharp = 0

    w = trial.suggest_float("short_strategy_weight", 0, 1)
    assert w

    return sharp


study = optuna.create_study()
study.optimize(objective, n_trials=100)
