#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/7/7 17:00
# @Author   : wsy
# @email    : 631535207@qq.com

import optuna
import os
from pathlib import Path
import fire
import pandas as pd
import qlib
from qlib.workflow import R
from qlib.config import C
from qlib.workflow.task.collect import RecorderCollector
from empyrical.stats import sharpe_ratio

from backtest.qlib_custom._dfbacktest import LongShortBacktest


def recorder_for_longshortback(experiment_name: str, recorder_id: str = None):
    """根据recorder中的pred.pkl和label.pkl做回测并返回longshortback回测结果"""

    if recorder_id is None:
        r = RecorderCollector(experiment=experiment_name)
        assert len(r.experiment.info["recorders"]) == 1
        recorder = R.get_recorder(experiment_name=experiment_name)
    else:
        recorder = R.get_recorder(
            experiment_name=experiment_name, recorder_id=recorder_id
        )

    pred = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl").dropna()
    label_df.columns = ["label"]

    dt_values = pred.index.get_level_values("datetime")

    start_time = dt_values[0]
    end_time = dt_values[-1]
    time_mask = (dt_values >= pd.to_datetime(start_time)) & (
        dt_values <= pd.to_datetime(end_time)
    )
    pred = pred.loc[time_mask]

    return pred


def objective(trial, experiment_name, recorder_id: str = None):
    pred = recorder_for_longshortback(experiment_name, recorder_id)

    w = trial.suggest_float("short_strategy_weight", 0, 1)
    back = LongShortBacktest(tabular_df=pred, topk=1, long_weight=(1.0 - w) / 2)
    result = back.run_backtest(
        freq="day",
        shift=1,
        open_cost=0,
        close_cost=0,
        min_cost=0,
    )

    sharp = sharpe_ratio(result["long_short"])

    return sharp


def main(
    experiment_name: str = "SmLinearModel_GroupVixHandler_LongShortBacktestRecord",
    uri_folder="k-means",
):
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(
        Path(os.getcwd()).resolve() / uri_folder
    )
    qlib.init(provider_uri="data/qlib_data", exp_manager=exp_manager)

    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db")
    study.optimize(
        lambda trial: objective(trial=trial, experiment_name=experiment_name),
        n_trials=100,
        timeout=600,
    )

    trial = study.best_trial
    for k, v in trial.params.items():
        print("{}:{}".format(k, v))


if __name__ == "__main__":
    fire.Fire(main)
