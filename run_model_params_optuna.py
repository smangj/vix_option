#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/21 16:43
# @Author   : wsy
# @email    : 631535207@qq.com
import os
import fire
from qlib.workflow.cli import sys_config
import ruamel.yaml as yaml
import qlib
from qlib.config import C, SIMPLE_DATASET_CACHE
from pathlib import Path
from qlib.model.trainer import task_train
import optuna
import copy
import uuid
from backtest.dynamic_pipeline import RollingBenchmark

STORAGE_NAME = "sqlite:///data/optuna.db"
STUDY_NAME = "OPT"


def objective(trial, config, experiment_name, uid=""):
    conf = copy.deepcopy(config)
    model_params = conf["task"].get("model")
    if model_params is None:
        raise AttributeError("model info NOT in config!")

    roll_config = conf.get("roll_config")

    trials = []
    model_trial = conf["task"]["model"]["kwargs"]
    trials.append(model_trial)
    if roll_config is not None:
        trials.append(roll_config)
    TSDataset_trial = conf["task"]["dataset"]["kwargs"].get("step_len")
    if TSDataset_trial is not None:
        trials.append(conf["task"]["dataset"]["kwargs"])

    for tr in trials:
        for k, v in tr.items():
            if isinstance(v, list):
                if len(v) == 2:
                    tr[k] = trial.suggest_float(k, v[0], v[1])
                else:
                    tr[k] = trial.suggest_categorical(k, v)

    if roll_config is None:
        recorder = task_train(conf.get("task"), experiment_name=experiment_name)
    else:
        conf["roll_exp_name"] = conf["experiment_name"] + "_rolling_exp" + str(uid)
        recorder = RollingBenchmark(conf).run_all()

    recorder.save_objects(config=conf)

    metrics = recorder.list_metrics()
    obj = metrics["IC"]

    return obj


def workflow(
    config_path,
    experiment_name="workflow",
    uri_folder="mlruns",
    n_trials: int = 100,
    if_load_study=False,
):
    """
    :param config_path:
    :param experiment_name:
    :param uri_folder:
    :param n_trials:
    :param if_load_study: 提前create study，只需要load_study该参数可变更为True，主要STORAGE_NAME和STUDY_NAME匹配
    :return:
    """
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # config the `sys` section
    sys_config(config, config_path)
    if "experiment_name" not in config:
        config["experiment_name"] = experiment_name

    if "exp_manager" in config.get("qlib_init"):
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
        exp = config["qlib_init"].get("exp_manager")
        if exp["kwargs"]["uri"] == "databricks":
            config["experiment_name"] = (
                "/Users/631535207@qq.com/" + config["experiment_name"]
            )
        if exp["kwargs"]["uri"] == "http://192.168.16.11:5001":
            # 需要download_atifact需要，自建的mlflow_server
            os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.16.11:5001"
        qlib.init(**config.get("qlib_init"), dataset_cache=SIMPLE_DATASET_CACHE)
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(
            Path(os.getcwd()).resolve() / uri_folder
        )
        qlib.init(
            **config.get("qlib_init"),
            exp_manager=exp_manager,
            dataset_cache=SIMPLE_DATASET_CACHE
        )

    if if_load_study:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
        uid = uuid.uuid4().hex[:8]
    else:
        study = optuna.create_study(direction="maximize")
        uid = ""
    study.optimize(
        lambda x: objective(x, config, experiment_name, uid=uid), n_trials=n_trials
    )


def main(name: str = "XGB.yaml", n_trials: int = 100, if_load_study=False):
    file_path = os.path.join("model_optuna_config", name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    else:
        workflow(
            file_path,
            experiment_name=name.split(".")[0],
            n_trials=n_trials,
            if_load_study=if_load_study,
        )


if __name__ == "__main__":
    fire.Fire(main)
