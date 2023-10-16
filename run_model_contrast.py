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

from backtest.dynamic_pipeline import RollingBenchmark


def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/workflow/cli.py`
    """
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # config the `sys` section
    sys_config(config, config_path)
    if "experiment_name" in config:
        experiment_name = config["experiment_name"]

    if "exp_manager" in config.get("qlib_init"):
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
        exp = config["qlib_init"].get("exp_manager")
        if exp["kwargs"]["uri"] == "databricks":
            config["experiment_name"] = "/Users/631535207@qq.com/" + experiment_name
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

    roll_config = config.get("roll_config")
    if roll_config is None:
        for task in config.get("task"):
            recorder = task_train(task, experiment_name=experiment_name)
            recorder.save_objects(config=config)
    else:
        RollingBenchmark(config).run_all()


def main(name: str = "models.yaml"):
    file_path = os.path.join("model_contrast_config", name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    else:
        workflow(file_path, experiment_name=name.split(".")[0])


if __name__ == "__main__":
    fire.Fire(main)
