#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/10/16 18:31
# @Author   : wsy
# @email    : 631535207@qq.com
import os
import typing
from enum import Enum
from pathlib import Path

import qlib
from qlib.config import SIMPLE_DATASET_CACHE, C
from qlib.model.trainer import task_train
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow.cli import sys_config
from ruamel import yaml as yaml

from backtest.dynamic_pipeline import RollingBenchmark


class MlflowType(Enum):
    LOCAL = (0, "local")
    DATABRICKS = (1, "databricks")
    OFFICE_LOCAL = (2, "office")
    CLOUD_DOCKER = (3, "cloud_docker")

    def __init__(self, value: int, name: str):
        super().__init__()
        self._value = value
        self._name = name

    @property
    def value(self) -> int:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_name(cls, name: str) -> typing.Optional["MlflowType"]:
        for obj in cls:
            if name == obj.name:
                return obj
        return None


def myworkflow(
    config_path, experiment_name="workflow", server_type: MlflowType = MlflowType.LOCAL
):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/workflow/cli.py`
    """
    with open(config_path) as fp:
        y = yaml.YAML(typ="safe", pure=True)
        config = y.load(fp)

    # config the `sys` section
    sys_config(config, config_path)
    if "experiment_name" not in config:
        config["experiment_name"] = experiment_name

    if server_type == MlflowType.LOCAL:
        uri_folder = "mlruns"
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(
            Path(os.getcwd()).resolve() / uri_folder
        )
    elif server_type == MlflowType.DATABRICKS:
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
        config["experiment_name"] = (
            "/Users/631535207@qq.com/" + config["experiment_name"]
        )
        exp_manager = {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": "databricks", "default_exp_name": "Experiment"},
        }
    elif server_type == MlflowType.OFFICE_LOCAL:
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
        exp_manager = {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "http://192.168.16.11:5001",
                "default_exp_name": "Experiment",
            },
        }
        os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.16.11:5001"
    elif server_type == MlflowType.CLOUD_DOCKER:
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
        exp_manager = {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "http://127.0.0.1:5001",
                "default_exp_name": "Experiment",
            },
        }
    else:
        raise NotImplementedError("unknown MlflowType")
    qlib.init(
        **config.get("qlib_init"),
        exp_manager=exp_manager,
        dataset_cache=SIMPLE_DATASET_CACHE
    )
    # if "exp_manager" in config.get("qlib_init"):
    #     os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
    #     exp = config["qlib_init"].get("exp_manager")
    #     if exp["kwargs"]["uri"] == "databricks":
    #         config["experiment_name"] = (
    #                 "/Users/631535207@qq.com/" + config["experiment_name"]
    #         )
    #     if exp["kwargs"]["uri"] == "http://192.168.16.11:5001":
    #         # 需要download_atifact需要，自建的mlflow_server
    #         os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.16.11:5001"
    #     qlib.init(**config.get("qlib_init"), dataset_cache=SIMPLE_DATASET_CACHE)
    # else:
    #     exp_manager = C["exp_manager"]
    #     exp_manager["kwargs"]["uri"] = "file:" + str(
    #         Path(os.getcwd()).resolve() / uri_folder
    #     )
    #     qlib.init(
    #         **config.get("qlib_init"),
    #         exp_manager=exp_manager,
    #         dataset_cache=SIMPLE_DATASET_CACHE
    #     )

    roll_config = config.get("roll_config")
    tasks = config.get("task")

    if roll_config is None:
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            recorder = task_train(task, experiment_name=experiment_name)
            recorder.save_objects(config=config)
    else:
        if isinstance(tasks, list):
            configs = []
            for task in tasks:
                conf = deepcopy_basic_type(config)
                conf["task"] = task
                configs.append(conf)
        else:
            configs = [config]
        for conf in configs:
            RollingBenchmark(conf).run_all()
