#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/21 16:43
# @Author   : wsy
# @email    : 631535207@qq.com
import os
import fire
from qlib.workflow.cli import workflow

from backtest.dynamic_pipeline import dynamicworkflow

LEGAL_WORKFLOW = ["origin", "dynamic"]


def main(name: str = "linear_dynamic_simpleback.yaml", work_flow: str = "dynamic"):
    assert work_flow in LEGAL_WORKFLOW
    if work_flow == "origin":
        func = workflow

    elif work_flow == "dynamic":
        func = dynamicworkflow
    else:
        raise NotImplementedError

    file_path = os.path.join("yaml_config", name)
    if os.path.isfile(file_path):
        func(file_path, experiment_name=name.split(".")[0])
    elif os.path.isdir(file_path):
        for file in os.listdir(file_path):
            func(os.path.join(file_path, file), experiment_name=file.split(".")[0])
    else:
        raise NotImplementedError


if __name__ == "__main__":
    fire.Fire(main)
