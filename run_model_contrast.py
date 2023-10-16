#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/21 16:43
# @Author   : wsy
# @email    : 631535207@qq.com
import os
import fire
from backtest.qlib_custom.pipeline import myworkflow


def main(name: str = "GRU_XGB_LINEAR.yaml"):
    file_path = os.path.join("model_contrast_config", name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    else:
        myworkflow(file_path, experiment_name=name.split(".")[0])


if __name__ == "__main__":
    fire.Fire(main)
