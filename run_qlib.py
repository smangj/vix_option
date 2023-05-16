#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/21 16:43
# @Author   : wsy
# @email    : 631535207@qq.com
import os
import fire
from qlib.workflow.cli import workflow


def main(name: str = "gru_xt_one_for_test.yaml"):
    workflow(os.path.join("yaml_config", name), experiment_name=name.split(".")[0])


if __name__ == "__main__":
    fire.Fire(main)
