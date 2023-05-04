#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/21 16:43
# @Author   : wsy
# @email    : 631535207@qq.com
import os

from qlib.workflow.cli import workflow

name = "gru_xt.yaml"


def main():
    workflow(os.path.join("yaml_config", name), experiment_name="vix_data_test")


if __name__ == "__main__":
    main()
