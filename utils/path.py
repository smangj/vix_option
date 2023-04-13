#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/10 17:04
# @Author   : wsy
# @email    : 631535207@qq.com

from pathlib import Path

# 根据当前文件的路径获取项目根目录
CUR_DIR = Path(__file__).resolve().parent
PROJ_ROOT_DIR = CUR_DIR.parent
DEFAULT_QLIB_PROVIDER_URI = str(PROJ_ROOT_DIR.joinpath("data/qlib_data"))


def check_and_mkdirs(dir_path):
    if dir_path is None:
        return
    import os

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
