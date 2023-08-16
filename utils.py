#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/8/15 16:52
# @Author   : wsy
# @email    : 631535207@qq.com


def check_and_mkdirs(dir_path):
    if dir_path is None:
        return
    import os

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
