#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/1/5 16:06
# @Author   : wsy
# @email    : 631535207@qq.com
import pytest
import qlib
from qlib.config import C
import os
from pathlib import Path

URI_FOLDER = "mlruns_test"


@pytest.fixture()
def qlib_init():
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(
        Path(os.getcwd()).resolve() / URI_FOLDER
    )
    qlib.init(provider_uri="data/qlib_data", region="us", exp_manager=exp_manager)
