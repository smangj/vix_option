#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/16 11:07
# @Author   : wsy
# @email    : 631535207@qq.com
from data_processor import DataProcessor

RAW_DATA = DataProcessor()


def test_end_date():
    se = RAW_DATA.end_date()

    assert se is not True
