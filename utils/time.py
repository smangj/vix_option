#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/17 18:30
# @Author   : wsy
# @email    : 631535207@qq.com
import functools
import typing
import datetime as dt
import pandas as pd
import time


def to_pydatetime(
    src_dt: typing.Union[dt.datetime, dt.date, str, pd.Timestamp, None]
) -> typing.Optional[dt.datetime]:
    dst_dt = None if src_dt is None else pd.to_datetime(src_dt).to_pydatetime()
    return dst_dt


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"***************函数{func.__name__} 运行时间为{run_time: .2f}秒************")

        return result

    return wrapper
