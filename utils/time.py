#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/17 18:30
# @Author   : wsy
# @email    : 631535207@qq.com
import typing
import datetime as dt
import pandas as pd


def to_pydatetime(
    src_dt: typing.Union[dt.datetime, dt.date, str, pd.Timestamp, None]
) -> typing.Optional[dt.datetime]:
    dst_dt = None if src_dt is None else pd.to_datetime(src_dt).to_pydatetime()
    return dst_dt
