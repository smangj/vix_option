#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/16 10:59
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
import datetime as dt
import numpy as np
from data_processor import DataProcessor
from utils.time import to_pydatetime


def generate_vt(const_maturity: int) -> pd.DataFrame:
    data = DataProcessor()

    date_view = data.date_view()

    result = []
    for k, v in date_view.items():

        if len(v) == 0:
            result.append({"date": k, "price": np.nan})
        elif len(v) == 1:
            """假设vix time structure是平的"""
            result.append({"date": k, "price": v[0].close})
        else:
            end_date = to_pydatetime(k) + dt.timedelta(const_maturity)
            near_sort_index = np.argsort(
                [abs((to_pydatetime(x.expiration) - end_date).days) for x in v]
            )
            weight_0 = (
                end_date - to_pydatetime(v[near_sort_index[0]].expiration)
            ).days / (
                to_pydatetime(v[near_sort_index[1]].expiration)
                - to_pydatetime(v[near_sort_index[0]].expiration)
            ).days
            price = v[near_sort_index[0]].close * weight_0 + v[
                near_sort_index[1]
            ].close * (1 - weight_0)
            result.append({"date": k, "price": price})

    return pd.DataFrame(result)
