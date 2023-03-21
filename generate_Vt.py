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
import typing


def generate_vt(
    const_maturity_list: typing.Union[typing.List[int], int]
) -> pd.DataFrame:

    if isinstance(const_maturity_list, int):
        const_maturity_list = [const_maturity_list]

    const_maturity_list = np.sort(const_maturity_list)

    data = DataProcessor()

    date_view = data.date_view()
    # np.argwhere(date_view.keys())

    result = []
    for k, v in date_view.items():

        price_dict = {"date": k}
        for const_maturity in const_maturity_list:
            if len(v) <= 1:
                price = np.nan
            else:
                end_date_index = list(date_view.keys()).index(k) + const_maturity
                if end_date_index < len(date_view):
                    end_date = to_pydatetime(list(date_view.keys())[end_date_index])
                else:
                    end_date = to_pydatetime(k) + dt.timedelta(int(const_maturity))
                near_sort_index = np.argsort(
                    [abs((to_pydatetime(x.expiration) - end_date).days) for x in v]
                )
                # 到期日实际是expiration - 1那一天
                weight_0 = (
                    (end_date - to_pydatetime(v[near_sort_index[0]].expiration)).days
                    + 1
                ) / (
                    to_pydatetime(v[near_sort_index[1]].expiration)
                    - to_pydatetime(v[near_sort_index[0]].expiration)
                ).days
                price = v[near_sort_index[0]].close * weight_0 + v[
                    near_sort_index[1]
                ].close * (1 - weight_0)
            price_dict[str(const_maturity) + "_DAYS"] = price
        result.append(price_dict)

    return pd.DataFrame(result)


def main():
    result = generate_vt([21, 42, 63, 84, 105, 126])
    result.to_csv("1-6M_trading_days.csv")


if __name__ == "__main__":
    main()
