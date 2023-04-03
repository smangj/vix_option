#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/31 17:07
# @Author   : wsy
# @email    : 631535207@qq.com
import numpy as np
import pandas as pd

from data_process.generate_Vt import generate_xt


def simple_backtest():
    roll = 20
    xt = generate_xt()
    xt["vix_ma"] = xt["ln_VIX"].rolling(window=roll).mean()
    xt["vix_std"] = xt["ln_VIX"].rolling(window=roll).std()

    std_times = 2
    position_num = 1
    net_values = []
    for month in range(1, 7):

        buy_signal = xt["ln_V" + str(month)] <= (
            xt["vix_ma"] - std_times * xt["vix_std"]
        )
        sell_signal = xt["ln_V" + str(month)] >= (
            xt["vix_ma"] + std_times * xt["vix_std"]
        )

        xt["signal" + str(month)] = (
            position_num * buy_signal - position_num * sell_signal
        )

        # 简单规则：遇到异号signal变position
        xt["position" + str(month)] = (
            xt["signal" + str(month)]
            .replace(0, np.nan)
            .fillna(method="ffill")
            .replace(np.nan, 0)
        )
        xt["trading_vol" + str(month)] = (
            xt["position" + str(month)] - xt["position" + str(month)].shift(1)
        ).fillna(0)
        xt["trading_price" + str(month)] = xt["ln_V" + str(month)]

        xt["equity" + str(month)] = (
            xt["position" + str(month)] * xt["ln_V" + str(month)]
        )

        # cash能cover住
        init_equity = abs(xt["equity" + str(month)]).max()

        cash = []
        for i in range(len(xt)):
            if i == 0:
                cash.append(init_equity)
            else:
                this_cash = (
                    cash[-1]
                    - xt["trading_vol" + str(month)][i]
                    * xt["trading_price" + str(month)][i]
                )
                cash.append(this_cash)
        xt["cash" + str(month)] = cash

        xt["total_equity" + str(month)] = (
            xt["cash" + str(month)] + xt["equity" + str(month)]
        )

        net_value = (xt["total_equity" + str(month)] / init_equity).rename(
            "net_value" + str(month)
        )
        net_values.append(net_value)
    result = pd.concat(net_values, axis=1)
    result.to_csv("vix_reversion_strategy.csv")
    return result


simple_backtest()
