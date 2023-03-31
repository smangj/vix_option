#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/31 17:07
# @Author   : wsy
# @email    : 631535207@qq.com
from data_process.generate_Vt import generate_xt


def simple_backtest():
    roll = 20
    xt = generate_xt()
    xt["vix_ma"] = xt["ln_VIX"].rolling(window=roll).mean()
    xt["vix_std"] = xt["ln_VIX"].rolling(window=roll).std()

    month = 1
    std_times = 2
    buy_signal = xt["ln_V" + str(month)] <= (xt["vix_ma"] - std_times * xt["vix_std"])
    sell_signal = xt["ln_V" + str(month)] >= (xt["vix_ma"] + std_times * xt["vix_std"])

    position_num = 1
    xt["signal"] = position_num * buy_signal - position_num * sell_signal
    print(xt)


simple_backtest()
