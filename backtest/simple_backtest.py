#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/31 17:07
# @Author   : wsy
# @email    : 631535207@qq.com
import numpy as np
import pandas as pd

from data_process.data_processor import DataProcessor
from data_process.generate_Vt import generate_xt


class SimpleBacktest:
    """基于Xt和ETF的矩阵回测"""

    def __init__(self):

        self.data = self._feature_and_etfprice()

    @staticmethod
    def _feature_and_etfprice() -> pd.DataFrame:
        etf_map = {
            "SPVXSP": "ETF1",
            "SPVIX2ME": "ETF2",
            "SPVIX3ME": "ETF3",
            "SPVIX4ME": "ETF4",
            "SPVXMP": "ETF5",
            "SPVIX6ME": "ETF6",
        }
        xt = generate_xt().fillna(method="bfill")
        trading_price = DataProcessor().generate_trade_price().fillna(method="bfill")
        xt = pd.merge(xt, trading_price, left_index=True, right_index=True).rename(
            columns=etf_map
        )
        return xt

    def _generate_signal(self, month) -> pd.DataFrame:

        xt = self.data.copy()
        roll = 20
        xt["vix_ma"] = xt["ln_VIX"].rolling(window=roll).mean()
        xt["vix_std"] = xt["ln_VIX"].rolling(window=roll).std()
        std_times = 2
        position_num = 1
        buy_signal = xt["ln_V" + str(month)] <= (
            xt["vix_ma"] - std_times * xt["vix_std"]
        )
        sell_signal = xt["ln_V" + str(month)] >= (
            xt["vix_ma"] + std_times * xt["vix_std"]
        )

        xt["signal"] = position_num * buy_signal - position_num * sell_signal
        return xt

    def run(self):

        net_values = []
        init_equity = 100000
        for month in range(1, 7):

            xt = self._generate_signal(month)

            # 简单规则：
            xt["trading_flag"] = (
                xt["signal"]
                .replace(0, np.nan)
                .fillna(method="ffill")
                .replace(np.nan, 0)
            )
            xt["trading_flag"] = (
                xt["trading_flag"] - xt["trading_flag"].shift(1)
            ).fillna(0)
            xt["trading_price"] = xt["ETF" + str(month)]

            cash = []
            position_list = []
            equity_list = []
            for i in range(len(xt)):
                if i == 0:
                    equity = init_equity
                else:
                    equity = cash[-1] + xt["trading_price"].iloc[i] * position_list[-1]
                equity_list.append(equity)

                if xt["trading_flag"].iloc[i] > 0:
                    position = equity / xt["trading_price"].iloc[i]
                elif xt["trading_flag"].iloc[i] < 0:
                    position = -equity / xt["trading_price"].iloc[i]
                else:
                    if i == 0:
                        position = 0
                    else:
                        position = position_list[-1]
                position_list.append(position)

                this_cash = equity - position * xt["trading_price"].iloc[i]
                cash.append(this_cash)
            xt["cash" + str(month)] = cash
            xt["total_equity" + str(month)] = equity_list
            xt["position" + str(month)] = position_list

            xt["equity" + str(month)] = (
                xt["position" + str(month)] * xt["trading_price"]
            )

            net_value = (xt["total_equity" + str(month)] / init_equity).rename(
                "net_value" + str(month)
            )
            net_values.append(net_value)
        result = pd.concat(net_values, axis=1)
        result.to_csv("vix_reversion_strategy.csv")
        return result


SimpleBacktest().run()
