#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/16 11:03
# @Author   : wsy
# @email    : 631535207@qq.com
from dataclasses import dataclass
import pandas as pd


@dataclass
class OneGIFutInfo:
    symbol: str
    expiration: str
    close: float


class DataProcessor:
    def __init__(self):
        self.df1 = pd.read_feather("./data/allVIXinfo.feather")
        self.df2 = pd.read_feather("./data/GIFutprices.feather")
        self.df3 = pd.read_feather("./data/indexdata.feather")

        self.data_clean()

    def end_date(self) -> pd.Series:
        result = self.df2.set_index("Symbol")["Expiration"].drop_duplicates()
        return result

    def date_view(self) -> dict:
        result = {}
        groups = self.df2.groupby("Date")
        for date, group in groups:
            result[date.strftime("%Y-%m-%d")] = [
                OneGIFutInfo(
                    symbol=record["Symbol"],
                    expiration=record["Expiration"].strftime("%Y-%m-%d"),
                    close=record["SettlementPrice"],
                )
                for index, record in group.iterrows()
            ]

        return result

    def data_clean(self):
        # 去掉时区
        self.df2["Expiration"] = self.df2["Expiration"].dt.tz_convert(None)
        self.df2["Date"] = self.df2["Date"].dt.tz_convert(None)

    def generate_trade_price(self):
        array = ["SPVXSP", "SPVIX2ME", "SPVIX3ME", "SPVIX4ME", "SPVXMP", "SPVIX6ME"]
        raw = self.df1.loc[self.df1["ID"].isin(array)]
        pivot = raw.pivot(index="Date", columns="ID", values="Close")
        result = pivot[array]
        return result
