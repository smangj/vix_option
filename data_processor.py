#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/16 11:03
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd


class DataProcessor:
    def __init__(self):
        self.df1 = pd.read_feather("./data/allVIXinfo.feather")
        self.df2 = pd.read_feather("./data/GIFutprices.feather")

    def end_date(self) -> pd.Series:
        result = self.df2.set_index("SecurityID")["Expiration"].drop_duplicates()
        return result
