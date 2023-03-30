#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "smangj"
__email__ = "631535207@qq.com"
import pandas as pd

df1 = pd.read_csv("1-6M_trading_days.csv")
df2 = pd.read_csv("./data/alldata.csv")

merge_df = pd.merge(df1, df2, on="Date")
days = [21, 42, 63, 84, 105, 126]
for i in range(6):
    merge_df["delta_v" + str(i + 1)] = (
        merge_df["VIX" + str(i + 1)] - merge_df[str(days[i]) + "_v"]
    )
    merge_df["delta_w" + str(i + 1)] = (
        merge_df["w" + str(i + 1)] - merge_df[str(days[i]) + "_w"]
    )
merge_df.to_csv("contrast.csv")
