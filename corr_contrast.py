#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'smangj'
__email__ = "631535207@qq.com"
import pandas as pd

df1 = pd.read_csv("1-6M_trading_days.csv")[["date", "20_DAYS", "40_DAYS", "60_DAYS", "80_DAYS", "100_DAYS", "120_DAYS"]]
df2 = pd.read_csv("./data/paperdata.csv")[["Date", "VIX1", "VIX2", "VIX3", "VIX4", "VIX5", "VIX6"]]

merge_df = pd.merge(df1, df2, left_on="date", right_on="Date")
merge_df.to_csv("corr_contrast.csv")
