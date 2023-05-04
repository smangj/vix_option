#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/4 16:16
# @Author   : wsy
# @email    : 631535207@qq.com
import os

import pandas as pd

from data_process.data_processor import DataProcessor
from data_process.generate_Vt import generate_xt
from qlib_scripts import dump_bin
from utils.path import check_and_mkdirs, PROJ_ROOT_DIR

csv_output_dir = str(PROJ_ROOT_DIR.joinpath("data/qlib_data_csv"))
qlib_dir = str(PROJ_ROOT_DIR.joinpath("data/qlib_data"))
check_and_mkdirs(csv_output_dir)
check_and_mkdirs(qlib_dir)


def etf_to_csv():
    etf_map = {
        "SPVXSP": "VIX_1M",
        "SPVIX2ME": "VIX_2M",
        "SPVIX3ME": "VIX_3M",
        "SPVIX4ME": "VIX_4M",
        "SPVXMP": "VIX_5M",
        "SPVIX6ME": "VIX_6M",
    }
    xt = (
        DataProcessor()
        .generate_trade_price()
        .fillna(method="bfill")
        .rename(columns=etf_map)
    )
    xt.index.name = "date"

    feature = generate_xt().fillna(method="bfill")

    for k, v in etf_map.items():
        security = xt[v].reset_index().dropna()
        security.loc[:, "open"] = security[v]
        security.loc[:, "close"] = security[v]
        security.loc[:, "high"] = security[v]
        security.loc[:, "low"] = security[v]
        security.loc[:, "volume"] = 10000
        security.loc[:, "factor"] = 1
        security = pd.merge(security, feature, left_on="date", right_index=True)
        security.to_csv(os.path.join(csv_output_dir, v + ".csv"), index=False)


def features_to_csv():
    xt = generate_xt()
    xt.index.name = "date"
    xt.reset_index().to_csv(os.path.join(csv_output_dir, "features.csv"), index=False)


def macro_to_csv():
    raw_data = DataProcessor()
    for id in ["SPY US Equity", "TLT US Equity"]:
        micro_data = raw_data.df3.loc[raw_data.df3["ID"] == id]
        micro_data.loc[:, "volume"] = 10000
        micro_data.loc[:, "factor"] = 1
        micro_data = micro_data.rename(
            columns={"Date": "date", "O": "open", "H": "high", "L": "low", "C": "close"}
        )[["date", "open", "high", "low", "close", "volume", "factor"]]
        micro_data.to_csv(
            os.path.join(csv_output_dir, id.split()[0] + ".csv"), index=False
        )


def csv_to_bin():
    dump_bin.DumpDataAll(
        csv_path=csv_output_dir, qlib_dir=qlib_dir, date_field_name="date"
    ).dump()


if __name__ == "__main__":
    csv_to_bin()
