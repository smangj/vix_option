#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/16 10:59
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
import numpy as np
from data_process.data_processor import DataProcessor
import typing


def generate_vt(
    const_maturity_list: typing.Union[typing.List[int], int]
) -> pd.DataFrame:
    if isinstance(const_maturity_list, int):
        const_maturity_list = [const_maturity_list]

    const_maturity_list = np.sort(const_maturity_list)

    data = DataProcessor()
    raw = data.df2.copy()
    # 筛选出month的数据
    month_contract_mask = raw["Symbol"].str.split().apply(lambda x: x[0] == "VX")
    raw = raw.loc[month_contract_mask]

    date_series = pd.Series(raw["Date"].unique()).sort_values()
    raw["Expiration_for_trade"] = raw["Expiration"].map(
        pd.Series(date_series.shift(1).values, index=date_series.values)
    )

    result = None
    for i in range(len(const_maturity_list)):
        map_series = pd.Series(
            date_series.shift(-const_maturity_list[i]).values, index=date_series.values
        )
        raw[str(const_maturity_list[i]) + "_end_date"] = raw["Date"].map(map_series)

        mask = ~raw[str(const_maturity_list[i]) + "_end_date"].isna()
        raw.loc[mask, str(const_maturity_list[i]) + "diff"] = (
            raw.loc[mask, "Expiration_for_trade"]
            - raw.loc[mask, str(const_maturity_list[i]) + "_end_date"]
        ).dt.days

        raw[str(const_maturity_list[i]) + "_rank_positive"] = (
            raw[raw[str(const_maturity_list[i]) + "diff"] >= 0]
            .groupby("Date")[str(const_maturity_list[i]) + "diff"]
            .rank(method="first")
        )
        raw[str(const_maturity_list[i]) + "_rank_negative"] = (
            raw[raw[str(const_maturity_list[i]) + "diff"] < 0]
            .groupby("Date")[str(const_maturity_list[i]) + "diff"]
            .rank(ascending=False, method="first")
        )

        # 选出距离最近的左右期货
        left = raw.loc[raw[str(const_maturity_list[i]) + "_rank_negative"] == 1]
        right = raw.loc[raw[str(const_maturity_list[i]) + "_rank_positive"] == 1]

        result_df = pd.merge(left, right, on="Date", suffixes=("_left", "_right"))

        result_df[str(const_maturity_list[i]) + "_w"] = (
            result_df["Expiration_for_trade_right"]
            - result_df[str(const_maturity_list[i]) + "_end_date" + "_right"]
        ).dt.days / (
            result_df["Expiration_for_trade_right"]
            - result_df["Expiration_for_trade_left"]
        ).dt.days

        result_df[str(const_maturity_list[i]) + "_v"] = result_df[
            "SettlementPrice_left"
        ] * result_df[str(const_maturity_list[i]) + "_w"] + result_df[
            "SettlementPrice_right"
        ] * (
            1 - result_df[str(const_maturity_list[i]) + "_w"]
        )

        result_df[str(const_maturity_list[i]) + "_roll"] = (
            (result_df["SettlementPrice_right"] - result_df["SettlementPrice_left"])
            / result_df[str(const_maturity_list[i]) + "_v"].shift(1)
            * (
                -252
                / (
                    result_df["Expiration_for_trade_right"]
                    - result_df["Expiration_for_trade_left"]
                ).dt.days
            )
        )

        result_df = result_df[
            [
                "Date",
                str(const_maturity_list[i]) + "_w",
                str(const_maturity_list[i]) + "_v",
                str(const_maturity_list[i]) + "_roll",
            ]
        ]
        if result is None:
            result = result_df
        else:
            result = pd.merge(result, result_df, on="Date")

    return result


def generate_xt() -> pd.DataFrame:
    constant_days = [21, 42, 63, 84, 105, 126]

    vt = generate_vt(constant_days)
    vt["Date"] = vt["Date"].dt.date

    raw_data = DataProcessor()
    VIX = raw_data.df1.loc[raw_data.df1["ID"] == "VIX"]
    micro_data = raw_data.df3.query("ID in ['SPY US Equity', 'TLT US Equity']")

    merged_df = pd.merge(vt, VIX, on="Date").set_index("Date")
    merged_df = pd.merge(
        merged_df,
        pd.pivot(micro_data, index="Date", columns="ID", values="C"),
        left_index=True,
        right_index=True,
    )

    result = merged_df[
        ["Close"]
        + [str(x) + "_v" for x in constant_days]
        + ["SPY US Equity", "TLT US Equity"]
    ]
    result.columns = (
        ["ln_VIX"]
        + ["ln_V" + str(i + 1) for i in range(len(constant_days))]
        + ["ln_SPY", "ln_TLT"]
    )
    result = np.log(result)
    result_roll = merged_df[[str(x) + "_roll" for x in constant_days]]
    result_roll.columns = ["roll" + str(i + 1) for i in range(len(constant_days))]

    return pd.concat([result, result_roll], axis=1)


def xt_with_etfprice() -> pd.DataFrame:
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


def main():
    result = generate_vt([21, 42, 63, 84, 105, 126])
    result.to_csv("1-6M_trading_days.csv")


if __name__ == "__main__":
    generate_xt()
