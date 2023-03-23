#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/16 10:59
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
import datetime as dt
import numpy as np
from data_process.data_processor import DataProcessor
from utils.time import to_pydatetime
import typing


def generate_vt(
    const_maturity_list: typing.Union[typing.List[int], int]
) -> pd.DataFrame:
    if isinstance(const_maturity_list, int):
        const_maturity_list = [const_maturity_list]

    const_maturity_list = np.sort(const_maturity_list)

    data = DataProcessor()
    raw = data.df2.copy()

    date_series = pd.Series(raw["Date"].unique()).sort_values()
    raw["Expiration_for_trade"] = raw["Expiration"].map(
        pd.Series(date_series.shift(1).values, index=date_series.values)
    )

    masks = []
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

        masks.append(raw[str(const_maturity_list[i]) + "_rank_positive"] == 1)
        masks.append(raw[str(const_maturity_list[i]) + "_rank_negative"] == 1)

    final_mask = masks[0]
    for m in masks[1:]:
        final_mask |= m

    result = raw.loc[final_mask]

    groups = raw.groupby("Date")

    result = []
    for date, group in groups:

        if len(group) <= 1:
            continue
        date_dict = {"date": date}
        for const_maturity in const_maturity_list:
            end_date_index = list(date_series.values).index(date) + const_maturity
            if end_date_index < len(groups):
                end_date = to_pydatetime(list(date_series.values)[end_date_index])
            else:
                end_date = to_pydatetime(date) + dt.timedelta(int(const_maturity))
            near_sort_index = np.argsort(
                group["Expiration_for_trade"]
                .apply(lambda x: abs((to_pydatetime(x) - end_date).days))
                .values
            )
            f = group.iloc[near_sort_index[0]]
            f1 = None
            for i in range(len(near_sort_index) - 1):
                # 保证Ti <= t+ thetai <= Ti+1
                if (
                    group.iloc[near_sort_index[i + 1]]["Expiration_for_trade"]
                    - end_date
                ).days * (f["Expiration_for_trade"] - end_date).days <= 0:
                    f1 = group.iloc[near_sort_index[i + 1]]
                    break
            if f1 is None:
                date_dict[str(const_maturity) + "_left"] = np.nan
                date_dict[str(const_maturity) + "_right"] = np.nan
                date_dict[str(const_maturity) + "_w"] = np.nan
                date_dict[str(const_maturity) + "_price"] = np.nan
                continue
            elif f["Expiration_for_trade"] >= end_date:
                f_left = f1
                f_right = f
            else:
                f_left = f
                f_right = f1
            fi = f_left["SettlementPrice"]
            fi_ = f_right["SettlementPrice"]
            w = (f_right["Expiration_for_trade"] - end_date).days / (
                f_right["Expiration_for_trade"] - f_left["Expiration_for_trade"]
            ).days
            date_dict[str(const_maturity) + "_left"] = fi
            date_dict[str(const_maturity) + "_right"] = fi_
            date_dict[str(const_maturity) + "_w"] = w
            date_dict[str(const_maturity) + "_price"] = fi * w + fi_ * (1 - w)

        result.append(date_dict)

    return pd.DataFrame.from_records(result)


def generate_xt() -> pd.DataFrame:
    constant_days = [21, 42, 63, 84, 105, 126]

    vt = generate_vt(constant_days)

    raw_data = DataProcessor()
    VIX = raw_data.df1.loc[raw_data.df1["ID"] == "VIX"]
    VIX["date"] = VIX["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))

    result = pd.merge(vt, VIX, on="date").set_index("date")
    result = result[["Close"] + [str(x) + "_DAYS_price" for x in constant_days]]
    result.columns = ["VIX"] + ["VIX" + str(i + 1) for i in range(len(constant_days))]
    result = np.log(result)
    result["roll1"] = None

    return vt


def main():
    result = generate_vt([21, 42, 63, 84, 105, 126])
    result.to_csv("1-6M_trading_days.csv")


if __name__ == "__main__":
    main()
