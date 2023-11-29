#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/16 13:56
# @Author   : wsy
# @email    : 631535207@qq.com
import abc
import pandas as pd
from qlib.data.dataset.processor import Processor, get_group_columns
from qlib.data.dataset.utils import fetch_df_by_index


class _MethodFill(Processor, abc.ABC):
    def __init__(self, fields_group=None):
        super().__init__()
        self.fields_group = fields_group

    def __call__(self, df):
        method = self.__class__.method()
        try:
            if self.fields_group is None:
                df.fillna(method=method, inplace=True)
            else:
                cols = get_group_columns(df, self.fields_group)
                # this implementation is extremely slow
                # df.fillna({col: self.fill_value for col in cols}, inplace=True)
                df.loc[:, df.columns.isin(cols)] = df.loc[
                    :, df.columns.isin(cols)
                ].fillna(method=method)
            return df
        except Exception as e:
            print(e)
            raise "valid method for df.fillna!"

    @classmethod
    def method(cls) -> str:
        """df.fillna(method=xx) return xx"""
        raise NotImplementedError


class BFill(_MethodFill):
    """Process NaN with bfill"""

    @classmethod
    def method(cls) -> str:
        return "bfill"


class FFill(_MethodFill):
    """Process NaN with ffill"""

    @classmethod
    def method(cls) -> str:
        return "ffill"


class LabelInstrumentNorm(Processor):
    """
    按照不同instrument的label的std进行norm
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group="label"):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(
            df, slice(self.fit_start_time, self.fit_end_time), level="datetime"
        )
        cols = get_group_columns(df, self.fields_group)
        std = df[cols].groupby("instrument", group_keys=False).std()
        scale = std.iloc[0, :] / std
        self.scale = pd.merge(
            left=pd.DataFrame(index=df.index),
            left_on=df.index.get_level_values("instrument"),
            right=scale,
            right_index=True,
            how="left",
        )
        self.cols = cols

    def __call__(self, df):
        df.loc(axis=1)[self.cols] = df[self.cols] * self.scale[self.cols]
        return df
