#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/16 13:56
# @Author   : wsy
# @email    : 631535207@qq.com
import abc

from qlib.data.dataset.processor import Processor, get_group_columns


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
