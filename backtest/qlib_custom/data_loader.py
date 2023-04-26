#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/25 16:45
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
from qlib.log import get_module_logger
from qlib.data.dataset.loader import DataLoaderDH


class CombineLoader(DataLoaderDH):
    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        if instruments is not None:
            get_module_logger(self.__class__.__name__).warning(
                f"instruments[{instruments}] is ignored"
            )

        if self.is_group:
            data_list = [
                dh.fetch(
                    selector=slice(start_time, end_time),
                    level="datetime",
                    **self.fetch_kwargs,
                )
                for grp, dh in self.handlers.items()
            ]
            df = data_list[0]
            for data in data_list[1:]:
                df = pd.merge(df, data, left_index=True, right_index=True, how="outer")
        else:
            df = self.handlers.fetch(
                selector=slice(start_time, end_time),
                level="datetime",
                **self.fetch_kwargs,
            )
        return df
