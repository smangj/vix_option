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
            data_dict = {
                grp: dh.fetch(
                    selector=slice(start_time, end_time),
                    level="datetime",
                    **self.fetch_kwargs,
                )
                for grp, dh in self.handlers.items()
        }
            instrument = data_dict.get("instruments_handler")
            if instrument is None:
                raise ValueError("instruments_handler is necessary!")
            df = instrument
            for name, data in data_dict.items():
                if name == "instruments_handler":
                    continue
                data.index = data.index.droplevel(1)
                df = pd.merge(df, data, left_index=True, right_index=True, how="outer")
        else:
            df = self.handlers.fetch(
                selector=slice(start_time, end_time),
                level="datetime",
                **self.fetch_kwargs,
            )
        return df
