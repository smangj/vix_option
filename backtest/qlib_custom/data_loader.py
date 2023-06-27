#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/25 16:45
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
import numpy as np
from qlib.log import get_module_logger
from qlib.data.dataset.loader import DataLoaderDH
from qlib.data.dataset import TSDataSampler


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
                df = pd.merge(df, data, left_index=True, right_index=True)
        else:
            df = self.handlers.fetch(
                selector=slice(start_time, end_time),
                level="datetime",
                **self.fetch_kwargs,
            )
        return df


class TSTabularSampler(TSDataSampler):
    def _get_idx_data(self, idx: int):

        indices = self._get_indices(*self._get_row_col(idx))

        # 1) for better performance, use the last nan line for padding the lost date
        # 2) In case of precision problems. We use np.float64.
        # precision problems. It will not cause any problems in my tests at least
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(
            int
        )

        if (
            np.diff(indices) == 1
        ).all():  # slicing instead of indexing for speeding up.
            data = self.data_arr[indices[0] : indices[-1] + 1]
        else:
            data = self.data_arr[indices]

        return data

    def __getitem__(self, idx: int):
        """
        返回tabular data产生的3维矩阵(instru, step_len, features)
        """

        t = self.data_index[idx][1]
        idxs = np.where(self.data_index.get_level_values(1) == t)
        if isinstance(idxs, tuple) and len(idxs) == 1:
            idxs = idxs[0]

        arrs = []
        for i in idxs:
            arrs.append(self._get_idx_data(i))

        data = np.array(arrs)

        return data
