#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/5 14:12
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.data.dataset import TSDatasetH
from copy import deepcopy
from backtest.qlib_custom.data_loader import TSTabularSampler


class TSTabularDatasetH(TSDatasetH):
    def _prepare_seg(self, slc: slice, **kwargs) -> TSTabularSampler:
        """
        split the _prepare_raw_seg is to leave a hook for data preprocessing before creating processing data
        NOTE: TSDatasetH only support slc segment on datetime !!!
        """
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super(TSDatasetH, self)._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super(TSDatasetH, self)._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = TSTabularSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
        )
        return tsds
