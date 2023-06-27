#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/24 10:16
# @Author   : wsy
# @email    : 631535207@qq.com
import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP, DataHandler
from qlib.contrib.data.handler import (
    _DEFAULT_LEARN_PROCESSORS,
    _DEFAULT_INFER_PROCESSORS,
    check_transform_proc,
)
from qlib.utils import init_instance_by_config, lazy_sort_index
from qlib.log import TimeInspector


class VixHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        feature_instruments="features",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        instruments_data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (["$maturity"], ["maturity"]),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        features_data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_features(feature_instruments),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        instruments_handler_config = {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "start_time": start_time,
                "end_time": end_time,
                "instruments": instruments,
                "data_loader": instruments_data_loader,
            },
        }

        features_handler_config = {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "start_time": start_time,
                "end_time": end_time,
                "instruments": feature_instruments,
                "data_loader": features_data_loader,
            },
        }
        instruments_handler = init_instance_by_config(
            instruments_handler_config, accept_types=DataHandler
        )
        features_handler = init_instance_by_config(
            features_handler_config, accept_types=DataHandler
        )
        data_loader = {
            "class": "CombineLoader",
            "module_path": "backtest.qlib_custom.data_loader",
            "kwargs": {
                "handler_config": {
                    "instruments_handler": instruments_handler,
                    "features_handler": features_handler,
                },
                "is_group": True,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def get_features(feature_instruments: str):
        if feature_instruments == "features":
            fields = [
                "$ln_VIX",
                "$ln_V1",
                "$ln_V2",
                "$ln_V3",
                "$ln_V4",
                "$ln_V5",
                "$ln_V6",
                "$ln_SPY",
                "$ln_TLT",
                "$roll1",
                "$roll2",
                "$roll3",
                "$roll4",
                "$roll5",
                "$roll6",
            ]
            names = [
                "ln_VIX",
                "ln_V1",
                "ln_V2",
                "ln_V3",
                "ln_V4",
                "ln_V5",
                "ln_V6",
                "ln_SPY",
                "ln_TLT",
                "roll1",
                "roll2",
                "roll3",
                "roll4",
                "roll5",
                "roll6",
            ]
        else:
            raise NotImplementedError("unknown feature columns")

        return fields, names


class SimpleVixHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="trable",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_features(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @classmethod
    def get_features(cls):
        fields = [
            "$maturity",
            "$ln_VIX",
            "$ln_V1",
            "$ln_V2",
            "$ln_V3",
            "$ln_V4",
            "$ln_V5",
            "$ln_V6",
            "$ln_SPY",
            "$ln_TLT",
            "$roll1",
            "$roll2",
            "$roll3",
            "$roll4",
            "$roll5",
            "$roll6",
            "$ln_V1 - $ln_VIX",
            "$ln_V2 - $ln_VIX",
            "$ln_V3 - $ln_VIX",
            "$ln_V4 - $ln_VIX",
            "$ln_V5 - $ln_VIX",
            "$ln_V6 - $ln_VIX",
        ]
        names = [
            "maturity",
            "ln_VIX",
            "ln_V1",
            "ln_V2",
            "ln_V3",
            "ln_V4",
            "ln_V5",
            "ln_V6",
            "ln_SPY",
            "ln_TLT",
            "roll1",
            "roll2",
            "roll3",
            "roll4",
            "roll5",
            "roll6",
            "mu1",
            "mu2",
            "mu3",
            "mu4",
            "mu5",
            "mu6",
        ]
        return fields, names


class GroupVixHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="trable",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_features(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @classmethod
    def get_common_fields(cls):
        fields = [
            "$maturity",
            "$ln_VIX",
            "$ln_SPY",
            "$ln_TLT",
        ]
        names = [
            "maturity",
            "ln_VIX",
            "ln_SPY",
            "ln_TLT",
        ]
        # 衍生common_fields, 结构信息
        fields += [
            "$ln_V1 - $ln_V2",
            "$ln_V2 - $ln_V3",
            "$ln_V3 - $ln_V4",
            "$ln_V4 - $ln_V5",
            "$ln_V5 - $ln_V6",
            "$roll1 - $roll2",
            "$roll2 - $roll3",
            "$roll3 - $roll4",
            "$roll4 - $roll5",
            "$roll5 - $roll6",
        ]
        names += [
            "delta_v12",
            "delta_v23",
            "delta_v34",
            "delta_v45",
            "delta_v56",
            "delta_roll12",
            "delta_roll23",
            "delta_roll34",
            "delta_roll45",
            "delta_roll56",
        ]
        return fields, names

    @classmethod
    def get_features(cls):
        fields, names = cls.get_common_fields()
        fields += [
            "$ln_V1",
            "$roll1",
            "$ln_V1 - $ln_VIX",
            "$ln_V2",
            "$roll2",
            "$ln_V2 - $ln_VIX",
            "$ln_V3",
            "$roll3",
            "$ln_V3 - $ln_VIX",
            "$ln_V4",
            "$roll4",
            "$ln_V4 - $ln_VIX",
            "$ln_V5",
            "$roll5",
            "$ln_V5 - $ln_VIX",
            "$ln_V6",
            "$roll6",
            "$ln_V6 - $ln_VIX",
        ]
        names += [
            "ln_V1",
            "roll1",
            "mu1",
            "ln_V2",
            "roll2",
            "mu2",
            "ln_V3",
            "roll3",
            "mu3",
            "ln_V4",
            "roll4",
            "mu4",
            "ln_V5",
            "roll5",
            "mu5",
            "ln_V6",
            "roll6",
            "mu6",
        ]
        return fields, names

    def setup_data(self, init_type: str = DataHandlerLP.IT_FIT_SEQ, **kwargs):
        """
        Set up the data in case of running initialization for multiple time

        Parameters
        ----------
        init_type : str
            The type `IT_*` listed above.
        enable_cache : bool
            default value is false:

            - if `enable_cache` == True:

                the processed data will be saved on disk, and handler will load the cached data from the disk directly
                when we call `init` next time
        """
        # Setup data.
        # _data may be with multiple column index level. The outer level indicates the feature set name
        with TimeInspector.logt("Loading data"):
            # make sure the fetch method is based on an index-sorted pd.DataFrame
            _data = lazy_sort_index(
                self.data_loader.load(self.instruments, self.start_time, self.end_time)
            )

        self._data = self._vix_process(_data)

        with TimeInspector.logt("fit & process data"):
            if init_type == DataHandlerLP.IT_FIT_IND:
                self.fit()
                self.process_data()
            elif init_type == DataHandlerLP.IT_LS:
                self.process_data()
            elif init_type == DataHandlerLP.IT_FIT_SEQ:
                self.fit_process_data()
            else:
                raise NotImplementedError("This type of input is not supported")

    def _vix_process(self, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()
        _data[("feature", "ln_V")] = _data[("feature", "ln_V1")]
        _data[("feature", "roll")] = _data[("feature", "roll1")]
        _data[("feature", "mu")] = _data[("feature", "mu1")]
        for i in range(5):
            mask = _data[("feature", "maturity")] == i + 2
            _data[("feature", "ln_V")].loc[mask] = _data.loc[mask][
                ("feature", "ln_V" + str(i + 2))
            ]
            _data[("feature", "roll")].loc[mask] = _data.loc[mask][
                ("feature", "roll" + str(i + 2))
            ]
            _data[("feature", "mu")].loc[mask] = _data.loc[mask][
                ("feature", "mu" + str(i + 2))
            ]
        return _data.drop(
            [
                ("feature", "ln_V1"),
                ("feature", "ln_V2"),
                ("feature", "ln_V3"),
                ("feature", "ln_V4"),
                ("feature", "ln_V5"),
                ("feature", "ln_V6"),
                ("feature", "roll1"),
                ("feature", "roll2"),
                ("feature", "roll3"),
                ("feature", "roll4"),
                ("feature", "roll5"),
                ("feature", "roll6"),
                ("feature", "mu1"),
                ("feature", "mu2"),
                ("feature", "mu3"),
                ("feature", "mu4"),
                ("feature", "mu5"),
                ("feature", "mu6"),
            ],
            axis=1,
        )
