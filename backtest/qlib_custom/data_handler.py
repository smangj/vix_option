#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/24 10:16
# @Author   : wsy
# @email    : 631535207@qq.com
import abc

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


class GroupVixHandler20230711(DataHandlerLP):
    windows = [5, 20, 60]
    # 与features的衍生匹配
    rolling_names = ["z-score", "MA", "std", "skew", "kurt"]
    data_category = ["ln_V", "roll", "mu"]

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
        return fields, names

    @classmethod
    def get_features(cls):
        fields, names = cls.get_common_fields()
        person_fields = [
            "$ln_V1",
            "$roll1",
            "$ln_V1 - $ln_VIX",
            "$ln_V2",
            "$roll2",
            "$ln_V2 - $ln_V1",
            "$ln_V3",
            "$roll3",
            "$ln_V3 - $ln_V2",
            "$ln_V4",
            "$roll4",
            "$ln_V4 - $ln_V3",
            "$ln_V5",
            "$roll5",
            "$ln_V5 - $ln_V4",
            "$ln_V6",
            "$roll6",
            "$ln_V6 - $ln_V5",
        ]
        person_names = [
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

        fields += person_fields
        names += person_names

        rolling_features = fields[1:]
        var_names = names[1:]

        # z-score
        fields += [
            "Mean({}, {}) / Std({}, {})".format(x, y, x, y)
            for x in person_fields
            for y in cls.windows
        ]
        names += [
            "{}_z-score{}".format(x, y) for x in person_names for y in cls.windows
        ]
        # rolling, 除了maturity
        fields += [
            "Mean({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_MA{}".format(x, y) for x in var_names for y in cls.windows]
        fields += [
            "Std({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_std{}".format(x, y) for x in var_names for y in cls.windows]
        fields += [
            "Skew({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_skew{}".format(x, y) for x in var_names for y in cls.windows]
        fields += [
            "Kurt({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_kurt{}".format(x, y) for x in var_names for y in cls.windows]
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

    @classmethod
    def _vix_process(cls, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()
        _data[("feature", "ln_V")] = _data[("feature", "ln_V1")]
        _data[("feature", "roll")] = _data[("feature", "roll1")]
        _data[("feature", "mu")] = _data[("feature", "mu1")]
        for window in cls.windows:
            for name in cls.rolling_names:
                for category in cls.data_category:
                    _data[
                        ("feature", "{}_{}{}".format(category, name, window))
                    ] = _data[("feature", "{}1_{}{}".format(category, name, window))]

        for i in range(5):
            mask = _data[("feature", "maturity")] == i + 2
            for category in cls.data_category:
                _data[("feature", "{}".format(category))].loc[mask] = _data.loc[mask][
                    ("feature", "{}".format(category) + str(i + 2))
                ]
            for window in cls.windows:
                for name in cls.rolling_names:
                    for category in cls.data_category:
                        _data[
                            ("feature", "{}_{}{}".format(category, name, window))
                        ].loc[mask] = _data.loc[mask][
                            (
                                "feature",
                                "{}{}_{}{}".format(category, i + 2, name, window),
                            )
                        ]

        drop_list = [
            ("feature", "{}{}".format(x, i + 1))
            for x in cls.data_category
            for i in range(6)
        ]
        drop_list += [
            ("feature", "{}{}_{}{}".format(x, i + 1, name, window))
            for x in cls.data_category
            for i in range(6)
            for name in cls.rolling_names
            for window in cls.windows
        ]
        return _data.drop(drop_list, axis=1)


class _GroupVixHandler(DataHandlerLP, abc.ABC):
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

    @classmethod
    def get_common_fields(cls):
        fields = [
            "$ln_VIX",
            "$ln_SPY",
            "$ln_TLT",
        ]
        names = [
            "ln_VIX",
            "ln_SPY",
            "ln_TLT",
        ]
        return fields, names

    @classmethod
    def get_features(cls):
        fields, names = cls.get_common_fields()
        p_fields, p_names = cls.get_private_features()
        fields += p_fields
        names += p_names
        return fields, names

    @classmethod
    def get_private_features(cls):
        raise NotImplementedError

    def _vix_process(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class OldBro(_GroupVixHandler):
    @classmethod
    def get_private_features(cls):
        fields = [
            "$ln_V1",
            "$roll1",
            "$ln_V2",
            "$roll2",
            "$ln_V3",
            "$roll3",
            "$ln_V4",
            "$roll4",
            "$ln_V5",
            "$roll5",
            "$ln_V6",
            "$roll6",
        ]
        names = [
            "ln_V1",
            "roll1",
            "ln_V2",
            "roll2",
            "ln_V3",
            "roll3",
            "ln_V4",
            "roll4",
            "ln_V5",
            "roll5",
            "ln_V6",
            "roll6",
        ]
        return fields, names

    def _vix_process(self, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()
        _data[("feature", "ln_V")] = _data[("feature", "ln_V1")]
        _data[("feature", "roll")] = _data[("feature", "roll1")]
        for i in range(5):
            mask = (
                _data.index.get_level_values("instrument") == "VIX_" + str(i + 2) + "M"
            )
            _data[("feature", "ln_V")].loc[mask] = _data.loc[mask][
                ("feature", "ln_V" + str(i + 2))
            ]
            _data[("feature", "roll")].loc[mask] = _data.loc[mask][
                ("feature", "roll" + str(i + 2))
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
            ],
            axis=1,
        )


class TermStructure(OldBro):
    @classmethod
    def get_common_fields(cls):
        fields, names = super().get_common_fields()
        # 衍生common_fields, 结构信息
        fields += [
            "$roll1 - $roll2",
            "$roll2 - $roll3",
            "$roll3 - $roll4",
            "$roll4 - $roll5",
            "$roll5 - $roll6",
        ]
        names += [
            "delta_roll12",
            "delta_roll23",
            "delta_roll34",
            "delta_roll45",
            "delta_roll56",
        ]
        return fields, names

    @classmethod
    def get_private_features(cls):
        fields, names = super().get_private_features()
        fields += [
            "$ln_V1 - $ln_VIX",
            "$ln_V2 - $ln_V1",
            "$ln_V3 - $ln_V2",
            "$ln_V4 - $ln_V3",
            "$ln_V5 - $ln_V4",
            "$ln_V6 - $ln_V5",
        ]
        names += [
            "mu1",
            "mu2",
            "mu3",
            "mu4",
            "mu5",
            "mu6",
        ]
        return fields, names

    def _vix_process(self, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()
        _data[("feature", "ln_V")] = _data[("feature", "ln_V1")]
        _data[("feature", "roll")] = _data[("feature", "roll1")]
        _data[("feature", "mu")] = _data[("feature", "mu1")]
        for i in range(5):
            mask = (
                _data.index.get_level_values("instrument") == "VIX_" + str(i + 2) + "M"
            )
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


class Derivative(TermStructure):
    windows = [5, 20, 60]
    # 与features的衍生匹配
    rolling_names = ["z-score", "MA", "std", "skew", "kurt"]
    data_category = ["ln_V", "roll", "mu"]

    @classmethod
    def get_features(cls):
        fields, names = super().get_features()
        p_fields, p_names = cls.get_private_features()
        rolling_features = fields.copy()
        var_names = names.copy()

        # z-score
        fields += [
            "Mean({}, {}) / Std({}, {})".format(x, y, x, y)
            for x in p_fields
            for y in cls.windows
        ]
        names += ["{}_z-score{}".format(x, y) for x in p_names for y in cls.windows]
        # rolling
        fields += [
            "Mean({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_MA{}".format(x, y) for x in var_names for y in cls.windows]
        fields += [
            "Std({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_std{}".format(x, y) for x in var_names for y in cls.windows]
        fields += [
            "Skew({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_skew{}".format(x, y) for x in var_names for y in cls.windows]
        fields += [
            "Kurt({}, {})".format(x, y) for x in rolling_features for y in cls.windows
        ]
        names += ["{}_kurt{}".format(x, y) for x in var_names for y in cls.windows]
        return fields, names

    @classmethod
    def _vix_process(cls, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()
        _data[("feature", "ln_V")] = _data[("feature", "ln_V1")]
        _data[("feature", "roll")] = _data[("feature", "roll1")]
        _data[("feature", "mu")] = _data[("feature", "mu1")]
        for window in cls.windows:
            for name in cls.rolling_names:
                for category in cls.data_category:
                    _data[
                        ("feature", "{}_{}{}".format(category, name, window))
                    ] = _data[("feature", "{}1_{}{}".format(category, name, window))]

        for i in range(5):
            mask = (
                _data.index.get_level_values("instrument") == "VIX_" + str(i + 2) + "M"
            )
            for category in cls.data_category:
                _data[("feature", "{}".format(category))].loc[mask] = _data.loc[mask][
                    ("feature", "{}".format(category) + str(i + 2))
                ]
            for window in cls.windows:
                for name in cls.rolling_names:
                    for category in cls.data_category:
                        _data[
                            ("feature", "{}_{}{}".format(category, name, window))
                        ].loc[mask] = _data.loc[mask][
                            (
                                "feature",
                                "{}{}_{}{}".format(category, i + 2, name, window),
                            )
                        ]

        drop_list = [
            ("feature", "{}{}".format(x, i + 1))
            for x in cls.data_category
            for i in range(6)
        ]
        drop_list += [
            ("feature", "{}{}_{}{}".format(x, i + 1, name, window))
            for x in cls.data_category
            for i in range(6)
            for name in cls.rolling_names
            for window in cls.windows
        ]
        return _data.drop(drop_list, axis=1)
