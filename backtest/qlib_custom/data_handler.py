#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/24 10:16
# @Author   : wsy
# @email    : 631535207@qq.com

from qlib.data.dataset.handler import DataHandlerLP, DataHandler
from qlib.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.utils import init_instance_by_config


class ExternalFeatures(DataHandlerLP):
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
        **kwargs
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )
        print(kwargs)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (["$close"], ["close"]),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        data_loader_2 = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_features(feature_instruments),
                    "label": kwargs.pop("label", self.get_label_config()),
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
                "infer_processors": infer_processors,
                "learn_processors": learn_processors,
                "data_loader": data_loader,
            },
        }

        features_handler_config = {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "start_time": start_time,
                "end_time": end_time,
                "instruments": feature_instruments,
                "infer_processors": infer_processors,
                "learn_processors": learn_processors,
                "data_loader": data_loader_2,
            },
        }
        instruments_handler = init_instance_by_config(
            instruments_handler_config, accept_types=DataHandler
        )
        features_handler = init_instance_by_config(
            features_handler_config, accept_types=DataHandler
        )
        data_loader_1 = {
            "class": "DataLoaderDH",
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
            data_loader=data_loader_1,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
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
