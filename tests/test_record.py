#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/7/19 14:20
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.workflow import R
from pathlib import Path

from backtest.qlib_custom.record import LongShortBacktestRecord, JiaQiRecord
from utils.time import timing_decorator


def test_LongShortBacktestRecord(qlib_init):
    recorder = R.get_recorder(
        recorder_id="ee16729a846348ceacbe20aedc76652f", experiment_name="gru_xt"
    )

    r = LongShortBacktestRecord(
        recorder=recorder,
    )
    r.generate()


@timing_decorator
def test_JiaQiRecord(qlib_init):
    recorder = R.get_recorder(
        recorder_id="084e905ffbf34e69b74f2efc2cca2afb", experiment_id="8"
    )
    h_path = (
        Path(recorder.get_local_dir()).parent
        / "GRU_GroupVixHandler20230711_LongShortBacktestRecord_handler_horizon0.pkl"
    )
    r = JiaQiRecord(
        recorder=recorder,
        handler=h_path,
    )
    r.generate()
