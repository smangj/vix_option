#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/7/19 14:20
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.workflow import R
from pathlib import Path

from backtest.qlib_custom.record import LongShortBacktestRecord, JiaQiRecord


def test_LongShortBacktestRecord(qlib_init):
    recorder = R.get_recorder(
        recorder_id="ee16729a846348ceacbe20aedc76652f", experiment_name="gru_xt"
    )

    r = LongShortBacktestRecord(
        recorder=recorder,
    )
    r.generate()


def test_JiaQiRecord(qlib_init):
    recorder = R.get_recorder(
        recorder_id="67ad416857fd45ce88a311d9c18a8f5b", experiment_id="43"
    )
    h_path = (
        Path(recorder.get_local_dir()).parent
        / "GRU_GroupVixHandler_LongShortBacktestRecord1_handler_horizon0.pkl"
    )
    r = JiaQiRecord(
        recorder=recorder,
        handler=h_path,
    )
    r.generate()
    print("haha")
