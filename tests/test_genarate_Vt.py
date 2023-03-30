#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/17 18:35
# @Author   : wsy
# @email    : 631535207@qq.com
import pytest

from data_process.generate_Vt import generate_vt, generate_xt
from utils.time import to_pydatetime


def test_generate_vt():
    result = generate_vt(21)

    assert result.loc[result["Date"] == to_pydatetime("2005-12-15"), "21_v"].iloc[
        0
    ] == pytest.approx(12.82964, abs=0.01)


def test_genarate_xt():
    result = generate_xt()

    assert result.loc[to_pydatetime("2005-12-22").date(), "ln_VIX"] == pytest.approx(
        2.33117, abs=0.01
    )
    assert result.loc[to_pydatetime("2022-08-15").date(), "roll4"] == pytest.approx(
        -0.06561, abs=0.0001
    )
