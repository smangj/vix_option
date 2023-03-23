#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/17 18:35
# @Author   : wsy
# @email    : 631535207@qq.com
import pytest

from data_process.generate_Vt import generate_vt, generate_xt


def test_generate_vt():
    result = generate_vt(21)

    assert result.loc[result["date"] == "2005-08-22"]["21_DAYS_price"].iloc[
        0
    ] == pytest.approx(14.95132, abs=0.01)


def test_genarate_xt():
    result = generate_xt()

    assert result.loc[result["date"] == "2005-08-22"]["VIX1"].iloc[0] == pytest.approx(
        14.95132, abs=0.01
    )
