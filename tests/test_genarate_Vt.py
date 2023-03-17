#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/17 18:35
# @Author   : wsy
# @email    : 631535207@qq.com
from generate_Vt import generate_vt


def test_generate_vt():
    result = generate_vt(30)

    assert result is not None
