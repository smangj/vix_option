#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/4 15:42
# @Author   : wsy
# @email    : 631535207@qq.com

import qlib
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position
import pandas as pd

qlib.init(provider_uri="data/qlib_data", region="us")


recorder = R.get_recorder(
    recorder_id="5e3d48c4d08541a986b2026bf3eb5863", experiment_name="vix_data_test"
)

pred_df = recorder.load_object("pred.pkl")
label_df = recorder.load_object("label.pkl")
label_df.columns = ["label"]

pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)

analysis_model.model_performance_graph(pred_label, N=6)
