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
    recorder_id="664e152a226a45cab958c1551a68edb3", experiment_name="gru_xt"
)

pred_df = recorder.load_object("pred.pkl")
label_df = recorder.load_object("label.pkl")
label_df.columns = ["label"]

ic = recorder.load_object("sig_analysis/ic.pkl")

pred_label = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)

analysis_model.model_performance_graph(pred_label, N=6)
