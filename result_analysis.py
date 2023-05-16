#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/4 15:42
# @Author   : wsy
# @email    : 631535207@qq.com

import qlib
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position
import pandas as pd


def analysis_result(recorder_id: str, experiment_name: str):
    qlib.init(provider_uri="data/qlib_data", region="us")

    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)

    # get_result
    pred_df = recorder.load_object("pred.pkl")

    label_df = recorder.load_object("label.pkl")
    label_df.columns = ["label"]

    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")

    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")

    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    pred_label = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(
        label_df.index
    )

    # analysis
    analysis_model.model_performance_graph(pred_label, N=6)

    analysis_position.cumulative_return_graph(
        positions, report_normal_df, label_df.swaplevel().sort_index()
    )

    analysis_position.report_graph(report_normal_df)

    analysis_position.risk_analysis_graph(analysis_df, report_normal_df)

    analysis_position.score_ic_graph(pred_label)


if __name__ == "__main__":
    analysis_result("dea9b1e915294896b66b4ff4c2d13d65", "gru_xt")
