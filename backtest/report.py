import typing
import pandas as pd
import ffn
from dataclasses import dataclass
import os
import datetime as dt
import pyecharts.options as opts
from pyecharts.charts import Line, Page
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


@dataclass
class Values:
    name: str
    values: pd.Series


def cal_stats(
    values: pd.Series,
    rf: typing.Optional[float] = 0.0,
) -> typing.List:
    assert isinstance(values, pd.Series), "The input is not a Pd.Series."
    assert (
        values.index.dtype == "datetime64[ns]"
    ), "The dtype of the index is not datetime64[ns]."
    assert len(values) > 1, "The length of the series is not greater than 1."
    perf = ffn.core.PerformanceStats(values, rf)
    return [
        perf.total_return,
        perf.incep,
        perf.daily_vol,
        perf.max_drawdown,
        perf.daily_sharpe,
    ]


def report(
    values_list: typing.List[Values],
    rf: typing.Optional[float] = 0.0,
    output_dir: str = "output",
    file_name: typing.Optional[str] = None,
) -> str:
    assert output_dir is not None and len(output_dir) > 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if file_name is None or len(file_name) == 0:
        file_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, file_name + ".html")
    strategy_num = len(values_list)
    cumulative_return_all = []
    rows = []
    for i in range(strategy_num):
        values = values_list[i]
        assert isinstance(values.values, pd.Series), "The input is not a Pd.Series."
        assert (
            values.values.index.dtype == "datetime64[ns]"
        ), "The dtype of the index is not datetime64[ns]"
        assert len(values.values) > 1, "The length of the series is not greater than 1."
        name = values.name
        cumulative_return_se = values.values / values.values.iloc[0]
        cumulative_return_se = (cumulative_return_se - 1) * 100
        cumulative_return_se.name = name
        cumulative_return_all.append(cumulative_return_se)
        perf = cal_stats(values.values, rf)
        total_rtn = perf[0]
        annual_rtn = perf[1]
        annual_vol = perf[2]
        max_drawdown = perf[3]
        sharpe = perf[4]
        rows.append(
            [
                name,
                "{:.2%}".format(total_rtn),
                "{:.2%}".format(annual_rtn),
                "{:.2%}".format(annual_vol),
                "{:.2%}".format(max_drawdown),
                "{:.2%}".format(sharpe),
            ]
        )
    headers = [
        "策略名称",
        "组合收益",
        "组合年化收益",
        "年化波动率",
        "最大回撤",
        "Sharpe",
    ]

    # 合并累积收益率曲线，对齐日期
    cum_return_df = pd.concat(cumulative_return_all, axis=1)
    cum_return_df.ffill(inplace=True)

    table = (
        Table()
        .add(headers, rows)
        .set_global_opts(title_opts=ComponentTitleOpts(title="评估指标"))
    )
    line_chart = Line()
    line_chart.add_xaxis(xaxis_data=cum_return_df.index)
    for name in cum_return_df.columns:
        line_chart.add_yaxis(
            series_name=name,
            y_axis=cum_return_df[name],
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )
    line_chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis"),
        title_opts=opts.TitleOpts(title="累积收益率图"),
        xaxis_opts=opts.AxisOpts(
            name="日期", type_="time", name_location="center", name_gap=40
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="累积收益率（%）",
            name_location="center",
            name_gap=40,
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    )
    page = Page()
    page.add(table, line_chart)
    page.render(file_path)
    return file_path
