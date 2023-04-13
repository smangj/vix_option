#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing
import pandas as pd
from qlib.backtest.position import BasePosition
from qlib.backtest.report import Indicator
from qlib.backtest.high_performance_ds import BaseOrderIndicator

__author__ = "Vitor Chen"
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

ACCOUNT_COLS = ["cash", "now_account_value"]
POSITION_COLS = ["amount", "price", "weight", "count_day"]

ORDER_COLS = ["amount", "deal_amount", "trade_price", "trade_value", "trade_cost"]


def gen_acct_pos_dfs(
    hist_positions: typing.Dict[pd.Timestamp, BasePosition]
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从Qlib历史仓位记录数据对象中提取账户记录表、持仓表

    :param hist_positions: Qlib的历史仓位记录数据对象，对应PortAnaRecord保存的"portfolio_analysis/positions_normal_{FREQ}.pkl“文件
    :return: (account_df, position_df)
        account_df: DataFrame
            账户记录表
        position_df: DataFrame
            持仓记录表

    """

    def _gen_chunk_df(
        _dt: pd.Timestamp, _position_chunk: BasePosition
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        _account_chunk_df = pd.DataFrame(
            {k: v for k, v in _position_chunk.position.items() if k in ACCOUNT_COLS},
            index=[_dt],
        )
        _account_chunk_df.index.name = "datetime"
        _account_chunk_df = _account_chunk_df.reset_index()

        _position_chunk_df = pd.DataFrame.from_dict(
            {
                k: v
                for k, v in _position_chunk.position.items()
                if k not in ACCOUNT_COLS
            },
            orient="index",
        )
        _position_chunk_df.index.name = "stock_id"
        if len(_position_chunk_df) > 0:
            _position_chunk_df.loc[:, "datetime"] = _dt
        _position_chunk_df = _position_chunk_df.reset_index()
        return _account_chunk_df, _position_chunk_df

    chunk_pairs = [
        _gen_chunk_df(_dt, _position_chunk)
        for _dt, _position_chunk in hist_positions.items()
    ]
    account_df = pd.concat([_pair[0] for _pair in chunk_pairs], ignore_index=True)
    position_df = pd.concat([_pair[1] for _pair in chunk_pairs], ignore_index=True)

    return account_df, position_df


def gen_orders_df(indicator_obj: Indicator) -> pd.DataFrame:
    """
    从Qlib回测交易指标对象中提取交易订单记录表

    :param indicator_obj: Qlib回测交易指标对象，对应PortAnaRecord保存的"portfolio_analysis/indicators_normal_{FREQ}_obj.pkl“文件
    :return: orders_df: DataFrame 交易订单记录表
    """

    def _gen_chunk_df(
        _dt: pd.Timestamp, _order_indicator: BaseOrderIndicator
    ) -> pd.DataFrame:
        metrics_dict = _order_indicator.to_series()
        chunk_df = pd.concat(
            [metrics_dict[_metric] for _metric in ORDER_COLS], axis=1
        ).reset_index()
        chunk_df.columns = ["stock_id"] + ORDER_COLS
        if len(chunk_df) > 0:
            chunk_df.loc[:, "datetime"] = _dt
        return chunk_df

    chunk_df_list = [
        _gen_chunk_df(_dt, _order_indicator)
        for _dt, _order_indicator in indicator_obj.order_indicator_his.items()
    ]

    return pd.concat(chunk_df_list, ignore_index=True)
