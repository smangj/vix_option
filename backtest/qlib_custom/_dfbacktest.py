#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/7/20 8:38
# @Author   : wsy
# @email    : 631535207@qq.com
import abc

import numpy as np
import pandas as pd
from qlib.backtest import get_exchange
from qlib.backtest.exchange import Exchange
from qlib.config import C
from qlib.data import D
from qlib.data.dataset import get_level_index
from qlib.utils import get_date_range
import cvxpy as cp


class _DfBacktest(abc.ABC):
    """通过tabular_df直接做回测的类"""

    def __init__(self, tabular_df: pd.DataFrame):
        if get_level_index(tabular_df, level="datetime") == 1:
            tabular_df = tabular_df.swaplevel().sort_index()
        self.data = tabular_df

    def _generate_exchange(
        self,
        trade_unit,
        limit_threshold,
        deal_price,
        subscribe_fields,
        shift,
        freq,
        open_cost,
        close_cost,
        min_cost,
    ) -> Exchange:
        if trade_unit is None:
            trade_unit = C.trade_unit
        if limit_threshold is None:
            limit_threshold = C.limit_threshold

        _pred_dates = self.data.index.get_level_values(level="datetime")
        predict_dates = D.calendar(
            start_time=_pred_dates.min(), end_time=_pred_dates.max()
        )
        trade_dates = np.append(
            predict_dates[shift:],
            get_date_range(predict_dates[-1], left_shift=1, right_shift=shift),
        )

        trade_exchange = get_exchange(
            start_time=predict_dates[0],
            end_time=trade_dates[-1],
            freq=freq,
            codes=list(self.data.index.get_level_values("instrument").unique()),
            deal_price=deal_price,
            subscribe_fields=subscribe_fields,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            trade_unit=trade_unit,
        )
        return trade_exchange

    def run_backtest(
        self,
        freq: str = "day",
        deal_price=None,
        shift=1,
        open_cost=0,
        close_cost=0,
        trade_unit=None,
        limit_threshold=None,
        min_cost=0,
        subscribe_fields=[],
    ):

        # deal date
        _pred_dates = self.data.index.get_level_values(level="datetime")
        predict_dates = D.calendar(
            start_time=_pred_dates.min(), end_time=_pred_dates.max()
        )
        trade_dates = np.append(
            predict_dates[shift:],
            get_date_range(predict_dates[-1], left_shift=1, right_shift=shift),
        )

        if deal_price is None:
            deal_price = C.deal_price
        if deal_price[0] != "$":
            deal_price = "$" + deal_price
        subscribe_fields = subscribe_fields.copy()
        profit_str = f"Ref({deal_price}, -1)/{deal_price} - 1"
        subscribe_fields.append(profit_str)

        trade_exchange = self._generate_exchange(
            trade_unit,
            limit_threshold,
            deal_price,
            subscribe_fields,
            shift,
            freq,
            open_cost,
            close_cost,
            min_cost,
        )
        long_returns = {}
        short_returns = {}
        ls_returns = {}
        for pdate, date in zip(predict_dates, trade_dates):

            print(pdate)
            stocks = self._generate_position(pdate)
            long_stocks = {k: v for k, v in stocks.items() if v > 0}
            short_stocks = {k: v for k, v in stocks.items() if v < 0}
            long_profit = 0
            short_profit = 0

            for stock, w in long_stocks.items():
                if not trade_exchange.is_stock_tradable(
                    stock_id=stock, start_time=pdate, end_time=pdate
                ):
                    continue
                profit = trade_exchange.get_quote_info(
                    stock_id=stock, start_time=pdate, end_time=pdate, field=profit_str
                )
                if np.isnan(profit):
                    long_profit += 0
                else:
                    long_profit += w * profit

            for stock, w in short_stocks.items():
                if not trade_exchange.is_stock_tradable(
                    stock_id=stock, start_time=pdate, end_time=pdate
                ):
                    continue
                profit = trade_exchange.get_quote_info(
                    stock_id=stock, start_time=pdate, end_time=pdate, field=profit_str
                )
                if np.isnan(profit):
                    short_profit += 0
                else:
                    short_profit += w * profit

            long_returns[date] = long_profit
            short_returns[date] = short_profit
            ls_returns[date] = long_profit + short_profit

        return dict(
            zip(
                ["long", "short", "long_short"],
                map(pd.Series, [long_returns, short_returns, ls_returns]),
            )
        )

    @abc.abstractmethod
    def _generate_position(self, date) -> dict:
        """{instrument: weight}"""
        raise NotImplementedError


class LongShortBacktest(_DfBacktest):
    def __init__(self, tabular_df, topk=1, long_weight=0.5):
        super().__init__(tabular_df)
        self.topk = topk
        self.long_weight = long_weight

    def _generate_position(self, date) -> dict:
        long_position = {}
        short_position = {}
        score = self.data.loc(axis=0)[date, :]
        score = score.reset_index().sort_values(by="score", ascending=False)

        long_stocks = list(score.iloc[: self.topk]["instrument"])
        for stock in long_stocks:
            long_position[stock] = 1 / len(long_stocks)
        short_stocks = list(score.iloc[-self.topk :]["instrument"])
        for stock in short_stocks:
            short_position[stock] = 1 / len(short_stocks)

        long_position.update(short_position)
        return long_position

    def run_backtest(
        self,
        freq: str = "day",
        deal_price=None,
        shift=1,
        open_cost=0,
        close_cost=0,
        trade_unit=None,
        limit_threshold=None,
        min_cost=0,
        subscribe_fields=[],
    ) -> dict:
        result = super(LongShortBacktest, self).run_backtest(
            freq,
            deal_price,
            shift,
            open_cost,
            close_cost,
            trade_unit,
            limit_threshold,
            min_cost,
            subscribe_fields,
        )

        result["long_short"] = (1 - self.long_weight) * result[
            "short"
        ] + self.long_weight * result["long"]
        return result


class CvxpyBacktest(_DfBacktest):
    def __init__(self, tabular_df: pd.DataFrame):
        super().__init__(tabular_df)
        self.rolling_cov = None

    def _generate_position(self, date) -> dict:
        self.cal_sigma()
        sigma = self.rolling_cov.loc(axis=0)[date, :]
        date_data = self.data.loc(axis=0)[date, :]
        if sigma.isna().any().any() or date_data["mu"].isna().any():
            return pd.Series(0, index=sigma.columns).to_dict()

        weight = self.mvo(date_data["mu"].values, sigma.values)
        return pd.Series(weight, index=sigma.columns).to_dict()

    def mvo(self, mu, sigma, Gamma=50, maxrisk=0.3):
        w = cp.Variable(len(sigma))
        risk = w @ sigma @ w.T
        objective = cp.Maximize(cp.sum(w * mu) - Gamma * risk)
        constraints = [
            cp.max(cp.abs(w)) <= 1,
            cp.sum(cp.abs(w)) <= 3,
            cp.abs(cp.sum(w)) <= 2,
            risk <= maxrisk**2 / 252,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return w.value

    def cal_sigma(self):
        if self.rolling_cov is not None:
            return self.rolling_cov
        else:
            ts_data = self.data.reset_index().pivot(
                index="datetime", columns="instrument", values="return"
            )
            window_size = 20
            self.rolling_cov = ts_data.ewm(span=window_size).cov()
            # self.rolling_cov = ts_data.rolling(window=window_size).cov()
            return self.rolling_cov


def long_short_backtest(
    pred,
    freq: str = "day",
    topk=1,
    deal_price=None,
    shift=1,
    open_cost=0,
    close_cost=0,
    trade_unit=None,
    limit_threshold=None,
    min_cost=0,
    subscribe_fields=[],
    long_weight=0.5,
):
    """
    A backtest for long-short strategy

    :param pred:        The trading signal produced on day `T`.
    :param freq:        freq.
    :param topk:       The short topk securities and long topk securities.
    :param deal_price:  The price to deal the trading.
    :param shift:       Whether to shift prediction by one day.  The trading day will be T+1 if shift==1.
    :param open_cost:   open transaction cost.
    :param close_cost:  close transaction cost.
    :param trade_unit:  100 for China A.
    :param limit_threshold: limit move 0.1 (10%) for example, long and short with same limit.
    :param min_cost:    min transaction cost.
    :param subscribe_fields: subscribe fields.
    :return:            The result of backtest, it is represented by a dict.
                        { "long": long_returns(excess),
                        "short": short_returns(excess),
                        "long_short": long_short_returns}
    """
    if get_level_index(pred, level="datetime") == 1:
        pred = pred.swaplevel().sort_index()

    if trade_unit is None:
        trade_unit = C.trade_unit
    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if deal_price is None:
        deal_price = C.deal_price
    if deal_price[0] != "$":
        deal_price = "$" + deal_price

    subscribe_fields = subscribe_fields.copy()
    profit_str = f"Ref({deal_price}, -1)/{deal_price} - 1"
    subscribe_fields.append(profit_str)

    _pred_dates = pred.index.get_level_values(level="datetime")
    predict_dates = D.calendar(start_time=_pred_dates.min(), end_time=_pred_dates.max())
    trade_dates = np.append(
        predict_dates[shift:],
        get_date_range(predict_dates[-1], left_shift=1, right_shift=shift),
    )

    trade_exchange = get_exchange(
        start_time=predict_dates[0],
        end_time=trade_dates[-1],
        freq=freq,
        codes=list(pred.index.get_level_values("instrument").unique()),
        deal_price=deal_price,
        subscribe_fields=subscribe_fields,
        limit_threshold=limit_threshold,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
        trade_unit=trade_unit,
    )

    long_returns = {}
    short_returns = {}
    ls_returns = {}

    for pdate, date in zip(predict_dates, trade_dates):
        score = pred.loc(axis=0)[pdate, :]
        score = score.reset_index().sort_values(by="score", ascending=False)

        long_stocks = list(score.iloc[:topk]["instrument"])
        short_stocks = list(score.iloc[-topk:]["instrument"])

        long_profit = []
        short_profit = []

        for stock in long_stocks:
            if not trade_exchange.is_stock_tradable(
                stock_id=stock, start_time=pdate, end_time=pdate
            ):
                continue
            profit = trade_exchange.get_quote_info(
                stock_id=stock, start_time=pdate, end_time=pdate, field=profit_str
            )
            if np.isnan(profit):
                long_profit.append(0)
            else:
                long_profit.append(profit)

        for stock in short_stocks:
            if not trade_exchange.is_stock_tradable(
                stock_id=stock, start_time=pdate, end_time=pdate
            ):
                continue
            profit = trade_exchange.get_quote_info(
                stock_id=stock, start_time=pdate, end_time=pdate, field=profit_str
            )
            if np.isnan(profit):
                short_profit.append(0)
            else:
                short_profit.append(profit * -1)

        long_returns[date] = np.mean(long_profit)
        short_returns[date] = np.mean(short_profit)
        ls_returns[date] = (1 - long_weight) * np.mean(
            short_profit
        ) + long_weight * np.mean(long_profit)

    return dict(
        zip(
            ["long", "short", "long_short"],
            map(pd.Series, [long_returns, short_returns, ls_returns]),
        )
    )
