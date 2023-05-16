#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 18:00
# @Author   : wsy
# @email    : 631535207@qq.com
import abc
import copy
import warnings
from dataclasses import dataclass, asdict
from typing import Union, Generator, Any, Sequence

import pandas as pd
import numpy as np
from qlib.backtest.decision import TradeDecisionWO, BaseTradeDecision, Order
from qlib.backtest.position import Position, BasePosition
from qlib.backtest.signal import SignalWCache
from qlib.strategy.base import BaseStrategy
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy


@dataclass(frozen=True)
class BaseRecord:
    stock_id: str
    datetime: pd.Timestamp

    def __post_init__(self):
        # 为Frozen dataclass的某个属性重新赋值 https://docs.python.org/3/library/dataclasses.html#frozen-instances
        # 确保datetime的类型为Pandas TimeStamp
        object.__setattr__(self, "datetime", pd.to_datetime(self.datetime))


@dataclass(frozen=True)
class TargetPositionRecord(BaseRecord):
    target_position: float
    """
    目标持仓数
    股票、ETF：股数
    期权、期货：合约数量
    """


@dataclass(frozen=True)
class OrderRecord(BaseRecord):
    volume: float
    """
    交易数量
    股票、ETF：股数
    期权、期货：合约数量
    """

    is_close_position: bool = False
    """
    是否清仓，若为True，则忽略volume，根据当前持仓情况生成订单，默认为False
    """


class ActualVolumeStrategyBase(BaseStrategy, metaclass=abc.ABCMeta):
    def __init__(
        self,
        records_list: Sequence[BaseRecord],
        calc_contract_multiplier: bool = True,
        adjust_volume_by_factor: bool = True,
        *,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ) -> None:
        """
        使用自定义数据集的策略，支持期权、股票，接受原始目标仓位数量/原始交易数量。
        自动获取期权乘数并计算期权合约实际的规模，并自动根据复权因子调整提交至回测系统的交易数量。

        :param records_list: 目标持仓/订单记录列表
        :param calc_contract_multiplier: 生成订单时是否考虑合约乘数，默认为True。
                                         Note：由于Qlib框架中并未区分股票、期权、期货，为正确计算账户市值，
                                              需要将期权、期货的交易数量、持仓数量转换为 {合约数量} * {合约乘数}
                                              {合约乘数}依赖于自定义添加至Qlib Data中的 {$contract_multi}字段
        :param adjust_volume_by_factor: 是否根据复权因子调整目标持仓数量，默认为Ture。
        """
        super().__init__(
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
            **kwargs,
        )

        assert records_list is not None and len(records_list) > 0
        records_list = [asdict(rec) for rec in records_list]
        records_df = (
            pd.DataFrame.from_records(records_list)
            .rename(
                columns={
                    "stock_id": "instrument",
                }
            )
            .set_index(["instrument", "datetime"])
        )

        assert records_df.index.duplicated().sum() == 0, f"输入的记录列表有重复记录: \n{records_df}"

        self._signal = SignalWCache(records_df)
        self._calc_contract_multiplier = calc_contract_multiplier
        self._adjust_volume_by_factor = adjust_volume_by_factor

    def _map_feature(
        self,
        stock_id: str,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        field: str,
        default: Union[None, int, float, bool] = None,
    ) -> Union[None, int, float, bool]:
        value = self.trade_exchange.quote.get_data(
            stock_id,
            trade_start_time,
            trade_end_time,
            field=field,
            method="ts_data_last",
        )
        if pd.isna(value):
            if default is not None:
                warnings.warn(
                    f"[{stock_id}]-[{trade_start_time}] {field} 值为NaN，返回默认值 {default}"
                )
                value = default
            else:
                raise ValueError(
                    f"[{stock_id}]-[{trade_start_time}] {field} 值为NaN，且未设置默认值"
                )
        return value

    @abc.abstractmethod
    def _generate_trade_decision(
        self,
        temp_current_position: BasePosition,
        now_records_df: pd.DataFrame,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def volume_col(self) -> str:
        raise NotImplementedError

    def generate_trade_decision(
        self, execute_result: list = None
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:

        temp_current_position = copy.deepcopy(self.trade_position)
        assert isinstance(temp_current_position, Position)  # Avoid InfPosition

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        # 获取当天的目标持仓DataFrame
        # Index："instrument"
        # Columns：["target_position"]
        now_records_df = self._signal.get_signal(
            start_time=trade_start_time, end_time=trade_end_time
        )
        if now_records_df is None:
            return TradeDecisionWO([], self)
        now_records_df = now_records_df.copy()

        if self._calc_contract_multiplier:
            now_records_df.loc[:, "contract_multi"] = now_records_df.index.map(
                lambda stock_id: self._map_feature(
                    stock_id=stock_id,
                    trade_start_time=trade_start_time,
                    trade_end_time=trade_end_time,
                    field="$contract_multi",
                )
            )
        else:
            now_records_df.loc[:, "contract_multi"] = 1

        if self._adjust_volume_by_factor:
            now_records_df.loc[:, "adjust_factor"] = now_records_df.index.map(
                lambda stock_id: self._map_feature(
                    stock_id=stock_id,
                    trade_start_time=trade_start_time,
                    trade_end_time=trade_end_time,
                    field="$factor",
                )
            )
        else:
            now_records_df.loc[:, "adjust_factor"] = 1

        now_records_df.loc[:, f"actual_{self.volume_col}"] = (
            now_records_df[self.volume_col]
            * now_records_df["contract_multi"]
            / now_records_df["adjust_factor"]
        )

        return self._generate_trade_decision(
            temp_current_position=temp_current_position,
            now_records_df=now_records_df,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )


class TargetPositionStrategy(ActualVolumeStrategyBase):
    """
    根据目标持仓数量进行自动调仓的策略，若某个交易日有记录，则未出现在记录表上的标的视为全部清仓
    """

    def __init__(
        self,
        records_list: Sequence[TargetPositionRecord],
        calc_contract_multiplier: bool = True,
        adjust_volume_by_factor: bool = True,
        *,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ) -> None:
        for rec in records_list:
            assert isinstance(rec, TargetPositionRecord)

        super().__init__(
            records_list,
            calc_contract_multiplier,
            adjust_volume_by_factor,
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

    def _generate_trade_decision(
        self,
        temp_current_position: BasePosition,
        now_records_df: pd.DataFrame,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        trade_target_dict = now_records_df[f"actual_{self.volume_col}"].to_dict()

        order_list = [
            order
            for order in self.trade_exchange.generate_order_for_target_amount_position(
                target_position=trade_target_dict,
                current_position=temp_current_position.get_stock_amount_dict(),
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            if abs(order.amount) >= 1e-5  # 去除数量极小的订单，防止float误差导致的多余无效订单
        ]

        return TradeDecisionWO(order_list, self)

    @property
    def volume_col(self) -> str:
        return "target_position"


class OrderStrategy(ActualVolumeStrategyBase):
    def __init__(
        self,
        records_list: Sequence[OrderRecord],
        calc_contract_multiplier: bool = True,
        adjust_volume_by_factor: bool = True,
        *,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ) -> None:
        for rec in records_list:
            assert isinstance(rec, OrderRecord)

        super().__init__(
            records_list,
            calc_contract_multiplier,
            adjust_volume_by_factor,
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

    def _generate_trade_decision(
        self,
        temp_current_position: BasePosition,
        now_records_df: pd.DataFrame,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        current_position_dict = temp_current_position.get_stock_amount_dict()
        buy_order_list = []
        sell_order_list = []
        for rec_row in now_records_df.itertuples():
            stock_id = rec_row.Index
            volume = getattr(rec_row, f"actual_{self.volume_col}")
            is_close_position = getattr(rec_row, "is_close_position")
            factor = getattr(rec_row, "adjust_factor")

            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue

            if is_close_position:
                current_position_volume = current_position_dict.get(stock_id, 0)
                if current_position_volume == 0:
                    continue
                # FIXME: 平仓时不对委托数量做round，虽然当前Qlib不支持空仓，但也暂时直接生成持仓方向相反的委托
                direction = Order.BUY if current_position_volume < 0 else Order.SELL
                order_list = (
                    buy_order_list if direction == Order.BUY else sell_order_list
                )
                order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=abs(current_position_volume),
                        direction=direction,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        factor=factor,
                    )
                )
            else:
                direction = Order.BUY if volume > 0 else Order.SELL
                sign = np.sign(volume)
                rounded_volume = sign * self.trade_exchange.round_amount_by_trade_unit(
                    abs(volume), factor
                )
                order_list = (
                    buy_order_list if direction == Order.BUY else sell_order_list
                )
                order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=abs(rounded_volume),
                        direction=direction,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        factor=factor,
                    )
                )
        # 确保先卖后买
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

    @property
    def volume_col(self) -> str:
        return "volume"


class SimpleSignalStrategy(BaseSignalStrategy):
    """配置signal为正的资产"""

    def generate_trade_decision(
        self, execute_result: list = None
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:

        temp_current_position = copy.deepcopy(self.trade_position)
        assert isinstance(temp_current_position, Position)  # Avoid InfPosition

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )
        pred_score = self.signal.get_signal(
            start_time=pred_start_time, end_time=pred_end_time
        )
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        # load score
        equity = temp_current_position.calculate_value()

        # 要卖的是昨日有持仓，今日score为负的
        buy = pred_score.loc[pred_score > 0].index
        # equal weighted
        trade_target_dict = {
            stock_id: equity * self.risk_degree / len(buy) if len(buy) > 0 else 0
            for stock_id in buy
        }

        order_list = [
            order
            for order in self.trade_exchange.generate_order_for_target_amount_position(
                target_position=trade_target_dict,
                current_position=temp_current_position.get_stock_amount_dict(),
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            if abs(order.amount) >= 1e-5  # 去除数量极小的订单，防止float误差导致的多余无效订单
        ]

        return TradeDecisionWO(order_list, self)
