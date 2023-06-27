#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc

import numpy as np
from dataclasses import dataclass
import logging
import os
import tempfile
import warnings
from pprint import pprint
from typing import Union, List, Optional
from qlib.data import D
from qlib.data.dataset.utils import get_level_index
import pandas as pd
from qlib.backtest import backtest as normal_backtest, get_exchange
from qlib.contrib.evaluate import risk_analysis, indicator_analysis
from qlib.config import C
from qlib.log import get_module_logger
from qlib.utils import flatten_dict, get_date_range
from qlib.utils.time import Freq
from qlib.workflow.record_temp import PortAnaRecord

from backtest.qlib_custom.data_handler import SimpleVixHandler
from backtest.qlib_custom.utils import gen_acct_pos_dfs, gen_orders_df
from backtest.report import report, Values

logger = get_module_logger("workflow", logging.INFO)


@dataclass
class HistFilePaths:
    position_hist_path: str
    account_hist_path: str
    orders_hist_path: str


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
        ls_returns[date] = 0.5 * np.mean(short_profit) + 0.5 * np.mean(long_profit)

    return dict(
        zip(
            ["long", "short", "long_short"],
            map(pd.Series, [long_returns, short_returns, ls_returns]),
        )
    )


class StandalonePortAnaRecord(PortAnaRecord):
    """
    可独立使用的PortAnaRecord，不依赖于SignalRecord；同时生成账户市值、持仓、交易订单历史记录表。
    基于qlib.workflow.record_temp.PortAnaRecord修改
    """

    depend_cls = None

    _POSITION_HIST_EXCEL_FORMAT = "position_{}.xlsx"
    _ACCOUNT_HIST_EXCEL_FORMAT = "account_{}.xlsx"
    _ORDERS_HIST_EXCEL_FORMAT = "orders_{}.xlsx"

    def __init__(
        self,
        recorder,
        config=None,
        risk_analysis_freq: Union[List, str] = None,
        indicator_analysis_freq: Union[List, str] = None,
        indicator_analysis_method=None,
        skip_existing=False,
        artifact_sub_path: Optional[str] = None,
        **kwargs,
    ):
        """

        :param artifact_sub_path: 结果文件子目录，用于分别存放同次回测的多个策略的结果，默认为None。
                                  若该参数为None时，则结果文件存放于{record_id}/artifacts/portfolio_analysis下；
                                  若该参数不为None时，则结果文件存放于{record_id}/artifacts/portfolio_analysis/{artifact_sub_path}

        """

        super().__init__(
            recorder,
            config,
            risk_analysis_freq,
            indicator_analysis_freq,
            indicator_analysis_method,
            skip_existing,
            **kwargs,
        )

        self._artifact_sub_path = artifact_sub_path

    def _save_df(self, df: pd.DataFrame, file_name: str, dir_path: str):
        file_path = os.path.join(dir_path, file_name)
        df.to_excel(file_path, index=False)
        pprint(file_path)
        self.save(local_path=file_path)

    def _generate(self, **kwargs):
        # Copy From PortAnaRecord._generate

        if self.backtest_config["start_time"] is None:
            raise ValueError("start_time is None!")
        if self.backtest_config["end_time"] is None:
            raise ValueError("end_time is None!")

        # custom strategy and get backtest
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=self.executor_config,
            strategy=self.strategy_config,
            **self.backtest_config,
        )

        with tempfile.TemporaryDirectory() as dir_path:
            for _freq, (
                report_normal,
                positions_normal,
            ) in portfolio_metric_dict.items():
                self.save(**{f"report_normal_{_freq}.pkl": report_normal})
                self.save(**{f"positions_normal_{_freq}.pkl": positions_normal})

                pprint(f"提取账户净值、持仓历史表，周期：{_freq}...")
                account_df, position_df = gen_acct_pos_dfs(
                    hist_positions=positions_normal
                )
                self._save_df(
                    df=account_df,
                    file_name=self._ACCOUNT_HIST_EXCEL_FORMAT.format(_freq),
                    dir_path=dir_path,
                )
                self._save_df(
                    df=position_df,
                    file_name=self._POSITION_HIST_EXCEL_FORMAT.format(_freq),
                    dir_path=dir_path,
                )

            for _freq, indicators_normal in indicator_dict.items():
                self.save(**{f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
                self.save(
                    **{f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]}
                )

                pprint(f"提取交易订单历史表，周期：{_freq}...")
                orders_df = gen_orders_df(indicator_obj=indicators_normal[1])
                self._save_df(
                    df=orders_df,
                    file_name=self._ORDERS_HIST_EXCEL_FORMAT.format(_freq),
                    dir_path=dir_path,
                )

            for _analysis_freq in self.risk_analysis_freq:
                if _analysis_freq not in portfolio_metric_dict:
                    warnings.warn(
                        f"the freq {_analysis_freq} report is not found,"
                        " please set the corresponding env with `generate_portfolio_metrics=True`"
                    )
                else:
                    report_normal, _ = portfolio_metric_dict.get(_analysis_freq)
                    analysis = dict()
                    analysis["excess_return_without_cost"] = risk_analysis(
                        report_normal["return"] - report_normal["bench"],
                        freq=_analysis_freq,
                    )
                    analysis["excess_return_with_cost"] = risk_analysis(
                        report_normal["return"]
                        - report_normal["bench"]
                        - report_normal["cost"],
                        freq=_analysis_freq,
                    )

                    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                    # log metrics
                    analysis_dict = flatten_dict(
                        analysis_df["risk"].unstack().T.to_dict()
                    )
                    self.recorder.log_metrics(
                        **{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()}
                    )
                    # save results
                    self.save(**{f"port_analysis_{_analysis_freq}.pkl": analysis_df})
                    logger.info(
                        f"Portfolio analysis record 'port_analysis_{_analysis_freq}.pkl'"
                        f" has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                    )
                    # print out results
                    pprint(
                        f"The following are analysis results of benchmark return({_analysis_freq})."
                    )
                    pprint(risk_analysis(report_normal["bench"], freq=_analysis_freq))
                    pprint(
                        f"The following are analysis results of the excess return without cost({_analysis_freq})."
                    )
                    pprint(analysis["excess_return_without_cost"])
                    pprint(
                        f"The following are analysis results of the excess return with cost({_analysis_freq})."
                    )
                    pprint(analysis["excess_return_with_cost"])

            for _analysis_freq in self.indicator_analysis_freq:
                if _analysis_freq not in indicator_dict:
                    warnings.warn(f"the freq {_analysis_freq} indicator is not found")
                else:
                    indicators_normal = indicator_dict.get(_analysis_freq)[0]
                    if self.indicator_analysis_method is None:
                        analysis_df = indicator_analysis(indicators_normal)
                    else:
                        analysis_df = indicator_analysis(
                            indicators_normal, method=self.indicator_analysis_method
                        )
                    # log metrics
                    analysis_dict = analysis_df["value"].to_dict()
                    self.recorder.log_metrics(
                        **{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()}
                    )
                    # save results
                    self.save(
                        **{f"indicator_analysis_{_analysis_freq}.pkl": analysis_df}
                    )
                    logger.info(
                        f"Indicator analysis record 'indicator_analysis_{_analysis_freq}.pkl'"
                        f" has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                    )
                    pprint(
                        f"The following are analysis results of indicators({_analysis_freq})."
                    )
                    pprint(analysis_df)

    def list(self):
        # 父类的list中缺了两个indicators文件，补充完整

        full_list = super().list()

        for _freq in self.all_freq:
            indicators_files = [
                f"indicators_normal_{_freq}.pkl",
                f"indicators_normal_{_freq}_obj.pkl",
                self._ORDERS_HIST_EXCEL_FORMAT.format(_freq),
                self._ACCOUNT_HIST_EXCEL_FORMAT.format(_freq),
                self._POSITION_HIST_EXCEL_FORMAT.format(_freq),
            ]
            for _file in indicators_files:
                if _file not in full_list:
                    full_list.append(_file)

        return full_list

    def get_path(self, path=None):
        """
        重载基类类方法get_path, 支持对子目录的设置
        """

        base_path = super().get_path(path)
        return (
            os.path.join(base_path, self._artifact_sub_path)
            if self._artifact_sub_path is not None
            else base_path
        )

    def get_hist_file_paths(self, freq="day") -> HistFilePaths:
        _count, _freq = Freq.parse(freq)
        freq_tag = f"{_count}{_freq}"
        if freq_tag not in self.all_freq:
            raise ValueError(f"the freq {freq_tag} history is not found")
        return HistFilePaths(
            position_hist_path=os.path.join(
                self.get_path(), self._POSITION_HIST_EXCEL_FORMAT.format(freq_tag)
            ),
            account_hist_path=os.path.join(
                self.get_path(), self._ACCOUNT_HIST_EXCEL_FORMAT.format(freq_tag)
            ),
            orders_hist_path=os.path.join(
                self.get_path(), self._ORDERS_HIST_EXCEL_FORMAT.format(freq_tag)
            ),
        )


class _SimpleBacktestRecord(PortAnaRecord, abc.ABC):
    """
    调用SimpleBacktest回测并保存相关结果
    只支持根据score生成signal的简单策略
    """

    _NET_VALUE_EXCEL_FORMAT = "net_value_{}.xlsx"

    def __init__(
        self,
        recorder,
        config=None,
        risk_analysis_freq: Union[List, str] = None,
        indicator_analysis_freq: Union[List, str] = None,
        indicator_analysis_method=None,
        skip_existing=False,
        **kwargs,
    ):
        super().__init__(
            recorder,
            config,
            risk_analysis_freq,
            indicator_analysis_freq,
            indicator_analysis_method,
            skip_existing,
            **kwargs,
        )
        self._fields, names = SimpleVixHandler.get_features()
        self._fields.append("$close")
        names.append("close")
        self._freq = "day"
        market_data_df = D.features(
            instruments=["VIX_1M", "VIX_2M", "VIX_3M", "VIX_4M", "VIX_5M", "VIX_6M"],
            fields=self._fields,
            freq=self._freq,
            disk_cache=1,
        )
        market_data_df.columns = names
        self._data = market_data_df.swaplevel().sort_index()

    def _save_df(self, df: pd.DataFrame, file_name: str, dir_path: str):
        file_path = os.path.join(dir_path, file_name)
        df.to_excel(file_path, index=True)
        pprint(file_path)
        self.save(local_path=file_path)

    def _generate(self, *args, **kwargs):
        pred = self.load("pred.pkl")
        label_df = self.load("label.pkl").dropna()
        label_df.columns = ["label"]

        pred_label = pd.concat([pred, label_df, self._data], axis=1, sort=True).reindex(
            label_df.index
        )
        assert "instrument" in pred_label.index.names
        assert "datetime" in pred_label.index.names
        instruments = list(pred_label.index.get_level_values("instrument").unique())
        dt_values = pred_label.index.get_level_values("datetime")

        start_time = (
            dt_values[0]
            if self.backtest_config["start_time"] is None
            else self.backtest_config["start_time"]
        )
        end_time = (
            dt_values[-1]
            if self.backtest_config["end_time"] is None
            else self.backtest_config["end_time"]
        )
        time_mask = (dt_values >= pd.to_datetime(start_time)) & (
            dt_values <= pd.to_datetime(end_time)
        )
        pred_label = pred_label.loc[time_mask]

        # temp
        # pred_label = pred_label.drop(index=pd.to_datetime("2018-02-05"))

        bt_dt_values = (
            pred_label.index.get_level_values("datetime").unique().sort_values()
        )

        net_values = []
        for instrument in instruments:
            xt = self._generate_signal(
                pred_label.loc[(slice(None), instrument), :], instrument
            )
            # 简单规则：
            xt["trading_flag" + "_" + instrument] = xt["signal"]

            xt["next_ret"] = (xt["close"].shift(-1) / xt["close"]) - 1
            xt["daily_ret"] = xt["next_ret"] * xt["trading_flag" + "_" + instrument]
            xt[instrument] = np.cumprod(xt["daily_ret"] + 1).shift(1).fillna(1)
            net_value = xt[[instrument, "trading_flag" + "_" + instrument]]
            net_value.index = bt_dt_values
            net_value.name = instrument
            net_values.append(net_value)
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            file_path = report(
                [Values(nv.name, nv[nv.name]) for nv in net_values],
                output_dir=tmp_dir_path,
                file_name=self.name + "_report",
            )
            self.recorder.log_artifact(local_path=file_path)
            values_df = pd.concat(net_values, axis=1)
            self._save_df(
                df=values_df,
                file_name=self.name
                + "_"
                + self._NET_VALUE_EXCEL_FORMAT.format(self._freq),
                dir_path=tmp_dir_path,
            )
            self._save_df(
                df=pred_label,
                file_name="pred_label.xlsx",
                dir_path=tmp_dir_path,
            )

    def _generate_signal(self, score_df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        score_df: columns=["score", "label"]
        return: columns新增"signal"
        """
        raise NotImplementedError

    def list(self):

        full_list = super().list()

        for _freq in self.all_freq:
            indicators_files = [
                f"indicators_normal_{_freq}.pkl",
                f"indicators_normal_{_freq}_obj.pkl",
                self._NET_VALUE_EXCEL_FORMAT.format(_freq),
            ]
            for _file in indicators_files:
                if _file not in full_list:
                    full_list.append(_file)

        return full_list

    @property
    def name(self) -> str:
        return "empty"


class Cross(_SimpleBacktestRecord):
    def _generate_signal(self, score_df, instrument) -> pd.DataFrame:
        xt = score_df.copy()
        month = instrument.split("_")[1][0]
        ma_long = 20
        ma_short = 5
        xt["ma_long"] = xt["ln_V" + str(month)].rolling(window=ma_long).mean()
        xt["ma_short"] = xt["ln_V" + str(month)].rolling(window=ma_short).mean()

        buy_signal = xt["ma_short"] > xt["ma_long"]
        sell_signal = xt["ma_short"] <= xt["ma_long"]

        xt["signal"] = 1 * buy_signal - 1 * sell_signal
        return xt

    @property
    def name(self) -> str:
        return "cross_strategy"


class MeanReversion(_SimpleBacktestRecord):
    def _generate_signal(self, score_df, instrument) -> pd.DataFrame:
        xt = score_df.copy()
        month = instrument.split("_")[1][0]
        roll = 20
        xt["vix_ma"] = xt["ln_VIX"].rolling(window=roll).mean()
        xt["vix_std"] = xt["ln_VIX"].rolling(window=roll).std()
        std_times = 2
        position_num = 1
        buy_signal = xt["ln_V" + str(month)] <= (
            xt["vix_ma"] - std_times * xt["vix_std"]
        )
        sell_signal = xt["ln_V" + str(month)] >= (
            xt["vix_ma"] + std_times * xt["vix_std"]
        )

        xt["signal"] = position_num * buy_signal - position_num * sell_signal
        xt["signal"] = (
            xt["signal"].replace(0, np.nan).fillna(method="ffill").replace(np.nan, 0)
        )
        return xt

    @property
    def name(self) -> str:
        return "vix_reversion_strategy"


class RollSignal(_SimpleBacktestRecord):
    def _generate_signal(self, score_df, instrument) -> pd.DataFrame:
        xt = score_df.copy()
        month = instrument.split("_")[1][0]
        buy_signal = xt["roll" + str(month)] > 0
        sell_signal = xt["roll" + str(month)] <= 0

        xt["signal"] = 1 * buy_signal - 1 * sell_signal
        return xt

    @property
    def name(self) -> str:
        return "roll_signal_strategy"


class MuSignal(_SimpleBacktestRecord):
    def _generate_signal(self, score_df, instrument) -> pd.DataFrame:
        xt = score_df.copy()
        month = instrument.split("_")[1][0]
        buy_signal = xt["mu" + str(month)] <= 0
        sell_signal = xt["mu" + str(month)] > 0

        xt["signal"] = 1 * buy_signal - 1 * sell_signal
        return xt

    @property
    def name(self) -> str:
        return "mu_signal_strategy"


class MuBufferSignal(_SimpleBacktestRecord):
    def _generate_signal(self, score_df, instrument) -> pd.DataFrame:
        xt = score_df.copy()
        month = instrument.split("_")[1][0]
        buy_signal = xt["mu" + str(month)] <= -0.05
        sell_signal = xt["mu" + str(month)] >= 0.05

        xt["signal"] = 1 * buy_signal - 1 * sell_signal
        return xt

    @property
    def name(self) -> str:
        return "mu_buffer_signal_strategy"


class ScoreSign(_SimpleBacktestRecord):
    def _generate_signal(self, score_df, instrument) -> pd.DataFrame:
        xt = score_df.copy()
        buy_signal = xt["score"] > 0
        sell_signal = xt["score"] <= 0

        xt["signal"] = 1 * buy_signal - 1 * sell_signal
        return xt

    @property
    def name(self) -> str:
        return "score_sign"


class LongShortBacktestRecord(_SimpleBacktestRecord):
    def _generate_signal(self, score_df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        pass

    def _generate(self, *args, **kwargs):
        pred = self.load("pred.pkl")
        label_df = self.load("label.pkl").dropna()
        label_df.columns = ["label"]

        dt_values = pred.index.get_level_values("datetime")

        start_time = (
            dt_values[0]
            if self.backtest_config["start_time"] is None
            else self.backtest_config["start_time"]
        )
        end_time = (
            dt_values[-1]
            if self.backtest_config["end_time"] is None
            else self.backtest_config["end_time"]
        )
        time_mask = (dt_values >= pd.to_datetime(start_time)) & (
            dt_values <= pd.to_datetime(end_time)
        )
        pred = pred.loc[time_mask]
        pred_label = pd.concat([pred, label_df, self._data], axis=1, sort=True).reindex(
            pred.index
        )

        result = long_short_backtest(
            pred,
            freq=self._freq,
            topk=1,
            shift=1,
            open_cost=0,
            close_cost=0,
            min_cost=0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            file_path = report(
                [Values(k, np.cumprod(nv + 1).dropna()) for k, nv in result.items()],
                output_dir=tmp_dir_path,
                file_name=self.name + "_report",
            )
            self.recorder.log_artifact(local_path=file_path)
            values_df = pd.DataFrame(
                {k: np.cumprod(nv + 1).dropna() for k, nv in result.items()}
            )
            self._save_df(
                df=values_df,
                file_name=self.name
                + "_"
                + self._NET_VALUE_EXCEL_FORMAT.format(self._freq),
                dir_path=tmp_dir_path,
            )
            self._save_df(
                df=pred_label,
                file_name="pred_label.xlsx",
                dir_path=tmp_dir_path,
            )

    @property
    def name(self) -> str:
        return "LongShortBacktestRecord"


if __name__ == "__main__":
    import qlib
    from qlib.workflow import R

    qlib.init(provider_uri="data/qlib_data", region="us")

    recorder = R.get_recorder(
        recorder_id="ee16729a846348ceacbe20aedc76652f", experiment_name="gru_xt"
    )

    r = LongShortBacktestRecord(
        recorder=recorder,
    )
    r.generate()
