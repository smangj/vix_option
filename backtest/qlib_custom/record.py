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
import pandas as pd
from qlib.backtest import backtest as normal_backtest
from qlib.contrib.evaluate import risk_analysis, indicator_analysis
from qlib.log import get_module_logger
from qlib.utils import flatten_dict
from qlib.utils.time import Freq
from qlib.workflow.record_temp import PortAnaRecord
from qlib.utils import init_instance_by_config

from backtest.qlib_custom._dfbacktest import (
    LongShortBacktest,
    CvxpyBacktest,
)
from backtest.qlib_custom.utils import gen_acct_pos_dfs, gen_orders_df
from backtest.report import report, Values

logger = get_module_logger("workflow", logging.INFO)


@dataclass
class HistFilePaths:
    position_hist_path: str
    account_hist_path: str
    orders_hist_path: str


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
        self._fields = [
            "$close",
            "$ln_VIX",
            "$ln_V1",
            "$ln_V2",
            "$ln_V3",
            "$ln_V4",
            "$ln_V5",
            "$ln_V6",
            "$ln_SPY",
            "$ln_TLT",
            "$roll1",
            "$roll2",
            "$roll3",
            "$roll4",
            "$roll5",
            "$roll6",
        ]
        self._freq = "day"
        market_data_df = D.features(
            instruments=["VIX_1M", "VIX_2M", "VIX_3M", "VIX_4M", "VIX_5M", "VIX_6M"],
            fields=self._fields,
            freq=self._freq,
            disk_cache=1,
        ).rename(columns={x: x[1:] for x in self._fields})
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


class _DfBacktestRecord(PortAnaRecord, abc.ABC):
    """
    调用_DfBacktest回测并保存相关结果
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
        handler = kwargs.pop("handler", None)
        if handler is None:
            handler = {
                "class": "VixHandler",
                "module_path": "backtest.qlib_custom.data_handler",
                "kwargs": {
                    "start_time": "2005-12-20",
                    "end_time": "2023-03-06",
                    "instruments": "trable",
                },
            }
        handler = init_instance_by_config(handler)
        super().__init__(
            recorder,
            config,
            risk_analysis_freq,
            indicator_analysis_freq,
            indicator_analysis_method,
            skip_existing,
            **kwargs,
        )

        self._freq = "day"
        market_data_df = D.features(
            instruments=["VIX_1M", "VIX_2M", "VIX_3M", "VIX_4M", "VIX_5M", "VIX_6M"],
            fields=["$close / Ref($close, 1) - 1"],
            freq=self._freq,
            disk_cache=1,
        )
        market_data_df.columns = ["return"]
        features = handler._data
        features.columns = features.columns.get_level_values(1)
        market_data_df = pd.merge(
            market_data_df.swaplevel(), features, left_index=True, right_index=True
        )
        self._data = market_data_df.sort_index()

    def _save_df(self, df: pd.DataFrame, file_name: str, dir_path: str):
        file_path = os.path.join(dir_path, file_name)
        if not os.path.exists(file_path):
            df.to_excel(file_path, index=True)
            pprint(file_path)
            self.save(local_path=file_path)

    @abc.abstractmethod
    def _generate(self, *args, **kwargs):

        raise NotImplementedError

    def _process_data(self):
        "time_mask and merge self._data"
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
        return pred, pred_label

    def _save(self, result: dict, data: pd.DataFrame):
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
                df=data,
                file_name="pred_label.xlsx",
                dir_path=tmp_dir_path,
            )

    def list(self):

        full_list = super().list()

        for _freq in self.all_freq:
            indicators_files = [
                self._NET_VALUE_EXCEL_FORMAT.format(_freq),
            ]
            for _file in indicators_files:
                if _file not in full_list:
                    full_list.append(_file)

        return full_list

    @property
    def name(self) -> str:
        return "empty"


class LongShortBacktestRecord(_DfBacktestRecord):
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
        self.short_weight = kwargs.pop("short_weight", 0.0)

        super().__init__(
            recorder,
            config,
            risk_analysis_freq,
            indicator_analysis_freq,
            indicator_analysis_method,
            skip_existing,
            **kwargs,
        )

    def _generate(self, *args, **kwargs):
        pred, pred_label = self._process_data()

        back = LongShortBacktest(
            tabular_df=pred, topk=1, long_weight=(1.0 - self.short_weight) / 2
        )
        result = back.run_backtest(
            freq=self._freq,
            shift=1,
            open_cost=0,
            close_cost=0,
            min_cost=0,
        )

        self._save(result, pred_label)

    @property
    def name(self) -> str:
        return "LongShortBacktestRecord"


class JiaQiRecord(_DfBacktestRecord):
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

    def _generate(self, *args, **kwargs):
        pred, pred_label = self._process_data()

        back = CvxpyBacktest(pred_label)
        result = back.run_backtest(
            freq=self._freq,
            shift=1,
            open_cost=0,
            close_cost=0,
            min_cost=0,
        )

        self._save(result, pred_label)

    @property
    def name(self) -> str:
        return "JiaQiRecord"


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
