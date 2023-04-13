#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 14:14
# @Author   : wsy
# @email    : 631535207@qq.com
import datetime as dt
import tempfile
from dataclasses import dataclass
from pprint import pprint
from typing import Union, Optional, List
import pandas as pd
import qlib
from qlib.backtest import get_exchange
from qlib.config import SIMPLE_DATASET_CACHE
from qlib.constant import REG_US
from . import report
from qlib.log import TimeInspector
from qlib.workflow import R

from utils.path import DEFAULT_QLIB_PROVIDER_URI
from .qlib_custom.record import StandalonePortAnaRecord


@dataclass
class Values:
    name: str
    values: pd.Series


EXCHANGE_CONFIG_BASE = {
    "freq": "day",
    "deal_price": "close",
    # "open_cost": 0.0005,
    # "close_cost": 0.0015,
    # "min_cost": 5,
    "open_cost": 0,
    "close_cost": 0,
    "min_cost": 0,
}

PORT_ANALYSIS_CONFIG_BASE = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
}


def _gen_backtest_config(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    account: float,
    exchange_config_or_inst: Union[dict, object],
    benchmark: str = None,
) -> dict:
    result = {
        "start_time": start_time,
        "end_time": end_time,
        "account": account,
        "benchmark": benchmark,
        "exchange_kwargs": {"exchange": exchange_config_or_inst},
    }
    return result


@dataclass
class PortfolioConfig:
    strategy_config_or_inst: Union[dict, object]
    name: Optional[str] = None


class BtPipeline:
    def __init__(
        self,
        experiment_name: str,
        port_configs: Union[PortfolioConfig, List[PortfolioConfig]],
        start_date: Union[pd.Timestamp, str, dt.datetime, dt.date],
        end_date: Union[pd.Timestamp, str, dt.datetime, dt.date],
        init_cash: float = 10000000,
        codes: Union[list, str] = "all",
        qlib_provider_uri: str = DEFAULT_QLIB_PROVIDER_URI,
        dataset_cache: str = SIMPLE_DATASET_CACHE,
    ) -> None:
        super().__init__()
        self._experiment_name = experiment_name

        # 暂时仅支持日度回测，确保start_date和end_date精确到日
        self._start_date = str(pd.to_datetime(start_date).date())
        self._end_date = str(pd.to_datetime(end_date).date())
        self._init_cash = init_cash

        self._qlib_provider_uri = qlib_provider_uri
        self._dataset_cache = dataset_cache

        self._exchange_config = EXCHANGE_CONFIG_BASE.copy()
        self._exchange_config["codes"] = codes
        self._exchange_config["start_time"] = self._start_date
        self._exchange_config["end_time"] = self._end_date

        if isinstance(port_configs, PortfolioConfig):
            port_configs = [port_configs]
        else:
            portfolio_names = set()
            for _port_conf in port_configs:
                assert (
                    _port_conf.name is not None and len(_port_conf.name) > 0
                ), "设置同时回测多个组合时，必须为每个组合设置name"
                portfolio_names.add(_port_conf.name)
            assert len(portfolio_names) == len(port_configs), "存在重复的组合名称"

        self._port_configs = port_configs

    def run(self):
        # 必须使用REG_US, 因为REG_CN会限制成交股数为100的整数，并且有涨停板限制，若回测涉及到期权，则这些限制都不符合实际情况，
        # qlib默认配置详情见：qlib.config模块
        qlib.init(
            provider_uri=self._qlib_provider_uri,
            region=REG_US,
            dataset_cache=self._dataset_cache,
        )

        with TimeInspector.logt("创建交易所对象，准备市场数据..."):
            exchange_instance = get_exchange(**self._exchange_config)

        shared_port_analysis_config = PORT_ANALYSIS_CONFIG_BASE.copy()
        shared_port_analysis_config["backtest"] = _gen_backtest_config(
            start_time=self._start_date,
            end_time=self._end_date,
            account=self._init_cash,
            benchmark="VIX_1M",
            exchange_config_or_inst=exchange_instance,
        )

        with TimeInspector.logt("回测阶段..."):
            with R.start(experiment_name=self._experiment_name):
                recorder = R.get_recorder()
                ba_rid = recorder.id

                report_value_list: List[Values] = []

                for port_config in self._port_configs:
                    _port_analysis_config = shared_port_analysis_config.copy()
                    _port_analysis_config[
                        "strategy"
                    ] = port_config.strategy_config_or_inst
                    # backtest & analysis
                    par = StandalonePortAnaRecord(
                        recorder,
                        _port_analysis_config,
                        artifact_sub_path=port_config.name,
                    )
                    par.generate()

                    hist_paths = par.get_hist_file_paths()
                    full_account_hist_path = recorder.download_artifact(
                        hist_paths.account_hist_path
                    )
                    account_df = pd.read_excel(
                        full_account_hist_path,
                        index_col="datetime",
                        parse_dates=["datetime"],
                    )
                    value_se = account_df["now_account_value"].sort_index()
                    report_value_list.append(
                        Values(name=port_config.name, values=value_se)
                    )

                with tempfile.TemporaryDirectory() as tmp_dir_path:
                    file_path = report.report(
                        report_value_list,
                        rf=0.03,
                        output_dir=tmp_dir_path,
                        file_name="report",
                    )
                    recorder.log_artifact(local_path=file_path)

                pprint(f"回测结束，记录ID：{ba_rid}")
