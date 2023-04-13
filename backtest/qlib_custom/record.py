#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import logging
import os
import tempfile
import warnings
from pprint import pprint
from typing import Union, List, Optional

import pandas as pd
from qlib.backtest import backtest as normal_backtest
from qlib.contrib.evaluate import risk_analysis, indicator_analysis
from qlib.log import get_module_logger
from qlib.utils import flatten_dict
from qlib.utils.time import Freq
from qlib.workflow.record_temp import PortAnaRecord

from .utils import gen_acct_pos_dfs, gen_orders_df

__author__ = "Vitor Chen"
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

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
