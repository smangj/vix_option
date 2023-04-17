#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 17:52
# @Author   : wsy
# @email    : 631535207@qq.com

from backtest.qlib_backtest import BtPipeline, PortfolioConfig
from backtest.qlib_custom.strategy import TargetPositionStrategy
from backtest.simple_backtest import RollSignalBacktest


def target_order_backtest():
    init_cash = 1000000
    model = RollSignalBacktest()

    port_configs = [
        PortfolioConfig(
            name=str(i) + "M",
            strategy_config_or_inst=TargetPositionStrategy(
                records_list=model.gen_target_position(i, init_cash),
                calc_contract_multiplier=False,
            ),
        )
        for i in range(1, 7)
    ]
    # 仅加载所有组合涉及到的标的数据，降低数据准备时间
    all_codes = ["VIX_" + str(i) + "M" for i in range(1, 7)]
    pipeline = BtPipeline(
        experiment_name="RollSignalBacktest",
        port_configs=port_configs,
        start_date="2005-12-20",
        end_date="2023-03-06",
        init_cash=init_cash,
        codes=all_codes,
    )

    pipeline.run()


def model_based_backtest():
    pass


if __name__ == "__main__":
    model_based_backtest()
