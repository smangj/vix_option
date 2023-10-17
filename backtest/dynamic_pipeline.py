#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/23 9:33
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.model.ens.ensemble import RollingEnsemble
import qlib
import os
from pathlib import Path
from qlib.model.trainer import TrainerR, TrainerRM, _log_task_info
from qlib.workflow.task.manage import TaskManager
from qlib.utils import init_instance_by_config
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow import R
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector


class RollingBenchmark:
    """
    roll_train_pipeline
    for one model!!
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self.step = config["roll_config"].get("step")
        self.horizon = config["roll_config"].get("horizon")
        self.rolling_type = config["roll_config"].get("rolling_type")

        self.COMB_EXP = config.get("experiment_name")
        self.rolling_exp = self.COMB_EXP + "_rolling_exp"
        self._handler_path = None

        if self.rolling_type is None or self.rolling_type == "expanding":
            self.rolling_type = RollingGen.ROLL_EX
        elif self.rolling_type == "rolling_window":
            self.rolling_type = RollingGen.ROLL_SD
        else:
            raise AttributeError("无法识别的rolling_type")

        task_pool = config.get("task_pool")
        if task_pool is None:
            self.trainer = TrainerR(experiment_name=self.rolling_exp)
        else:
            self.task_pool = task_pool
            self.trainer = TrainerRM(self.rolling_exp, self.task_pool)

    def basic_task(self):
        conf = deepcopy_basic_type(self._config)

        task = conf["task"]

        name = task.get("name")
        if name is None:
            name = task["model"]["class"]
        h_path = Path.cwd() / "{}_handler_horizon{}.pkl".format(name, self.horizon)
        if not h_path.exists():
            h_conf = task["dataset"]["kwargs"]["handler"]
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)

        self._handler_path = h_path
        task["dataset"]["kwargs"]["handler"] = h_path
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]

        return task

    def reset(self):
        print("========== reset ==========")
        if isinstance(self.trainer, TrainerRM):
            TaskManager(task_pool=self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.rolling_exp)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    def create_rolling_tasks(self):
        task = self.basic_task()
        task_l = task_generator(
            task,
            RollingGen(
                step=self.step, trunc_days=self.horizon + 1, rtype=self.rolling_type
            ),
        )  # the last two days should be truncated to avoid information leakage
        return task_l

    def train_rolling_tasks(self, task_l=None):
        if task_l is None:
            task_l = self.create_rolling_tasks()
        self.reset()
        print("========== task_training ==========")
        self.trainer.train(task_l)

    def ens_rolling(self):
        print("========== task_collecting ==========")

        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()

        with R.start(
            experiment_name=self.COMB_EXP,
            recorder_name=self._config["task"].get("name"),
        ):
            _log_task_info(self._config["task"])
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
            recorder = R.get_recorder()
            return recorder

    def update_rolling_rec(self, recorder):
        """
        Evaluate the combined rolling results
        """

        for record in self._config["task"]["record"]:
            r = init_instance_by_config(
                record,
                recorder=recorder,
                default_module="qlib.workflow.record_temp",
                try_kwargs={"handler": self._handler_path},
            )
            r.generate()

        print(
            f"Your evaluation results can be found in the experiment named `{self.COMB_EXP}`."
        )

    def run_all(self):
        try:
            # 1) each rolling task is saved in rolling_models
            self.train_rolling_tasks()
            # 2) combined rolling tasks and evaluation results are saved in rolling
            recorder = self.ens_rolling()
            self.update_rolling_rec(recorder)
        finally:
            # 删除临时handler文件
            if self._handler_path is not None:
                os.remove(self._handler_path)
        return recorder


if __name__ == "__main__":

    qlib.init(provider_uri="data/qlib_data", region="us")
    config = {}
    config["roll_config"] = {}
    config["roll_config"]["step"] = 20
    config["roll_config"]["horizon"] = 0
    config["roll_config"][
        "rolling_exp"
    ] = "rolling_GRU_SimpleVixHandler_SimpleSignalStrategy"
    config["experiment_name"] = "GRU_SimpleVixHandler_SimpleSignalStrategy"
    RollingBenchmark(config).ens_rolling()
