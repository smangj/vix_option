#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/23 9:33
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.model.ens.ensemble import RollingEnsemble
import ruamel.yaml as yaml
from qlib.workflow.cli import sys_config
import qlib
import os
from qlib.config import C
from pathlib import Path
from qlib.model.trainer import TrainerR
from qlib.utils import init_instance_by_config
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow import R
from qlib.workflow.recorder import MLflowRecorder
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector


def dynamicworkflow(
    config_path, experiment_name="dynamicworkflow", uri_folder="mlruns"
):
    """
    需要动态train的流程`
    """
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(
            Path(os.getcwd()).resolve() / uri_folder
        )
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]

    config["roll_config"]["rolling_exp"] = "rolling_" + experiment_name
    i = 0
    while True:

        try:
            record = R.get_recorder(experiment_name=config["experiment_name"])
        except Exception as e:
            print(e)
            with R.start(experiment_name=config["experiment_name"]):
                record = R.get_recorder()
        assert isinstance(record, MLflowRecorder)
        h_path = Path(record.get_local_dir()).parent
        if not os.path.exists(os.path.join(h_path, "config.yaml")):
            import shutil

            shutil.copyfile(config_path, os.path.join(h_path, "config.yaml"))
            break
        else:
            i += 1
            config["experiment_name"] = config["experiment_name"] + str(i)

    RollingBenchmark(config).run_all()


class RollingBenchmark:
    """
    roll_train_pipeline
    """

    def __init__(self, config=dict) -> None:
        self._config = config
        self.step = config["roll_config"]["step"]
        self.horizon = config["roll_config"]["horizon"]
        self.rolling_exp = config["roll_config"]["rolling_exp"]
        self.COMB_EXP = config["experiment_name"]
        self._handler_path = None

    def basic_task(self):
        conf = deepcopy_basic_type(self._config)

        task = conf["task"]

        record = R.get_recorder(experiment_name=self.COMB_EXP)
        assert isinstance(record, MLflowRecorder)
        h_path = Path(
            record.get_local_dir()
        ).parent / "{}_handler_horizon{}.pkl".format(
            conf["experiment_name"], self.horizon
        )
        if not h_path.exists():
            h_conf = task["dataset"]["kwargs"]["handler"]
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)

        self._handler_path = (
            "file://"
            + os.path.join(*h_path.parts[0:-1])
            + "/{}".format(h_path.parts[-1])
        )
        task["dataset"]["kwargs"]["handler"] = self._handler_path
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task

    def create_rolling_tasks(self):
        task = self.basic_task()
        task_l = task_generator(
            task, RollingGen(step=self.step, trunc_days=self.horizon + 1)
        )  # the last two days should be truncated to avoid information leakage
        return task_l

    def train_rolling_tasks(self, task_l=None):
        if task_l is None:
            task_l = self.create_rolling_tasks()
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    def ens_rolling(self):
        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()
        r = RecorderCollector(experiment=self.COMB_EXP)
        assert len(r.experiment.info["recorders"]) == 1
        recorder = R.get_recorder(experiment_name=self.COMB_EXP)
        recorder.log_params(exp_name=self.rolling_exp)
        recorder.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})

    def update_rolling_rec(self):
        """
        Evaluate the combined rolling results
        """
        for rid, rec in R.list_recorders(experiment_name=self.COMB_EXP).items():
            for record in self._config["task"]["record"]:
                # Some recorder require the parameter `model` and `dataset`.
                # try to automatically pass in them to the initialization function
                # to make defining the tasking easier
                r = init_instance_by_config(
                    record,
                    recorder=rec,
                    default_module="qlib.workflow.record_temp",
                    try_kwargs={"handler": self._handler_path},
                )
                r.generate()

        print(
            f"Your evaluation results can be found in the experiment named `{self.COMB_EXP}`."
        )

    def run_all(self):
        # the results will be  save in mlruns.
        # 1) each rolling task is saved in rolling_models
        self.train_rolling_tasks()
        # 2) combined rolling tasks and evaluation results are saved in rolling
        self.ens_rolling()
        self.update_rolling_rec()


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
