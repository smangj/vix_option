#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/5/23 9:33
# @Author   : wsy
# @email    : 631535207@qq.com
import ruamel.yaml as yaml
from qlib.workflow.cli import sys_config
import qlib
import os
from qlib.config import C
from pathlib import Path
from qlib.model.trainer import TrainerR, TrainerRM
from qlib.workflow.task.manage import TaskManager
from qlib.utils import init_instance_by_config
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow import R
from qlib.workflow.recorder import MLflowRecorder
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.group import RollingGroup


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

    def __init__(self, config: dict) -> None:
        self._config = config
        self.step = config["roll_config"].get("step")
        self.horizon = config["roll_config"].get("horizon")
        self.rolling_type = config["roll_config"].get("rolling_type")

        self.COMB_EXP = config.get("experiment_name")
        self.rolling_exp = self.COMB_EXP + "_rolling_exp"
        self._handler_path = {}

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

        tasks = conf["task"]

        # 兼容多task和单task
        if isinstance(tasks, list):
            for task in tasks:
                # try:
                #     record = R.get_recorder(experiment_name=self.COMB_EXP)
                # except Exception as e:
                #     print(e)
                #     with R.start(experiment_name=self.COMB_EXP):
                #         record = R.get_recorder()
                # assert isinstance(record, MLflowRecorder)
                h_path = Path.cwd() / "{}_handler_horizon{}.pkl".format(
                    task["name"], self.horizon
                )
                if not h_path.exists():
                    h_conf = task["dataset"]["kwargs"]["handler"]
                    h = init_instance_by_config(h_conf)
                    h.to_pickle(h_path, dump_all=True)

                self._handler_path[task["name"]] = h_path
                task["dataset"]["kwargs"]["handler"] = h_path
                task["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        else:
            try:
                record = R.get_recorder(experiment_name=self.COMB_EXP)
            except Exception as e:
                print(e)
                with R.start(experiment_name=self.COMB_EXP):
                    record = R.get_recorder()
            assert isinstance(record, MLflowRecorder)
            h_path = Path(
                record.get_local_dir()
            ).parent / "{}_handler_horizon{}.pkl".format(
                conf["experiment_name"], self.horizon
            )
            if not h_path.exists():
                h_conf = tasks["dataset"]["kwargs"]["handler"]
                h = init_instance_by_config(h_conf)
                h.to_pickle(h_path, dump_all=True)

            self._handler_path[tasks["name"]] = h_path
            tasks["dataset"]["kwargs"]["handler"] = h_path
            tasks["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return tasks

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

        def rec_key(recorder):
            task_config = recorder.load_object("task")
            model_key = task_config["name"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return model_key, rolling_key

        rc = RecorderCollector(
            experiment=self.rolling_exp,
            process_list=RollingGroup(),
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
            rec_key_func=rec_key,
        )
        res = rc()
        # r = RecorderCollector(experiment=self.COMB_EXP)
        # assert len(r.experiment.info["recorders"]) == 1
        # recorder = R.get_recorder(experiment_name=self.COMB_EXP)
        # recorder.log_params(exp_name=self.rolling_exp)
        # recorder.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})

        for task in self._config["task"]:
            with R.start(experiment_name=self.COMB_EXP, recorder_name=task["name"]):
                recorder = R.get_recorder()
                recorder.save_objects(
                    **{
                        "pred.pkl": res["pred"][(task["name"],)],
                        "label.pkl": res["label"][(task["name"],)],
                    }
                )

    def update_rolling_rec(self):
        """
        Evaluate the combined rolling results
        """
        for task in self._config["task"]:
            rec = R.get_recorder(
                experiment_name=self.COMB_EXP, recorder_name=task["name"]
            )
            for record in task["record"]:
                r = init_instance_by_config(
                    record,
                    recorder=rec,
                    default_module="qlib.workflow.record_temp",
                    try_kwargs={"handler": self._handler_path[task["name"]]},
                )
                r.generate()
        # for rid, rec in R.list_recorders(experiment_name=self.COMB_EXP).items():
        #     for record in self._config["task"]["record"]:
        #         # Some recorder require the parameter `model` and `dataset`.
        #         # try to automatically pass in them to the initialization function
        #         # to make defining the tasking easier
        #         r = init_instance_by_config(
        #             record,
        #             recorder=rec,
        #             default_module="qlib.workflow.record_temp",
        #             try_kwargs={"handler": self._handler_path},
        #         )
        #         r.generate()

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

        # 删除临时handler文件
        for k, v in self._handler_path.items():
            os.remove(v)


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
