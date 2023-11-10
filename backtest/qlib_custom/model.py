#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 16:51
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.contrib.model import LinearModel, XGBModel
from qlib.contrib.model.pytorch_nn import DNNModelPytorch, AverageMeter
from qlib.contrib.meta.data_selection.utils import ICLoss
from qlib.model.base import Model
from qlib.model.utils import ConcatDataset
from qlib.log import get_module_logger
from qlib.utils import get_or_create_path, auto_filter_kwargs
from collections import defaultdict
import statsmodels.api as sm
import gc
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from qlib.contrib.model.pytorch_utils import count_parameters
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_gru_ts import GRU
from qlib.contrib.model.pytorch_gru import GRU as GRU_normal
from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
from qlib.data.dataset.weight import Reweighter
from typing import Text, Union
from qlib.log import TimeInspector
from mlflow.utils.time_utils import get_current_time_millis
from mlflow.entities.metric import Metric
from qlib.workflow import R
from qlib.data.dataset import DatasetH
from backtest.qlib_custom.nn_module import GRUModel, GRUModelMultiOutput


class SmLinearModel(LinearModel):
    def __init__(self, estimator="ols", alpha=0.0, fit_intercept=False):
        super().__init__(estimator, alpha, fit_intercept)
        self.summary = None

    def _fit(self, X, y, w):
        assert w is None
        if self.fit_intercept:
            X = sm.add_constant(X)
        if self.estimator == self.OLS:
            model = sm.OLS(y, X)
        else:
            raise NotImplementedError
        result = model.fit()

        self.coef_ = result.params[1:]
        self.intercept_ = result.params[0]
        self.summary = str(result.summary())


class FixedLGBModel(LGBModel):
    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}  # in case of unsafety of Python default values
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds
            if early_stopping_rounds is None
            else early_stopping_rounds
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],  # training dataset
            num_boost_round=self.num_boost_round
            if num_boost_round is None
            else num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
                # log_evaluation_to_mlflow()
            ],
            **kwargs,
        )

        with TimeInspector.logt(name="record metrics by log_batch", show_start=True):
            metrics = []
            current_time = get_current_time_millis()
            for k in names:
                for key, val in evals_result[k].items():
                    name = f"{key}.{k}"
                    for epoch, m in enumerate(val):
                        metrics.append(
                            Metric(
                                key=name.replace("@", "_"),
                                value=m,
                                timestamp=current_time,
                                step=epoch,
                            )
                        )
            recorder = R.get_exp(start=True).get_recorder(start=True)
            recorder.client.log_batch(run_id=recorder.id, metrics=metrics)


class XgbFix(XGBModel):
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(
            self.model.predict(xgb.DMatrix(x_test.values)), index=x_test.index
        )


class myGRU(GRU):
    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        dl_valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if dl_train.empty or dl_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=min(self.batch_size, len(ConcatDataset(dl_valid, wl_valid))),
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()


class EmbeddingGRU(Model):
    """GRU Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        d_instru=6,
        embedding_dim=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=2,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("Embedding_GRU")
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.d_instru = d_instru
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GRU_model = GRUModel(
            d_feat=self.d_feat,
            d_instru=self.d_instru,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.GRU_model))
        self.logger.info(
            "model size: {:.4f} MB".format(count_parameters(self.GRU_model))
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

        self.fitted = False
        self.GRU_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):

        self.GRU_model.train()

        for (data, weight) in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.GRU_model.eval()

        scores = []
        losses = []

        for (data, weight) in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        dl_valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if dl_train.empty or dl_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare(
            "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(
            dl_test, batch_size=self.batch_size, num_workers=self.n_jobs
        )
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class MultiOutputGRU(GRU_normal):
    """GRU Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        d_instru=6,  # output dim
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=2,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("MultiOutputGRU")
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.d_instru = d_instru
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
            "\noutput_dim : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                d_instru,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gru_model = GRUModelMultiOutput(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.d_instru,
        )
        self.logger.info("model:\n{:}".format(self.gru_model))
        self.logger.info(
            "model size: {:.4f} MB".format(count_parameters(self.gru_model))
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

        self.fitted = False
        self.gru_model.to(self.device)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        if self.loss == "ls_mse":
            pass

        raise ValueError("unknown loss `%s`" % self.loss)

    def train_epoch(self, x_train, y_train):

        x_train_values = x_train.values.reshape(-1, self.d_instru, self.d_feat)
        y_train_values = y_train.values.reshape(-1, self.d_instru)

        self.gru_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            pred = self.gru_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):

        # prepare training data
        x_values = data_x.values.reshape(-1, self.d_instru, self.d_feat)
        y_values = data_y.values.reshape(-1, self.d_instru)

        self.gru_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:

            feature = (
                torch.from_numpy(x_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.gru_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index
        self.gru_model.eval()
        x_values = x_test.values.reshape(-1, self.d_instru, self.d_feat)
        # sample_num = x_values.shape[0]
        # preds = []
        #
        # for begin in range(sample_num)[:: self.batch_size]:
        #
        #     if sample_num - begin < self.batch_size:
        #         end = sample_num
        #     else:
        #         end = begin + self.batch_size
        #
        #     x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
        #
        #     with torch.no_grad():
        #         pred = self.gru_model(x_batch).detach().cpu().numpy()
        #
        #     preds.append(pred)

        pred = (
            self.gru_model(torch.from_numpy(x_values).float().to(self.device))
            .detach()
            .cpu()
            .numpy()
        )

        return pd.Series(pred.reshape(-1, 1).squeeze(), index=index)


class DNNModelPytorchFix(DNNModelPytorch):
    def get_metric(self, pred, target, index):
        # NOTE: the order of the index must follow <datetime, instrument> sorted order
        return -ICLoss()(pred, target, index, skip_size=6)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
        reweighter=None,
    ):
        has_valid = "valid" in dataset.segments
        segments = ["train", "valid"]
        vars = ["x", "y", "w"]
        all_df = defaultdict(dict)  # x_train, x_valid y_train, y_valid w_train, w_valid
        all_t = defaultdict(dict)  # tensors
        for seg in segments:
            if seg in dataset.segments:
                # df_train df_valid
                df = dataset.prepare(
                    seg,
                    col_set=["feature", "label"],
                    data_key=self.valid_key if seg == "valid" else DataHandlerLP.DK_L,
                )
                all_df["x"][seg] = df["feature"]
                all_df["y"][seg] = df[
                    "label"
                ].copy()  # We have to use copy to remove the reference to release mem
                if reweighter is None:
                    all_df["w"][seg] = pd.DataFrame(
                        np.ones_like(all_df["y"][seg].values), index=df.index
                    )
                elif isinstance(reweighter, Reweighter):
                    all_df["w"][seg] = pd.DataFrame(reweighter.reweight(df))
                else:
                    raise ValueError("Unsupported reweighter type.")

                # get tensors
                for v in vars:
                    all_t[v][seg] = torch.from_numpy(all_df[v][seg].values).float()
                    # if seg == "valid": # accelerate the eval of validation
                    all_t[v][seg] = all_t[v][seg].to(
                        self.device
                    )  # This will consume a lot of memory !!!!

                evals_result[seg] = []
                # free memory
                del df
                del all_df["x"]
                gc.collect()

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        # train
        self.logger.info("training...")
        self.fitted = True
        # return
        # prepare training data
        train_num = all_t["y"]["train"].shape[0]

        for step in range(1, self.max_steps + 1):
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.dnn_model.train()
            self.train_optimizer.zero_grad()
            choice = np.random.choice(train_num, self.batch_size)
            x_batch_auto = all_t["x"]["train"][choice].to(self.device)
            y_batch_auto = all_t["y"]["train"][choice].to(self.device)
            w_batch_auto = all_t["w"]["train"][choice].to(self.device)

            # forward
            preds = self.dnn_model(x_batch_auto)
            cur_loss = self.get_loss(preds, w_batch_auto, y_batch_auto, self.loss_type)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())
            # R.log_metrics(train_loss=loss.avg, step=step)

            # validation
            train_loss += loss.val
            # for evert `eval_steps` steps or at the last steps, we will evaluate the model.
            if step % self.eval_steps == 0 or step == self.max_steps:
                if has_valid:
                    stop_steps += 1
                    train_loss /= self.eval_steps

                    with torch.no_grad():
                        self.dnn_model.eval()

                        # forward
                        preds = self._nn_predict(all_t["x"]["valid"], return_cpu=False)
                        cur_loss_val = self.get_loss(
                            preds,
                            all_t["w"]["valid"],
                            all_t["y"]["valid"],
                            self.loss_type,
                        )
                        loss_val = cur_loss_val.item()
                        metric_val = (
                            self.get_metric(
                                preds.reshape(-1),
                                all_t["y"]["valid"].reshape(-1),
                                all_df["y"]["valid"].index,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .item()
                        )
                        # R.log_metrics(val_loss=loss_val, step=step)
                        # R.log_metrics(val_metric=metric_val, step=step)

                        if self.eval_train_metric:
                            metric_train = (
                                self.get_metric(
                                    self._nn_predict(
                                        all_t["x"]["train"], return_cpu=False
                                    ),
                                    all_t["y"]["train"].reshape(-1),
                                    all_df["y"]["train"].index,
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .item()
                            )
                            # R.log_metrics(train_metric=metric_train, step=step)
                        else:
                            metric_train = np.nan
                    if verbose:
                        self.logger.info(
                            f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}, "
                            f"train_metric {metric_train:.6f}, valid_metric {metric_val:.6f}"
                        )
                    evals_result["train"].append(train_loss)
                    evals_result["valid"].append(loss_val)
                    if loss_val < best_loss:
                        if verbose:
                            self.logger.info(
                                "\tvalid loss update from {:.6f} to {:.6f}, save checkpoint.".format(
                                    best_loss, loss_val
                                )
                            )
                        best_loss = loss_val
                        self.best_step = step
                        # R.log_metrics(best_step=self.best_step, step=step)
                        stop_steps = 0
                        torch.save(self.dnn_model.state_dict(), save_path)
                    train_loss = 0
                    # update learning rate
                    if self.scheduler is not None:
                        auto_filter_kwargs(self.scheduler.step, warning=False)(
                            metrics=cur_loss_val, epoch=step
                        )
                    # R.log_metrics(lr=self.get_lr(), step=step)
                else:
                    # retraining mode
                    if self.scheduler is not None:
                        self.scheduler.step(epoch=step)

        if has_valid:
            # restore the optimal parameters after training
            self.dnn_model.load_state_dict(
                torch.load(save_path, map_location=self.device)
            )
        if self.use_gpu:
            torch.cuda.empty_cache()


class myTransformer(TransformerModel):
    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):

        dl_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        dl_valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )

        if dl_train.empty or dl_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dl_valid,
            batch_size=min(self.batch_size, len(dl_valid)),
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()
