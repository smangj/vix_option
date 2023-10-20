#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/11 16:51
# @Author   : wsy
# @email    : 631535207@qq.com
from qlib.contrib.model import LinearModel, XGBModel
from qlib.model.base import Model
from qlib.model.utils import ConcatDataset
from qlib.log import get_module_logger
from qlib.utils import get_or_create_path
import statsmodels.api as sm
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


class MultiOutputGRU(Model):
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

        self.GRU_model = GRUModelMultiOutput(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.d_instru,
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
            feature = data[:, 0, :, 0:-1].to(self.device)
            label = data[:, :, -1, -1].to(self.device)

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.GRU_model.eval()

        scores = []
        losses = []

        for (data, weight) in data_loader:
            feature = data[:, 0, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, :, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label)
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
            dl_test,
            batch_size=int(len(dl_test) / self.d_instru),
            num_workers=self.n_jobs,
        )
        self.GRU_model.eval()
        preds = []

        for i, data in enumerate(test_loader):
            feature = data[:, 0, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()[:, i]

            preds.append(pred)

        return pd.DataFrame(np.concatenate(preds), index=dl_test.get_index())
