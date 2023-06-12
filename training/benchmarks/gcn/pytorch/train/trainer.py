# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import sys
import math

import torch.nn.functional as F
from torch.types import Device

from model import create_model
from optimizers import create_optimizer
from train.evaluator import Evaluator
from train.training_state import TrainingState
from utils.utils import accuracy

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 features, labels, adj, idx_train, idx_val, idx_test):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

        self.features = features
        self.labels = labels
        self.adj = adj
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def init(self):
        dist_pytorch.main_proc_print("Init progress")
        self.model = create_model(self.config, self.features, self.labels)
        self.model.to(self.config.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)

        if self.config.cuda:
            self.features = self.features.cuda()
            self.labels = self.labels.cuda()
            self.adj = self.adj.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

    def train_one_epoch(self):

        t = time.time()
        model = self.model
        config = self.config
        optimizer = self.optimizer
        state = self.training_state

        epoch = self.training_state.epoch

        model.train()
        optimizer.zero_grad()
        output = model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train],
                                self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train],
                             self.labels[self.idx_train])

        loss_train.backward()
        optimizer.step()

        if not config.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(self.features, self.adj)

        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        state.epoch += 1
        self.detect_training_status(state)

    def detect_training_status(self, state):
        config = self.config
        if state.eval_acc >= config.target_acc:
            print(
                f"converged_success. eval_acc: {state.eval_acc}, target_acc: {config.target_acc}"
            )
            state.converged_success()

        if state.epoch >= config.max_epochs:
            state.end_training = True

        return state.end_training