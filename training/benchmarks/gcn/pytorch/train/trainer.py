# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import time

import torch.nn.functional as F
from torch.types import Device

from model import create_model
from optimizers import create_optimizer
from train.evaluator import Evaluator
from train.training_state import TrainingState
from utils.utils import accuracy
from driver import Driver, dist_pytorch

class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 features, labels):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

        self.features = features
        self.labels = labels

    def init(self):
        dist_pytorch.main_proc_print("Init process")
        self.model = create_model(self.config, self.features, self.labels)
        self.model.to(self.config.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)

    def train_one_epoch(self, features, labels, adj, idx_train, idx_val):

        t = time.time()
        config = self.config
        model = self.model
        state = self.training_state

        model.train()
        self.optimizer.zero_grad()
        
        output = model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        self.optimizer.step()

        if not config.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        state.eval_acc = acc_val.item()
        state.eval_loss = loss_val.item()

        state.epoch += 1
        print('Epoch: {:04d}'.format(state.epoch),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        self.detect_training_status(state)

    def detect_training_status(self, state):
        config = self.config
        if state.eval_acc >= config.target_acc:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_acc: {state.eval_acc}, target_acc: {config.target_acc}"
            )
            state.converged_success()

        if state.epoch >= config.max_epochs:
            state.end_training = True

        return state.end_training