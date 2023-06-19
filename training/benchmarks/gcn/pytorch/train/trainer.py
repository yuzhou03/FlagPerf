# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.types import Device

from model import create_model
from optimizers import create_optimizer
from train.evaluator import Evaluator
from train.training_state import TrainingState
from utils.utils import accuracy
from driver import Driver, Event, dist_pytorch


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

        self.criterion = F.nll_loss
        self.optimizer = create_optimizer(self.model, self.config)

    def train_one_epoch(self, train_dataloader, adj, idx_val):

        t = time.time()
        driver = self.driver
        config = self.config
        model = self.model
        state = self.training_state

        model.train()
        self.optimizer.zero_grad()

        if dist_pytorch.is_dist_avail_and_initialized():
            train_dataloader.sampler.set_epoch(state.epoch)

        for batch_idx, batch in enumerate(train_dataloader):
            features, labels = batch
            print(f"batch_idx: {batch_idx} features.shape: {features.shape}")
            if config.cuda:
                features = features.cuda()
                labels = labels.cuda()
                adj = adj.cuda()
                self.labels = self.labels.cuda()

            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch, adj)
            state.global_steps += 1

        if not config.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(
                self.features[idx_val], adj[min(idx_val):max(idx_val) + 1,
                                            min(idx_val):max(idx_val) + 1])

        loss_val = self.criterion(output, self.labels[idx_val])
        acc_val = accuracy(output, self.labels[idx_val])

        state.eval_acc = acc_val.item()
        state.eval_loss = loss_val.item()

        self.detect_training_status(state)
        state.num_trained_samples += len(train_dataloader.dataset)
        print('Epoch: {:04d}'.format(state.epoch), 'loss_train: {:.4f}'.format(
            state.train_loss), 'acc_train: {:.4f}'.format(state.train_acc),
              'loss_val: {:.4f}'.format(state.eval_loss),
              'acc_val: {:.4f}'.format(state.eval_acc),
              'num_trained_samples: {:.4f}'.format(state.num_trained_samples),
              'time: {:.4f}s'.format(time.time() - t))

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

    def train_one_step(self, batch, adj):
        # move data to the same device as model
        batch = self.process_batch(batch, self.config.device)
        state = self.training_state
        self.model.train()

        _, state.train_loss, state.train_acc = self.forward(batch, adj)
        self.adapter.backward(state.train_loss, self.optimizer)

        if dist_pytorch.is_dist_avail_and_initialized():
            if state.train_loss is None or state.train_acc is None:
                total = torch.tensor([0, 0],
                                     dtype=torch.float32,
                                     device=self.config.device)

            else:
                total = torch.tensor([state.train_loss, state.train_acc],
                                     dtype=torch.float32,
                                     device=self.config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            total = total / dist.get_world_size()
            state.train_loss, state.train_acc = total.tolist()

        self.driver.event(Event.BACKWARD, state.global_steps, state.train_loss,
                          state.train_acc)

    def process_batch(self, batch, device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch

    def forward(self, batch, adj):
        features, labels = batch
        if self.config.cuda:
            labels = labels.cuda()

        state = self.training_state
        index_start = len(features) * state.global_steps
        index_end = len(features) * (state.global_steps + 1)

        if index_start >= adj.shape[0]:
            return None, None, None

        output = self.model(features, adj[index_start:index_end,
                                          index_start:index_end])

        loss = self.criterion(output, labels)
        acc = accuracy(output, labels)
        return output, loss, acc

    def can_do_eval(self, state):
        config = self.config
        do_eval = all([
            config.eval_data is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps %
            math.ceil(config.eval_interval_samples /
                      dist_pytorch.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination