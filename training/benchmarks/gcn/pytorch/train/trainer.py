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
        dist_pytorch.main_proc_print("Init process")
        self.model = create_model(self.config, self.features, self.labels)
        self.model.to(self.config.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.criterion = F.nll_loss
        self.optimizer = create_optimizer(self.model, self.config)

    def train_one_epoch(self, train_dataloader):

        t = time.time()
        driver = self.driver
        model = self.model
        state = self.training_state
        adj = self.adj

        model.train()
        self.optimizer.zero_grad()

        if dist_pytorch.is_dist_avail_and_initialized():
            train_dataloader.sampler.set_epoch(state.epoch)

        for batch_idx, batch in enumerate(train_dataloader):
            state.global_steps += 1
            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch, batch_idx, adj)

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_loss, state.eval_acc = self.evaluator.evaluate(self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_loss=state.eval_loss,
                                   eval_acc=state.eval_acc,
                                   time=eval_end - eval_start)

            state.end_training = self.detect_training_status(state)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if state.end_training:
                break

        state.num_trained_samples += len(train_dataloader.dataset)

        print('Epoch: {:04d}'.format(state.epoch), 'loss_train: {:.4f}'.format(
            state.train_loss), 'acc_train: {:.4f}'.format(state.train_acc),
              'loss_val: {:.4f}'.format(state.eval_loss),
              'acc_val: {:.4f}'.format(state.eval_acc),
              'num_trained_samples: {:8d}'.format(state.num_trained_samples),
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

    def train_one_step(self, batch, batch_idx, adj):
        # move data to the same device as model
        batch = self.process_batch(batch, self.config.device)
        state = self.training_state
        self.model.train()

        _, state.train_loss, state.train_acc = self.forward(
            batch, batch_idx, adj)
        self.adapter.backward(state.train_loss, self.optimizer)

        if dist_pytorch.is_dist_avail_and_initialized():
            if state.train_loss is None or state.train_acc is None:
                print("train_loss or train_acc is None")
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

    def forward(self, batch, batch_idx, adj):
        features, labels = batch
        config = self.config

        index_start = min(self.idx_train).item() + dist_pytorch.global_batch_size(config) * batch_idx
        index_end = index_start + len(features)

        print(
            f"index_start:{index_start} index_end:{index_end} len(features):{len(features)} batch_idx:{batch_idx}"
        )

        if index_start >= len(adj):
            return None, None, None

        # NOTE: must keep second param as NxN matrix, N = len(features)
        output = self.model(features, adj[index_start:index_end, index_start:index_end])

        loss = self.criterion(output, labels)
        acc = accuracy(output, labels)
        return output, loss, acc

    def inference(self, batch, batch_idx, adj, is_testing: bool = False):
        self.model.eval()

        features, labels = batch
        config = self.config

        if is_testing:
            index_start = min(self.idx_test).item() + (config.test_batch_size * dist.get_world_size()) * batch_idx
        else:
            index_start = min(self.idx_val).item() + (config.eval_batch_size * dist.get_world_size()) * batch_idx

        index_end = index_start + len(features)

        if index_start >= len(adj):
            return None, None, None

        # NOTE: must keep second param as NxN matrix, N = len(features)
        output = self.model(features, adj[index_start:index_end,
                                          index_start:index_end])

        print(
            f"inference output shape: {output.shape} epoch: {self.training_state.epoch}  batch_idx:{batch_idx}"
        )

        loss = self.criterion(output, labels)
        acc = accuracy(output, labels)
        return output, loss, acc

    def can_do_eval(self, state):
        config = self.config
        do_eval = all([
            # config.eval_data is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps %
            math.ceil(config.eval_interval_samples /
                      dist_pytorch.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 0,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination