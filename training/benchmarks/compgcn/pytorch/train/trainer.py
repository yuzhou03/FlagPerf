from time import time

import torch
import torch.utils.data
from torch.types import Device
import torch.distributed as dist

from dataloaders.dataloader import Data
from model import create_model
from optimizers import create_optimizer
from train.evaluator import Evaluator
from train.training_state import TrainingState
from driver import Driver, dist_pytorch, Event
from utils.utils import in_out_norm


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 data: Data):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

        self.data = data
        graph = data.g.to(device)
        self.graph = in_out_norm(graph)

    def init(self):
        dist_pytorch.main_proc_print("Init progress:")
        self.model = create_model(self.config, self.data)
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.optimizer.zero_grad()

        self.criterion = torch.nn.BCELoss()

    def train_one_epoch(self, data):
        model = self.model
        state = self.training_state
        driver = self.driver

        data_iter = data.data_iter

        # Training and validation using a full graph
        model.train()
        t0 = time()

        for step, batch in enumerate(data_iter["train"]):
            state.global_steps += 1
            state.num_trained_samples += batch[0].shape[0]
            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            self.train_one_step(batch)

        t1 = time()
        self.validate()
        t2 = time()

        print(
            "In epoch {:3d}, Train Loss: {:.4f}, Valid MRR: {:.5},  Valid Hits@1: {:.5}, Train time: {}, Valid time: {}"
            .format(state.epoch, state.loss, state.eval_MRR, state.eval_Hit1,
                    t1 - t0, t2 - t1))

    def validate(self):
        state = self.training_state
        model = self.model
        graph = self.graph
        device = self.config.device
        data_iter = self.data.data_iter
        config = self.config

        # init variables
        best_mrr = 0.0
        kill_cnt = 0

        val_results = self.evaluator.evaluate(model,
                                              graph,
                                              device,
                                              data_iter,
                                              split="valid")

        state.eval_MRR, state.eval_MR = val_results['mrr'], val_results['mr']
        state.eval_Hit1, state.eval_Hit3, state.eval_Hit10 = val_results[
            'hits@1'], val_results['hits@3'], val_results['hits@10']

        if dist_pytorch.is_dist_avail_and_initialized():
            total = torch.tensor([
                state.eval_MRR, state.eval_MR, state.eval_Hit1, state.eval_Hit3,
                state.eval_Hit10
            ],
                                 dtype=torch.float32,
                                 device=self.config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM)
            total = total / dist.get_world_size()
            state.eval_MRR, state.eval_MR, state.eval_Hit1, state.eval_Hit3, state.eval_Hit10 = \
                total.tolist()

        # validate
        if state.eval_MRR > best_mrr:
            best_mrr = state.eval_MRR
            best_epoch = state.epoch
            torch.save(model.state_dict(), "comp_link" + "_" + config.dataset)
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt > 100:
                print("early stop.")

        self.detect_training_status(state)

    def train_one_step(self, batch):
        # move data to the same device as model
        batch = self.process_batch(batch, self.config.device)
        state = self.training_state
        self.model.train()

        state.loss = self.forward(batch)
        self.adapter.backward(state.loss, self.optimizer)

        if dist_pytorch.is_dist_avail_and_initialized():
            total = torch.tensor([state.loss],
                                 dtype=torch.float32,
                                 device=self.config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM)
            total = total / dist.get_world_size()
            state.loss = total.tolist()[0]

        self.driver.event(Event.BACKWARD, state.global_steps, state.loss,
                          self.optimizer)

    def process_batch(self, batch, device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch

    def forward(self, batch):
        triple, label = batch
        graph = self.graph
        sub, rel, obj, label = (
            triple[:, 0],
            triple[:, 1],
            triple[:, 2],
            label,
        )
        output = self.model(graph, sub, rel)

        # compute loss
        loss = self.criterion(output, label)
        return loss

    def can_do_eval(self, state):
        do_eval = all([
            state.global_steps >= 1,
        ])
        return do_eval

    def detect_training_status(self, state):
        config = self.config
        if state.eval_MRR >= config.target_MRR and state.eval_Hit1 >= config.target_Hit1:
            dist_pytorch.main_proc_print(f"converged_success")
            state.converged_success()

        if state.epoch > config.max_epochs:
            state.end_training = True

        return state.end_training