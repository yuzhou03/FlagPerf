from .meter import AverageMeter, ProgressMeter, accuracy
from driver import Driver, Event, dist_pytorch
import config
from train.training_state import TrainingState
from train.evaluator import Evaluator
from schedulers import create_scheduler
from model import create_model

import shutil
import torch
import torch.distributed as dist
from torch.types import Device
import os
import sys
import time
import math


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config, ngpus_per_node:int=1):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None
        self.device = device

        self.optimizer = None
        self.config = config
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.overflow_buf = None
        self.ngpus_per_node = ngpus_per_node

    def init(self):
        self.model = create_model(config, self.ngpus_per_node)
        self.model = self._init_model(self.model, self.device)
        self.model = self.adapter.convert_model(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model, self.config)
        self.model = self.adapter.model_to_fp16(self.model, self.optimizer)
        self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer)
        # if self.config.fp16 and self.optimizer is not None:
        #     self.optimizer._model_params_to_master_params()
        self.grad_scaler = self.adapter.create_grad_scaler()

    def _init_model(self, model, device):
        # checkpoint = torch.load(config.init_checkpoint, map_location="cpu")
        # if "model" in checkpoint:
        #     checkpoint = checkpoint["model"]
        # model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
        return model

    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, device):

        print(f"==== train_one_epoch device:{device} =====")
        print(f"==== train_one_epoch epoch:{epoch} =====")
        print(f"==== train_one_epoch model:{model} =====")

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader),
                                 [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()

        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        step_start_time = time.time()

        epoch_start_num_sample = state.num_trained_samples

        for batch_idx, (images, target) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            state.global_steps += 1
            # TODO: Maybe we should update num_trained_samples after all epochs.
            state.num_trained_samples = state.global_steps * dist_pytorch.global_batch_size(self.config)

            driver.event(Event.STEP_BEGIN, step=state.global_steps)

            # move data to the same device as model
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            # model.to(device)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                # show progress
                progress.display(batch_idx + 1)

                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (dist_pytorch.global_batch_size(
                    self.config) * self.config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second


            eval_result = None

            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_accuracy = self.evaluator.evaluate(self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_accuracy=state.eval_accuracy,
                                   time=eval_end - eval_start)

            end_training = self.detect_training_status(state)

            step_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END, message=step_info,
                         step=state.global_steps, loss=state.loss)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if end_training:
                break

        epoch_start_num_sample += len(train_loader.dataset)
        state.num_trained_samples = epoch_start_num_sample

        driver.event(Event.EPOCH_END, state.epoch)

    def can_do_eval(self, state):
        config = self.config
        do_eval = all([
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps %
            math.ceil(config.eval_interval_samples /
                      dist_pytorch.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.num_trained_samples >= config.max_samples_termination

    def detect_training_status(self, state):
        config = self.config
        if state.eval_accuracy >= config.target_accuracy:
            state.converged_success()

        if state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training    


def save_checkpoint(state: object, is_best: bool, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
