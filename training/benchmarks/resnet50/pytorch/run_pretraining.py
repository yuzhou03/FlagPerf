"""ResNet50 Pretraining"""

from dataloaders import WorkerInitializer, build_train_dataloader
from driver import Driver, Event, dist_pytorch

from driver.helper import InitHelper
import driver
from train import trainer_adapter
from train.device import Device
from train.training_state import TrainingState
from train.trainer import Trainer
from train.evaluator import Evaluator

import shutil

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import torch

import os
import sys
import time
import random
from typing import Tuple, Any

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))


logger = None



def main() -> Tuple[Any, Any, Any]:
    """ main training workflow """
    import config
    global logger

    print(f"learning_rate: {config.learning_rate}")
    print(f"num_workers: {config.num_workers}")
    print(f"n_device: {config.n_device}")
    print(f"target_accuracy: {config.target_accuracy}")
    print(f"weight_decay_rate: {config.weight_decay_rate}")
    print(f"print_freq: {config.print_freq}")

     # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver()  # _base.py增加模型名称name
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time

    print(f"main max_steps: {config.max_steps}")

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    world_size = dist.get_world_size()
    config.distributed = world_size > 1 or config.multiprocessing_distributed
    print(f"world_size={world_size}, config.distributed:{config.distributed}")

    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    rand_seed = config.seed
    worker_init = WorkerInitializer.default(rand_seed)

    # train && val dataset
    traindir = os.path.join(config.data_dir, 'train')
    valdir = os.path.join(config.data_dir, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    print(f"train_dataloader num_workers: {config.num_workers}")
    print(f"train_dataloader train_batch_size: {config.train_batch_size}")

    eval_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size,
                                                  shuffle=False, num_workers=config.num_workers,
                                                  pin_memory=True, sampler=val_sampler)

    train_dataloader = build_train_dataloader(
        config, train_dataset, worker_init)

    print(f"train_dataset length:{len(train_dataloader.dataset)}")
    print(f"train length:{len(train_dataloader)}")
    print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
    print(f"eval length:{len(eval_dataloader)}")

    # prepare paramter for training
    device = Device.get_device(config)
    criterion = nn.CrossEntropyLoss().to(device)
    evaluator = Evaluator(config, eval_dataloader, criterion)

    training_state = TrainingState()
    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=device,
                      config=config)

    training_state._trainer = trainer

    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    if config.evaluate:
        init_evaluation_start = time.time()
        training_state.eval_accuracy = evaluator.evaluate(trainer)
        init_evaluation_end = time.time()

        init_evaluation_info = dict(
            eval_accuracy=training_state.eval_accuracy,
            time=init_evaluation_end - init_evaluation_start,
        )
        # training_event.on_init_evaluate(init_evaluation_info)
        model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
        return

    if not config.do_train:
        return config, training_state

    # training_event.on_init_end()
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier(config.vendor)
    epoch = -1
    # training_event.on_train_begin()
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time

    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader, criterion, epoch)
        trainer.lr_scheduler.step()

    # training_event.on_train_end()
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (
        raw_train_end_time - raw_train_start_time) / 1e+3
    return config, training_state, helper


if __name__ == "__main__":
    start = time.time()
    config, state, helper = main()
    if not dist_pytorch.is_main_process():
        exit()
    finished_info = helper.get_finished_info(start, state)
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
