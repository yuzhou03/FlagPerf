"""ResNet50 Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn  as nn
import numpy as np
import torch

import argparse
import os
import sys
import time
import random
from typing import Tuple, Any

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from train import trainer_adapter
import driver
from driver import Driver, Event, dist_pytorch, check
from model.losses.device import Device
from model.models.modeling import create_model
from optimizers import create_optimizer


logger = None


def main() -> Tuple[Any, Any]:
    """ main training workflow """
    import config
    from config import mutable_params
    global logger

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    resnet_driver = Driver(config, config.mutable_params)
    resnet_driver.setup_config(argparse.ArgumentParser("ResNet50"))
    resnet_driver.setup_modules(driver, globals(), locals())

    logger = resnet_driver.logger
    dist_pytorch.init_dist_training_env(config)

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    # check.check_config(config, "cpm_model_states_medium.pt")

    dist_pytorch.barrier()
    resnet_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    traindir = os.path.join(config.data_dir, 'train')
    valdir = os.path.join(config.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                   shuffle=(train_sampler is None), num_workers=config.num_workers, 
                                                   pin_memory=True, sampler=train_sampler)

    eval_dataloader = torch.utils.data.DataLoader(  val_dataset, batch_size=config.eval_batch_size, 
                                                    shuffle=False, num_workers=config.num_workers, 
                                                    pin_memory=True, sampler=val_sampler)

    print(f"train_dataset length:{len(train_dataloader.dataset)}")
    print(f"train length:{len(train_dataloader)}")
    print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
    print(f"eval length:{len(eval_dataloader)}")


    # prepare paramter for training
    model = create_model(config, ngpus_per_node)
    device = Device.get_device(config)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer =  create_optimizer(model, config)

    evaluator = Evaluator(eval_dataloader, model, criterion, config, device)

    training_state = TrainingState()
    trainer = Trainer(driver=resnet_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=Device.get_device(config),
                      config=config)

    training_state._trainer = trainer

    dist_pytorch.barrier()
    trainer.init()

    if config.evaluate:
        # validate(val_loader, model, criterion, args)
        init_evaluation_start = time.time()
        training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(trainer)
        init_evaluation_end = time.time()

        init_evaluation_info = dict(
            eval_loss=training_state.eval_avg_loss,
            eval_embedding_average=training_state.eval_embedding_average,
            time=init_evaluation_end - init_evaluation_start,
        )
        # training_event.on_init_evaluate(init_evaluation_info)
        resnet_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
        return
    
    if not config.do_train:
        return config, training_state

    # training_event.on_init_end()
    resnet_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_pytorch.barrier()
    epoch = -1
    # training_event.on_train_begin()
    resnet_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        # train_one_epoch(self, dataloader, model, criterion, optimizer, epoch, device)
        trainer.train_one_epoch(train_dataloader, model, criterion, optimizer, epoch, device)
    
    # training_event.on_train_end()
    resnet_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e+3
    return config, training_state


if __name__ == "__main__":
    if not dist_pytorch.is_main_process():
        exit()

    start = time.time()
    config, state = main()
    e2e_time = time.time() - start
    training_perf = (dist_pytorch.global_batch_size(config) * state.global_steps) / state.raw_train_time

    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_avg_loss,
            "final_mlm_accuracy": state.eval_embedding_average,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
