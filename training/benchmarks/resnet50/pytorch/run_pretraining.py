"""ResNet50 Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models
import torch.nn  as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
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
from train.trainer import Trainer, save_checkpoint
from train.training_state import TrainingState
from train import trainer_adapter
import driver
from driver import Driver, Event, dist_pytorch, check
from model.losses.device import Device
from model.models.modeling import create_model
from optimizers import create_optimizer
from schedulers import create_scheduler


logger = None


def main() -> Tuple[Any, Any]:
    """ main training workflow """
    import config
    from config import mutable_params


    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    
    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed


    
    

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    print(f"config.local_rank: {config.local_rank}")
    print(f"config.distributed: {config.distributed}")    
    print(f"ngpus_per_node: {ngpus_per_node}")
    print(f"config.multiprocessing_distributed: {config.multiprocessing_distributed}") 


    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)    

def main_worker(gpu, ngpus_per_node, config):

    global logger
    resnet_driver = Driver(config, config.mutable_params)
    resnet_driver.setup_config(argparse.ArgumentParser("ResNet50"))
    resnet_driver.setup_modules(driver, globals(), locals())
    logger = resnet_driver.logger


    dist_pytorch.init_dist_training_env(config)


    global best_acc1
    config.gpu = gpu

    print(f"main_worker config.gpu: {config.gpu}")
    print(f"main_worker config.distributed: {config.distributed}")

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed and False:

        if config.dist_url == "env://" and config.local_rank == -1:
            config.local_rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.local_rank = config.local_rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.local_rank)


    model = create_model(config, ngpus_per_node=ngpus_per_node)

    # if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    #     print('using CPU, this will be slow')
    # elif config.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if torch.cuda.is_available():
    #         if config.gpu is not None:
    #             torch.cuda.set_device(config.gpu)
    #             model.cuda(config.gpu)
    #             # When using a single GPU per process and per
    #             # DistributedDataParallel, we need to divide the batch size
    #             # ourselves based on the total number of GPUs of the current node.
    #             config.batch_size = int(config.batch_size / ngpus_per_node)
    #             config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
    #             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    #         else:
    #             model.cuda()
    #             # DistributedDataParallel will divide and allocate batch_size to all
    #             # available GPUs if device_ids are not set
    #             model = torch.nn.parallel.DistributedDataParallel(model)
    # elif config.gpu is not None and torch.cuda.is_available():
    #     torch.cuda.set_device(config.gpu)
    #     model = model.cuda(config.gpu)
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     model = model.to(device)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if config.arch.startswith('alexnet') or config.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()

    # if torch.cuda.is_available():
    #     if config.gpu:
    #         device = torch.device('cuda:{}'.format(config.gpu))
    #     else:
    #         device = torch.device("cuda")
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")


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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = create_scheduler(optimizer=optimizer)

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
    dist_pytorch.barrier()

    if config.evaluate:
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
        trainer.train_one_epoch(train_dataloader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        acc1 = evaluator.evaluate(trainer)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.local_rank % ngpus_per_node == 0):
            stats = {
                'epoch': epoch + 1,
                'arch': config.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_checkpoint(stats, is_best)
    
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
