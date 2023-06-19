# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

import config
from driver.dist_pytorch import main_proc_print, is_dist_avail_and_initialized


def convert_model(model: nn.Module) -> nn.Module:
    return model


def model_to_fp16(model: nn.Module) -> nn.Module:
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[config.local_rank])
    return model


def backward(loss: torch.Tensor, optimizer: Optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()