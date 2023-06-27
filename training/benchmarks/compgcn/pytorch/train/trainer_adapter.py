from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from driver.dist_pytorch import is_dist_avail_and_initialized, main_proc_print
import config


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


def backward(loss: Tensor, optimizer: Optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()