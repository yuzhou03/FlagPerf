import os
import sys

import torch
import torch.distributed as dist
from torch.types import Device

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from .utils import accuracy


class Evaluator:

    def __init__(self, config, eval_dataloader):
        self.config = config
        self.eval_dataloader = eval_dataloader

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        return batch

    @torch.no_grad()
    def evaluate(self, model, data_loader, device, criterion):
        model.eval()

        acc1_total, acc5_total = 0.0, 0.0
        loss_total = 0.0

        steps = 0
        for step, batch in enumerate(data_loader):
            if step % self.config.log_freq == 0:
                print("Eval Step " + str(step) + "/" + str(len(data_loader)))
            batch = self.process_batch(batch, device)
            images, target = batch
            output = model(images)

            eval_loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, (1, 5))
            acc1_total += acc1
            acc5_total += acc5
            loss_total += eval_loss
            steps += 1

        acc1 = torch.tensor([acc1_total], dtype=torch.float32, device=device)
        acc5 = torch.tensor([acc5_total], dtype=torch.float32, device=device)
        eval_loss = torch.tensor([loss_total],
                                 dtype=torch.float32,
                                 device=device)

        world_size = 1
        if self.config.distributed:
            dist.all_reduce(acc1, dist.ReduceOp.SUM)
            dist.all_reduce(acc5, dist.ReduceOp.SUM)
            dist.all_reduce(eval_loss, dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        acc1 = acc1 / (steps * world_size)
        acc5 = acc5 / (steps * world_size)
        eval_loss = eval_loss / (steps * world_size)
        print("Eval Acc1: " + str(float(acc1)) + "%, Eval Acc5:" +
              str(float(acc5)) + "%" + ", eval_loss:" + str(float(eval_loss)))
        return float(eval_loss), float(acc1), float(acc5)