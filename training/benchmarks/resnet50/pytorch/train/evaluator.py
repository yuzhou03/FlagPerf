# coding=utf-8
import time
from typing import Any
import torch

from .meter import ProgressMeter, AverageMeter, Summary, accuracy
from torch.utils.data import Subset

class Evaluator:

    def __init__(self, val_loader, model, criterion, config, device):
        self.val_loader = val_loader
        self.model = model 
        self.criterion = criterion
        self.config = config
        self.device = device

    
    def evaluate(self, trainer)->Any:

        val_loader = self.val_loader
        model = self.model
        config = self.config
        criterion = self.criterion
        device = self.device

        print(f"== evaluate: device: {device}")

        def run_validate(loader, base_progress=0):
            with torch.no_grad():

                # print(f"== run_validate: torch version: {torch.__version__}")
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    if config.gpu is not None and torch.cuda.is_available():
                        images = images.cuda(config.gpu, non_blocking=True)
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        images = images.to('mps')
                        target = target.to('mps')
                    if torch.cuda.is_available():
                        target = target.cuda(config.gpu, non_blocking=True)

                    # compute output
                    output = model(images)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % config.gradient_accumulation_steps == 0:
                        progress.display(i + 1)


        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(val_loader) + (config.distributed and (len(val_loader.sampler) * config.world_size < len(val_loader.dataset))),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        run_validate(val_loader)
        if config.distributed:
            top1.all_reduce()
            top5.all_reduce()

        if config.distributed and (len(val_loader.sampler) * config.world_size < len(val_loader.dataset)):
            aux_val_dataset = Subset(val_loader.dataset, range(len(val_loader.sampler) * config.world_size, len(val_loader.dataset)))
            aux_val_loader = torch.utils.data.DataLoader(
                aux_val_dataset, batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True)
            run_validate(aux_val_loader, len(val_loader))

        progress.display_summary()

        return top1.avg
