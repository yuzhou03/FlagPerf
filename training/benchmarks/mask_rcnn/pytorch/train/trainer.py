import torch
from torch.types import Device
import os
import sys

from model.modeling import create_model
from schedulers import create_scheduler

import utils.train.train_eval_utils as utils
from train.evaluator import Evaluator
from train.training_state import TrainingState

import config

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_pytorch


def process_batch(batch, device):
    """Process batch and produce inputs for the model."""
    batch = {t: batch[t].to(device) for t in batch if t != 'answer_idx'}

    return batch


class Trainer:
    """
    Trainer
    """

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
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

    def init(self):
        config = self.config

        pretrain_path = os.path.join(config.data_dir, config.pretrained_path)
        coco_weights_pretrained_path = os.path.join(
            config.data_dir, config.coco_weights_pretrained_path)

        dist_pytorch.main_proc_print(
            f"pretrain_path:{pretrain_path}, coco_weights_pretrained_path:{coco_weights_pretrained_path}"
        )
        self.model = create_model(
            num_classes=config.num_classes,
            load_pretrain_weights=True,
            pretrain_path=pretrain_path,
            coco_weights_path=coco_weights_pretrained_path)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.optimizer = self.adapter.create_optimizer(self.model)
        # self.model = self.adapter.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler()

    def train_one_epoch(self,
                        dataloader,
                        epoch,
                        print_freq=50,
                        warmup=True,
                        scaler=None):
        
        state = self.training_state
        driver = self.driver
        device = self.device
        model = self.model
        optimizer = self.optimizer
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        mean_loss, lr = utils.train_one_epoch(model,
                                              optimizer,
                                              dataloader,
                                              device,
                                              epoch,
                                              print_freq=print_freq,
                                              warmup=warmup,
                                              scaler=scaler)

        return mean_loss, lr

    def forward(self, batch):
        """forward pass"""
        data = batch
        tokens, labels, position_ids, attention_mask = data['text'], data[
            'label'], data['position'], data['mask']
        target_ids, logit_mask = data['target'], data['logit_mask']

        result = self.model(tokens, position_ids, attention_mask, target_ids,
                            logit_mask)
        logits, *mems = result

        loss_mask = data["loss_mask"]
        logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.contiguous().float(), labels)

        return loss, mems
