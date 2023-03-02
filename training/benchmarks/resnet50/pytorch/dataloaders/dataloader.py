# coding=utf-8

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


import torch.distributed as dist
from torch.utils.data.dataloader import default_collate

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


class WorkerInitializer(object):

    _instance = None

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(seed=self.seed + idx)
        random.seed(self.seed + idx)

    @classmethod
    def default(cls, seed=0):
        if cls._instance is None:
            cls._instance = cls(seed)
        return cls._instance


def build_data_loader(
    dataset,
    batch_size,
    num_workers,
    drop_last,
    shuffle=True,
    worker_init_fn: WorkerInitializer = None,
):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    world_size = dist.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, shuffle=shuffle
    )
    dist_pytorch.main_proc_print(
        f"use sampler: DistributedSampler, num_replicas:{world_size}"
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return data_loader


def build_train_dataset(config):
    """build train dataset"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    traindir = os.path.join(config.data_dir, "train")
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return train_dataset


def build_eval_dataset(config):
    """build evaluation dataset"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    valdir = os.path.join(config.data_dir, "val")

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    return val_dataset


def build_train_dataloader(
    args, train_dataset, worker_init_fn: WorkerInitializer = None
):
    """Build training dataloader"""
    dist_pytorch.main_proc_print("building train dataloader ...")

    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default(args.seed)

    train_dataloader = build_data_loader(
        train_dataset,
        args.train_batch_size,
        args.num_workers,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    dist_pytorch.main_proc_print(
        f"train samples:{len(train_dataset)}, batch size:{args.train_batch_size}"
    )
    return train_dataloader


def build_eval_dataloader(config, eval_dataset):
    """build validation dataloader"""
    dist_pytorch.main_proc_print("building eval dataloaders ...")

    if config.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=True
        )
    else:
        val_sampler = None

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, shuffle=False, drop_last=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    dist_pytorch.main_proc_print(
        f"eval samples:{len(eval_dataset)}, batch size:{config.eval_batch_size}"
    )
    return eval_dataloader
