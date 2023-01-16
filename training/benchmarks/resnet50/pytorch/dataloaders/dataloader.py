# coding=utf-8

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
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


def my_collate(batch):
    # if train text position mask target logit_mask
    # if eval text position mask target logit_mask answer_idx

    choice_nums = 0
    for sample in batch:
        choice_nums = max(choice_nums, len(sample['text']))

    def pad_choice_dim(data, choice_num):
        if len(data) < choice_num:
            data = np.concatenate([data] + [data[0:1]] *
                                  (choice_num - len(data)))
        return data

    new_batch = []
    answers = []
    for i, sample in enumerate(batch):
        new_sample = {}
        text_len = len(sample['text'])
        loss_mask = np.array([1] * text_len + [0] * (choice_nums - text_len),
                             dtype=np.int64)
        new_sample['loss_mask'] = loss_mask
        new_sample['label'] = 0
        for key, value in sample.items():
            if key != "answer_idx":
                new_sample[key] = pad_choice_dim(value, choice_nums)
            else:
                answers.append(sample['answer_idx'])

        new_batch.append(new_sample)

    new_batch = default_collate(new_batch)
    if len(answers):
        new_batch['answer_idx'] = answers

    return new_batch


def build_data_loader(dataset,
                      batch_size,
                      num_workers,
                      drop_last,
                      shuffle=True,
                      only_rank0=False,
                      worker_init_fn: WorkerInitializer = None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    world_size = dist.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, shuffle=shuffle)
    dist_pytorch.main_proc_print(
        f"use sampler: DistributedSampler, num_replicas:{world_size}")

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=my_collate,
                                              worker_init_fn=worker_init_fn)
    return data_loader


def build_train_dataloader(args, worker_init_fn: WorkerInitializer = None):
    """Traing and validation dataloaders."""
    dist_pytorch.main_proc_print('building train dataloaders ...')
    train_dataset = H5pyDataSet("train", args)
    train_dataloader = build_data_loader(train_dataset,
                                         args.train_batch_size,
                                         args.num_workers,
                                         drop_last=False,
                                         worker_init_fn=worker_init_fn)

    dist_pytorch.main_proc_print(
        f'train samples:{len(train_dataset)}, batch size:{args.train_batch_size}'
    )
    return train_dataloader


def build_eval_dataloaders(args):
    dist_pytorch.main_proc_print('building eval dataloaders ...')
    eval_dataset = H5pyDataSet("eval", args)
    eval_dataloader = build_data_loader(eval_dataset,
                                        args.eval_batch_size,
                                        args.num_workers,
                                        shuffle=False,
                                        drop_last=False)
    dist_pytorch.main_proc_print(
        f'eval samples:{len(eval_dataset)}, batch size:{args.eval_batch_size}')
    return eval_dataloader


if __name__ == "__main__":
    import config
    config.eval_data = "/mnt/dataset/mlperf/glm/ReCoRD/eval_hdf5/eval_sparse.hdf5"
    dataset = H5pyDataSet('eval', config)
    print("len:", len(dataset))
    sample = dataset[9000]