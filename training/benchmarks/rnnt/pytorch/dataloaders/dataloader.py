# 本文件部分实现参考 https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/train.py
import numpy as np

from driver import dist_pytorch
from model import config
from common.data.dali import sampler as dali_sampler
from common.data.text import Tokenizer
from common.data.dali.data_loader import DaliDataLoader


def build_train_dataset(cfg):
    dist_pytorch.main_proc_print('Setting up training datasets...')
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_specaugm_kw,
    ) = config.input(cfg, 'train')

    return train_dataset_kw, train_features_kw, train_splicing_kw, train_specaugm_kw


def build_eval_dataset(cfg):
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, 'val')

    return val_dataset_kw, val_features_kw, val_splicing_kw, val_specaugm_kw


def get_tokenizer(cfg):
    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)
    return tokenizer


def get_sampler(args):
    world_size = args.n_device

    np_rng = None
    # np_rng is used for buckets generation, and needs the same seed on every worker
    if args.seed is not None:
        np_rng = np.random.default_rng(seed=args.seed)

    if args.num_buckets is not None:
        sampler = dali_sampler.BucketingSampler(args.num_buckets,
                                                args.train_batch_size, world_size,
                                                args.max_epochs, np_rng)
    else:
        sampler = dali_sampler.SimpleSampler()

    return sampler


def build_train_dataloader(args, train_dataset_kw, train_features_kw,
                           tokenizer):
    

    sampler = get_sampler(args)
    train_loader = DaliDataLoader(
        gpu_id=args.local_rank,
        dataset_path=args.data_dir,
        config_data=train_dataset_kw,
        config_features=train_features_kw,
        json_names=args.train_manifests,
        batch_size=args.train_batch_size,
        sampler=sampler,
        grad_accumulation_steps=args.gradient_accumulation_steps,
        pipeline_type="train",
        device_type=args.dali_device,
        tokenizer=tokenizer)
    return train_loader


def build_eval_dataloader(args, val_dataset_kw, val_features_kw, tokenizer):
    val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                dataset_path=args.data_dir,
                                config_data=val_dataset_kw,
                                config_features=val_features_kw,
                                json_names=args.val_manifests,
                                batch_size=args.val_batch_size,
                                sampler=dali_sampler.SimpleSampler(),
                                pipeline_type="val",
                                device_type=args.dali_device,
                                tokenizer=tokenizer)
    return val_loader