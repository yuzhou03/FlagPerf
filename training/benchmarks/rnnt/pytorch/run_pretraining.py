# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

# 标准库
import os
import sys
import time
from typing import Any, Tuple

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

# 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
# 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
from dataloaders.dataloader import build_train_dataset, build_eval_dataset, \
    build_train_dataloader, build_eval_dataloader, get_tokenizer
from model import config as yamlconfig
from utils.mlperf import logging
from common.tb_dllogger import flush_log, init_log, log

logger = None


def check_config(args):
    assert args.gradient_accumulation_steps >= 1
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, f'{args.train_batch_size} % {args.gradient_accumulation_steps} != 0'


def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    check_config(config)

    print(f"config.model_config:{config.model_config}")
    cfg = yamlconfig.load(config.model_config)


    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    init_helper.set_seed(config.seed, model_driver.config.vendor)

    world_size = dist_pytorch.get_world_size()
    config.distributed = world_size > 1

    # 构建dataset, dataloader 【train && validate】
    train_dataset_kw, train_features_kw, train_splicing_kw, train_specaugm_kw = build_train_dataset(
        cfg)
    val_dataset_kw, val_features_kw, val_splicing_kw, val_specaugm_kw = build_eval_dataset(
        cfg)
    tokenizer = get_tokenizer(cfg)

    print("train_dataset_kw", len(train_dataset_kw))
    print("val_dataset_kw", len(val_dataset_kw))


    train_dataloader = build_train_dataloader(config, train_dataset_kw,
                                              train_features_kw, tokenizer)

    val_dataloader = build_eval_dataloader(config, val_dataset_kw,
                                           val_features_kw, tokenizer)

    # 根据 eval_dataloader 构建evaluator
    evaluator = Evaluator(config, val_dataloader)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    trainer = Trainer(
        driver=model_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=config.device,
        config=config,
        world_size=world_size,
        tokenizer=tokenizer,
        train_specaugm_kw=train_specaugm_kw,
        train_splicing_kw=train_splicing_kw,
        val_specaugm_kw=val_specaugm_kw,
        val_splicing_kw=val_splicing_kw,
    )
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    # do evaluation
    if not config.do_train:
        return config, training_state

    # init 统计
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    training_state.init_time = (init_end_time -
                                init_start_time) / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = time.time()

    # 训练过程
    training_state.epoch = 1
    while (config.start_epoch + 1 <= training_state.epoch < config.max_epochs + 1) and \
        (not training_state.end_training):

        logging.log_start(logging.constants.BLOCK_START,
                          metadata=dict(first_epoch_num=training_state.epoch,
                                        epoch_count=1))
        logging.log_start(logging.constants.EPOCH_START,
                          metadata=dict(epoch_num=training_state.epoch))
        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()

        trainer.train_one_epoch(train_dataloader, epoch_utts,
                                accumulated_batches, epoch_start_time)
        logging.log_end(logging.constants.EPOCH_STOP,
                        metadata=dict(epoch_num=training_state.epoch))
        training_state.epoch += 1

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)

    # 训练时长，单位为秒
    training_state.raw_train_time = time.time() - raw_train_start_time

    return config, training_state, trainer


if __name__ == "__main__":
    start = time.time()
    config_update, state, trainer = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    args = config_update

    if state.epoch == args.epochs:
        trainer.evaluator.evaluate(state.epoch, state.step, trainer.val_loader,
                                   trainer.val_feat_proc,
                                   trainer.tokenizer.detokenize,
                                   trainer.ema_model, trainer.criterion,
                                   trainer.greedy_decoder, args.amp)

    flush_log()
    # if config_update.save_at_the_end:
    #     checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)

    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:
        finished_info = {
            "e2e_time":
            e2e_time,
            "converged":
            state.converged,
            "raw_train_time":
            state.raw_train_time,
            "init_time":
            state.init_time,
            "epoch":
            state.epoch,
            "global_steps":
            state.global_steps,
            "train_loss":
            state.train_loss,
            "val_loss":
            state.val_loss,
            "num_trained_samples":
            state.num_mels,
            "pure_training_computing_time":
            state.pure_compute_time,
            "throughput(ips)_raw":
            state.num_mels / state.raw_train_time,
            "throughput(ips)_no_eval":
            state.num_mels / state.no_eval_time,
            "throughput(ips)_pure_compute":
            state.num_mels / state.pure_compute_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
