# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

# 标准库
import os
import sys
import time
from typing import Any, Tuple

import torch

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from utils.utils import load_data

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    config.cuda = not config.no_cuda and torch.cuda.is_available()
    print(f"config.cuda: {config.cuda}")

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    # Load data
    print(f"load_data path: {config.data_dir}")

    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        path=config.data_dir)

    seed = config.seed

    init_helper.set_seed(seed, model_driver.config.vendor)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    evaluator = Evaluator(config, features, labels, adj, idx_test)

    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config,
                      features=features,
                      labels=labels,
                      adj=adj,
                      idx_train=idx_train,
                      idx_val=idx_val,
                      idx_test=idx_test)
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    training_state.init_time = (init_end_time -
                                init_start_time) / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time  # 训练起始时间，单位为ms

    # 训练过程
    while not training_state.end_training and training_state.epoch <= config.max_epochs:
        trainer.train_one_epoch()

    dist_pytorch.main_proc_print(f"Optimization Finished!")

    training_state.eval_acc, training_state.eval_loss = trainer.evaluator.evaluate(
        trainer)

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time  # 训练结束时间，单位为ms

    # 训练时长，单位为秒
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - start
    finished_info = {"e2e_time": e2e_time}

    if config_update.do_train:
        training_perf = state.num_trained_samples / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_samples_per_second": training_perf,
            "converged": state.converged,
            "final_eval_acc": state.eval_acc,
            "final_eval_loss": state.eval_loss,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
            "num_trained_samples": state.num_trained_samples,
        }
        print(f"finished_info:{finished_info}")
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)