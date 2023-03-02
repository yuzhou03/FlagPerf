"""ResNet50 Pretraining"""
# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库
import torch.nn as nn

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(
    os.path.abspath(os.path.join(CURR_PATH, "../../"))
)  # add benchmarks directory


# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper, get_finished_info

# TODO 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from train import trainer_adapter
from train.device import Device
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

# TODO 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
from dataloaders.dataloader import (
    build_train_dataset,
    build_eval_dataset,
    build_train_dataloader,
    build_eval_dataloader,
)

logger = None


def main() -> Tuple[Any, Any]:
    """training entrypoint"""
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver()
    config = model_driver.config

    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    world_size = dist_pytorch.get_world_size()
    config.distributed = world_size > 1 or config.multiprocessing_distributed

    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    # train && val dataset
    train_dataset = build_train_dataset(config)
    val_dataset = build_eval_dataset(config)
    train_dataloader = build_train_dataloader(
        config, train_dataset, worker_init_fn=None
    )
    eval_dataloader = build_eval_dataloader(config, val_dataset)

    if config.debug_mode:
        dist_pytorch.main_proc_print(f"config.vendor: {config.vendor}")
        dist_pytorch.main_proc_print(f"learning_rate: {config.learning_rate}")
        dist_pytorch.main_proc_print(f"num_workers: {config.num_workers}")
        dist_pytorch.main_proc_print(f"n_device: {config.n_device}")
        dist_pytorch.main_proc_print(f"target_accuracy: {config.target_accuracy}")
        dist_pytorch.main_proc_print(f"weight_decay_rate: {config.weight_decay_rate}")
        dist_pytorch.main_proc_print(f"print_freq: {config.print_freq}")

        dist_pytorch.main_proc_print(f"train_dataset length:{len(train_dataloader.dataset)}")
        dist_pytorch.main_proc_print(f"train length:{len(train_dataloader)}")
        dist_pytorch.main_proc_print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
        dist_pytorch.main_proc_print(f"eval length:{len(eval_dataloader)}")
        dist_pytorch.main_proc_print(f"main max_steps: {config.max_steps}")
        dist_pytorch.main_proc_print(f"world_size={world_size}, config.distributed:{config.distributed}")

    # prepare parameters for training
    device = Device.get_device(config)
    criterion = nn.CrossEntropyLoss().to(device)
    evaluator = Evaluator(config, eval_dataloader, criterion)

    training_state = TrainingState()
    trainer = Trainer(
        driver=model_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=device,
        config=config,
    )
    training_state.set_trainer(trainer)

    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    if not config.do_train:
        init_evaluation_start = time.time()
        training_state.eval_accuracy = evaluator.evaluate(trainer)
        init_evaluation_end = time.time()

        init_evaluation_info = dict(
            eval_accuracy=training_state.eval_accuracy,
            time=init_evaluation_end - init_evaluation_start,
        )
        model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
        return config, training_state

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e3

    dist_pytorch.barrier(config.vendor)
    epoch = -1
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time

    while (
        training_state.global_steps < config.max_steps
        and not training_state.end_training
    ):
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader, criterion, epoch)
        trainer.lr_scheduler.step()

    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e3
    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config, state = main()
    if not dist_pytorch.is_main_process():
        exit()
    finished_info = get_finished_info(start, state)
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
