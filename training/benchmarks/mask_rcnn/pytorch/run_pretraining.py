"""Mask R-CNN Pretraining"""
# 标准库
import datetime
import os
import sys
import time
from typing import Any, Tuple

# 三方库
import torch

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
import utils.train.train_eval_utils as utils

# 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

from utils.train.device import Device
# 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader

from utils.train import save_on_master, mkdir

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())

    config = model_driver.config
    device = Device.get_device(config)

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = os.path.join(config.output_dir, "result",
                                    f"det_results_{now}.txt")
    seg_results_file = os.path.join(config.output_dir, "result",
                                    f"seg_results_{now}.txt")

    # mkdir if necessary
    if config.output_dir:
        mkdir(config.output_dir)

    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    world_size = dist_pytorch.get_world_size()
    config.distributed = world_size > 1 or config.multiprocessing_distributed

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    # 得到seed
    """
    这里获取seed的可行方式：
    1. 配置文件中的seed
    2. 自定义seed的生成方式：dist_pytorch.setup_seeds得到work_seeds数组，取其中某些元素。参考GLM-Pytorch的run_pretraining.py的seed生成方式
    3. 其他自定义方式
    """
    init_helper.set_seed(config.seed, model_driver.config.vendor)

    # 构建dataset, dataloader 【train && validate】
    train_dataset = build_train_dataset(config)
    eval_dataset = build_eval_dataset(config)
    train_dataloader, train_sampler = build_train_dataloader(
        config, train_dataset)
    eval_dataloader = build_eval_dataloader(config, train_dataset,
                                            eval_dataset)

    # 根据 eval_dataloader 构建evaluator
    evaluator = Evaluator(config, eval_dataloader)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    # evaluation统计
    init_evaluation_start = time.time()  # evaluation起始时间，单位为秒
    """
    实现Evaluator 类的evaluate()方法，用于返回关键指标信息，如loss，eval_embedding_average等。
    例如：training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(trainer)
    """

    init_evaluation_end = time.time()  # evaluation结束时间，单位为秒
    """
    收集eval关键信息，用于日志输出
    例如： init_evaluation_info = dict(
        eval_loss=training_state.eval_avg_loss,
        eval_embedding_average=training_state.eval_embedding_average,
        time=init_evaluation_end - init_evaluation_start)
    """
    # time单位为秒
    init_evaluation_info = dict(time=init_evaluation_end -
                                init_evaluation_start)
    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    # do evaluation
    if not config.do_train:
        return config, training_state

    # init计时
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    elapsed_ms = init_end_time - init_start_time
    training_state.init_time = elapsed_ms / 1e+3  # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time  # 训练起始时间，单位为ms

    # 训练指标
    train_loss = []
    learning_rate = []
    val_map = []

    # 训练过程
    epoch = -1
    while training_state.global_steps < config.max_steps and \
            not training_state.end_training:

        if config.distributed:
            train_sampler.set_epoch(epoch)

        epoch += 1
        training_state.epoch = epoch
        mean_loss, lr = trainer.train_one_epoch(train_dataloader,
                                                epoch,
                                                print_freq=config.print_freq,
                                                scaler=trainer.grad_scaler)

        # update learning rate
        trainer.lr_scheduler.step()

        # evaluate after every epoch
        det_info, seg_info = utils.evaluate(trainer.model,
                                            eval_dataloader,
                                            device=device)

        if det_info is not None:
            training_state.eval_mAP = det_info[1]
            print(f"training_state.eval_mAP:{training_state.eval_mAP}")

        # 只在主进程上进行写操作
        if config.local_rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            val_map.append(det_info[1])  # pascal mAP

            # 写det结果
            with open(det_results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [
                    f"{i:.4f}" for i in det_info + [mean_loss.item()]
                ] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # 写seg结果
            with open(seg_results_file, "a") as f:
                # 写入的数据包括coco指标, 还有loss和learning rate
                result_info = [
                    f"{i:.4f}" for i in seg_info + [mean_loss.item()]
                ] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        if config.output_dir:
            # 只在主进程上执行保存权重操作
            model_without_ddp = trainer.model
            if config.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    trainer.model, device_ids=[config.gpu])
                model_without_ddp = model.module

            save_files = {
                'model': model_without_ddp.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'lr_scheduler': trainer.lr_scheduler.state_dict(),
                'epoch': epoch
            }
            if config.amp:
                save_files["scaler"] = trainer.grad_scaler.state_dict()

            checkpoint_path = os.path.join(config.output_dir, "checkpoint", 
                                           f'model_{epoch}.pth')
            save_on_master(save_files, checkpoint_path)

        trainer.detect_training_status()

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time  # 训练结束时间，单位为ms

    # 训练时长，单位为秒
    raw_train_time_ms = raw_train_end_time - raw_train_start_time
    training_state.raw_train_time = raw_train_time_ms / 1e+3

    # 绘图
    plot_train_result(config, train_loss, learning_rate, val_map)

    return config, training_state


def plot_train_result(config, train_loss: list, learning_rate: list,
                      val_map: list):
    # 绘图
    if config.local_rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from utils.plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)

        # plot mAP curve
        if len(val_map) != 0:
            from utils.plot_curve import plot_map
            plot_map(val_map)


if __name__ == "__main__":

    start = time.time()
    updated_config, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - start
    if updated_config.do_train:
        # 构建训练所需的统计信息，包括不限于：e2e_time、training_samples_per_second、
        # converged、final_accuracy、raw_train_time、init_time
        training_perf = (dist_pytorch.global_batch_size(updated_config) *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_samples_per_second": training_perf,
            "converged": state.converged,
            "final_mAP": state.eval_mAP,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
