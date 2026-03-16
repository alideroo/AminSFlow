# 导入操作系统相关功能的标准库
import os
# 导入类型提示功能
from typing import List
# 从src.tasks模块导入on_prediction_mode上下文管理器
from src.tasks import on_prediction_mode

# 导入PyTorch的神经网络模块
from torch import nn
# 导入Hydra配置管理库
import hydra
# 导入OmegaConf的DictConfig类，用于类型标注配置对象
from omegaconf import DictConfig
# 导入PyTorch Lightning的核心类
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
# 导入PyTorch Lightning的日志记录器
from pytorch_lightning.loggers import Logger

# 导入项目自定义工具函数
from src import utils

# 获取当前模块的日志记录器
log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """包含测试/预测管道的最小示例。在测试集上评估给定的检查点。

    参数:
        config (DictConfig): 由Hydra组合的配置。

    返回:
        None
    """

    # 为PyTorch、Numpy和Python.random中的随机数生成器设置种子
    # 确保实验可重现性
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # 如果检查点路径是相对路径，则转换为绝对路径
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = utils.resolve_ckpt_path(ckpt_dir=config.paths.ckpt_dir, ckpt_path=config.ckpt_path)    
    
    # 加载通用管道组件：数据模块、模型、日志记录器和回调函数
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)

    # 初始化Lightning训练器
    log.info(f"实例化训练器 <{config.trainer._target_}>")
    # 使用Hydra实例化配置中指定的训练器，并传入日志记录器和回调函数
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    # 记录超参数
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    # 获取运行模式
    mode = config.mode

    # 开始预测
    log.info(f"开始运行模式='{mode}'!")

    # (1) 通过配置datamodule.test_split指定测试数据集
    # 使用config.get()获取配置，如果不存在则使用默认值'test'
    data_split = config.get('data_split') or config.datamodule.get('test_split', 'test')
    # 设置数据模块的测试分割参数
    datamodule.hparams.test_split = data_split
    log.info(f"从'{data_split}'数据集加载测试数据...")

    # PyTorch Lightning对预测和测试有不同处理方式
    # 使用上下文管理器和trainer.test来按预期运行预测
    # 当mode为'predict'时启用预测模式
    with on_prediction_mode(pl_module, enable=mode == 'predict'):
        # 执行测试，传入模型、数据模块和检查点路径
        trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # 记录完成信息
    log.info(f"在'{data_split}'数据集上完成模式='{mode}'。")