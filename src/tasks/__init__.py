# 导入基础库
import copy                     # 用于深拷贝对象
import glob                     # 用于文件路径模式匹配
import importlib                # 用于动态导入模块
import os                       # 操作系统功能
from contextlib import contextmanager  # 上下文管理器装饰器
from typing import Any, Callable, Dict, List, Optional, Union  # 类型提示

# 导入数据科学和深度学习库
import numpy as np              # 科学计算库
import torch                    # PyTorch深度学习库
from src import utils           # 项目自定义工具
from src.utils.lr_scheduler import get_scheduler  # 学习率调度器
from src.utils.optim import get_optimizer  # 优化器工具
from omegaconf import DictConfig # 配置管理
from pytorch_lightning import LightningModule  # PyTorch Lightning基础模块
from pytorch_lightning.utilities.types import _METRIC  # 指标类型
from torch import distributed as dist  # 分布式训练支持
from torch import nn            # 神经网络模块
from torch.nn import functional as F  # 神经网络函数
from torchmetrics import MaxMetric, MeanMetric, Metric, MinMetric, SumMetric  # 指标计算工具
from torchmetrics.text.bleu import BLEUScore as BLEU  # BLEU文本相似度评分


# 获取日志器
log = utils.get_logger(__name__)


@contextmanager
def on_prediction_mode(pl_module: LightningModule, enable=True):
    """上下文管理器，用于临时交换模型的测试和预测方法
    
    Args:
        pl_module: PyTorch Lightning模块
        enable: 是否启用交换功能，默认为True
    """
    if not enable:
        yield  # 如果不启用，直接返回
        return

    # 定义要交换的方法名模板列表
    _methods = [
        '{}_step',
        '{}_step_end',
        # 'on_{}_epoch_end',
        # '{}_epoch_end',
        # 'on_{}_batch_start',
        # 'on_{}_batch_end',
        # 'on_{}_epoch_start',
        'on_{}_epoch_end',
        # 'on_{}_start',
        # 'on_{}_end',
    ]

    # 交换测试和预测方法
    for _method in _methods:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')

        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)

        # 交换测试和预测方法/钩子
        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)

    yield  # 执行上下文内的代码

    # 恢复原始方法（再次交换回来）
    for _method in _methods:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')

        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)

        # 交换测试和预测方法/钩子
        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)


class TaskLitModule(LightningModule):
    """LightningModule的基类，用于序列到序列学习任务
    
    一个LightningModule将PyTorch代码组织为5个部分:
        - 计算（初始化）
        - 训练循环（training_step）
        - 验证循环（validation_step）
        - 测试循环（test_step）
        - 优化器配置（configure_optimizers）
    
    文档:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: List[nn.Module],          # 模型列表
        criterion: nn.Module = None,     # 损失函数
        optimizer: Union[Callable, torch.optim.Optimizer] = None,  # 优化器
        lr_scheduler: Union[Callable, torch.optim.lr_scheduler._LRScheduler] = None,  # 学习率调度器
    ):
        super().__init__()

        # 保存超参数，并记录到日志
        self.save_hyperparameters(logger=True)

        # 损失函数
        self.criterion = criterion

        # 用于记录验证指标的字典
        self.valid_logged = {}

    def setup(self, stage=None) -> None:
        """设置阶段，在训练前调用"""
        self._stage = stage
        super().setup(stage)

    @property
    def lrate(self):
        """获取当前学习率"""
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    @property
    def stage(self):
        """获取当前阶段"""
        return self._stage

    def log(self, name: str, value: _METRIC, prog_bar: bool = False, logger: bool = True, on_step: Optional[bool] = None, on_epoch: Optional[bool] = None, **kwargs) -> None:
        """记录指标，并在验证阶段保存到valid_logged字典"""
        if on_epoch and not self.training:
            self.valid_logged[name] = value
        return super().log(name, value, prog_bar, logger, on_step, on_epoch, **kwargs)

    # -------# 训练相关方法 #-------- #
    def step(self, batch):
        """单个步骤的计算，子类需实现"""
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        """训练步骤，子类需实现"""
        raise NotImplementedError

    def training_step_end(self, step_output: Union[torch.Tensor, Dict[str, Any]]) -> Union[torch.Tensor, Dict[str, Any]]:
        """训练步骤结束处理"""
        return super().training_step_end(step_output)

    def on_train_epoch_end(self, outputs: List[Any]):
        """训练轮次结束处理"""
        pass

    # -------# 评估相关方法 #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        """验证步骤，子类需实现"""
        raise NotImplementedError

    def validation_step_end(self, *args, **kwargs) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        """验证步骤结束处理"""
        return super().validation_step_end(*args, **kwargs)
        # raise NotImplementedError

    def on_validation_epoch_end(self):
        """验证轮次结束时输出日志信息"""
        logging_info = ", ".join(f"{key}={val:.3f}" for key, val in self.valid_logged.items())
        logging_info = f"Validation Info @ (Epoch {self.current_epoch}, global step {self.global_step}): {logging_info}"
        log.info(logging_info)

    def test_step(self, batch: Any, batch_idx: int):
        """测试步骤，默认复用验证步骤"""
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, *args, **kwargs) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        """测试步骤结束处理，默认复用验证步骤结束处理"""
        return self.validation_step_end(*args, **kwargs)

    def on_test_epoch_end(self):
        """测试轮次结束处理，默认复用验证轮次结束处理"""
        return self.on_validation_epoch_end()

    # -------# 推理/预测相关方法 #-------- #
    def forward(self, batch):
        """前向传播，子类需实现"""
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """预测步骤，子类需实现"""
        raise NotImplementedError

    def on_predict_epoch_end(self, results: List[Any], log_pref=None) -> None:
        """预测轮次结束处理，子类需实现"""
        raise NotImplementedError

    # -------# 优化器和学习率调度器配置 #-------- #
    def configure_optimizers(self):
        """配置优化器和学习率调度器
        
        文档:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # 获取优化器
        optimizer = get_optimizer(self.hparams.optimizer, self.parameters())
        # 如果配置了学习率调度器，则返回优化器和调度器
        if 'lr_scheduler' in self.hparams and self.hparams.lr_scheduler is not None:
            lr_scheduler, extra_kwargs = get_scheduler(self.hparams.lr_scheduler, optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {"scheduler": lr_scheduler, **extra_kwargs}
            }
        # 否则只返回优化器
        return optimizer

    # -------# 其他方法 #-------- #
    def on_train_epoch_end(self) -> None:
        """训练轮次结束时更新分布式采样器的epoch"""
        if dist.is_initialized() and hasattr(self.trainer.datamodule, 'train_batch_sampler'):
            self.trainer.datamodule.train_batch_sampler.set_epoch(self.current_epoch + 1)
            self.trainer.datamodule.train_batch_sampler._build_batches()

    def on_epoch_end(self):
        """轮次结束通用处理"""
        pass


class AutoMetric(nn.Module):
    """自动指标类，可动态创建和管理不同类型的指标"""
    
    # 指标类型简称映射
    _type_shortnames = dict(
        mean=MeanMetric,   # 平均指标
        sum=SumMetric,     # 总和指标
        max=MaxMetric,     # 最大值指标
        min=MinMetric,     # 最小值指标
    )

    def __init__(self) -> None:
        super().__init__()
        # 注册一个参数，用于确定设备
        self.register_parameter('_device', torch.zeros(1))

    @property
    def device(self):
        """获取当前设备"""
        return self._device.device

    def update(self, name, value, type='mean', **kwds):
        """更新指定名称的指标
        
        如果指标不存在，会创建一个新的指标
        """
        if not hasattr(self, name):
            if isinstance(type, str):
                type = self._type_shortnames[type]
            setattr(self, name, type(**kwds))

            getattr(self, name).to(self.device)

        getattr(self, name).update(value)

    def compute(self, name):
        """计算指定名称的指标值"""
        return getattr(self, name).compute()

    def reset(self, name):
        """重置指定名称的指标"""
        getattr(self, name).reset()


# 任务注册表，用于存储所有已注册的任务
TASK_REGISTRY = {}


def register_task(name):
    """任务注册装饰器，用于将任务类注册到TASK_REGISTRY中
    
    Args:
        name: 任务名称
    """
    def decorator(cls):
        cls._name_ = name
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


# 自动导入src/tasks目录下的所有Python文件
utils.import_modules(os.path.dirname(__file__), "src.tasks")