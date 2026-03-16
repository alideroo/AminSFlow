# 导入必要的库
import importlib  # 用于动态导入模块
import os  # 操作系统接口，处理文件路径等
from contextlib import contextmanager  # 提供上下文管理器功能
from copy import deepcopy  # 用于创建对象的深拷贝
from pathlib import Path  # 提供面向对象的文件路径处理
from typing import Any, List, Sequence  # 类型提示
import logging  # 日志记录功能
from pytorch_lightning.utilities import rank_zero_only  # PyTorch Lightning工具，确保只在主进程执行

# 导入Hydra配置管理库
import hydra  # 用于管理复杂应用的配置
from omegaconf import DictConfig, OmegaConf  # 层次化配置系统

def get_logger(name=__name__) -> logging.Logger:
    """初始化多GPU友好的Python命令行日志记录器。
    
    Args:
        name: 日志记录器名称，默认为调用模块的名称
        
    Returns:
        配置好的日志记录器
    """

    # 获取指定名称的日志记录器
    logger = logging.getLogger(name)

    # 确保所有日志级别都使用rank_zero_only装饰器标记
    # 否则在多GPU设置中，日志会被每个GPU进程重复记录
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        # 动态地为每个日志级别方法应用rank_zero_only装饰器
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# 获取当前模块的日志记录器
log = get_logger(__name__)


def make_config(**kwargs):
    """创建结构化的配置对象。
    
    Args:
        **kwargs: 关键字参数，将被转换为配置
        
    Returns:
        结构化的OmegaConf配置对象
    """
    return OmegaConf.structured(kwargs)


def compose_config(**kwds):
    """从关键字参数创建配置对象。
    
    Args:
        **kwds: 关键字参数，将被转换为配置
        
    Returns:
        OmegaConf配置对象
    """
    return OmegaConf.create(kwds)


def merge_config(default_cfg, override_cfg):
    """合并两个配置对象，override_cfg会覆盖default_cfg中的值。
    
    Args:
        default_cfg: 默认配置
        override_cfg: 覆盖配置
        
    Returns:
        合并后的配置对象
    """
    return OmegaConf.merge(default_cfg, override_cfg)


def load_yaml_config(fpath: str) -> OmegaConf:
    """从YAML文件加载配置。
    
    Args:
        fpath: YAML文件路径
        
    Returns:
        加载的配置对象
    """
    return OmegaConf.load(fpath)


def parse_cli_override_args():
    """解析命令行覆盖参数。
    
    Returns:
        包含命令行覆盖配置的OmegaConf对象
    """
    # 从命令行获取覆盖参数
    _overrides = OmegaConf.from_cli()
    print(_overrides)
    # 处理参数名称，移除开头的'+'
    overrides = compose_config(**{kk if not kk.startswith('+') else kk[1:]: vv for kk, vv in _overrides.items()})
    return overrides


def resolve_experiment_config(config: DictConfig):
    """解析实验配置，从现有实验加载配置并应用覆盖。
    
    Args:
        config: 输入配置
        
    Returns:
        解析后的配置
    """
    # 从现有Hydra实验加载训练配置
    if config.experiment_path is not None:
        # 转换为绝对路径
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        # 加载实验配置
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        from omegaconf import open_dict
        # 打开配置字典以进行修改
        with open_dict(config):
            # 复制实验配置的各个部分
            config.datamodule = experiment_config.datamodule
            config.model = experiment_config.model
            config.task = experiment_config.task
            config.train = experiment_config.train
            config.paths = experiment_config.paths
            config.name = experiment_config.name
            config.paths.log_dir = config.experiment_path
            
            # 处理命令行覆盖参数
            cli_overrides = parse_cli_override_args()
            config = merge_config(config, cli_overrides)
            print(cli_overrides)
            # 更改当前工作目录
            os.chdir(config.paths.log_dir)
    return config


def _convert_target_to_string(t: Any) -> Any:
    """将目标对象转换为字符串表示。
    
    Args:
        t: 目标对象，可以是可调用对象或其他类型
        
    Returns:
        字符串表示或原始对象
    """
    if callable(t):
        # 如果是可调用对象，返回其模块和限定名称
        return f"{t.__module__}.{t.__qualname__}"
    else:
        # 否则返回原始对象
        return t


def get_obj_from_str(string, reload=False):
    """从字符串获取对象（类或函数）。
    
    Args:
        string: 形如"module.submodule.Class"的字符串
        reload: 是否重新加载模块
        
    Returns:
        解析的对象
    """
    # 分割最后一个点，得到模块路径和类/函数名
    module, cls = string.rsplit(".", 1)
    if reload:
        # 如果需要重新加载
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 从模块获取指定的类或函数
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(cfg: OmegaConf, group=None, **override_kwargs):
    """从配置实例化对象。
    
    Args:
        cfg: 配置对象，必须包含_target_键
        group: 模块组名称，用于从注册表获取模块
        **override_kwargs: 覆盖配置的关键字参数
        
    Returns:
        实例化的对象
        
    Raises:
        KeyError: 如果配置中没有_target_键或注册表中没有指定的模块
    """
    # 检查配置中是否有_target_键
    if "_target_" not in cfg:
        raise KeyError("Expected key `_target_` to instantiate.")

    if group is None:
        # 如果没有指定组，直接使用Hydra实例化
        return hydra.utils.instantiate(cfg, **override_kwargs)
    else:
        # 使用注册表获取模块
        from . import registry
        _target_ = cfg.pop('_target_')
        target = registry.get_module(group_name=group, module_name=_target_)
        if target is None:
            # 如果模块不在注册表中，抛出异常
            raise KeyError(
                f'{_target_} is not a registered <{group}> class [{registry.get_registered_modules(group)}].')
        # 将目标转换为字符串
        target = _convert_target_to_string(target)
        log.info(f"    Resolving {group} <{_target_}> -> <{target}>")

        # 从字符串获取类
        target_cls = get_obj_from_str(target)
        try:
            # 尝试实例化类
            return target_cls(**cfg, **override_kwargs)
        except:
            # 如果失败，尝试合并配置后再实例化
            cfg = merge_config(cfg, override_kwargs)
            return target_cls(cfg)