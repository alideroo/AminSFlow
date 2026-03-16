# 导入标准库模块
import os  # 提供文件和路径操作功能，用于保存结果和加载模型
from typing import Any, Callable, List, Union  # 提供类型注解，增强代码可读性和IDE支持
from pathlib import Path  # 提供面向对象的文件路径操作，更现代化的路径处理方式
import numpy as np  # 导入NumPy进行数值计算，用于矩阵操作和数学函数
import torch  # 导入PyTorch框架，用于构建和训练神经网络模型

# 导入项目自定义模块
from src import utils  # 导入通用工具函数，包括配置处理和模型实例化
from src.models.fixedbb.generator import IterativeRefinementGenerator, maybe_remove_batch_dim  # 导入迭代细化生成器，用于序列生成
from src.modules import metrics, noise_schedule, diffusion_utils, cross_entropy  # 导入评估指标、噪声调度、扩散工具和损失函数
from src.tasks import TaskLitModule, register_task  # 导入任务基类和任务注册装饰器，用于模型注册
from src.utils.config import compose_config as Cfg, merge_config  # 导入配置工具，用于管理模型配置

# 导入第三方库
from omegaconf import DictConfig  # 导入OmegaConf的配置字典类，用于结构化配置
from torch import nn  # 导入PyTorch神经网络模块，提供神经网络层和组件
from torch.nn import functional as F  # 导入PyTorch函数式API，提供激活函数和操作
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric  # 导入指标计算工具，用于模型评估

from tqdm import tqdm  # 导入进度条库，用于显示迭代过程的进度
from ikan import KAN
from src.datamodules.datasets.data_utils import Alphabet  # 导入字母表工具类，用于序列编码解码

# 获取日志记录器
log = utils.get_logger(__name__)  # 初始化模块级日志记录器，用于记录训练和评估信息


def new_arange(x, *size):
    """
    返回一个与x相同设备的张量，包含范围[0, size[-1])的值并扩展到指定尺寸
    
    Args:
        x: 参考设备的张量
        *size: 输出张量的尺寸，如果未指定则使用x的尺寸
        
    Returns:
        包含扩展后的整数序列的张量
    """
    if len(size) == 0:  # 如果没有提供尺寸参数
        size = x.size()  # 使用x的尺寸作为默认值
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()  # 创建张量并扩展到指定尺寸，确保内存连续


@register_task('fixedbb/mb_pifold')  # 使用装饰器注册任务，允许通过名称实例化
class MarkovBridge(TaskLitModule):  # 定义马尔可夫桥模型类，继承自TaskLitModule基类
    """实现Bridge-IF算法的马尔可夫桥模型，用于从蛋白质结构生成氨基酸序列"""

    # 默认配置字典，定义模型参数的默认值
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='no_noise',  # 训练时使用的噪声类型：无噪声/全掩码/随机掩码
            use_context=False,  # 是否使用上下文信息指导生成
            num_unroll=0,  # 用于循环网络的展开步数
        ),
        generator=Cfg(
            max_iter=1,  # 生成过程的最大迭代次数
            strategy='denoise',  # 生成策略：去噪或掩码预测
            noise='full_mask',  # 生成时使用的噪声类型
            replace_visible_tokens=False,  # 是否在生成过程中替换可见标记
            temperature=0,  # 采样温度，控制生成的多样性
            eval_sc=False,  # 是否评估结构一致性
        ),
        version=Cfg(
            dataset='cath_4.2',  # 使用的数据集版本
        ),
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],  # 模型实例或模型配置
        alphabet: DictConfig,  # 字母表配置，定义了标记到氨基酸的映射
        criterion: Union[nn.Module, DictConfig],  # 损失函数实例或配置
        optimizer: DictConfig,  # 优化器配置
        lr_scheduler: DictConfig = None,  # 学习率调度器配置(可选)
        *,  # 后续参数必须使用关键字
        learning=_DEFAULT_CFG.learning,  # 学习相关配置，默认使用_DEFAULT_CFG中定义的值
        generator=_DEFAULT_CFG.generator,  # 生成器配置，控制采样和生成过程
        version=_DEFAULT_CFG.version  # 版本配置，包括数据集选择
    ):
        # 调用父类初始化方法，设置基本组件
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # 保存初始化参数到hparams属性，并记录到日志，便于实验跟踪
        self.save_hyperparameters(logger=True)

        # 初始化字母表，用于序列编码解码
        self.alphabet = Alphabet(**alphabet)
        # 构建神经网络模型
        self.build_model() 
        # 构建序列生成器
        self.build_generator()
                # if not hasattr(self, 'test_predictions'):
        self.test_predictions = []

        # 扩散过程设置 - 对应伪代码中的超参数设置
        # self.T = self.hparams.generator.diffusion_steps  # 扩散总步数T
        self.T = 25
        self.hparams.generator.eval_sc = True
        self.hparams.learning.reparam = False
        self.alpha_t = lambda t: 1 + (12.0 * (t ** 2.0)) * ((1 - t) ** 3.0)
        self.beta_t = lambda t: self.alpha_t(t) - 1  # beta(t)函数
        # 初始化预定义的离散噪声调度 - 定义噪声添加的速率
        self.noise_schedule = noise_schedule.PredefinedNoiseScheduleDiscrete(
            noise_schedule=self.hparams.generator.diffusion_noise_schedule,  # 噪声调度类型
            timesteps=self.T  # 时间步数
        )
        # 初始化转换模型 - 定义如何在状态之间转换
        self.transition_model = noise_schedule.InterpolationTransition(
            x_classes=len(self.alphabet),  # 类别数量等于字母表大小(氨基酸种类数)
        )
        self.token_embeddings = nn.Embedding(len(self.alphabet), 64)
        self.token_embeddings.weight.requires_grad = False

        # 是否使用上下文信息指导生成
        self.use_context = self.hparams.learning.use_context
        # self.use_context = True
        self.alpha = 5.0  # 条件流匹配权重
        self.balance_factor = 0.01  # 质量平衡因子(0=完全平衡, 1=完全不平衡)

        # 加载预训练的结构编码器 - 对应伪代码中的E
        if version == 'cath_4.2':  # 根据数据集版本选择检查点
            self.load_encoder_from_ckpt('/home/zrc/Bridge-IF/ckpts/cath_4.2/lm_design_esm1b_650m_pifold/checkpoints/best.ckpt')
        elif version == 'cath_4.3':  # 另一个数据集版本
            self.load_encoder_from_ckpt('/home/zrc/Bridge-IF/ckpts/cath_4.3/lm_design_esm1b_650m_pifold/checkpoints/best.ckpt')
        # 冻结编码器参数，在训练中不更新
        for param in self.model.encoder.parameters():
            param.requires_grad_(False)

    def setup(self, stage=None) -> None:
        # 在训练/验证/测试开始前调用，初始化阶段特定组件
        super().setup(stage)  # 调用父类setup方法，设置stage属性

        # 构建损失函数，用于模型训练
        self.build_criterion()
        # 构建评估指标，用于模型评估
        self.build_torchmetric()

        # 如果是训练阶段，打印模型架构信息便于调试
        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_model(self):
        # 从配置实例化神经网络模型
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        # 使用工具函数从配置创建模型实例
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_generator(self):
        # 合并默认生成器配置和用户提供的覆盖配置
        self.hparams.generator = merge_config(
            default_cfg=self._DEFAULT_CFG.generator,
            override_cfg=self.hparams.generator
        )
        # 实例化迭代细化生成器，用于序列生成
        self.generator = IterativeRefinementGenerator(
            alphabet=self.alphabet,  # 传入字母表用于序列处理
            **self.hparams.generator  # 展开生成器配置
        )
        
        # 记录生成器配置信息
        log.info(f"Generator config: {self.hparams.generator}")

    def build_criterion(self):
        # 从配置实例化主损失函数
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        # 设置忽略索引为填充标记的索引，计算损失时忽略填充标记
        self.criterion.ignore_index = self.alphabet.padding_idx
        # 实例化交叉熵损失函数，用于序列预测
        self.criterion_ce = cross_entropy.Coord2SeqCrossEntropyLoss(label_smoothing=0.0, ignore_index=1)
        # 同样设置忽略索引
        self.criterion_ce.ignore_index = self.alphabet.padding_idx
        
    def build_torchmetric(self):
        # 初始化用于模型评估的各种指标
        self.eval_loss = MeanMetric()  # 评估损失均值
        self.eval_nll_loss = MeanMetric()  # 评估负对数似然损失均值

        self.val_ppl_best = MinMetric()  # 最佳验证困惑度(越小越好)

        self.acc = MeanMetric()  # 序列恢复准确率均值
        self.acc_best = MaxMetric()  # 最佳准确率(越大越好)

        self.acc_median = CatMetric()  # 准确率中位数计算器
        self.acc_median_best = MaxMetric()  # 最佳准确率中位数(越大越好)

    def load_from_ckpt(self, ckpt_path):
        # 从检查点加载模型状态
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']  # 加载状态字典到CPU

        # 加载状态字典，允许某些键缺失或多余
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # 打印加载信息，便于调试
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:  # 如果有缺失的键，打印详情
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
    
    def load_encoder_from_ckpt(self, ckpt_path):
        # 从检查点加载编码器状态
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']  # 加载状态字典到CPU

        # 只选择包含'encoder'的键值对，过滤出编码器相关参数
        encoder_state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}

        # 加载编码器状态字典
        missing, unexpected = self.load_state_dict(encoder_state_dict, strict=False)
        # 打印加载信息
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:  # 打印缺失的键
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_epoch_start(self) -> None:
        # print("9*****************************************************************")
        # 每个训练轮次开始时调用的钩子函数
        if self.hparams.generator.eval_sc:  # 如果启用了结构一致性评估
        # if self.hparams.generator.eval_sc:
            import esm  # 导入ESM模块，用于蛋白质结构预测
            log.info(f"Eval structural self-consistency enabled. Loading ESMFold model...")
            # 加载ESMFold模型并设为评估模式
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            # 将模型移至当前设备(CPU或GPU)
            self._folding_model = self._folding_model.to(self.device)

    # -------# 训练相关方法 - 对应算法1的实现 #-------- #
    @torch.no_grad()  # 禁用梯度计算，节省内存
    def inject_noise(self, tokens, coord_mask, noise=None, sel_mask=None, mask_by_unk=False):
        # print("10*****************************************************************")
        # 实现扩散过程的噪声注入 - 对应伪代码中生成中间状态z_t的过程
        padding_idx = self.alphabet.padding_idx  # 填充标记索引
        # 根据参数选择使用的掩码标记索引
        if mask_by_unk:
            mask_idx = self.alphabet.unk_idx  # 使用未知标记
        else:
            mask_idx = self.alphabet.mask_idx  # 使用掩码标记

        def _full_mask(target_tokens):
            # 全部掩码策略：将除特殊标记外的所有标记替换为掩码
            target_mask = (
                target_tokens.ne(padding_idx)  # 非填充标记
                & target_tokens.ne(self.alphabet.cls_idx)  # 非CLS标记
                & target_tokens.ne(self.alphabet.eos_idx)  # 非EOS标记
            )
            # 将满足掩码条件的标记替换为掩码标记
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            # 随机掩码策略：随机选择一定比例的标记进行掩码
            target_masks = (
                target_tokens.ne(padding_idx) & coord_mask  # 非填充且有坐标的标记
            )
            # 为每个标记生成随机分数
            target_score = target_tokens.clone().float().uniform_()
            # 将非目标掩码位置的分数设为大值
            target_score.masked_fill_(~target_masks, 2.0)
            # 计算目标长度并添加随机性
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            # 确保至少掩盖一个标记
            target_length = target_length + 1

            # 对分数排序并创建掩码
            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            # 应用掩码
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            # 选择性掩码：根据提供的掩码选择性地替换标记
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        def _adaptive_mask(target_tokens):
            # 自适应掩码函数存根，未实现
            raise NotImplementedError

        # 选择噪声类型，如果未指定则使用默认值
        noise = noise or self.hparams.noise

        # 根据噪声类型执行相应的掩码策略
        if noise == 'full_mask':  # 全部掩码
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':  # 随机掩码
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':  # 选择性掩码
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':  # 不添加噪声
            masked_tokens = tokens
        else:
            # 如果噪声类型未定义，引发错误
            raise ValueError(f"Noise type ({noise}) not defined.")

        # 设置前一时间步的标记
        prev_tokens = masked_tokens
        # 创建标记掩码，标识哪些位置是掩码标记
        prev_token_mask = prev_tokens.eq(mask_idx) & coord_mask
        # 返回处理后的标记和掩码
        return prev_tokens, prev_token_mask

    def sequence_structure_similarity(self, seq, target_seq):
        """计算序列与结构表示的匹配度"""
        # 可使用简单的匹配度计算，如序列相似性比例
        matches = (seq == target_seq).float()
        
        # Create boolean masks first, then combine with logical AND, then convert to float
        mask1 = (seq != self.alphabet.padding_idx)
        mask2 = (target_seq != self.alphabet.padding_idx)
        valid_positions = (mask1 & mask2).float()
        
        # Calculate similarity
        similarity = (matches.sum(dim=1) / (valid_positions.sum(dim=1) + 1e-6)).unsqueeze(1)
        return similarity
    
    def apply_noise(self, X, X_T, node_mask):
        # print("11*****************************************************************")
        # 实现扩散过程添加噪声 - 对应伪代码中的z_t ~ Cat(z_t; Q̄_t-1 x)
# ##################### 采样时间步t - 对应伪代码中的t ~ U(0,...,T-1)
        # lowest_t = 0 if self.training else 1  # 训练时从0开始，评估时从1开始
        # # 随机生成整数时间步t
        # t_int = torch.randint(lowest_t, self.T, size=(X.size(0), 1), device=X.device).float()
        # # print(t_int)
        # # 将时间步归一化到[0,1]区间
        # t_float = t_int / self.T
        # # print(t_float)
        # # 计算噪声参数 - 对应伪代码中的噪声调度计算
        # beta_t = self.noise_schedule(t_normalized=t_float)  # 获取时间t的噪声强度β_t
        # # print(beta_t)
        # # 计算累积噪声系数 α̅_t
        # alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float).to(X.device)
        # print(alpha_t_bar)
# ####################################################

        
# ###########################
        t = torch.full((len(X), 1), 0.04, device=X.device)
#         # ts = torch.linspace(0.01, 1.0, self.T).to(X.device)        
#         # indices = torch.randint(0, self.T, size=(X.size(0), 1), device=X.device)
#         # t = ts[indices].to(X.device)
#         # t = torch.rand(len(X), device=X.device).unsqueeze(1)
#         # t = torch.randint(0, self.T, size=(X.size(0), 1), device=X.device).float()
#         # t = t  / self.T
#         # alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t).to(X.device)
#         # print(alpha_t_bar)
        aligned_X = self.align_sequence_tokens(X, X_T, node_mask)
        # aligned_X = X

#         # 计算X和X_T的匹配度作为重加权基础
        alpha_t_bar = self.kappa(t)
# ###########################
        with torch.no_grad():
            similarity = self.sequence_structure_similarity(X, X_T)
            # 在训练阶段调整alpha_t_bar以实现不平衡流匹配
            if self.training:
                # 根据相似度调整混合系数，但保持原始数值范围
                modulation = 1.0 + self.balance_factor * (similarity - 0.5)
                # alpha_t_bar = alpha_t_bar * modulation.clamp(0.5, 1.0)


        xt_eq_x0_mask = torch.bernoulli(alpha_t_bar).int().to(X.device)
        X_t = (1 - xt_eq_x0_mask) * aligned_X +  xt_eq_x0_mask * X_T

        # t = torch.rand(len(X), device=X.device).unsqueeze(1)
        # print(alpha_t_bar)
        # print(t)
        # X_0 = self.x2prob(X).to(X.device)
        # X_1 = self.x2prob(X_T).to(X_T.device)
        # X_t = self.sample_cond_pt(X_0, X_1, t)
        # print(X_t)
        # print(X)
        # 确保形状一致性
        assert (X.shape == X_t.shape)

        # 收集并返回噪声数据
        noisy_data = {
            't': t,  # 整数时间步
            # 't': t_float,  # 归一化时间步
            # 'beta_t': beta_t,  # 噪声强度
            'alpha_t_bar': 1-t,  # 累积噪声系数
            'alpha_t_bar':alpha_t_bar,
            'X_t': X_t,  # 添加噪声后的数据
            'node_mask': node_mask,  # 节点掩码
            'xt_eq_x0_mask': xt_eq_x0_mask,  # 保留原始值的位置掩码
            'similarity': similarity  # 保存相似度信息用于训练
        }
        return noisy_data
################################################################################
    def kappa(self,t):
        
        # return -2 * (t**3) + 3 * (t**2) + 2.0 * (t**3 - 2 * t**2 + t) + 3.0 * (t**3 - t**2)
        return t

    def sample_cond_pt(self, p0, p1, t):
        """
        在给定时间t条件下，从p0到p1的中间分布采样
        
        Args:
            p0: 源分布概率，形状为 [batch_size, dict_size, seq_len]
            p1: 目标分布概率，形状为 [batch_size, dict_size, seq_len]
            t: 时间参数，取值范围[0,1]，形状为 [batch_size]
            kappa: 时间调度器
            
        Returns:
            采样的标记序列，形状为 [batch_size, seq_len]
        """
        # 调整t的维度以便于广播
        t = t.reshape(-1, 1, 1).to(p0.device)
        
        # 使用kappa调度器插值p0和p1
        pt = (1 - self.kappa(t)) * p0 + self.kappa(t) * p1
        pt = pt.to(t.device)
        
        # 从中间分布采样
        return self.sample_p(pt)

    def x2prob(self, x):
        """
        将标记序列转换为one-hot编码的概率分布表示
        
        Args:
            x: 形状为 [batch_size, seq_len] 的标记序列
            dict_size: 词典大小(类别数量)
            
        Returns:
            形状为 [batch_size, dict_size, seq_len] 的概率分布表示
        """
        # 将输入序列转换为one-hot编码
        x_one_hot = torch.nn.functional.one_hot(x, num_classes=33).to(x.device)
        # 调整维度顺序以符合流匹配要求
        return x_one_hot.permute(0, 2, 1)  # [batch_size, dict_size, seq_len]

    def sample_p(self, pt):
        """
        从类别概率分布中采样离散标记
        
        Args:
            pt: 形状为 [batch_size, dict_size, seq_len] 的概率分布
            
        Returns:
            形状为 [batch_size, seq_len] 的采样标记序列
        """
        batch_size, dict_size, seq_len = pt.shape
        
        # 调整维度以便于采样
        pt = pt.permute(0, 2, 1)  # [batch_size, seq_len, dict_size]
        pt = pt.reshape(-1, dict_size)  # [(batch_size * seq_len), dict_size]
        
        # 从多项分布中采样
        # xt = torch.multinomial(pt, 1).to(pt.device)  # [(batch_size * seq_len), 1]
        _, xt = pt.max(-1)  # 直接得到形状为[batch_size, seq_len]的张量
        # 重新调整形状
        return xt.reshape(batch_size, seq_len)  # [batch_size, seq_len]

    def align_sequence_tokens(self, init_pred, target_tokens, valid_mask):
        """内存高效的序列对齐实现"""
        # batch_size, seq_len = init_pred.shape
        
        # # 1. 提取序列特征
        # with torch.no_grad():  # 避免存储计算图
        #     if not hasattr(self, 'token_embeddings'):
        #         # 使用轻量级嵌入
        #         num_tokens = len(self.alphabet)
        #         self.token_embeddings = nn.Embedding(num_tokens, 64).to(init_pred.device)
                
            # 计算嵌入
        pred_embeds = self.token_embeddings(init_pred)  # [batch_size, seq_len, embed_dim]
        target_embeds = self.token_embeddings(target_tokens)  # [batch_size, seq_len, embed_dim]
            
            # 适用掩码
        pred_embeds = pred_embeds * valid_mask.unsqueeze(-1)
        target_embeds = target_embeds * valid_mask.unsqueeze(-1)
            
            # 计算序列级特征
        pred_features = pred_embeds.sum(dim=1)  # [batch_size, embed_dim]
        target_features = target_embeds.sum(dim=1)  # [batch_size, embed_dim]
            
            # 归一化特征
        pred_norm = F.normalize(pred_features, p=2, dim=1)
        target_norm = F.normalize(target_features, p=2, dim=1)
        
        # 2. 使用CPU计算相似度矩阵，避免CUDA内存压力
        pred_cpu = pred_norm.cpu()
        target_cpu = target_norm.cpu()
        
        cost_matrix = -torch.mm(pred_cpu, target_cpu.t()).detach().numpy()  # 负相似度作为成本
        
        # 3. 使用匈牙利算法找到最优匹配
        from scipy.optimize import linear_sum_assignment
        pred_perm, target_perm = linear_sum_assignment(cost_matrix)
        
        # 4. 应用排列
        aligned_pred = init_pred[torch.tensor(pred_perm).to(init_pred.device)]
        
        return aligned_pred

    # def align_sequence_tokens(self, X, X_T, node_mask):
    #     """内存优化版的序列令牌对齐
        
    #     Args:
    #         X: 初始预测序列令牌，形状为 [batch_size, seq_len]
    #         X_T: 目标序列令牌，形状为 [batch_size, seq_len]
    #         node_mask: 有效令牌掩码，形状为 [batch_size, seq_len]
            
    #     Returns:
    #         aligned_X: 对齐后的预测序列令牌
    #     """
    #     # 获取批量大小和序列长度
    #     batch_size, seq_len = X.shape
    #     device = X.device
        
    #     # 如果批量大小较小，可以直接返回原始序列
    #     if batch_size <= 1:
    #         return X
        
    #     # 创建用于嵌入的层（如果不存在）
    #     if not hasattr(self, 'token_embeddings'):
    #         embed_dim = 128  # 降低维度减少内存使用
    #         self.token_embeddings = nn.Embedding(
    #             num_embeddings=len(self.alphabet),
    #             embedding_dim=embed_dim
    #         ).to(device)
    #         # 初始化嵌入权重
    #         nn.init.normal_(self.token_embeddings.weight, std=0.02)
        
    #     # 直接计算每个批次内的相似度而不是所有批次对
    #     # 这显著减少了内存需求
    #     aligned_X = X.clone()
        
    #     # 分批处理以减少内存使用
    #     max_pairs = 16  # 每批处理的最大对数
    #     num_batches = (batch_size + max_pairs - 1) // max_pairs
        
    #     for i in range(num_batches):
    #         start_idx = i * max_pairs
    #         end_idx = min((i + 1) * max_pairs, batch_size)
    #         curr_batch_size = end_idx - start_idx
            
    #         # 获取当前批次的嵌入
    #         with torch.no_grad():  # 避免存储不必要的梯度
    #             X_curr = X[start_idx:end_idx]
    #             X_T_curr = X_T[start_idx:end_idx]
    #             mask_curr = node_mask[start_idx:end_idx]
                
    #             # 获取嵌入
    #             X_embeds = self.token_embeddings(X_curr)
    #             X_T_embeds = self.token_embeddings(X_T_curr)
                
    #             # 对每个序列位置规范化嵌入
    #             X_norm = F.normalize(X_embeds, p=2, dim=2)
    #             X_T_norm = F.normalize(X_T_embeds, p=2, dim=2)
                
    #             # 计算简化的相似度度量
    #             # 使用序列级别的平均嵌入而不是逐位置计算
    #             X_avg = torch.sum(X_norm * mask_curr.unsqueeze(-1), dim=1) / (torch.sum(mask_curr, dim=1, keepdim=True) + 1e-8)
    #             X_T_avg = torch.sum(X_T_norm * mask_curr.unsqueeze(-1), dim=1) / (torch.sum(mask_curr, dim=1, keepdim=True) + 1e-8)
                
    #             # 计算序列级别的余弦相似度
    #             sim_matrix = torch.mm(X_avg, X_T_avg.t())  # [curr_batch_size, curr_batch_size]
                
    #             # 使用贪婪匹配而非匈牙利算法，更快且内存需求更低
    #             _, indices = sim_matrix.topk(1, dim=1)
                
    #             # 应用匹配
    #             for j in range(curr_batch_size):
    #                 aligned_X[start_idx + j] = X_curr[indices[j, 0]]
        
    #     return aligned_X

    def step(self, batch, batch_idx):
        # print("12*****************************************************************")
        """
        执行训练或评估的一步 - 对应算法1的完整实现
        
        Args:
            batch: 包含输入数据的字典
            batch_idx: 批次索引
            
        Returns:
            loss: 计算的损失值
            logging_output: 包含损失和指标的日志字典
        """
        # 提取批次数据
        coords = batch['coords']  # 蛋白质结构坐标 (s)
        coord_mask = batch['coord_mask']  # 坐标掩码，标识有效坐标位置
        tokens = batch['tokens']  # 氨基酸序列标记 (y)
        # 获取批次转换器，用于序列处理
        batch_converter = self.alphabet._alphabet.get_batch_converter()

        # 注入噪声 - 对应伪代码中的初始样本准备
        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, coord_mask, noise=self.hparams.learning.noise)
        # 更新批次数据
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = label_mask = prev_token_mask        

        # 1) 使用结构编码器生成初始预测 - 对应伪代码中的x = E(s)
        encoder_logits, encoder_out = self.model.encoder(batch, return_feats=True)

        # 分离特征，防止梯度传播到编码器
        encoder_out['feats'] = encoder_out['feats'].detach()

        # 获取初始预测 - 编码器输出的最可能序列
        init_pred = encoder_logits.argmax(-1)
        # 在有坐标的位置使用预测值，其他位置保留原始标记
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        ###### 对齐pifold和esm - 实现细节，处理不同模型间的对齐 ######
        # 解码标记为氨基酸序列
        seqs = self.alphabet.decode(tokens, remove_special=True)
        # 使用批次转换器重新编码为对齐的标记
        aligned_tokens = batch_converter([('seq', seq) for seq in seqs])[-1].to(tokens)
        # 创建对齐的标签掩码，排除特殊标记
        aligned_label_mask = (
                aligned_tokens.ne(1)  # 非填充标记
                & aligned_tokens.ne(0)  # 非CLS标记
                & aligned_tokens.ne(2)  # 非EOS标记
            )
        # 创建对齐的特征张量，初始化为零
        encoder_out['aligned_feats'] = torch.zeros(aligned_tokens.shape[0],aligned_tokens.shape[1],self.hparams.model.encoder.d_model).to(encoder_out['feats'])
        # 将特征从原始位置复制到对齐位置
        encoder_out['aligned_feats'][aligned_label_mask] = encoder_out['feats'][coord_mask]
        # 保存对齐的标签掩码
        encoder_out['aligned_label_mask'] = aligned_label_mask
        
        # 对初始预测进行相同的对齐处理
        init_seqs = self.alphabet.decode(init_pred, remove_special=True)
        aligned_init_pred = batch_converter([('seq', seq) for seq in init_seqs])[-1].to(init_pred)
        ##################################
        # print("1$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(aligned_init_pred)
        # tensor([[ 0,  5,  5,  ...,  1,  1,  1],
        #         [ 0, 20, 15,  ...,  1,  1,  1],
        #         [ 0, 14, 10,  ...,  1,  1,  1],
        #         ...,
        #         [ 0, 20,  7,  ...,  1,  1,  1],
        #         [ 0,  8,  9,  ...,  9,  4,  2],
        #         [ 0, 17,  7,  ...,  1,  1,  1]], device='cpu')
        # print(aligned_init_pred.shape) torch.Size([12, 491])

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(aligned_tokens)
        # tensor([[ 0,  6, 20,  ...,  1,  1,  1],
        #         [ 0,  4, 13,  ...,  1,  1,  1],
        #         [ 0,  5, 17,  ...,  1,  1,  1],
        #         ...,
        #         [ 0,  7,  7,  ...,  1,  1,  1],
        #         [ 0, 20,  9,  ...,  9,  4,  2],
        #         [ 0,  4,  4,  ...,  1,  1,  1]], device='cpu')
        # print(aligned_tokens.shape) torch.Size([12, 491])
        # print("2$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # 2) 训练扩散桥模型 - 对应算法1的主体部分
        # 2.1: 获取噪声数据 - 对应伪代码中的z_t ~ Cat(z_t; Q̄_t-1 x)
        # print(aligned_init_pred.shape)
        # print(aligned_tokens.shape)
        # print(len(self.alphabet))
        # print("zzzzzzzzzzzzzzzzz")
        # print(aligned_init_pred)
        # print("ooooooooooo00000000000")
        # print(self.x2prob(aligned_init_pred))
        # print("ppppppppppppppppppppppppp")
        # print(self.sample_p(self.x2prob(aligned_init_pred)))
        noisy_data = self.apply_noise(
            X=aligned_init_pred,  # 初始预测作为起点x
            X_T=aligned_tokens,   # 目标序列作为终点y
            node_mask=aligned_label_mask  # 节点掩码标识有效位置
            )
        # 提取噪声掩码
        xt_eq_x0_mask = noisy_data['xt_eq_x0_mask']
        
        # 2.2: 使用解码器进行预测 - 对应伪代码中的ŷ ← φ_θ(z_t, t)
        # 如果使用上下文，设置为初始预测的克隆
        context = aligned_init_pred.clone() if self.use_context else None
        # context = context.to(noisy_data['alpha_t_bar'].device)
        # 使用解码器预测，传入噪声数据和时间步
        logits = self.model.decoder(
            tokens=noisy_data['X_t'],  # 添加噪声的标记z_t
            alpha_t_bar=noisy_data['alpha_t_bar'],  # 累积噪声系数
            context=context,  # 上下文信息(可选)
            # timesteps=noisy_data['t'],
            timesteps=noisy_data['t']*100,  # 时间步t
            encoder_out=encoder_out,  # 编码器输出特征
        )['logits']  # 提取logits，对应ŷ
        # print(logits)
        
        # print(aligned_tokens)
        # 处理模型输出并计算损失 - 对应伪代码中的最小化KL散度
        if isinstance(logits, tuple):
            # print("111111111111111111")
            # 如果模型返回多组logits(联合训练的情况)
            logits, encoder_logits = logits
            # 计算主要损失
            loss, logging_output = self.criterion(
                logits, tokens,  # 预测和目标标记
                label_mask=label_mask  # 计算损失的掩码
            )
            # 计算编码器损失
            encoder_loss, encoder_logging_output = self.criterion(encoder_logits, tokens, label_mask=label_mask)

            # 组合损失
            loss = loss + encoder_loss
            # 记录编码器损失
            logging_output['encoder/nll_loss'] = encoder_logging_output['nll_loss']
            logging_output['encoder/ppl'] = encoder_logging_output['ppl']
        else:
                
                # 核心改进：不平衡加权的损失计算
                # 1. 基本交叉熵损失
                # 在所有有效位置计算损失
                loss, logging_output = self.criterion_ce(logits, aligned_tokens, label_mask=aligned_label_mask)
                # 2. 根据相似度调整每个样本的损失权重
                # 相似度低的样本权重降低（不平衡传输的核心思想）
                sample_weights = (1.0 - self.balance_factor + self.balance_factor * noisy_data['similarity']).squeeze()

                weighted_loss = loss * sample_weights.mean()
                
        # 返回损失和日志输出
        return weighted_loss, logging_output
    
    def training_step(self, batch: Any, batch_idx: int):
        # print("13*****************************************************************")
        # 执行一步训练，由Lightning框架调用
        loss, logging_output = self.step(batch, batch_idx)  # 计算损失和日志

        # 记录训练指标
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)  # 全局步数
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)  # 学习率

        # 记录所有日志输出到TensorBoard或其他记录器
        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)

        # 返回损失
        return {"loss": loss}

    def compute_training_CE_loss_and_metrics(self, true, pred, batch_idx):
        # 计算训练交叉熵损失和指标的方法(未实现)
        pass

    def compute_training_VLB(self, true, pred, node_mask, noisy_data, batch_idx):
        # 计算变分下界损失，适用于VLB训练模式
        bsz = true.shape[0]  # 批次大小
        print("kkkkkkkkkkkkkkkkkkkkkkk")
        # 计算标记数量
        n_tokens = true.numel()  # 总标记数
        if self.criterion.ignore_index is not None:
            # 计算非填充标记的数量
            n_nonpad_tokens = true.ne(self.criterion.ignore_index).float().sum()
        # 有效样本大小
        sample_size = node_mask.sum()

        # 提取所需数据
        z_t = noisy_data['X_t']  # 当前时间步的标记
        z_T_true = true  # 真实目标
        z_T_pred = pred  # 预测目标
        t = noisy_data['t_int'] + 1  # 时间步加1，用于转移概率计算

        # 计算条件概率 - 对应伪代码中的转移分布计算
        # 计算真实条件概率q(z_s|z_t,z_T=true)
        true_pX = self.compute_q_zs_given_q_zt(self.alphabet.one_hot(z_t), self.alphabet.one_hot(z_T_true), node_mask, t=t)
        # 计算模型预测的条件概率p(z_s|z_t)
        pred_pX = self.compute_p_zs_given_p_zt(self.alphabet.one_hot(z_t), z_T_pred, node_mask, t=t)

        # 计算损失 - KL散度最小化
        loss = self.criterion(
            masked_pred_X=pred_pX,  # 预测分布
            true_X=true_pX,  # 真实分布
            label_mask=node_mask  # 掩码标识有效位置
        )

        # 收集日志信息
        logging_output = {
            'loss_sum': loss.data,  # 损失和
            'bsz': bsz,  # 批次大小
            'sample_size': self.criterion.node_loss.total_samples,  # 样本大小
            'sample_ratio': self.criterion.node_loss.total_samples / n_tokens,  # 样本比例
            'nonpad_ratio': n_nonpad_tokens / n_tokens  # 非填充比例
        }
        # 重置损失计算器
        self.criterion.reset()
        return loss, logging_output

    # -------# 评估相关方法 #-------- #
    def on_test_epoch_start(self) -> None:
        # print("6*****************************************************************")
        # 测试轮次开始时调用，设置测试参数
        self.hparams.noise = 'full_mask'  # 测试时使用全掩码噪声

    def validation_step(self, batch: Any, batch_idx: int):
        # print("7*****************************************************************")
        # 执行验证步骤，由Lightning框架调用
        loss, logging_output = self.step(batch, batch_idx)  # 计算损失和日志

        # 更新评估指标
        sample_size = logging_output['sample_size']  # 样本大小
        self.eval_loss.update(loss, weight=sample_size)  # 累积评估损失
        self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        # 如果是拟合阶段，执行预测步骤
        if self.stage == 'fit':
            pred_outs = self.predict_step(batch, batch_idx)
        # 返回损失
        return {"loss": loss}

    def on_validation_epoch_end(self):
        # print("8*****************************************************************")
        # 验证轮次结束时调用，计算和记录整体指标
        # 确定日志前缀，测试或验证
        log_key = 'test' if self.stage == 'test' else 'val'

        # 计算整个数据集的平均指标
        eval_loss = self.eval_loss.compute()  # 计算平均评估损失

        self.eval_loss.reset()  # 重置评估损失指标
        eval_nll_loss = self.eval_nll_loss.compute()  # 计算平均负对数似然损失
        # print("1++++++++++++++++++++++++++++")
        # print(eval_nll_loss)
        self.eval_nll_loss.reset()  # 重置负对数似然损失指标
        eval_ppl = torch.exp(eval_nll_loss)  # 计算困惑度(perplexity)

        # 记录指标到日志
        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        # 如果是拟合阶段，更新最佳指标
        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)  # 更新最佳困惑度
            self.log("val/ppl_best", self.val_ppl_best.compute(), on_epoch=True, prog_bar=True)

            # 执行预测轮次结束处理
            self.on_predict_epoch_end()

        # 调用父类的验证轮次结束方法
        super().on_validation_epoch_end()

    # -------# 推理/预测相关方法 - 对应算法2的实现 #-------- #
    def forward(self, batch, return_ids=False):
        # 前向传播方法，用于模型推理 - 算法2的入口
        # 获取输入序列标记
        # print("5*****************************************************************")
        tokens = batch['tokens']

        # 向输入序列注入噪声 - 准备初始状态
        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, batch['coord_mask'],
            noise=self.hparams.generator.noise,  # 使用配置的噪声类型
        )
        # 更新批次数据
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        # 执行采样过程 - 对应算法2的迭代采样
        output_tokens, output_scores, logits, history = self.sample(
            batch=batch, alphabet=self.alphabet, 
            max_iter=self.T,  # 最大迭代次数T
            strategy=self.hparams.generator.strategy,  # 采样策略
            replace_visible_tokens=self.hparams.generator.replace_visible_tokens,  # 是否替换可见标记
            temperature=self.hparams.generator.temperature  # 采样温度
        )
        
        # 返回结果，根据需要选择返回ID或解码序列
        if not return_ids:
            return self.alphabet.decode(output_tokens)  # 解码为氨基酸序列
        return output_tokens, logits, history  # 返回标记ID、logits和生成历史

    # @torch.no_grad()  # 禁用梯度计算，节省内存
#     def sample(self, batch, alphabet=None, 
#                max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
#                need_attn_weights=False):
#         # print("4*****************************************************************")
#         # 采样方法，实现算法2的完整过程
#         alphabet = alphabet or self.alphabet  # 使用提供的字母表或默认字母表
#         padding_idx = alphabet.padding_idx  # 填充标记索引
#         mask_idx = alphabet.mask_idx  # 掩码标记索引

#         max_iter = max_iter  # 最大迭代次数T
#         strategy = strategy  # 采样策略
#         temperature = temperature  # 采样温度，控制随机性

#         # 0) 编码阶段 - 对应伪代码中的z_0 ← E(s)
#         encoder_out = self.model.forward_encoder(batch)  # 使用编码器处理输入结构

#         # 1) 初始化全部为掩码标记
#         initial_output_tokens, initial_output_scores = self.model.initialize_output_tokens(
#             batch, encoder_out=encoder_out)  # 初始化输出标记
        
#         ###### 对齐pifold和esm ######
#         batch_converter = alphabet._alphabet.get_batch_converter()
#         # 解码初始输出标记
#         init_seqs = self.alphabet.decode(initial_output_tokens, remove_special=True)
#         # 重新转换为对齐的标记
#         initial_output_tokens = batch_converter([('seq', seq) for seq in init_seqs])[-1].to(initial_output_tokens)
               
#         # 创建对齐的标签掩码
#         aligned_label_mask = (
#             initial_output_tokens.ne(1)  # 非填充标记
#             & initial_output_tokens.ne(0)  # 非CLS标记
#             & initial_output_tokens.ne(2)  # 非EOS标记
#         )
#         # 创建对齐的特征
#         encoder_out['aligned_feats'] = torch.zeros(initial_output_tokens.shape[0],initial_output_tokens.shape[1],self.hparams.model.encoder.d_model).to(encoder_out['feats'])
#         # 复制特征
#         encoder_out['aligned_feats'][aligned_label_mask] = encoder_out['feats'][batch['coord_mask']]
#         # 保存对齐的标签掩码
#         encoder_out['aligned_label_mask'] = aligned_label_mask
#         ###### 对齐pifold和esm ######
        
#         # 初始化解码器输出 - 存储采样状态和历史
#         prev_decoder_out = dict(
#             output_tokens=initial_output_tokens,  # 输出标记
#             output_scores=torch.zeros_like(initial_output_tokens).float(),  # 输出分数
#             logits=torch.zeros_like(initial_output_tokens).float().unsqueeze(-1).repeat(1,1,33),  # logits
#             output_masks=None,  # 输出掩码
#             attentions=None,  # 注意力权重
#             step=0,  # 当前步骤
#             max_step=max_iter,  # 最大步骤
#             history=[initial_output_tokens.clone()],  # 记录生成历史
#             temperature=temperature,  # 采样温度
#             xt_neq_xT=torch.full_like(initial_output_tokens, True, dtype=torch.bool)  # 标记是否不等于目标
#         )

#         # 如果使用上下文，设置为初始输出标记的克隆
#         context = initial_output_tokens.clone() if self.use_context else None

#         # 如果需要注意力权重，初始化列表
#         if need_attn_weights:
#             attns = []  # 用于存储注意力权重
# ##################################
#         t_min = 1e-1
#         n_steps = 4
#         t = t_min * torch.ones(batch['prev_tokens'].size(0), device=batch['prev_tokens'].device)

#         default_h = 1 / n_steps
#         dirac_xt = self.x2prob(initial_output_tokens)

#         t = self.pad_like_x(t, dirac_xt)

#         # 迭代采样过程 - 对应伪代码中的for t in 0, ..., T - 1:循环
#         while t.max() <= 1 - default_h:
#         # for s_int in tqdm(range(0, max_iter)):
#             # 创建时间步张量
#             # s_array = s_int * torch.ones((batch['prev_tokens'].size(0), 1)).type_as(batch['coords'])
#             # t_array = s_array + 1  # 下一时间步
#             # 采样z_s - 对应伪代码中的q_θ(z_t+1|z_t)和z_t+1 ~ q_θ(z_t+1|z_t)

#             sampled_s, output_scores, logits, new_xt_neq_xT, h, t, X_s_n = self.sample_p_zs_given_zt(
#                 s=default_h,  # 当前时间步
#                 t=t,  # 下一时间步
#                 prev_decoder_out=prev_decoder_out,  # 前一步解码器输出
#                 X=initial_output_tokens,  # 初始标记
#                 node_mask=aligned_label_mask,  # 节点掩码
#                 context=context,  # 上下文
#                 encoder_out=encoder_out,  # 编码器输出
#                 argmax_decoding=True  # 使用argmax解码方式
#             )
#             # print(t.flatten())
#             # 如果需要替换可见标记，进行替换
#             if replace_visible_tokens:
#                 visible_token_mask = ~batch['prev_token_mask']  # 可见标记的掩码
#                 visible_tokens = batch['prev_tokens']  # 可见标记
#                 output_tokens = torch.where(
#                     visible_token_mask, visible_tokens, output_tokens)  # 替换可见标记

#             # 如果需要注意力权重，保存注意力信息
#             if need_attn_weights:
#                 attns.append(
#                     dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
#                          output=maybe_remove_batch_dim(output_tokens),
#                          attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
#                 )

#             # 更新解码器输出
#             prev_decoder_out.update(
#                 # output_tokens=sampled_s,  # 更新当前标记
#                 output_tokens=X_s_n,
#                 output_tokens_n = sampled_s,
#                 output_scores=output_scores,  # 更新分数
#                 logits=logits,  # 更新logits
#                 # step=self.T-s_int,  # 更新步骤
#                 setp = h,
#                 xt_neq_xT=new_xt_neq_xT,  # 更新标记掩码
#             )
#             # t += h
#             # 添加当前采样结果到历史记录
#             prev_decoder_out['history'].append(sampled_s)

#         # 设置最终解码器输出
#         decoder_out = prev_decoder_out

#         # 返回结果，根据需要选择返回格式
#         if need_attn_weights:
#             return decoder_out['output_tokens_n'], decoder_out['output_scores'], decoder_out['logits'], attns
#         return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], prev_decoder_out['history']
    
    def derivative(self, t):
        # return -6 * (t**2) + 6 * t + 2.0 * (3 * t**2 - 4 * t + 1) + 3.0 * (3 * t**2 - 2 * t)
        return torch.ones_like(t).to(t.device)


#     def adaptative_h(self, h, t):
#        # 校正采样器的自适应步长计算
#        alpha_term = self.alpha_t(t) * self.derivative(t) / (1 - self.kappa(t))
#        beta_term = self.beta_t(t) * self.derivative(t) / self.kappa(t)
#        coeff = 1 / (alpha_term + beta_term)  # 根据校正项计算步长系数

#        h = torch.tensor(h, device=t.device)
#        h_adapt = torch.minimum(h, coeff)  # 取默认步长和自适应步长的较小值
#        return h_adapt

    def u(self, t, xt, pred, X):
       # 使用校正版本的向量场，通过alpha_t和beta_t调整
    #    return self.bar_u(t, xt, self.alpha_t(t), self.beta_t(t), pred, X)
        return self.forward_u(t, xt, pred)

    def bar_u(self, t, xt, alpha_t, beta_t, pred, X):
       # 计算校正向量场，结合前向和反向向量场
       return alpha_t * self.forward_u(t, xt, pred) - beta_t * self.backward_u(t, xt, X)

    def forward_u(self, t, xt, pred):
       # 计算前向向量场，定义了从x_t到x_1的传输方向
    #    print(xt.shape)
       dirac_xt = self.x2prob(xt)  # 将当前状态转为one-hot表示
       p1t = torch.softmax(pred, dim=-1).permute(0, 2, 1)  # 预测t时刻条件下的目标分布
       t = self.pad_like_x(t, dirac_xt)
       kappa_coeff = self.derivative(t) / (1 - self.kappa(t))  # 计算kappa系数
       kappa_coeff = kappa_coeff.view(-1, 1, 1)
    #    print("+++++++++++++++++++")
    #    print(p1t.shape)
    #    print(dirac_xt.shape)
    #    print(kappa_coeff.shape)
       return kappa_coeff * (p1t - dirac_xt)  # 返回缩放后的向量场方向

    def backward_u(self, t, xt, X):
       # 计算反向向量场，定义了从x_t到x_0的传输方向
       dirac_xt = self.x2prob(xt)  # 将当前状态转为one-hot表示

       # TODO: adapt to Ccoupling
       x0 = X.to(X.device)  # 创建初始状态x_0
       p = self.x2prob(x0)  # 将x_0转为one-hot表示

       kappa_coeff = self.derivative(t) / self.kappa(t)  # 计算kappa系数
    #    kappa_coeff = kappa_coeff.view(-1, 1, 1)
    #    print(dirac_xt.shape)
    #    print(p.shape)
       return kappa_coeff * (dirac_xt - p)  # 返回缩放后的向量场方向

    def pad_like_x(self, x, y):
        # add dims to x to match number of dims in y
        return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))

#     def sample_p_zs_given_zt(self, s, t, prev_decoder_out, X, node_mask, context=None, encoder_out=None, argmax_decoding=True):
#         # 采样p(z_s|z_t)的实现 - 对应算法2中的预测和采样步骤
#         # 提取当前状态
#         X_t = prev_decoder_out['output_tokens']  # 当前标记z_t
#         # output_scores = prev_decoder_out['output_scores']  # 当前分数
#         xt_neq_xT = prev_decoder_out['xt_neq_xT']  # 标记掩码
#         # bs, n = X_t.shape[:2]  # 批次大小和序列长度
        
#         # 计算噪声参数
#         # beta_t = self.noise_schedule(t_int=t)  # 噪声强度β_t
#         # alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s)  # 累积噪声系数α̅_s

#         # 神经网络预测 - 对应伪代码中的ŷ ← φ_θ(z_t, t)
#         noisy_data = {'X_t': X_t, 't': t, 'node_mask': node_mask}  # 准备输入数据
#         # alpha_term = self.alpha_t(t) * self.derivative(t) / (1 - self.kappa(t))
#         # beta_term = self.beta_t(t) * self.derivative(t) / self.kappa(t)
#         # coeff = 1 / (alpha_term + beta_term)  # 根据校正项计算步长系数
#         # h = torch.tensor(s, device=X.device)
#         # h = torch.minimum(h, coeff)  # 取默认步长和自适应步长的较小值
#         h = self.adaptative_h(s, t) # 获取当前步长
#         dirac_xt = self.x2prob(X_t)
        
#         ######### forward_u
        
#         # 使用解码器预测
#         pred = self.model.decoder(
#             tokens=noisy_data['X_t'],  # 输入标记z_t
#             # alpha_t_bar=alpha_s_bar,  # 累积噪声系数
#             context=context,  # 上下文信息
#             timesteps=noisy_data['t'].flatten()*25,  # 时间步
#             encoder_out=encoder_out  # 编码器输出
#         )['logits']  # 提取logits，对应ŷ
#         #####def bar_u
#         pt = dirac_xt + h * self.u(t, X_t, pred, X)  # 欧拉法更新概率分布
#         # pt = dirac_xt + h *()

#         X_s_n = self.sample_p(pt)
#         # discretefm.bar_u(t, xt, self.alpha_t(t), self.beta_t(t))
# #####################################################################
#         # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         # 计算分数(概率) - 对应伪代码中的q_θ(z_t+1|z_t)
#         scores = torch.softmax(pred, dim=-1)  # 应用softmax得到概率分布
#         # print(scores)
#         # 根据解码策略选择标记 - 对应伪代码中的z_t+1 ~ q_θ(z_t+1|z_t)
#         if argmax_decoding:
#             # 贪婪解码：选择概率最高的标记
#             cur_scores, cur_tokens = scores.max(-1)
#             # print(cur_tokens.shape)
#             # print(scores.shape)
#             # print("1")
#             # 计算负熵，衡量预测的确定性
#             cur_entropy = -torch.distributions.Categorical(probs=scores).entropy()
#         else:
#             # 随机采样：从分类分布中采样
#             cur_tokens = torch.distributions.Categorical(logits=pred / 0.1).sample()  # 添加温度参数
#             # 获取采样标记的概率分数
#             cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)
#             # print(cur_tokens)
#         X_s = cur_tokens
#         # 使用怀疑解码获取掩码 - 创新性扩展，不在原始伪代码中
#         # lowest_k_mask = self.skeptical_unmasking(cur_scores, node_mask, t=s, rate_schedule='linear', topk_mode='deterministic')    
            
#         # 生成掩码，决定哪些位置要更新 - 对应改进的采样策略
#         # not_v1_t = lowest_k_mask  # 使用'uncond'模式
#         # not_v2_t = lowest_k_mask  # 使用相同的掩码
        
#         # 创建更新掩码 - 指示哪些位置更新为新预测的标记
#         # masked_to_xT = ~not_v2_t  # 取反，得到要更新的位置
#         # masked_to_xT = (masked_to_xT & node_mask).int()  # 限制在有效节点范围内
        
#         # 生成新状态：部分保留当前标记，部分更新为新预测
#         # X_s = (1 - masked_to_xT) * X_t + masked_to_xT * cur_tokens
        
#         # 更新标记掩码
#         new_xt_neq_xT = xt_neq_xT
                
#         # 确保形状一致性
#         assert (X_t.shape == X_s.shape)
#         t += h
#         # 返回采样结果
#         return (
#             X_s.type_as(X_t),  # 新采样的标记序列
#             cur_scores,  # 当前分数
#             pred,  # 原始预测logits
#             new_xt_neq_xT,  # 更新的标记掩码
#             h,
#             t,
#             X_s_n
#         )
    def sample(self, batch, alphabet=None, 
               max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
               need_attn_weights=False):
        # print("4*****************************************************************")
        # 采样方法，实现算法2的完整过程
        alphabet = alphabet or self.alphabet  # 使用提供的字母表或默认字母表
        padding_idx = alphabet.padding_idx  # 填充标记索引
        mask_idx = alphabet.mask_idx  # 掩码标记索引

        max_iter = max_iter  # 最大迭代次数T
        strategy = strategy  # 采样策略
        temperature = temperature  # 采样温度，控制随机性

        # 0) 编码阶段 - 对应伪代码中的z_0 ← E(s)
        encoder_out = self.model.forward_encoder(batch)  # 使用编码器处理输入结构

        # 1) 初始化全部为掩码标记
        initial_output_tokens, initial_output_scores = self.model.initialize_output_tokens(
            batch, encoder_out=encoder_out)  # 初始化输出标记
        
        ###### 对齐pifold和esm ######
        batch_converter = alphabet._alphabet.get_batch_converter()
        # 解码初始输出标记
        init_seqs = self.alphabet.decode(initial_output_tokens, remove_special=True)
        # 重新转换为对齐的标记
        initial_output_tokens = batch_converter([('seq', seq) for seq in init_seqs])[-1].to(initial_output_tokens)
               
        # 创建对齐的标签掩码
        aligned_label_mask = (
            initial_output_tokens.ne(1)  # 非填充标记
            & initial_output_tokens.ne(0)  # 非CLS标记
            & initial_output_tokens.ne(2)  # 非EOS标记
        )
        # 创建对齐的特征
        encoder_out['aligned_feats'] = torch.zeros(initial_output_tokens.shape[0],initial_output_tokens.shape[1],self.hparams.model.encoder.d_model).to(encoder_out['feats'])
        # 复制特征
        encoder_out['aligned_feats'][aligned_label_mask] = encoder_out['feats'][batch['coord_mask']]
        # 保存对齐的标签掩码
        encoder_out['aligned_label_mask'] = aligned_label_mask
        ###### 对齐pifold和esm ######
        
        # 初始化解码器输出 - 存储采样状态和历史
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,  # 输出标记
            output_scores=torch.zeros_like(initial_output_tokens).float(),  # 输出分数
            logits=torch.zeros_like(initial_output_tokens).float().unsqueeze(-1).repeat(1,1,33),  # logits
            output_masks=None,  # 输出掩码
            attentions=None,  # 注意力权重
            step=0,  # 当前步骤
            max_step=max_iter,  # 最大步骤
            history=[initial_output_tokens.clone()],  # 记录生成历史
            temperature=temperature,  # 采样温度
            xt_neq_xT=torch.full_like(initial_output_tokens, True, dtype=torch.bool)  # 标记是否不等于目标
        )

        # 如果使用上下文，设置为初始输出标记的克隆
        context = initial_output_tokens.clone() if self.use_context else None

        # 如果需要注意力权重，初始化列表
        if need_attn_weights:
            attns = []  # 用于存储注意力权重

        iters = 30
        ts = torch.linspace(0.01, 1.0, iters)

        # 迭代采样过程 - 对应伪代码中的for t in 0, ..., T - 1:循环
        # for s_int in tqdm(range(0, iters)):
        for s_int in tqdm(range(1, iters)):
            # 创建时间步张量
            s_array = ts[s_int-1] * torch.ones((batch['prev_tokens'].size(0), 1)).type_as(batch['coords'])
            # s_array = s_int * torch.ones((batch['prev_tokens'].size(0), 1)).type_as(batch['coords'])
            # t_array = s_array + 1  # 下一时间步
            # print(context.shape)
            # 采样z_s - 对应伪代码中的q_θ(z_t+1|z_t)和z_t+1 ~ q_θ(z_t+1|z_t)
            sampled_s, output_scores, logits, new_xt_neq_xT = self.sample_p_zs_given_zt(
                s=s_array,  # 当前时间步
                # t=t_array,  # 下一时间步
                t = ts[1]-ts[0],
                prev_decoder_out=prev_decoder_out,  # 前一步解码器输出
                X=initial_output_tokens,  # 初始标记
                node_mask=aligned_label_mask,  # 节点掩码
                context=context,  # 上下文
                encoder_out=encoder_out,  # 编码器输出
                argmax_decoding=True  # 使用argmax解码方式
            )
        # for s_int in tqdm(range(0, max_iter)):
        #     # 创建时间步张量
        #     s_array = s_int * torch.ones((batch['prev_tokens'].size(0), 1)).type_as(batch['coords'])
        #     t_array = s_array + 1  # 下一时间步

        #     # 采样z_s - 对应伪代码中的q_θ(z_t+1|z_t)和z_t+1 ~ q_θ(z_t+1|z_t)
        #     sampled_s, output_scores, logits, new_xt_neq_xT = self.sample_p_zs_given_zt(
        #         s=s_array,  # 当前时间步
        #         t=t_array,  # 下一时间步
        #         prev_decoder_out=prev_decoder_out,  # 前一步解码器输出
        #         X=initial_output_tokens,  # 初始标记
        #         node_mask=aligned_label_mask,  # 节点掩码
        #         context=context,  # 上下文
        #         encoder_out=encoder_out,  # 编码器输出
        #         argmax_decoding=True  # 使用argmax解码方式
        #     )

            # 如果需要替换可见标记，进行替换
            if replace_visible_tokens:
                visible_token_mask = ~batch['prev_token_mask']  # 可见标记的掩码
                visible_tokens = batch['prev_tokens']  # 可见标记
                output_tokens = torch.where(
                    visible_token_mask, visible_tokens, output_tokens)  # 替换可见标记

            # 如果需要注意力权重，保存注意力信息
            if need_attn_weights:
                attns.append(
                    dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
                         output=maybe_remove_batch_dim(output_tokens),
                         attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
                )

            # 更新解码器输出
            prev_decoder_out.update(
                output_tokens=sampled_s,  # 更新当前标记
                output_scores=output_scores,  # 更新分数
                logits=logits,  # 更新logits
                step=self.T-s_int,  # 更新步骤
                xt_neq_xT=new_xt_neq_xT,  # 更新标记掩码
            )
            # 添加当前采样结果到历史记录
            prev_decoder_out['history'].append(sampled_s)

        # 设置最终解码器输出
        decoder_out = prev_decoder_out

        # 返回结果，根据需要选择返回格式
        if need_attn_weights:
            return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], attns
        return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], prev_decoder_out['history']

#############################-------------------
    def sample_p_zs_given_zt(self, s, t, prev_decoder_out, X, node_mask, context=None, encoder_out=None, argmax_decoding=True, quality_threshold=0.7):
        # 采样p(z_s|z_t)的实现 - 对应算法2中的预测和采样步骤
        # 提取当前状态
        X_t = prev_decoder_out['output_tokens']  # 当前标记z_t
        output_scores = prev_decoder_out['output_scores']  # 当前分数
        xt_neq_xT = prev_decoder_out['xt_neq_xT']  # 标记掩码
        bs, n = X_t.shape[:2]  # 批次大小和序列长度
        
        # 计算噪声参数
        # beta_t = self.noise_schedule(t_int=t)  # 噪声强度β_t
        # alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s)  # 累积噪声系数α̅_s

        # 神经网络预测 - 对应伪代码中的ŷ ← φ_θ(z_t, t)
        noisy_data = {'X_t': X_t, 't_int': s, 'node_mask': node_mask}  # 准备输入数据
        
        alpha_s_bar = 1-s

        # 使用解码器预测
        # print(context.shape)
        
        pred = self.model.decoder(
            tokens=noisy_data['X_t'],  # 输入标记z_t
            alpha_t_bar=alpha_s_bar,  # 累积噪声系数
            context=context,  # 上下文信息
            timesteps=noisy_data['t_int']*100,  # 时间步
            encoder_out=encoder_out  # 编码器输出
        )['logits']  # 提取logits，对应ŷ
#############################-------------------
    # def sample_p_zs_given_zt(self, s, t, prev_decoder_out, X, node_mask, context=None, encoder_out=None, argmax_decoding=True):
    #     # 采样p(z_s|z_t)的实现 - 对应算法2中的预测和采样步骤
    #     # 提取当前状态
    #     X_t = prev_decoder_out['output_tokens']  # 当前标记z_t
    #     output_scores = prev_decoder_out['output_scores']  # 当前分数
    #     xt_neq_xT = prev_decoder_out['xt_neq_xT']  # 标记掩码
    #     bs, n = X_t.shape[:2]  # 批次大小和序列长度
        
    #     # 计算噪声参数
    #     beta_t = self.noise_schedule(t_int=t)  # 噪声强度β_t
    #     t = t[0]
    #     alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s)  # 累积噪声系数α̅_s

    #     # 神经网络预测 - 对应伪代码中的ŷ ← φ_θ(z_t, t)
    #     noisy_data = {'X_t': X_t, 't_int': s, 'node_mask': node_mask}  # 准备输入数据
    #     quality_threshold =0.7
    #     # 使用解码器预测
    #     pred = self.model.decoder(
    #         tokens=noisy_data['X_t'],  # 输入标记z_t
    #         alpha_t_bar=alpha_s_bar,  # 累积噪声系数
    #         context=context,  # 上下文信息
    #         timesteps=noisy_data['t_int'],  # 时间步
    #         encoder_out=encoder_out  # 编码器输出
    #     )['logits']  # 提取logits，对应ŷ
        
        # 计算分数(概率) - 对应伪代码中的q_θ(z_t+1|z_t)
        scores = torch.softmax(pred, dim=-1)  # 应用softmax得到概率分布

        # # 根据解码策略选择标记 - 对应伪代码中的z_t+1 ~ q_θ(z_t+1|z_t)
        # if argmax_decoding:
        #     # 贪婪解码：选择概率最高的标记
        #     cur_scores, cur_tokens = scores.max(-1)
        #     # 计算负熵，衡量预测的确定性
        #     cur_entropy = -torch.distributions.Categorical(probs=scores).entropy()
        # else:
        #     # 随机采样：从分类分布中采样
        #     cur_tokens = torch.distributions.Categorical(logits=pred / 0.1).sample()  # 添加温度参数
        #     # 获取采样标记的概率分数
        #     cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

        # # 使用怀疑解码获取掩码 - 创新性扩展，不在原始伪代码中
        # lowest_k_mask = self.skeptical_unmasking(cur_scores, node_mask, t=s, rate_schedule='linear', topk_mode='deterministic')    
            
        # # 生成掩码，决定哪些位置要更新 - 对应改进的采样策略
        # not_v1_t = lowest_k_mask  # 使用'uncond'模式
        # not_v2_t = lowest_k_mask  # 使用相同的掩码
        
        # # 创建更新掩码 - 指示哪些位置更新为新预测的标记
        # masked_to_xT = ~not_v2_t  # 取反，得到要更新的位置
        # masked_to_xT = (masked_to_xT & node_mask).int()  # 限制在有效节点范围内
        
        # # 生成新状态：部分保留当前标记，部分更新为新预测
        # X_s = (1 - masked_to_xT) * X_t + masked_to_xT * cur_tokens
        
        # # 更新标记掩码
        # new_xt_neq_xT = xt_neq_xT
                
        # # 确保形状一致性
        # assert (X_t.shape == X_s.shape)

        # # 返回采样结果
        # return (
        #     X_s.type_as(X_t),  # 新采样的标记序列
        #     cur_scores,  # 当前分数
        #     pred,  # 原始预测logits
        #     new_xt_neq_xT  # 更新的标记掩码
        # )
        


        # 根据解码策略选择标记 - 对应伪代码中的z_t+1 ~ q_θ(z_t+1|z_t)
        if argmax_decoding:
            # 贪婪解码：选择概率最高的标记
            cur_scores, cur_tokens = scores.max(-1)
            # 计算负熵，衡量预测的确定性
            # cur_entropy = -torch.distributions.Categorical(probs=scores).entropy()
        else:
            # 随机采样：从分类分布中采样
            cur_tokens = torch.distributions.Categorical(logits=pred / 0.1).sample()  # 添加温度参数
            # 获取采样标记的概率分数
            cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

        # 使用怀疑解码获取掩码 - 创新性扩展，不在原始伪代码中
        # lowest_k_mask = self.skeptical_unmasking(cur_scores, node_mask, t=s, rate_schedule='linear', topk_mode='deterministic')    
            
        # 生成掩码，决定哪些位置要更新 - 对应改进的采样策略
        # not_v1_t = lowest_k_mask  # 使用'uncond'模式
        # not_v2_t = lowest_k_mask  # 使用相同的掩码
        
        # 创建更新掩码 - 指示哪些位置更新为新预测的标记
        # masked_to_xT = ~not_v2_t  # 取反，得到要更新的位置
        # masked_to_xT = (masked_to_xT & node_mask).int()  # 限制在有效节点范围内
        

        # 生成新状态：部分保留当前标记，部分更新为新预测
        # X_s = (1 - masked_to_xT) * X_t + masked_to_xT * cur_tokens

########################################
        # 找出不同的位置
        # 定义向量场"大小" - 随时间变化的转移率
        # vector_field = (cur_tokens - X_t) / (1 - s)
        # expected_change = t * vector_field
        # p_update = (expected_change != 0).float() * t / (1 - s)
        # update_mask = torch.bernoulli(p_update).bool()
        # X_s = torch.where(update_mask, cur_tokens, X_t)
########################################
        # 核心改进：不平衡流匹配思想的离散实现
        # 1. 识别低质量位置（质量低于阈值的位置更有可能被更新）
        # 计算序列位置质量得分
        
        position_quality = self.compute_position_quality(X_t, scores, encoder_out)
        quality_factor = 1.0 + self.balance_factor * (0.5 - position_quality)
        
        low_quality_mask = position_quality < quality_threshold
        # p_update_1 = (1.0 - position_quality) * (1.0 - s[0])

        # 定义向量场"大小" - 随时间变化的转移率
        # field_magnitude = s[0] / (1 - s[0])
        field_magnitude = self.derivative(s[0]) / (1 - self.kappa(s[0]))
        adaptive_magnitude = field_magnitude * quality_factor
        # 向量场"方向" - 只在当前状态与预测状态不同的位置非零
        field_direction = (X_t != cur_tokens)
        # 计算更新概率 - 向量场大小乘以时间步长，限制在[0,1]范围内
        p_update = torch.clamp(field_magnitude * t, 0.0, 1.0).item()
        # update_probs = torch.full_like(diff_mask, p_transition, dtype=torch.float)
        # update_mask = (torch.bernoulli(update_probs).bool() & diff_mask)
         # 生成更新掩码 - 只在向量场方向非零的位置考虑更新
        update_mask = torch.bernoulli(torch.full_like(field_direction, p_update, dtype=torch.float)).bool() & field_direction & low_quality_mask
        # & low_quality_mask & (torch.bernoulli(p_update_1) > 0)
        X_s = torch.where(update_mask, cur_tokens, X_t)
########################################        kappa_coeff = self.derivative(t) / (1 - self.kappa(t))
        # dirac_xt = self.x2prob(X_t)
        # pt = dirac_xt + t * self.u(s, X_t, pred, X)  # 欧拉法更新概率分布
        # X_s = self.sample_p(pt)
        # 更新标记掩码
        new_xt_neq_xT = xt_neq_xT

        # 确保形状一致性
        assert (X_t.shape == X_s.shape)

        # 返回采样结果
        return (
            X_s.type_as(X_t),  # 新采样的标记序列
            cur_scores,  # 当前分数
            pred,  # 原始预测logits
            new_xt_neq_xT  # 更新的标记掩码
        )

    def compute_position_quality(self, tokens, scores, encoder_out):
        """计算每个序列位置的质量得分"""
        # 基本质量评估：预测分布的熵和结构兼容性
        
        # 1. 计算预测分布的熵（低熵=高确定性=高质量）
        entropy = -torch.sum(scores * torch.log(scores + 1e-10), dim=-1)
        max_entropy = torch.log(torch.tensor(scores.size(-1), dtype=torch.float, device=scores.device))
        entropy_quality = 1.0 - entropy / max_entropy  # 归一化反熵
        
        # 2. 结构兼容性得分（如有可能，基于结构编码器特征和token嵌入的相似度）
        # 简化计算，实际应用中可以使用更复杂的相似度度量
        token_emb = self.model.decoder.embed_tokens(tokens)  # [batch, seq_len, emb_dim]
        struct_emb = encoder_out['aligned_feats']  # [batch, seq_len, feat_dim]
        
        # 如果嵌入维度不同，可以使用投影
        if token_emb.size(-1) != struct_emb.size(-1):
            # 简单投影
            token_emb = token_emb[:, :, :struct_emb.size(-1)]
        
        # 计算余弦相似度
        token_emb_norm = token_emb / (torch.norm(token_emb, dim=-1, keepdim=True) + 1e-8)
        struct_emb_norm = struct_emb / (torch.norm(struct_emb, dim=-1, keepdim=True) + 1e-8)
        compatibility = torch.sum(token_emb_norm * struct_emb_norm, dim=-1)
        compatibility = (compatibility + 1) / 2  # 归一化到[0,1]
        
        # 组合得分，加权平均
        quality = 0.4 * entropy_quality + 0.6 * compatibility
        
        return quality
    
    def apply_sequence_filtering(self, tokens, encoder_out, mask):
        """应用序列过滤，实现不平衡OT中的目标分布筛选"""
        # 在实际应用中，此函数可能需要生成多个候选序列并选择最佳的
        # 这里我们简单地对当前序列进行微调
        
        # 计算序列质量
        position_quality = self.compute_position_quality(
            tokens,
            torch.softmax(self.model.decoder(
                tokens=tokens,
                alpha_t_bar=torch.zeros(1, 1).to(tokens.device),
                context=None,
                timesteps=torch.zeros(1, 1).to(tokens.device),
                encoder_out=encoder_out,
            )['logits'], dim=-1),
            encoder_out
        )
        
        # 识别低质量位置
        low_quality_mask = (position_quality < 0.5) & mask
        
        # 如果有低质量位置，尝试修复
        if low_quality_mask.any():
            # 最后一步细化的预测
            logits = self.model.decoder(
                tokens=tokens,
                alpha_t_bar=torch.zeros(1, 1).to(tokens.device),
                context=None,
                timesteps=torch.zeros(1, 1).to(tokens.device),
                encoder_out=encoder_out,
            )['logits']
            
            # 替换低质量位置
            improved_tokens = tokens.clone()
            improved_tokens[low_quality_mask] = logits[low_quality_mask].argmax(dim=-1)
            
            return improved_tokens
        
        return tokens
    
    # def sample(self, batch, alphabet=None, 
    #            max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
    #            need_attn_weights=False):
    #     # print("4*****************************************************************")
    #     # 采样方法，实现算法2的完整过程
    #     alphabet = alphabet or self.alphabet  # 使用提供的字母表或默认字母表
    #     padding_idx = alphabet.padding_idx  # 填充标记索引
    #     mask_idx = alphabet.mask_idx  # 掩码标记索引

    #     max_iter = max_iter  # 最大迭代次数T
    #     strategy = strategy  # 采样策略
    #     temperature = temperature  # 采样温度，控制随机性

    #     # 0) 编码阶段 - 对应伪代码中的z_0 ← E(s)
    #     encoder_out = self.model.forward_encoder(batch)  # 使用编码器处理输入结构

    #     # 1) 初始化全部为掩码标记
    #     initial_output_tokens, initial_output_scores = self.model.initialize_output_tokens(
    #         batch, encoder_out=encoder_out)  # 初始化输出标记
        
    #     ###### 对齐pifold和esm ######
    #     batch_converter = alphabet._alphabet.get_batch_converter()
    #     # 解码初始输出标记
    #     init_seqs = self.alphabet.decode(initial_output_tokens, remove_special=True)
    #     # 重新转换为对齐的标记
    #     initial_output_tokens = batch_converter([('seq', seq) for seq in init_seqs])[-1].to(initial_output_tokens)
               
    #     # 创建对齐的标签掩码
    #     aligned_label_mask = (
    #         initial_output_tokens.ne(1)  # 非填充标记
    #         & initial_output_tokens.ne(0)  # 非CLS标记
    #         & initial_output_tokens.ne(2)  # 非EOS标记
    #     )
    #     # 创建对齐的特征
    #     encoder_out['aligned_feats'] = torch.zeros(initial_output_tokens.shape[0],initial_output_tokens.shape[1],self.hparams.model.encoder.d_model).to(encoder_out['feats'])
    #     # 复制特征
    #     encoder_out['aligned_feats'][aligned_label_mask] = encoder_out['feats'][batch['coord_mask']]
    #     # 保存对齐的标签掩码
    #     encoder_out['aligned_label_mask'] = aligned_label_mask
    #     ###### 对齐pifold和esm ######
        
    #     # 初始化解码器输出 - 存储采样状态和历史
    #     prev_decoder_out = dict(
    #         output_tokens=initial_output_tokens,  # 输出标记
    #         output_scores=torch.zeros_like(initial_output_tokens).float(),  # 输出分数
    #         logits=torch.zeros_like(initial_output_tokens).float().unsqueeze(-1).repeat(1,1,33),  # logits
    #         output_masks=None,  # 输出掩码
    #         attentions=None,  # 注意力权重
    #         step=0,  # 当前步骤
    #         max_step=max_iter,  # 最大步骤
    #         history=[initial_output_tokens.clone()],  # 记录生成历史
    #         temperature=temperature,  # 采样温度
    #         xt_neq_xT=torch.full_like(initial_output_tokens, True, dtype=torch.bool)  # 标记是否不等于目标
    #     )

    #     # 如果使用上下文，设置为初始输出标记的克隆
    #     context = initial_output_tokens.clone() if self.use_context else None

    #     # 如果需要注意力权重，初始化列表
    #     if need_attn_weights:
    #         attns = []  # 用于存储注意力权重

    #     # 迭代采样过程 - 对应伪代码中的for t in 0, ..., T - 1:循环
    #     for s_int in tqdm(range(0, max_iter)):
    #         # 创建时间步张量
    #         s_array = s_int * torch.ones((batch['prev_tokens'].size(0), 1)).type_as(batch['coords'])
    #         t_array = s_array + 1  # 下一时间步

    #         # 采样z_s - 对应伪代码中的q_θ(z_t+1|z_t)和z_t+1 ~ q_θ(z_t+1|z_t)
    #         sampled_s, output_scores, logits, new_xt_neq_xT = self.sample_p_zs_given_zt(
    #             s=s_array,  # 当前时间步
    #             t=t_array,  # 下一时间步
    #             prev_decoder_out=prev_decoder_out,  # 前一步解码器输出
    #             X=initial_output_tokens,  # 初始标记
    #             node_mask=aligned_label_mask,  # 节点掩码
    #             context=context,  # 上下文
    #             encoder_out=encoder_out,  # 编码器输出
    #             argmax_decoding=True  # 使用argmax解码方式
    #         )

    #         # 如果需要替换可见标记，进行替换
    #         if replace_visible_tokens:
    #             visible_token_mask = ~batch['prev_token_mask']  # 可见标记的掩码
    #             visible_tokens = batch['prev_tokens']  # 可见标记
    #             output_tokens = torch.where(
    #                 visible_token_mask, visible_tokens, output_tokens)  # 替换可见标记

    #         # 如果需要注意力权重，保存注意力信息
    #         if need_attn_weights:
    #             attns.append(
    #                 dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
    #                      output=maybe_remove_batch_dim(output_tokens),
    #                      attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
    #             )

    #         # 更新解码器输出
    #         prev_decoder_out.update(
    #             output_tokens=sampled_s,  # 更新当前标记
    #             output_scores=output_scores,  # 更新分数
    #             logits=logits,  # 更新logits
    #             step=self.T-s_int,  # 更新步骤
    #             xt_neq_xT=new_xt_neq_xT,  # 更新标记掩码
    #         )
    #         # 添加当前采样结果到历史记录
    #         prev_decoder_out['history'].append(sampled_s)

    #     # 设置最终解码器输出
    #     decoder_out = prev_decoder_out

    #     # 返回结果，根据需要选择返回格式
    #     if need_attn_weights:
    #         return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], attns
    #     return decoder_out['output_tokens'], decoder_out['output_scores'], decoder_out['logits'], prev_decoder_out['history']
    
    # def sample_p_zs_given_zt(self, s, t, prev_decoder_out, X, node_mask, context=None, encoder_out=None, argmax_decoding=True):
    #     # 采样p(z_s|z_t)的实现 - 对应算法2中的预测和采样步骤
    #     # 提取当前状态
    #     X_t = prev_decoder_out['output_tokens']  # 当前标记z_t
    #     output_scores = prev_decoder_out['output_scores']  # 当前分数
    #     xt_neq_xT = prev_decoder_out['xt_neq_xT']  # 标记掩码
    #     bs, n = X_t.shape[:2]  # 批次大小和序列长度
        
    #     # 计算噪声参数
    #     beta_t = self.noise_schedule(t_int=t)  # 噪声强度β_t
    #     alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s)  # 累积噪声系数α̅_s

    #     # 神经网络预测 - 对应伪代码中的ŷ ← φ_θ(z_t, t)
    #     noisy_data = {'X_t': X_t, 't_int': s, 'node_mask': node_mask}  # 准备输入数据
        
    #     # 使用解码器预测
    #     pred = self.model.decoder(
    #         tokens=noisy_data['X_t'],  # 输入标记z_t
    #         alpha_t_bar=alpha_s_bar,  # 累积噪声系数
    #         context=context,  # 上下文信息
    #         timesteps=noisy_data['t_int'],  # 时间步
    #         encoder_out=encoder_out  # 编码器输出
    #     )['logits']  # 提取logits，对应ŷ
        
    #     # 计算分数(概率) - 对应伪代码中的q_θ(z_t+1|z_t)
    #     scores = torch.softmax(pred, dim=-1)  # 应用softmax得到概率分布
        
    #     # 根据解码策略选择标记 - 对应伪代码中的z_t+1 ~ q_θ(z_t+1|z_t)
    #     if argmax_decoding:
    #         # 贪婪解码：选择概率最高的标记
    #         cur_scores, cur_tokens = scores.max(-1)
    #         # 计算负熵，衡量预测的确定性
    #         cur_entropy = -torch.distributions.Categorical(probs=scores).entropy()
    #     else:
    #         # 随机采样：从分类分布中采样
    #         cur_tokens = torch.distributions.Categorical(logits=pred / 0.1).sample()  # 添加温度参数
    #         # 获取采样标记的概率分数
    #         cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

    #     # 使用怀疑解码获取掩码 - 创新性扩展，不在原始伪代码中
    #     lowest_k_mask = self.skeptical_unmasking(cur_scores, node_mask, t=s, rate_schedule='linear', topk_mode='deterministic')    
            
    #     # 生成掩码，决定哪些位置要更新 - 对应改进的采样策略
    #     not_v1_t = lowest_k_mask  # 使用'uncond'模式
    #     not_v2_t = lowest_k_mask  # 使用相同的掩码
        
    #     # 创建更新掩码 - 指示哪些位置更新为新预测的标记
    #     masked_to_xT = ~not_v2_t  # 取反，得到要更新的位置
    #     masked_to_xT = (masked_to_xT & node_mask).int()  # 限制在有效节点范围内
        
    #     # 生成新状态：部分保留当前标记，部分更新为新预测
    #     X_s = (1 - masked_to_xT) * X_t + masked_to_xT * cur_tokens
        
    #     # 更新标记掩码
    #     new_xt_neq_xT = xt_neq_xT
                
    #     # 确保形状一致性
    #     assert (X_t.shape == X_s.shape)

    #     # 返回采样结果
    #     return (
    #         X_s.type_as(X_t),  # 新采样的标记序列
    #         cur_scores,  # 当前分数
    #         pred,  # 原始预测logits
    #         new_xt_neq_xT  # 更新的标记掩码
    #     )
    
    def skeptical_unmasking(self, cur_scores, label_mask, t, rate_schedule, topk_mode='deterministic'):
        # 怀疑解码方法：控制每步解码的标记数量 - 算法扩展，不在原始伪代码中
        # 根据时间步计算更新率
        if rate_schedule == "linear":
            # 线性调度：随时间线性减少掩码率
            rate = 1 - (t + 1) / self.T
        elif rate_schedule == "cosine":
            # 余弦调度：更平滑的减少曲线
            rate = torch.cos((t + 1) / self.T * np.pi * 0.5)
        elif rate_schedule == "beta":
            # Beta调度：使用噪声调度函数
            rate = 1 - self.noise_schedule(t_int=t+1)
        else:
            # 未知调度类型
            raise NotImplementedError
        
        # 计算当前步要更新的标记数量
        cutoff_len = (
            label_mask.sum(1, keepdim=True).type_as(cur_scores) * rate
            ).long()
        # 将非有效位置的分数设为大值，确保不会被选中
        _scores_for_topk = cur_scores.masked_fill(~label_mask, 1000.0)
        
        # 根据topk模式选择处理方法
        if topk_mode.startswith("stochastic"):
            # 随机模式：添加Gumbel噪声引入随机性
            noise_scale = float(topk_mode.replace("stochastic", ""))
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
            _scores = _scores_for_topk + noise_scale * rate * gumbel_noise
        elif topk_mode == "deterministic":
            # 确定性模式：直接使用分数
            _scores = _scores_for_topk
            
        # 对分数排序并找出截断阈值
        sorted_scores = _scores.sort(-1)[0]
        cutoff = sorted_scores.gather(dim=-1, index=cutoff_len) + 1e-10
        
        # 创建掩码：分数低于截断阈值的位置
        masking = _scores < cutoff
        
        return masking  # 返回要掩码的位置

    def compute_q_zs_given_q_zt(self, X_t, X_T, node_mask, t):
        # 计算条件概率q(z_s|z_t) - 对应伪代码中的转移分布计算
        bs, n = X_t.shape[:2]  # 批次大小和序列长度
        beta_t = self.noise_schedule(t_int=t)  # 获取噪声强度

        # 计算转移矩阵 - 从z_t到z_s的转移概率
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,  # 噪声强度
            X_T=X_T,  # 目标状态
            node_mask=node_mask,  # 节点掩码
            device='cpu',  # 设备
        )  # 形状为(bs, n, dx_in, dx_out)

        # 计算节点转移概率
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt  # 矩阵乘法计算转移概率
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # 压缩维度
        
        # 处理概率和为0的边缘情况
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        
        # 归一化概率使总和为1
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

        # 验证概率和为1
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X  # 返回条件概率

    def compute_p_zs_given_p_zt(self, z_t, pred, node_mask, t):
        # 计算p(z_s|z_t) - 预测分布与转移矩阵的组合
        # 对预测logits应用softmax得到p(X_T)
        p_X_T = F.softmax(pred, dim=-1)  # 形状为bs, n, d

        # 初始化概率张量
        prob_X = torch.zeros_like(p_X_T)  # 形状与p_X_T相同

        # 对每个可能的目标类别计算加权概率
        for i in range(len(self.alphabet)):
            # 创建one-hot编码张量表示类别i
            X_T_i = self.alphabet.one_hot(torch.ones_like(p_X_T[..., 0]).long() * i)
            z_T = X_T_i  # 目标状态
            
            # 计算条件概率q(z_s|z_t,z_T=i)
            prob_X_i = self.compute_q_zs_given_q_zt(z_t, z_T, node_mask, t)
            
            # 加权求和：p(z_s|z_t) = ∑_i p(z_T=i) * q(z_s|z_t,z_T=i)
            prob_X += prob_X_i * p_X_T[..., i].unsqueeze(-1)

        # 验证概率和为1
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X  # 返回条件概率

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        # 预测步骤，由Lightning框架调用
        # print("2*****************************************************************")
        coord_mask = batch['coord_mask']  # 坐标掩码
        tokens = batch['tokens']  # 输入标记

        # 执行前向传播，返回预测结果 - 调用算法2的实现
        pred_tokens, logits, history = self.forward(batch, return_ids=True)
        # print(pred_tokens)
        # 对齐处理
        batch_converter = self.alphabet._alphabet.get_batch_converter()
        # 解码标记为序列再重新编码以对齐
        seqs = self.alphabet.decode(tokens, remove_special=True)
        tokens = batch_converter([('seq', seq) for seq in seqs])[-1].to(tokens)
               
        # 创建坐标掩码，排除特殊标记
        coord_mask = (
            tokens.ne(1)  # 非填充标记
            & tokens.ne(0)  # 非CLS标记
            & tokens.ne(2)  # 非EOS标记
        )
        
        # 创建对齐的坐标，用于结构评估
        batch['aligned_coords'] = torch.zeros(tokens.shape[0], tokens.shape[1], 4, 3).to(batch['coords'])
        batch['aligned_coords'][coord_mask] = batch['coords'][batch['coord_mask']]

        # 如果需要记录指标
        if log_metrics:
            # 计算每个样本的恢复准确率
            recovery_acc_per_sample = metrics.accuracy_per_sample(pred_tokens, tokens, mask=coord_mask)
            self.acc_median.update(recovery_acc_per_sample)

            # 计算全局准确率
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=coord_mask)
            self.acc.update(recovery_acc, weight=coord_mask.sum())

        # 收集结果
        pred_tokens = pred_tokens[:len(tokens)]
        results = {
            'pred_tokens': pred_tokens,  # 预测标记
            'names': batch['names'],  # 样本名称
            'native': batch['seqs'],  # 原生序列
            'recovery': recovery_acc_per_sample,  # 恢复准确率
            'sc_tmscores': np.zeros(pred_tokens.shape[0])  # 结构一致性TM分数初始值
        }

        # 如果启用了结构一致性评估
        # if self.hparams.generator.eval_sc:
        if self.hparams.generator.eval_sc:
            # 清空GPU缓存，减少内存占用
            if not hasattr(self, '_folding_model'):
                import esm
                log.info(f"Loading ESMFold model...")
                # 'cpu' = 'cpu'
                self._folding_model = esm.pretrained.esmfold_v1().eval()
                self._folding_model = self._folding_model.to(self.device)
            torch.cuda.empty_cache()
            # 评估结构一致性
            sc_tmscores, mean_plddt, pdb_results = self.eval_self_consistency(
                pred_tokens, batch['aligned_coords'], mask=tokens.ne(self.alphabet.padding_idx))
            # 保存评估结果
            results['sc_tmscores'] = sc_tmscores
            results['mean_plddt'] = mean_plddt
            results['pdb_results'] = pdb_results
        self.test_predictions.append(results)
        # print(results)
        return results  # 返回预测结果
    
    def on_predict_epoch_end(self) -> None:
        # 预测轮次结束处理，计算并记录整体指标
        # 确定日志前缀
        log_key = 'test' if self.stage == 'test' else 'val'
        # print("1*****************************************************************")
        
        # 检查是否有预测结果
        if not hasattr(self, 'test_predictions') or not self.test_predictions:
            print("No prediction results found.")
            return
        
        results = self.test_predictions
            # 清空预测结果
        if self.stage == 'fit':
            self.test_predictions=[]
        # print(results)
        # print("1############################")
        # 计算平均准确率
        acc = self.acc.compute() * 100
        self.acc.reset()
        # 记录准确率
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # 计算中位准确率
        acc_median = torch.median(self.acc_median.compute()) * 100
        self.acc_median.reset()
        # 记录中位准确率
        self.log(f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True)

        # 根据阶段执行不同操作
        if self.stage == 'fit':
            # 拟合阶段：更新最佳指标
            self.acc_best.update(acc)
            self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)

            self.acc_median_best.update(acc_median)
            self.log(f"{log_key}/acc_median_best", self.acc_median_best.compute(), on_epoch=True, prog_bar=True)
        else:
            # 测试阶段：评估结构一致性并保存预测结果
            # if self.hparams.generator.eval_sc:
            if self.hparams.generator.eval_sc:
                import itertools
                # 收集所有结果的TM分数和pLDDT分数
                sc_tmscores = list(itertools.chain(*[result['sc_tmscores'] for result in results]))
                mean_plddt = list(itertools.chain(*[result['mean_plddt'] for result in results]))
                # 记录平均分数
                self.log(f"{log_key}/sc_tmscores", np.mean(sc_tmscores), on_epoch=True, prog_bar=True)
                self.log(f"{log_key}/mean_plddt", np.mean(mean_plddt), on_epoch=True, prog_bar=True)
            
            # 保存预测结果到FASTA文件
            self.save_prediction(results, saveto=f'./test_tau{self.hparams.generator.temperature}.fasta')
            
    # def on_predict_epoch_end(self) -> None:
    #     # 预测轮次结束处理，计算并记录整体指标
    #     # 确定日志前缀
    #     log_key = 'test' if self.stage == 'test' else 'val'
        
    #     # 检查是否有预测结果
    #     if not hasattr(self, 'test_predictions') or not self.test_predictions:
    #         print("No prediction results found.")
    #         return
        
    #     results = self.test_predictions
    #     # 清空预测结果
    #     if self.stage == 'fit':
    #         self.test_predictions = []

    #     # 计算平均准确率
    #     acc = self.acc.compute() * 100
    #     self.acc.reset()
    #     # 记录准确率
    #     self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    #     # 计算中位准确率
    #     acc_median = torch.median(self.acc_median.compute()) * 100
    #     self.acc_median.reset()
    #     # 记录中位准确率
    #     self.log(f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True)

    #     # 准备数据分析
    #     all_names = []
    #     all_predictions = []
    #     all_natives = []
    #     all_recoveries = []
    #     all_tmscores = []
    #     all_lengths = []
        
    #     # 整合所有批次的数据，确保转移到CPU
    #     for entry in results:
    #         for name, prediction, native, recovery, scTM in zip(
    #             entry['names'],
    #             self.alphabet.decode(entry['pred_tokens'], remove_special=True),
    #             entry['native'],
    #             entry['recovery'],
    #             entry['sc_tmscores'],
    #         ):
    #             all_names.append(name)
    #             all_predictions.append(prediction)
    #             all_natives.append(native)
    #             # 确保tensor移至CPU
    #             recovery_val = recovery.cpu().item() if isinstance(recovery, torch.Tensor) else recovery
    #             all_recoveries.append(recovery_val)
    #             all_tmscores.append(scTM)
    #             all_lengths.append(len(prediction))
        
    #     # 创建分类标记
    #     is_short = [length < 101 for length in all_lengths]
    #     # 假设名称中包含 "_A" 表示单链蛋白质，可根据数据格式调整
    #     is_single_chain = ["_A" in name or "_a" in name or (name.count("_") == 0) for name in all_names]
        
    #     # 按类别分组计算指标
    #     # 全部蛋白质
    #     all_recovery_mean = np.mean(all_recoveries) * 100
    #     all_recovery_median = np.median(all_recoveries) * 100
        
    #     # 短蛋白质 (< 100残基)
    #     short_recoveries = [rec for rec, short in zip(all_recoveries, is_short) if short]
    #     short_recovery_mean = np.mean(short_recoveries) * 100 if short_recoveries else 0
    #     short_recovery_median = np.median(short_recoveries) * 100 if short_recoveries else 0
        
    #     # 单链蛋白质
    #     single_recoveries = [rec for rec, single in zip(all_recoveries, is_single_chain) if single]
    #     single_recovery_mean = np.mean(single_recoveries) * 100 if single_recoveries else 0
    #     single_recovery_median = np.median(single_recoveries) * 100 if single_recoveries else 0
        
    #     # 记录不同类别的指标
    #     self.log(f"{log_key}/all_recovery_mean", all_recovery_mean, on_epoch=True, prog_bar=True)
    #     self.log(f"{log_key}/all_recovery_median", all_recovery_median, on_epoch=True, prog_bar=True)
    #     self.log(f"{log_key}/short_recovery_mean", short_recovery_mean, on_epoch=True)
    #     self.log(f"{log_key}/short_recovery_median", short_recovery_median, on_epoch=True)
    #     self.log(f"{log_key}/single_recovery_mean", single_recovery_mean, on_epoch=True)
    #     self.log(f"{log_key}/single_recovery_median", single_recovery_median, on_epoch=True)
        
    #     # 如果启用了结构一致性评估，也按类别报告TM分数
    #     if self.hparams.generator.eval_sc:
    #         all_tm_mean = np.mean(all_tmscores)
    #         short_tmscores = [tm for tm, short in zip(all_tmscores, is_short) if short]
    #         short_tm_mean = np.mean(short_tmscores) if short_tmscores else 0
    #         single_tmscores = [tm for tm, single in zip(all_tmscores, is_single_chain) if single]
    #         single_tm_mean = np.mean(single_tmscores) if single_tmscores else 0
            
    #         self.log(f"{log_key}/all_tm_mean", all_tm_mean, on_epoch=True)
    #         self.log(f"{log_key}/short_tm_mean", short_tm_mean, on_epoch=True)
    #         self.log(f"{log_key}/single_tm_mean", single_tm_mean, on_epoch=True)
        
    #     # 额外编写详细的类别分析报告
    #     category_report_path = f'./{log_key}_category_report.txt'
    #     with open(category_report_path, 'w') as f:
    #         f.write("==== Protein Category Analysis Report ====\n\n")
    #         f.write(f"Total samples: {len(all_names)}\n")
    #         f.write(f"Short proteins (<100): {sum(is_short)}\n")
    #         f.write(f"Single-chain proteins: {sum(is_single_chain)}\n\n")
            
    #         f.write("=== Recovery Accuracy (%) ===\n")
    #         f.write(f"All proteins - Mean: {all_recovery_mean:.2f}, Median: {all_recovery_median:.2f}\n")
    #         f.write(f"Short proteins - Mean: {short_recovery_mean:.2f}, Median: {short_recovery_median:.2f}\n")
    #         f.write(f"Single-chain proteins - Mean: {single_recovery_mean:.2f}, Median: {single_recovery_median:.2f}\n\n")
            
    #         if self.hparams.generator.eval_sc:
    #             f.write("=== Structure Consistency (TM-score) ===\n")
    #             f.write(f"All proteins - Mean: {all_tm_mean:.4f}\n")
    #             f.write(f"Short proteins - Mean: {short_tm_mean:.4f}\n")
    #             f.write(f"Single-chain proteins - Mean: {single_tm_mean:.4f}\n")
        
    #     print(f"Category analysis report saved to {category_report_path}")
        
    #     # 根据阶段执行不同操作
    #     if self.stage == 'fit':
    #         # 拟合阶段：更新最佳指标
    #         self.acc_best.update(acc)
    #         self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)

    #         self.acc_median_best.update(acc_median)
    #         self.log(f"{log_key}/acc_median_best", self.acc_median_best.compute(), on_epoch=True, prog_bar=True)
    #     else:
    #         # 测试阶段：评估结构一致性并保存预测结果
    #         if self.hparams.generator.eval_sc:
    #             import itertools
    #             # 收集所有结果的TM分数和pLDDT分数
    #             sc_tmscores = []
    #             mean_plddt = []
                
    #             # 确保张量移至CPU
    #             for result in results:
    #                 for score in result['sc_tmscores']:
    #                     score_val = score.cpu().item() if isinstance(score, torch.Tensor) else score
    #                     sc_tmscores.append(score_val)
                    
    #                 if 'mean_plddt' in result:
    #                     for plddt in result['mean_plddt']:
    #                         plddt_val = plddt.cpu().item() if isinstance(plddt, torch.Tensor) else plddt
    #                         mean_plddt.append(plddt_val)
                
    #             # 记录平均分数
    #             if sc_tmscores:
    #                 self.log(f"{log_key}/sc_tmscores", np.mean(sc_tmscores), on_epoch=True, prog_bar=True)
    #             if mean_plddt:
    #                 self.log(f"{log_key}/mean_plddt", np.mean(mean_plddt), on_epoch=True, prog_bar=True)
            
    #         # 保存预测结果到FASTA文件，增加分类信息
    #         self.save_prediction_with_categories(results, saveto=f'./test_tau{self.hparams.generator.temperature}.fasta')

    # def save_prediction_with_categories(self, results, saveto=None):
    #     """保存预测结果到文件，附带类别信息"""
    #     save_dict = {}  # 初始化结果字典
        
    #     if saveto:
    #         saveto = os.path.abspath(saveto)  # 转换为绝对路径
    #         log.info(f"Saving categorized predictions to {saveto}...")
            
    #         # 打开输出文件
    #         fp = open(saveto, 'w')
    #         fp_native = open('./native.fasta', 'w')
            
    #         # 打开分类统计文件
    #         fp_stats = open('./category_stats.tsv', 'w')
    #         fp_stats.write("name\tlength\tcategory\trecovery\tTM-score\n")
        
    #     # 遍历所有结果批次
    #     for entry in results:
    #         for name, prediction, native, recovery, scTM in zip(
    #             entry['names'],
    #             self.alphabet.decode(entry['pred_tokens'], remove_special=True),
    #             entry['native'],
    #             entry['recovery'],
    #             entry['sc_tmscores'],
    #         ):
    #             # 确保值在CPU上
    #             recovery_val = recovery.cpu().item() if isinstance(recovery, torch.Tensor) else recovery
    #             scTM_val = scTM.cpu().item() if isinstance(scTM, torch.Tensor) else scTM
                
    #             # 确定蛋白质类别
    #             length = len(prediction)
    #             categories = []
    #             if length < 100:
    #                 categories.append("short")
    #             if "_A" in name or "_a" in name or (name.count("_") == 0):
    #                 categories.append("single_chain")
                
    #             category_str = ",".join(categories) if categories else "other"
                
    #             # 保存到结果字典
    #             save_dict[name] = {
    #                 'prediction': prediction,
    #                 'native': native,
    #                 'recovery': recovery_val,
    #                 'length': length,
    #                 'category': category_str
    #             }
                
    #             if saveto:
    #                 # 写入FASTA格式，包含名称、长度、类别、恢复准确率和结构一致性分数
    #                 fp.write(f">name={name} | L={length} | cat={category_str} | AAR={recovery_val:.2f} | scTM={scTM_val:.2f}\n")
    #                 fp.write(f"{prediction}\n\n")
    #                 fp_native.write(f">name={name}\n{native}\n\n")
                    
    #                 # 写入TSV统计
    #                 fp_stats.write(f"{name}\t{length}\t{category_str}\t{recovery_val:.4f}\t{scTM_val:.4f}\n")
        
    #     if saveto:
    #         # 关闭输出文件
    #         fp.close()
    #         fp_native.close()
    #         fp_stats.close()
            
    #     return save_dict  # 返回结果字典

    def save_prediction(self, results, saveto=None):
        # 保存预测结果到文件和字典
        # print("3*****************************************************************")
        save_dict = {}  # 初始化结果字典
        # print(results)
        if saveto:
            # 获取绝对路径
            # print("1--------------------------")
            saveto = os.path.abspath(saveto)  # 转换为绝对路径
            log.info(f"Saving predictions to {saveto}...")  # 记录保存路径
            # 打开输出文件
            fp = open(saveto, 'w')  # 写入预测结果
            fp_native = open('./native.fasta', 'w')  # 写入原生序列
        # print("2--------------------------")
        # 遍历所有结果批次
        for entry in results:
            # print("3--------------------------")
            # 遍历每个批次中的样本
            for name, prediction, native, recovery, scTM in zip(
                entry['names'],  # 样本名称
                self.alphabet.decode(entry['pred_tokens'], remove_special=True),  # 解码预测序列
                entry['native'],  # 原生序列
                entry['recovery'],  # 恢复准确率
                entry['sc_tmscores'],  # 结构一致性分数
            ):
                # 保存到结果字典
                save_dict[name] = {
                    'prediction': prediction,  # 预测序列
                    'native': native,  # 原生序列
                    'recovery': recovery  # 恢复准确率
                }
                # print("4--------------------------")
                if saveto:
                    # print("5--------------------------")
                    # print(name)
                    # print(prediction)
                    # print(recovery)
                    # 写入FASTA格式，包含名称、长度、恢复准确率和结构一致性分数
                    fp.write(f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | scTM={scTM:.2f}\n")
                    fp.write(f"{prediction}\n\n")  # 写入预测序列
                    fp_native.write(f">name={name}\n{native}\n\n")  # 写入原生序列
        
        # 注释：可选的PDB文件保存代码(已注释)
        for entry in results:
            # print("8888888888")
            for name, prediction, native, recovery, scTM, pdb in zip(
                entry['names'],
                self.alphabet.decode(entry['pred_tokens'], remove_special=True),
                entry['native'],
                entry['recovery'],
                entry['sc_tmscores'],
                entry['pdb_results'],
            ):
                save_dict[name] = {
                    'prediction': prediction,
                    'native': native,
                    'recovery': recovery
                }
                if saveto:
                    fp.write(f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | scTM={scTM:.2f}\n")
                    fp.write(f"{prediction}\n\n")
                    fp_native.write(f">name={name}\n{native}\n\n")
                # print("./predicted_pdb_initial/{}.pdb".format(name))
                # print("000000000000000000000000")
                with open("/home/zrc/Bridge-IF/results_82/{}.pdb".format(name), "w") as f:
                    f.write(pdb)                  

        if saveto:
            # 关闭输出文件
            fp.close()
            fp_native.close()
            
        return save_dict  # 返回结果字典

    def esm_refine(self, pred_ids, only_mask=False):
        """使用ESM-1b细化模型预测 - 可选的后处理步骤
        
        Args:
            pred_ids: 预测的标记ID
            only_mask: 是否只细化掩码位置
            
        Returns:
            refined_ids: 细化后的标记ID
        """
        # 如果尚未加载ESM模型，首次运行时加载
        if not hasattr(self, 'esm'):
            import esm  # 导入ESM库
            # 加载预训练的ESM-1b模型
            self.esm, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            # 获取批次转换器
            self.esm_batcher = self.esm_alphabet.get_batch_converter()
            # 将模型移至当前设备并设为评估模式
            self.esm.to('cpu')
            self.esm.eval()

        # 创建掩码，标识需要细化的位置(掩码标记)
        mask = pred_ids.eq(self.alphabet.mask_idx)

        # 使用预测ID作为输入
        input_ids = pred_ids
        # 执行ESM模型前向传播
        results = self.esm(
            input_ids.to('cpu'),  # 输入ID
            repr_layers=[33],  # 提取表示层
            return_contacts=False  # 不返回接触图
        )
        # 获取预测logits
        logits = results['logits']
        # 取argmax得到细化的ID
        refined_ids = logits.argmax(-1)
        # 转换字母表，从ESM字母表转回模型字母表
        refined_ids = convert_by_alphabets(refined_ids, self.esm_alphabet, self.alphabet)

        # 如果只细化掩码位置，保留其他位置的原始值
        if only_mask:
            refined_ids = torch.where(mask, refined_ids, pred_ids)
            
        return refined_ids  # 返回细化后的ID

    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # 自定义前向传播，转换输入为float32
    # def eval_self_consistency(self, pred_ids, positions, mask=None):
    #     """评估结构一致性：预测序列折叠后的结构与原始结构比较
        
    #     Args:
    #         pred_ids: 预测的标记ID
    #         positions: 原始结构坐标
    #         mask: 有效位置掩码
            
    #     Returns:
    #         sc_tmscores: 结构一致性TM分数
    #         mean_plddt: 平均pLDDT置信度分数
    #         pdb_outputs: PDB格式的结构输出
    #     """
    #     # 将预测ID解码为氨基酸序列
    #     pred_seqs = self.alphabet.decode(pred_ids, remove_special=True)

    #     # 初始化结果列表
    #     sc_tmscores = []  # 存储结构一致性TM分数
    #     pdb_outputs = []  # 存储PDB输出
        
    #     # 不计算梯度
    #     with torch.no_grad():
    #         # 使用ESMFold模型进行蛋白质结构预测
    #         output = self._folding_model.infer(sequences=pred_seqs, num_recycles=4)
            
    #         # 对每个样本计算TM分数
    #         for i in range(positions.shape[0]):
    #             pred_seq = pred_seqs[i]  # 获取预测序列
    #             seqlen = len(pred_seq)  # 序列长度
                
    #             # 计算TM分数：比较预测结构与原始坐标
    #             _, sc_tmscore = metrics.calc_tm_score(
    #                 positions[i, 1:seqlen + 1, :3, :].cpu().numpy(),  # 原始坐标
    #                 output['positions'][-1, i, :seqlen, :3, :].cpu().numpy(),  # 预测坐标
    #                 pred_seq, pred_seq  # 序列
    #             )
    #             sc_tmscores.append(sc_tmscore)  # 添加到结果列表
                
    #             # 生成PDB格式的结构输出
    #             pdb_output = self._folding_model.infer_pdb(pred_seq)
    #             pdb_outputs.append(pdb_output)
                
    #     # 返回评估结果
    #     return sc_tmscores, output['mean_plddt'].tolist(), pdb_outputs

    def eval_self_consistency(self, pred_ids, positions, mask=None):
        """评估结构一致性：预测序列折叠后的结构与原始结构比较"""
        # 将预测ID解码为氨基酸序列
        
        pred_seqs = self.alphabet.decode(pred_ids, remove_special=True)
        
        # 初始化结果列表
        sc_tmscores = []
        pdb_outputs = []
        
        # 不计算梯度
        with torch.no_grad():
            # 使用ESMFold模型进行蛋白质结构预测
            output = self._folding_model.infer(sequences=pred_seqs, num_recycles=2)
            
            # 对每个样本计算TM分数
            for i in range(positions.shape[0]):
                pred_seq = pred_seqs[i]
                seqlen = len(pred_seq)
                
                # 获取坐标并确保长度一致
                try:
                    # 获取原始坐标和预测坐标
                    orig_coords = positions[i, 1:seqlen+1, :3, :].cpu().numpy()
                    pred_coords = output['positions'][-1, i, :seqlen, :3, :].cpu().numpy()
                    
                    # 调试信息
                    log.info(f"Original coords shape: {orig_coords.shape}, Predicted coords shape: {pred_coords.shape}")
                    
                    # 确保长度一致
                    if orig_coords.shape[0] != pred_coords.shape[0]:
                        log.warning(f"Shape mismatch for sequence {i}: Expected {orig_coords.shape[0]} but got {pred_coords.shape[0]}")
                        
                        # 使用较短的长度
                        min_len = min(orig_coords.shape[0], pred_coords.shape[0])
                        orig_coords = orig_coords[:min_len]
                        pred_coords = pred_coords[:min_len]
                        adjusted_seq = pred_seq[:min_len]
                        sc_tmscore = 0.0
                    else:
                        adjusted_seq = pred_seq
                    
                        # 计算TM分数
                        _, sc_tmscore = metrics.calc_tm_score(
                            orig_coords,
                            pred_coords,
                            adjusted_seq, adjusted_seq
                        )
                    
                except Exception as e:
                    log.error(f"Error calculating TM-score for sequence {i}: {e}")
                    sc_tmscore = 0.0  # 出错时使用默认值
                
                sc_tmscores.append(sc_tmscore)
                
                # 生成PDB格式的结构输出
                pdb_output = self._folding_model.infer_pdb(pred_seq)
                pdb_outputs.append(pdb_output)
        
        # 计算平均pLDDT分数
        mean_plddt = output['mean_plddt'].tolist()
        
        # 返回评估结果
        return sc_tmscores, mean_plddt, pdb_outputs


def convert_by_alphabets(ids, alphabet1, alphabet2, relpace_unk_to_mask=True):
    """在不同字母表之间转换ID，用于模型间的对齐
    
    Args:
        ids: 输入ID张量
        alphabet1: 源字母表
        alphabet2: 目标字母表
        relpace_unk_to_mask: 是否将未知标记替换为掩码标记
        
    Returns:
        转换后的ID张量
    """
    # 获取输入张量的尺寸
    sizes = ids.size()
    
    # 执行字母表映射：逐一转换每个标记ID
    mapped_flat = ids.new_tensor(
        [alphabet2.get_idx(alphabet1.get_tok(ind)) for ind in ids.flatten().tolist()]
    )
    
    # 如果需要，将未知标记替换为掩码标记
    if relpace_unk_to_mask:
        mapped_flat[mapped_flat.eq(alphabet2.unk_idx)] = alphabet2.mask_idx
        
    # 重塑张量为原始尺寸并返回
    return mapped_flat.reshape(*sizes)