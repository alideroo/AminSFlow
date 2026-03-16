# 导入PyTorch基础模块
import torch  # PyTorch深度学习库
from torch import Tensor, nn  # 张量类和神经网络模块
from torch.nn import functional as F  # 函数式API，包含激活函数等

# 定义带标签平滑的负对数似然损失函数
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """计算带标签平滑的负对数似然损失
    
    标签平滑是一种正则化技术，通过将标签向量中的硬目标(如[0,0,1,0])替换为软目标(如[0.01,0.01,0.97,0.01])
    来防止模型过于自信，提高泛化能力。
    
    Args:
        lprobs: 对数概率，形状为[batch_size, ..., num_classes]
        target: 目标索引，形状为[batch_size, ...]
        epsilon: 平滑因子，控制平滑程度，通常取小值如0.1
        ignore_index: 忽略的标签索引(如填充标记)
        reduce: 是否对批次求和
        
    Returns:
        loss: 平滑后的损失
        nll_loss: 原始负对数似然损失(不带平滑)
    """
    # 检查target维度是否比lprobs少一维，若是则增加一维
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)  # 增加最后一维，使维度匹配
    
    # 计算负对数似然损失：收集每个样本在其目标类别上的对数概率，并取负
    nll_loss = -lprobs.gather(dim=-1, index=target)
    
    # 计算平滑损失：所有类别对数概率之和的负值
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    
    # 如果有指定忽略索引，则将对应位置的损失置为0
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)  # 创建填充掩码
        nll_loss.masked_fill_(pad_mask, 0.0)  # 忽略填充位置的nll损失
        smooth_loss.masked_fill_(pad_mask, 0.0)  # 忽略填充位置的平滑损失
    
    # 如果之前扩展了维度，现在去掉扩展的维度
    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    
    # 如果需要规约(一般是对批次求和)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    
    # 计算每个类别的平滑权重
    eps_i = epsilon / (lprobs.size(-1) - 1)
    
    # 组合nll损失和平滑损失，计算最终损失
    # (1-epsilon-eps_i)是正确类别的权重，eps_i是错误类别的权重
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    
    return loss, nll_loss


# 定义交叉熵损失类，继承自PyTorch的CrossEntropyLoss
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """扩展的交叉熵损失，支持掩码和额外日志信息
    
    相比标准CrossEntropyLoss，该类添加了对掩码的支持，
    并返回更详细的日志信息如困惑度(PPL)等。
    """
    def forward(self, scores: Tensor, target: Tensor, mask=None) -> Tensor:
        """前向传播计算损失
        
        Args:
            scores: 未规范化的分数，[N, ..., C]，其中C是类别数
            target: 目标索引，[N, ...]
            mask: 掩码张量，[N, ...]，True表示有效位置，False表示忽略的位置
            
        Returns:
            loss_avg: 平均损失值
            logging_output: 包含损失相关信息的字典
        """
        # 计算总标记数和非填充标记数
        n_tokens = target.numel()  # 总标记数
        n_nonpad_tokens = target.ne(self.ignore_index).long().sum()  # 非填充标记数
        
        bsz, num_classes = scores.shape[0], scores.shape[-1]  # 批次大小和类别数
        
        # 如果提供了掩码，只考虑掩码内的位置
        if mask is not None:
            scores = scores[mask]  # [N * len, C]
            target = target[mask]  # [N]
        
        # 将分数和目标重塑为2D和1D形式，便于计算
        scores = scores.reshape(-1, num_classes)
        target = target.reshape(-1)
        
        # 计算样本大小(有效标记数)
        if self.ignore_index is not None:
            sample_size = target.ne(self.ignore_index).long().sum()
        else:
            sample_size = torch.tensor(target.numel(), device=target.device)
        
        # 计算带标签平滑的损失和原始nll损失
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),  # 将分数转为对数概率
            target=target,
            epsilon=self.label_smoothing,  # 标签平滑因子
            ignore_index=self.ignore_index,
            reduce=True,  # 对批次求和
        )
        
        # 计算平均损失和困惑度
        loss_avg = loss / sample_size  # 平均损失
        ppl = torch.exp(nll_loss / sample_size)  # 困惑度(perplexity)
        
        # 整理日志信息
        logging_output = {
            'nll_loss_sum': nll_loss.data,  # nll损失总和
            'loss_sum': loss.data,  # 损失总和
            'ppl': ppl.data,  # 困惑度
            'bsz': bsz,  # 批次大小
            'sample_size': sample_size,  # 样本大小
            'sample_ratio': sample_size / n_tokens,  # 样本比例
            'nonpad_ratio': n_nonpad_tokens / n_tokens  # 非填充比例
        }
        
        return loss_avg, logging_output


# 定义坐标到序列的交叉熵损失类，用于结构到序列预测任务
class Coord2SeqCrossEntropyLoss(nn.CrossEntropyLoss):
    """坐标到序列的交叉熵损失，专为逆蛋白质折叠设计
    
    针对从蛋白质结构坐标预测氨基酸序列的任务，支持坐标掩码和标签掩码。
    """
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, coord_mask=None, weights=None) -> Tensor:
        """前向传播计算损失
        
        Args:
            scores: 未规范化的分数，[N, L, C]，批次大小×序列长度×类别数
            target: 目标索引，[N, L]
            label_mask: 标签掩码，[N, L]，标识有效标签位置
            coord_mask: 坐标掩码，[N, L]，标识有效坐标位置
            weights: 权重张量，[N, L]，为不同位置分配不同权重
            
        Returns:
            loss: 损失值
            logging_output: 包含损失相关信息的字典
        """
        # 如果未提供标签掩码，使用坐标掩码
        if label_mask is None:
            label_mask = coord_mask
            
        # 获取批次大小和类别数
        bsz, num_classes = scores.shape[0], scores.shape[-1]
        
        # 计算总标记数和非填充标记数
        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens
            
        # 计算带标签平滑的损失和原始nll损失，不对批次求和
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),  # 将分数转为对数概率
            target=target,
            epsilon=self.label_smoothing,  # 标签平滑因子
            ignore_index=self.ignore_index,
            reduce=False,  # 不对批次求和，保留每个位置的损失
        )
        
        # 如果提供了权重，按权重缩放损失
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights
            
        # 计算完整序列的平均损失
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size
        
        # 如果提供了标签掩码，只考虑掩码内的位置
        if label_mask is not None:
            label_mask = label_mask.float()  # 转为浮点型便于计算
            sample_size = label_mask.sum()  # 更新样本大小为有效坐标数
            
            # 计算掩码内的平均损失和准确率
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
            acc = ((scores.argmax(-1) == target) * label_mask).sum() / sample_size
        else:
            # 如果没有掩码，使用完整序列的损失
            loss, nll_loss = fullseq_loss, fullseq_nll_loss
            
        # 计算困惑度
        ppl = torch.exp(nll_loss)
        
        # 整理日志信息
        logging_output = {
            'nll_loss': nll_loss.data,  # nll损失
            'ppl': ppl.data,  # 困惑度
            'acc': acc.data,  # 准确率
            'fullseq_loss': fullseq_loss.data,  # 完整序列损失
            'fullseq_nll_loss': fullseq_nll_loss.data,  # 完整序列nll损失
            'bsz': bsz,  # 批次大小
            'sample_size': sample_size,  # 样本大小
            'sample_ratio': sample_size / n_tokens,  # 样本比例
            'nonpad_ratio': n_nonpad_tokens / n_tokens  # 非填充比例
        }
        
        return loss, logging_output