import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import RealFakeDataset
from .FFPP import FFPPRealFakeDataset

    

def get_bal_sampler(dataset):
    """
    根据数据集中各类样本的数量生成一个权重随机采样器，以平衡各类样本在训练中的权重。
    
    参数:
    dataset: Dataset对象，包含了要训练的子数据集。

    返回:
    WeightedRandomSampler对象，用于在数据加载时根据样本权重进行随机采样。
    """
    # 初始化目标标签列表
    targets = []
    # 遍历传入的Dataset对象中的每个子数据集
    for d in dataset.datasets:
        # 将每个子数据集的标签添加到目标标签列表中
        targets.extend(d.targets)

    # 计算每个类别的频数
    ratio = np.bincount(targets)
    # 对每个类别的频数取倒数，作为初步的权重
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    # 根据初步权重和目标标签生成最终的样本权重
    sample_weights = w[targets]
    # 创建并返回权重随机采样器
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt, preprocess=None):
    """
    创建一个数据加载器，根据选项配置数据集和采样器。

    参数:
    - opt: 包含配置选项的对象，如训练标志、批次大小、数据集类型等。
    - preprocess: 可选的预处理函数，用于数据转换。

    返回:
    - data_loader: 配置好的数据加载器实例。
    """
    # 根据训练模式和类别平衡选项决定是否打乱数据顺序
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    # 初始化数据集实例
    dataset = RealFakeDataset(opt)
    # 如果架构包含'2b'，则应用预处理函数
    if '2b' in opt.arch:
        dataset.transform = preprocess
    # 根据类别平衡选项，可能初始化一个平衡采样器
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    # 初始化数据加载器，配置数据集、批次大小、是否打乱顺序、采样器和工作线程数
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
