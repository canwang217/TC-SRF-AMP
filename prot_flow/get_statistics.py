import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# ⚠️ 这里改为导入你的精装房 config
from tc_srf_amp_config import create_tc_srf_amp_smoke_config, create_tc_srf_amp_config
from encoders import ESM2EncoderModel
from utils import load_fasta_file


def get_loader(config,  batch_size):
    train_dataset = load_fasta_file(config.data.train_dataset_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20, # 建议本地跑把 num_workers 调小一点，防止内存爆炸
        pin_memory=True,
    )
    return train_loader


def compute_statistics( # 改个名字，表明我们算的是所有统计量
        config,
        encoder,
        model_name, 
        dataset_name,
):
    sum_ = None
    sqr_sum_ = None
    max_ = None  # 🌟 新增：全局最大值
    min_ = None  # 🌟 新增：全局最小值
    
    num = 0
    batch_size = 64

    train_loader = get_loader(
        config=config,
        batch_size=batch_size
    )
    T = tqdm(train_loader)

    for i, X in enumerate(T):
        with torch.no_grad():
            output, _ = encoder.batch_encode(X)

        # 当前 Batch 的统计量
        cur_sum = torch.sum(output, dim=[0, 1])
        cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])
        cur_num = output.shape[0] * output.shape[1]
        
        # 🌟 新增：获取当前 Batch 的最大值和最小值 (沿着 batch 和 seq_len 维度)
        cur_max = torch.amax(output, dim=[0, 1])
        cur_min = torch.amin(output, dim=[0, 1])

        # 更新全局统计量
        sum_ = cur_sum if sum_ is None else cur_sum + sum_
        sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
        
        # 🌟 新增：持续更新全局最大和最小值
        max_ = cur_max if max_ is None else torch.maximum(max_, cur_max)
        min_ = cur_min if min_ is None else torch.minimum(min_, cur_min)
        
        num += cur_num

        mean = sum_[:3] / num
        std = torch.sqrt(sqr_sum_[:3] / num - mean ** 2)
        T.set_description(f"Processing... mean_0: {mean[0].item():.4f}")

    # 计算最终的均值和标准差
    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    # 提取 config 里定义好的路径文件夹
    folder_path = os.path.dirname(config.data.enc_mean)
    os.makedirs(folder_path, exist_ok=True)
    
    # 🌟 统一保存 4 个文件
    torch.save(mean, config.data.enc_mean)
    torch.save(std, config.data.enc_std)
    torch.save(max_, config.data.enc_max)
    torch.save(min_, config.data.enc_min)
    
    print(f"\n✅ 成功！所有统计文件(Mean, Std, Max, Min)已保存在: {folder_path}")


if __name__ == "__main__":
    # ⚠️ 关键修改：这里根据你要跑的任务选择 config
    # 如果你在做冒烟测试，就用 create_tc_srf_amp_smoke_config()
    # 如果你准备跑那 200 万条数据预训练，就用 create_tc_srf_amp_config()
    config = create_tc_srf_amp_config() 
    
    print(f"正在为数据集 [{config.data.dataset}] 提取统计信息...")
    
    encoder = ESM2EncoderModel(
        config.model.hg_name, 
        device="cuda:0", 
        decoder_path=None, 
        max_seq_len=config.data.max_sequence_len,
        enc_normalizer=None,
    )

    compute_statistics(
        config,
        encoder,
        model_name=config.model.hg_name_hash,
        dataset_name=config.data.dataset
    )