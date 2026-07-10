import pandas as pd
from config import Config
from utils.torch_utils import measure_time, measure_memory_usage
import torch
import os
class BaseCollector:
    def __init__(self, model, data_loader):
        self.model = model.to(Config.DEVICE)
        self.data_loader = data_loader
        self.results = []  # 存储 (latency_increase, memory_saving) 等数据

    def _get_baseline(self):
        """测量基准性能（无优化时的延迟和内存）"""
        # 定义前向+反向传播函数（训练场景）
        def forward_backward(data, target):
            self.model.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

        # 取一个batch的数据
        data, target = next(iter(self.data_loader))
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        
        # 测量基准延迟和内存
        base_latency = measure_time(forward_backward, data, target)
        base_memory = measure_memory_usage(forward_backward, data, target)
        return base_latency, base_memory

    def save_results(self, filename):
        """保存结果到CSV"""
        df = pd.DataFrame(self.results)
        save_path = os.path.join(Config.RAW_DATA_DIR, filename)
        df.to_csv(save_path, index=False)
        print(f"数据已保存至: {save_path}")