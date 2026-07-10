import torch
from .base_collector import BaseCollector
from config import Config
from utils.torch_utils import measure_time, measure_memory_usage
class SplitCollector(BaseCollector):
    def __init__(self, model, data_loader):
        super().__init__(model, data_loader)
        self.base_latency, self.base_memory = self._get_baseline()
        self.original_model = model  # 保存原始模型用于恢复

    def split_operator(self, split_ratio):
        """按比例分裂模型中的大算子（以卷积层为例）"""
        # 遍历模型，分裂第一个卷积层（可扩展为指定层）
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # 按输出通道分裂卷积层
                out_channels = module.out_channels
                split_channels = int(out_channels * split_ratio)
                # 创建两个子卷积层
                self.model.split_conv1 = torch.nn.Conv2d(
                    module.in_channels, split_channels, 
                    kernel_size=module.kernel_size, 
                    stride=module.stride,
                    padding=module.padding
                ).to(Config.DEVICE)
                self.model.split_conv2 = torch.nn.Conv2d(
                    module.in_channels, out_channels - split_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                ).to(Config.DEVICE)
                # 替换原始forward方法
                def new_forward(x):
                    x1 = self.model.split_conv1(x)
                    x2 = self.model.split_conv2(x)
                    return torch.cat([x1, x2], dim=1)
                module.forward = new_forward
                break  # 仅分裂第一个卷积层（可扩展）

    def collect(self):
        """采集不同分裂比例下的性能数据"""
        for ratio in Config.SPLIT_RATIOS:
            # 恢复原始模型并应用分裂
            self.model = self.original_model.to(Config.DEVICE)
            self.split_operator(ratio)
            
            # 定义测量函数
            def forward_backward(data, target):
                self.model.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

            data, target = next(iter(self.data_loader))
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            
            # 测量分裂后的延迟和内存
            split_latency = measure_time(forward_backward, data, target)
            split_memory = measure_memory_usage(forward_backward, data, target)
            
            # 计算增长率和节省率
            latency_increase = (split_latency - self.base_latency) / self.base_latency
            memory_saving = (self.base_memory - split_memory) / self.base_memory
            
            # 存储结果
            self.results.append({
                "split_ratio": ratio,
                "latency_increase_rate": max(0, min(1, latency_increase)),
                "memory_saving_rate": max(0, min(1, memory_saving))
            })
            print(f"分裂比例 {ratio} 完成: 延迟增加 {latency_increase:.2%}, 内存节省 {memory_saving:.2%}")

        # 保存结果
        self.save_results("operator_splitting_data.csv")