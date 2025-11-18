import torch
from .base_collector import BaseCollector
from config import Config
from utils.torch_utils import measure_time, measure_memory_usage
class RecomputeCollector(BaseCollector):
    def __init__(self, model, data_loader):
        super().__init__(model, data_loader)
        self.base_latency, self.base_memory = self._get_baseline()

    def enable_recompute(self, layer_name):
        """为指定层启用重计算（PyTorch的checkpoint机制）"""
        # 遍历模型子模块，对目标层应用checkpoint
        for name, module in self.model.named_modules():
            if layer_name in name:
                # 替换为带重计算的包装器
                module.forward = torch.utils.checkpoint.checkpoint(module.forward)

    def collect(self):
        """采集不同层重计算时的性能数据"""
        for layer in Config.RECOMPUTE_LAYERS:
            # 启用当前层重计算
            self.enable_recompute(layer)
            
            # 定义测量函数
            def forward_backward(data, target):
                self.model.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

            data, target = next(iter(self.data_loader))
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            
            # 测量重计算时的延迟和内存
            recompute_latency = measure_time(forward_backward, data, target)
            recompute_memory = measure_memory_usage(forward_backward, data, target)
            
            # 计算增长率和节省率
            latency_increase = (recompute_latency - self.base_latency) / self.base_latency
            memory_saving = (self.base_memory - recompute_memory) / self.base_memory
            
            # 存储结果（确保值在合理范围）
            self.results.append({
                "layer": layer,
                "latency_increase_rate": max(0, min(1, latency_increase)),  # 限制在[0,1]
                "memory_saving_rate": max(0, min(1, memory_saving))
            })
            print(f"重计算 {layer} 完成: 延迟增加 {latency_increase:.2%}, 内存节省 {memory_saving:.2%}")

        # 保存结果
        self.save_results("recompute_data.csv")