import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from data_collector.recompute_collector import RecomputeCollector
from data_collector.split_collector import SplitCollector
from visualizer.plotter import Plotter
from models.simple_cnn import SimpleCNN

def generate_dummy_data():
    """生成测试用随机数据"""
    data = torch.randn(Config.BATCH_SIZE * 10, *Config.INPUT_SHAPE)  # 10个batch
    target = torch.randint(0, 10, (Config.BATCH_SIZE * 10,))
    dataset = TensorDataset(data, target)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

def main():
    # 初始化模型和数据
    model = SimpleCNN()
    data_loader = generate_dummy_data()
    
    # 采集重计算数据
    print("开始采集重计算数据...")
    recompute_collector = RecomputeCollector(model, data_loader)
    recompute_collector.collect()
    
    # 采集算子分裂数据
    print("\n开始采集算子分裂数据...")
    split_collector = SplitCollector(model, data_loader)
    split_collector.collect()
    
    # 可视化结果
    print("\n生成可视化结果...")
    Plotter.plot_recompute_tradeoff()
    Plotter.plot_split_tradeoff()
    print("所有实验完成！")

if __name__ == "__main__":
    main()