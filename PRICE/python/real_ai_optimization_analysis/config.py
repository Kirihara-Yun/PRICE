import os

# 基础配置
class Config:
    # 保存路径
    SAVE_ROOT = "./results"
    RAW_DATA_DIR = os.path.join(SAVE_ROOT, "raw_data")
    PLOT_DIR = os.path.join(SAVE_ROOT, "plots")
    
    # 实验参数
    DEVICE = "cuda" if os.cuda.is_available() else "cpu"
    WARMUP_ITER = 5  # 热身迭代（排除初始化开销）
    MEASURE_ITER = 20  # 测量迭代次数（取平均降低误差）
    BATCH_SIZE = 32
    INPUT_SHAPE = (3, 224, 224)  # 输入数据形状（可根据模型调整）
    
    # 重计算配置
    RECOMPUTE_LAYERS = ["layer1", "layer2"]  # 测试重计算的层名
    # 算子分裂配置
    SPLIT_RATIOS = [0.2, 0.4, 0.6, 0.8]  # 算子分裂比例（如按通道拆分）

# 创建保存目录
for dir_path in [Config.RAW_DATA_DIR, Config.PLOT_DIR]:
    os.makedirs(dir_path, exist_ok=True)