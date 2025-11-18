import pandas as pd
import numpy as np
import os
from config import Config

class DataAnalyzer:
    """数据处理器，负责清洗、转换和分析原始采集数据"""
    
    @staticmethod
    def load_raw_data(filename: str) -> pd.DataFrame:
        """加载原始数据CSV文件"""
        data_path = os.path.join(Config.RAW_DATA_DIR, filename)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"原始数据文件不存在: {data_path}")
        return pd.read_csv(data_path)
    
    @staticmethod
    def clean_data(raw_data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        清洗数据：处理异常值、修正不合理值
        scenario: "recompute" 或 "split"，区分场景
        """
        cleaned = raw_data.copy()
        
        # 1. 处理延迟增加率（不应为负，超过1的极端值截断）
        cleaned["latency_increase_rate"] = cleaned["latency_increase_rate"].apply(
            lambda x: max(0.0, min(1.0, x))  # 限制在 [0, 1] 范围内
        )
        
        # 2. 处理内存节省率（不应为负，超过1的极端值截断）
        cleaned["memory_saving_rate"] = cleaned["memory_saving_rate"].apply(
            lambda x: max(0.0, min(1.0, x))
        )
        
        # 3. 针对不同场景的额外清洗
        if scenario == "recompute":
            # 重计算场景：过滤无效图层名
            valid_layers = Config.RECOMPUTE_LAYERS
            cleaned = cleaned[cleaned["layer"].isin(valid_layers)]
        elif scenario == "split":
            # 算子分裂场景：过滤无效分裂比例（应在0-1之间）
            cleaned = cleaned[(cleaned["split_ratio"] >= 0) & (cleaned["split_ratio"] <= 1)]
        
        # 4. 去除重复数据
        cleaned = cleaned.drop_duplicates()
        
        return cleaned
    
    @staticmethod
    def calculate_derived_metrics(cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """计算衍生指标，辅助分析权衡关系"""
        data = cleaned_data.copy()
        
        # 1. 计算"性价比"：内存节省率 / (延迟增加率 + 1e-6)（避免除零）
        # 数值越高，说明单位延迟增加带来的内存节省越划算
        data["cost_effectiveness"] = data["memory_saving_rate"] / (data["latency_increase_rate"] + 1e-6)
        
        # 2. 标记是否为"高效策略"（内存节省高且延迟增加低）
        # 阈值可根据实际场景调整
        data["is_efficient"] = (data["memory_saving_rate"] > 0.6) & (data["latency_increase_rate"] < 0.3)
        
        # 3. 标记是否为"低效策略"（内存节省低且延迟增加高）
        data["is_inefficient"] = (data["memory_saving_rate"] < 0.4) & (data["latency_increase_rate"] > 0.5)
        
        return data
    
    @staticmethod
    def aggregate_data(processed_data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """按场景聚合数据，计算统计量（均值、标准差等）"""
        if scenario == "recompute":
            # 重计算场景：按图层聚合
            agg = processed_data.groupby("layer").agg({
                "latency_increase_rate": ["mean", "std", "count"],
                "memory_saving_rate": ["mean", "std"],
                "cost_effectiveness": ["mean"]
            }).reset_index()
            # 重命名列名（扁平化MultiIndex）
            agg.columns = ["_".join(col).strip() for col in agg.columns.values]
            agg = agg.rename(columns={"layer_": "layer"})
            return agg
        elif scenario == "split":
            # 算子分裂场景：按分裂比例聚合
            agg = processed_data.groupby("split_ratio").agg({
                "latency_increase_rate": ["mean", "std", "count"],
                "memory_saving_rate": ["mean", "std"],
                "cost_effectiveness": ["mean"]
            }).reset_index()
            agg.columns = ["_".join(col).strip() for col in agg.columns.values]
            agg = agg.rename(columns={"split_ratio_": "split_ratio"})
            return agg
        else:
            raise ValueError(f"不支持的场景: {scenario}")
    
    @staticmethod
    def save_processed_data(processed_data: pd.DataFrame, filename: str) -> None:
        """保存处理后的数据"""
        save_path = os.path.join(Config.RAW_DATA_DIR, f"processed_{filename}")
        processed_data.to_csv(save_path, index=False)
        print(f"处理后的数据已保存至: {save_path}")
    
    def process_recompute_data(self) -> pd.DataFrame:
        """处理重计算场景数据的完整流程"""
        # 1. 加载原始数据
        raw_data = self.load_raw_data("recompute_data.csv")
        # 2. 清洗数据
        cleaned_data = self.clean_data(raw_data, scenario="recompute")
        # 3. 计算衍生指标
        processed_data = self.calculate_derived_metrics(cleaned_data)
        # 4. 聚合统计
        aggregated_data = self.aggregate_data(processed_data, scenario="recompute")
        # 5. 保存结果
        self.save_processed_data(processed_data, "recompute_data.csv")
        self.save_processed_data(aggregated_data, "recompute_aggregated.csv")
        return processed_data
    
    def process_split_data(self) -> pd.DataFrame:
        """处理算子分裂场景数据的完整流程"""
        raw_data = self.load_raw_data("operator_splitting_data.csv")
        cleaned_data = self.clean_data(raw_data, scenario="split")
        processed_data = self.calculate_derived_metrics(cleaned_data)
        aggregated_data = self.aggregate_data(processed_data, scenario="split")
        self.save_processed_data(processed_data, "operator_splitting_data.csv")
        self.save_processed_data(aggregated_data, "operator_splitting_aggregated.csv")
        return processed_data