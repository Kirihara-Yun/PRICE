import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from config import Config

# 字体设置（支持中文）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

class Plotter:
    @staticmethod
    def plot_recompute_tradeoff():
        """绘制重计算的内存-延迟权衡散点图"""
        data_path = os.path.join(Config.RAW_DATA_DIR, "recompute_data.csv")
        if not os.path.exists(data_path):
            print(f"未找到重计算数据: {data_path}")
            return
        
        data = pd.read_csv(data_path)
        latency = data["latency_increase_rate"]
        memory = data["memory_saving_rate"]
        layers = data["layer"]
        
        plt.figure(figsize=(10, 6))
        # 目标区域（内存节省少+延迟增加大）
        focus_mask = (memory < 0.4) & (latency > 0.6)
        # 散点图
        plt.scatter(latency[~focus_mask], memory[~focus_mask], 
                   color="#4682B4", alpha=0.7, label="普通区域")
        plt.scatter(latency[focus_mask], memory[focus_mask], 
                   color="#E63946", alpha=0.9, edgecolors="black", label="需优化区域")
        
        # 添加标签
        for i, layer in enumerate(layers):
            plt.annotate(layer, (latency[i], memory[i]), fontsize=8, alpha=0.7)
        
        plt.xlabel("延迟增加率", fontsize=12)
        plt.ylabel("内存节省率", fontsize=12)
        plt.title("重计算策略的内存节省与延迟增加权衡", fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(Config.PLOT_DIR, "recompute_tradeoff.png")
        plt.savefig(save_path, dpi=300)
        print(f"重计算可视化结果已保存至: {save_path}")

    @staticmethod
    def plot_split_tradeoff():
        """绘制算子分裂的内存-延迟权衡散点图"""
        data_path = os.path.join(Config.RAW_DATA_DIR, "operator_splitting_data.csv")
        if not os.path.exists(data_path):
            print(f"未找到算子分裂数据: {data_path}")
            return
        
        data = pd.read_csv(data_path)
        latency = data["latency_increase_rate"]
        memory = data["memory_saving_rate"]
        ratios = data["split_ratio"]
        
        plt.figure(figsize=(10, 6))
        focus_mask = (memory < 0.4) & (latency > 0.5)
        plt.scatter(latency[~focus_mask], memory[~focus_mask], 
                   color="#4682B4", alpha=0.7, label="普通区域")
        plt.scatter(latency[focus_mask], memory[focus_mask], 
                   color="#E63946", alpha=0.9, edgecolors="black", label="需优化区域")
        
        # 添加分裂比例标签
        for i, ratio in enumerate(ratios):
            plt.annotate(f"r={ratio}", (latency[i], memory[i]), fontsize=8, alpha=0.7)
        
        plt.xlabel("延迟增加率", fontsize=12)
        plt.ylabel("内存节省率", fontsize=12)
        plt.title("算子分裂策略的内存节省与延迟增加权衡", fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(Config.PLOT_DIR, "operator_splitting_tradeoff.png")
        plt.savefig(save_path, dpi=300)
        print(f"算子分裂可视化结果已保存至: {save_path}")