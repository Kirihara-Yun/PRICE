from dhg_ir import DHG_IR, Operator
from ci_calculator import CICalculator
from typing import Dict, List, Tuple, Optional, Any
import sympy as sp
class OptimizationDecision:
    """优化决策基类"""
    def __init__(self, dhg_ir: DHG_IR, ci_calculator: CICalculator):
        self.dhg_ir = dhg_ir
        self.ci_calculator = ci_calculator
        self.ci_calculator.compile_time_modeling()


class RecomputationDecision(OptimizationDecision):
    """重计算决策算法"""
    def __init__(self, dhg_ir: DHG_IR, ci_calculator: CICalculator, 
                 init_threshold: float = 2.0):
        super().__init__(dhg_ir, ci_calculator)
        self.init_threshold = init_threshold
        self.recomputation_candidates = []  # 重计算候选集
        
    def compile_time_preprocessing(self) -> None:
        """编译时预处理，筛选重计算候选集"""
        self.recomputation_candidates = []
        
        for target in self.ci_calculator.symbolic_flops.keys():
            # 获取CI符号化表达式
            flops_expr = self.ci_calculator.symbolic_flops[target].expr
            mem_expr = self.ci_calculator.symbolic_mem[target].expr
            ci_expr = flops_expr / mem_expr
            
            # 分析CI的最大值（基于DVar的最大值）
            max_vals = {name: dvar.max_val for name, dvar in self.dhg_ir.dvars.items()}
            substitutions = {sp.Symbol(name): val for name, val in max_vals.items()}
            
            try:
                max_ci = float(ci_expr.subs(substitutions))
                if max_ci < self.init_threshold:
                    self.recomputation_candidates.append(target)
            except:
                # 处理无法计算的情况
                continue
                
    def runtime_decision(self, target: str) -> bool:
        """运行时精准决策"""
        if target not in self.recomputation_candidates:
            return False
            
        # 计算实际计算密集度
        _, _, actual_ci = self.ci_calculator.runtime_calculation(target)
        
        # 动态调整阈值
        # 确定存储层级（这里简化处理，假设target是张量）
        if target in self.dhg_ir.tensors:
            level = self.dhg_ir.mem_hierarchy.get_actual_level(target)
        else:
            # 对于算子，使用其输出张量的存储层级
            op = self.dhg_ir.operators.get(target)
            if op and op.output_tensors:
                level = self.dhg_ir.mem_hierarchy.get_actual_level(op.output_tensors[0])
            else:
                level = "DRAM"
                
        # 根据存储层级调整阈值
        if level == "DRAM":
            threshold = self.init_threshold * 0.5
        elif level in ["L1", "L2"]:
            threshold = self.init_threshold * 2
        else:
            threshold = self.init_threshold
            
        # 决策
        return actual_ci < threshold


class OperatorSplittingDecision(OptimizationDecision):
    """算子分裂决策算法"""
    def __init__(self, dhg_ir: DHG_IR, ci_calculator: CICalculator,
                 split_threshold: float = 3.0, 
                 potential_granularities: List[int] = [2, 4, 8, 16]):
        super().__init__(dhg_ir, ci_calculator)
        self.split_threshold = split_threshold
        self.potential_granularities = potential_granularities
        self.split_candidates = {}  # 分裂候选集，{算子: 候选粒度}
        
    def compile_time_preprocessing(self) -> None:
        """编译时预处理，筛选分裂候选与预划粒度"""
        self.split_candidates = {}
        
        for op_name in self.dhg_ir.operators.keys():
            # 获取原算子的CI符号化表达式
            flops_expr = self.ci_calculator.symbolic_flops[op_name].expr
            mem_expr = self.ci_calculator.symbolic_mem[op_name].expr
            ci_expr = flops_expr / mem_expr
            
            # 分析原算子CI的最大值
            max_vals = {name: dvar.max_val for name, dvar in self.dhg_ir.dvars.items()}
            substitutions = {sp.Symbol(name): val for name, val in max_vals.items()}
            
            try:
                max_ci = float(ci_expr.subs(substitutions))
                if max_ci >= self.split_threshold:
                    continue  # 计算密集度高，不纳入候选
            except:
                continue  # 处理无法计算的情况
                
            # 预划粒度范围
            candidate_granularities = []
            for k in self.potential_granularities:
                # 简化处理：假设分裂后CI表达式与原表达式相似，但内存开销减少
                # 实际实现中应生成分裂后小算子的CI表达式
                min_vals = {name: dvar.min_val for name, dvar in self.dhg_ir.dvars.items()}
                min_substitutions = {sp.Symbol(name): val for name, val in min_vals.items()}
                
                try:
                    # 分裂后CI应保持较低水平
                    min_ci = float(ci_expr.subs(min_substitutions))
                    if min_ci < self.split_threshold:
                        candidate_granularities.append(k)
                except:
                    continue
                    
            if candidate_granularities:
                self.split_candidates[op_name] = candidate_granularities
                
    def _get_hardware_overhead(self, granularity: int) -> float:
        """获取硬件开销模型，输入粒度，输出新增硬件开销占比"""
        # 简化模型：粒度越大，硬件开销（如内核启动）越小
        # 实际实现中应基于硬件实测数据
        base_overhead = 0.2  # 基础开销
        return base_overhead / (granularity **0.5)
        
    def runtime_decision(self, op_name: str) -> Tuple[bool, Optional[int]]:
        """运行时精准决策，返回是否分裂及最优粒度"""
        if op_name not in self.split_candidates:
            return False, None
            
        # 计算原算子的实际计算密集度
        _, original_mem, actual_ci = self.ci_calculator.runtime_calculation(op_name)
        
        if actual_ci >= self.split_threshold:
            return False, None  # 临时取消分裂
            
        # 获取候选粒度
        candidate_granularities = self.split_candidates[op_name]
        best_granularity = None
        max_total_gain = -float('inf')
        
        # 评估每个候选粒度
        for k in candidate_granularities:
            # 简化处理：计算分裂后的内存收益和硬件开销
            # 实际实现中应计算分裂后小算子的CI和内存开销
            split_mem = original_mem / (k** 0.8)  # 假设内存随粒度增加而减少
            mem_gain = (original_mem - split_mem) / original_mem
            
            # 获取硬件开销
            hw_cost = self._get_hardware_overhead(k)
            
            # 计算总收益
            total_gain = mem_gain - hw_cost
            
            if total_gain > max_total_gain and total_gain > 0:
                max_total_gain = total_gain
                best_granularity = k
                
        return (best_granularity is not None), best_granularity