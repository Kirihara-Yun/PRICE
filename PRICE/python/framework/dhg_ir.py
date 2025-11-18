import sympy as sp
from typing import Dict, List, Tuple, Callable, Optional, Any

class DVar:
    """动态维度变量，支持维度约束和依赖关系"""
    def __init__(self, name: str, min_val: int, max_val: int, 
                 dependency: Optional[Callable] = None):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.dependency = dependency  # 维度依赖关系，如 lambda x: x // 2
        self.current_value = None  # 运行时实际值
        
    def resolve(self, input_values: Dict[str, int]) -> int:
        """根据输入值解析当前维度值"""
        if self.dependency:
            self.current_value = self.dependency(input_values)
        return self.current_value
    
    def __repr__(self) -> str:
        return f"DVar({self.name}, [{self.min_val}, {self.max_val}])"


class RuntimeHook:
    """运行时钩子，用于采集动态参数"""
    def __init__(self):
        self.parameters = {}  # 存储采集到的动态参数
        
    def collect(self, param_name: str, value: Any) -> None:
        """采集参数值"""
        self.parameters[param_name] = value
        
    def get(self, param_name: str, default: Any = None) -> Any:
        """获取参数值"""
        return self.parameters.get(param_name, default)


class CacheReuseTag:
    """缓存复用标签，标注张量的缓存驻留级别和访问局部性"""
    def __init__(self, level: str, reuse_count: int, 
                 spatial_loc: bool = False, temporal_loc: bool = False):
        self.level = level  # L1, L2, DRAM等
        self.reuse_count = reuse_count  # 复用次数
        self.spatial_loc = spatial_loc  # 空间局部性
        self.temporal_loc = temporal_loc  # 时间局部性
        self.actual_reuse = None  # 运行时实际复用次数
        
    def __repr__(self) -> str:
        return f"CacheReuseTag({self.level}, reuse={self.reuse_count})"


class DataFormatMeta:
    """数据格式元信息，包含稀疏度、精度等信息"""
    def __init__(self, dtype: str = "FP32", sparse_format: Optional[str] = None,
                 sparse_rate: float = 1.0, quant_info: Optional[Dict] = None):
        self.dtype = dtype  # 数据类型
        self.sparse_format = sparse_format  # 稀疏格式，如CSR
        self.sparse_rate = sparse_rate  # 稀疏率，1.0表示稠密
        self.quant_info = quant_info  # 量化信息
        self.actual_sparse_rate = None  # 运行时实际稀疏率
        
    def get_bit_width(self) -> int:
        """获取数据类型的位宽"""
        bit_map = {"FP32": 32, "FP16": 16, "INT8": 8, "INT16": 16}
        return bit_map.get(self.dtype, 32)
        
    def __repr__(self) -> str:
        return f"DataFormatMeta({self.dtype}, sparse={self.sparse_format})"


class MemHierMap:
    """硬件存储层级映射，关联张量访问与硬件存储层级"""
    def __init__(self):
        self.hierarchy = {
            "DRAM": {"cost_coeff": 100.0, "latency": 100},
            "L3": {"cost_coeff": 30.0, "latency": 30},
            "L2": {"cost_coeff": 10.0, "latency": 10},
            "L1": {"cost_coeff": 1.0, "latency": 1},
            "REG": {"cost_coeff": 0.1, "latency": 0.1}
        }
        self.tensor_mapping = {}  # 张量到存储层级的映射
        
    def set_tensor_mapping(self, tensor_name: str, level: str) -> None:
        """设置张量的存储层级"""
        if level in self.hierarchy:
            self.tensor_mapping[tensor_name] = level
            
    def get_cost_coeff(self, tensor_name: str) -> float:
        """获取张量访问的代价系数"""
        level = self.tensor_mapping.get(tensor_name, "DRAM")
        return self.hierarchy[level]["cost_coeff"]
        
    def get_actual_level(self, tensor_name: str) -> str:
        """获取运行时实际存储层级"""
        # 在实际实现中，这可以通过硬件监控获取
        return self.tensor_mapping.get(tensor_name, "DRAM")


class Operator:
    """算子类，包含计算FLOPs的函数和相关元信息"""
    def __init__(self, name: str, op_type: str, 
                 flops_func: Callable[..., float],
                 input_vars: List[DVar],
                 data_format: DataFormatMeta = DataFormatMeta()):
        self.name = name
        self.op_type = op_type
        self.flops_func = flops_func  # 计算FLOPs的函数
        self.input_vars = input_vars  # 输入动态维度变量
        self.data_format = data_format
        self.cache_tag = CacheReuseTag("DRAM", 1)  # 默认缓存标签
        self.output_tensors = []  # 输出张量名称
        
    def set_cache_tag(self, cache_tag: CacheReuseTag) -> None:
        """设置缓存标签"""
        self.cache_tag = cache_tag
        
    def set_output_tensors(self, tensors: List[str]) -> None:
        """设置输出张量"""
        self.output_tensors = tensors
        
    def __repr__(self) -> str:
        return f"Operator({self.name}, {self.op_type})"


class DHG_IR:
    """动态硬件感知图IR(DHG-IR)的顶层容器"""
    def __init__(self):
        self.dvars = {}  # 动态维度变量集合
        self.operators = {}  # 算子集合
        self.tensors = {}  # 张量集合，存储张量元信息
        self.mem_hierarchy = MemHierMap()  # 内存层级映射
        self.runtime_hook = RuntimeHook()  # 运行时钩子
        self.graph = []  # 计算图，存储算子执行顺序和依赖关系
        
    def add_dvar(self, dvar: DVar) -> None:
        """添加动态维度变量"""
        self.dvars[dvar.name] = dvar
        
    def add_operator(self, op: Operator) -> None:
        """添加算子"""
        self.operators[op.name] = op
        
    def add_tensor(self, tensor_name: str, data_format: DataFormatMeta,
                  cache_tag: CacheReuseTag) -> None:
        """添加张量"""
        self.tensors[tensor_name] = {
            "data_format": data_format,
            "cache_tag": cache_tag
        }
        # 默认映射到DRAM
        self.mem_hierarchy.set_tensor_mapping(tensor_name, cache_tag.level)
        
    def add_edge(self, src: str, dst: str) -> None:
        """添加计算图中的边，表示依赖关系"""
        self.graph.append((src, dst))
        
    def get_dvar_values(self) -> Dict[str, int]:
        """获取所有动态维度变量的当前值"""
        return {name: dvar.current_value for name, dvar in self.dvars.items() 
                if dvar.current_value is not None}
                
    def resolve_dvars(self, input_values: Dict[str, int]) -> None:
        """解析所有动态维度变量的值"""
        for dvar in self.dvars.values():
            dvar.resolve(input_values)