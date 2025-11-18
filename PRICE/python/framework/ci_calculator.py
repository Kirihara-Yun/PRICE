import sympy as sp
from dhg_ir import DHG_IR, Operator, DVar
from typing import Dict, Tuple, List, Any, Callable

class SymbolicExpression:
    """符号化表达式类，用于编译时符号化建模"""
    def __init__(self, expr: sp.Expr, variables: List[str]):
        self.expr = expr  # 符号表达式
        self.variables = variables  # 变量列表
        
    def substitute(self, values: Dict[str, Any]) -> float:
        """代入变量值计算表达式结果"""
        substitutions = {var: values[var] for var in self.variables if var in values}
        result = self.expr.subs(substitutions)
        return float(result) if result.is_number else result
        
    def __repr__(self) -> str:
        return f"SymbolicExpression({self.expr})"


class CICalculator:
    """计算密集度计算器，实现编译时符号化建模和运行时实例化计算"""
    def __init__(self, dhg_ir: DHG_IR):
        self.dhg_ir = dhg_ir
        self.symbolic_flops = {}  # 算子/张量的FLOPs符号化表达式
        self.symbolic_mem = {}    # 算子/张量的内存开销符号化表达式
        
    def _create_dvar_symbols(self) -> Dict[str, sp.Symbol]:
        """为所有动态维度变量创建符号"""
        return {name: sp.Symbol(name) for name in self.dhg_ir.dvars.keys()}
        
    def compile_time_modeling(self) -> None:
        """编译时符号化建模，生成FLOPs和内存开销的符号化表达式"""
        # 创建动态维度变量的符号
        dvar_symbols = self._create_dvar_symbols()
        
        # 为每个算子生成符号化表达式
        for op_name, op in self.dhg_ir.operators.items():
            # 生成FLOPs符号化表达式
            flops_expr = op.flops_func(dvar_symbols, op.data_format)
            self.symbolic_flops[op_name] = SymbolicExpression(
                flops_expr, list(dvar_symbols.keys())
            )
            
            # 生成内存开销符号化表达式
            # 1. 计算数据量表达式
            # 这里简化处理，实际应根据算子类型计算输出数据量
            data_size_expr = self._get_data_size_expr(op, dvar_symbols)
            
            # 2. 应用缓存复用
            if op.cache_tag.reuse_count > 0:
                data_size_expr = data_size_expr / op.cache_tag.reuse_count
                
            # 3. 绑定硬件代价权重
            mem_expr = 0
            for tensor in op.output_tensors:
                if tensor in self.dhg_ir.tensors:
                    cost_coeff = self.dhg_ir.mem_hierarchy.get_cost_coeff(tensor)
                    mem_expr += data_size_expr * cost_coeff
            
            self.symbolic_mem[op_name] = SymbolicExpression(
                mem_expr, list(dvar_symbols.keys())
            )
            
            # 为输出张量生成表达式（与算子相同）
            for tensor in op.output_tensors:
                self.symbolic_flops[tensor] = self.symbolic_flops[op_name]
                self.symbolic_mem[tensor] = self.symbolic_mem[op_name]
    
    def _get_data_size_expr(self, op: Operator, dvar_symbols: Dict[str, sp.Symbol]) -> sp.Expr:
        """获取数据量符号化表达式"""
        # 根据算子类型计算输出数据量，这里简化处理
        bit_width = op.data_format.get_bit_width()
        
        if op.op_type == "conv":
            # 卷积算子输出尺寸: batch * out_c * out_h * out_w
            batch = dvar_symbols.get("batch", 1)
            out_c = dvar_symbols.get("out_c", 1)
            out_h = dvar_symbols.get("out_h", 1)
            out_w = dvar_symbols.get("out_w", 1)
            elem_num = batch * out_c * out_h * out_w
            
        elif op.op_type == "matmul":
            # 矩阵乘法输出尺寸: m * n
            m = dvar_symbols.get("m", 1)
            n = dvar_symbols.get("n", 1)
            elem_num = m * n
            
        else:
            # 默认情况，假设一个简单的维度组合
            elem_num = 1
            for dvar in op.input_vars:
                elem_num *= dvar_symbols.get(dvar.name, 1)
                
        # 转换为字节数
        return elem_num * bit_width / 8
        
    def runtime_calculation(self, target: str) -> Tuple[float, float, float]:
        """运行时实例化计算，输出精准计算密集度"""
        if target not in self.symbolic_flops or target not in self.symbolic_mem:
            raise ValueError(f"Target {target} not found in symbolic expressions")
            
        # 获取动态参数值
        dvar_values = self.dhg_ir.get_dvar_values()
        
        # 实例化FLOPs计算
        flops_expr = self.symbolic_flops[target]
        actual_flops = flops_expr.substitute(dvar_values)
        
        # 实例化内存开销计算
        mem_expr = self.symbolic_mem[target]
        actual_mem = mem_expr.substitute(dvar_values)
        
        # 计算密集度
        ci = actual_flops / actual_mem if actual_mem != 0 else float('inf')
        
        return actual_flops, actual_mem, ci
        
    def analyze_ci_trend(self, target: str, dvar_name: str) -> str:
        """分析计算密集度随动态维度变量的变化趋势"""
        if target not in self.symbolic_flops or target not in self.symbolic_mem:
            raise ValueError(f"Target {target} not found in symbolic expressions")
            
        # 简化分析，检查CI表达式中dvar的次数
        flops_expr = self.symbolic_flops[target].expr
        mem_expr = self.symbolic_mem[target].expr
        ci_expr = flops_expr / mem_expr
        
        # 检查dvar_name在表达式中的次数
        dvar = sp.Symbol(dvar_name)
        degree = sp.poly(ci_expr, dvar).degree() if dvar in ci_expr.free_symbols else 0
        
        if degree > 0:
            return f"CI ∝ {dvar_name}^{degree}"
        elif degree < 0:
            return f"CI ∝ {dvar_name}^{degree} (decreasing with {dvar_name})"
        else:
            return f"CI is independent of {dvar_name}"