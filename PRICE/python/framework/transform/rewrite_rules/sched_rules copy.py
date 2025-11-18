from .base import *


class SwappingRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.relu(x)

        """
        The returned input_ids and output_ids technically have no distinction... 
        Their union represents the nodes in the original sub-graph that are connected to outside. 
        The only difference is that input-ops can be used directly within _create_new_subgraph. 
        Due to some historical reasons and it's not convenient to modify such interface.
        """
        return P, [x, y], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        return not g_op.is_mem_op()

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy = inp_ops
        gx_id, gy_id = pms.g_inp_ids
        arg_idx = pms.ori_graph.etag_between(gx_id, gy_id)
        S = OpGraph()
        x = S.add_node(gx)
        y = S.store(x)
        y = S.load(y)
        y = S.add_op_with_args(gy, [y], [arg_idx])
        return S, [x, y], []


# recompute for unary-op
class RecomputeRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.relu(x)  # 模式中的中间算子（一元）
        z1 = P.relu(y)
        z2 = P.relu(y)
        self.y_op = P[y]  # 记录待重计算的中间算子模式
        return P, [x, y, z1, z2], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        # 原有逻辑：排除“中间算子是一元但实际算子输入数≠1”的情况
        base_condition = not (self.y_op is p_op and len(g_op.inp_shapes) != 1)
        
        # 新增：过滤计算密集型算子（包含最新补充的算子）
        if self.y_op is p_op:  # 仅对中间算子生效
            # 计算密集型算子的实际tag列表（精确匹配固定tag）
            fixed_heavy_tags = {
                # 矩阵乘法类
                "matmul", "flex_matmul",
                # 线性层与归一化
                "linear", "layer_norm", "layer_norm_bwd",
                # 注意力与前馈网络
                "attention", "feedforward",
                # 卷积类（正向+反向）
                "conv2d", "conv2d_bwd_inp", "conv2d_bwd_wgt",
                # 池化类（正向+反向）
                "pool2d.avg", "pool2d.max", "pool2d_bwd.avg", "pool2d_bwd.max",
                # 插值类（正向+反向）
                "interpolate", "interpolate_bwd",
                # 激活函数类（正向+反向）
                "softmax", "softmax_bwd"
            }
            # 动态匹配前缀（如reduce.sum、reduce.mean等）
            is_reduce_op = g_op.tag.startswith("reduce.")
            
            # 满足任一条件则过滤
            if g_op.tag in fixed_heavy_tags or is_reduce_op:
                return False
        
        return base_condition

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy, gz1, gz2 = inp_ops
        _, gy_id, gz1_id, gz2_id = pms.g_inp_ids
        arg_idx1 = pms.ori_graph.etag_between(gy_id, gz1_id)
        arg_idx2 = pms.ori_graph.etag_between(gy_id, gz2_id)
        S = OpGraph()
        x = S.add_node(gx)
        y1 = S.add_op_with_args(gy, [x])
        y2 = S.add_op_with_args(gy, [x])
        z1 = S.add_op_with_args(gz1, [y1], [arg_idx1])
        z2 = S.add_op_with_args(gz2, [y2], [arg_idx2])
        return S, [x, y1, z1, z2], []


# recompute for binary-op
class RecomputeRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1])
        x2 = P.placeholder([1])
        self.y = y = P.add(x1, x2)  # 模式中的中间算子（二元）
        self.z1 = z1 = P.relu(y)
        self.z2 = z2 = P.relu(y)
        self.y_op = P[y]  # 记录待重计算的中间算子模式
        return P, [x1, x2, y, z1, z2], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        # 原有逻辑：排除“中间算子是二元但实际算子输入数≠2”的情况
        base_condition = not (self.y_op is p_op and len(g_op.inp_shapes) != 2)
        
        # 新增：过滤计算密集型算子（与RecomputeRule1保持一致）
        if self.y_op is p_op:  # 仅对中间算子生效
            fixed_heavy_tags = {
                # 矩阵乘法类
                "matmul", "flex_matmul",
                # 线性层与归一化
                "linear", "layer_norm", "layer_norm_bwd",
                # 注意力与前馈网络
                "attention", "feedforward",
                # 卷积类（正向+反向）
                "conv2d", "conv2d_bwd_inp", "conv2d_bwd_wgt",
                # 池化类（正向+反向）
                "pool2d.avg", "pool2d.max", "pool2d_bwd.avg", "pool2d_bwd.max",
                # 插值类（正向+反向）
                "interpolate", "interpolate_bwd",
                # 激活函数类（正向+反向）
                "softmax", "softmax_bwd"
            }
            is_reduce_op = g_op.tag.startswith("reduce.")
            
            if g_op.tag in fixed_heavy_tags or is_reduce_op:
                return False
        
        return base_condition

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        *gxs, gy, gz1, gz2 = inp_ops
        *_, gy_id, gz1_id, gz2_id = pms.g_inp_ids
        arg_idx1 = pms.ori_graph.etag_between(gy_id, gz1_id)
        arg_idx2 = pms.ori_graph.etag_between(gy_id, gz2_id)
        S = OpGraph()
        x1, x2 = S.add_nodes(gxs)
        y1 = S.add_op_with_args(gy, [x1, x2])
        y2 = S.add_op_with_args(gy, [x1, x2])
        z1 = S.add_op_with_args(gz1, [y1], [arg_idx1])
        z2 = S.add_op_with_args(gz2, [y2], [arg_idx2])
        return S, [x1, x2, y1, z1, z2], []


# 以下规则保持不变
class DeSwappingRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y = P.load(y)
        return P, [x], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x]


class DeSwappingRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y1 = P.load(y)
        y2 = P.load(y)
        return P, [x], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x, x]


class DeSwappingRule3(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y1 = P.load(y)
        y2 = P.load(y)
        y3 = P.load(y)
        return P, [x], [y1, y2, y3]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x, x, x]


class DeRecomputeRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y1 = P.relu(x)
        y2 = P.relu(x)
        return P, [x, y1, y2], []

    def _match_op(self, g_op: Operator, p_op: Operator):
        return p_op.tag == "placeholder" or len(g_op.inp_shapes) == len(p_op.inp_shapes)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy1, gy2 = inp_ops
        if gy1.key != gy2.key:
            return False
        S = OpGraph()
        x = S.add_node(gx)
        y = S.add_op_with_args(gy1, [x])
        return S, [x, y, y], []


class DeRecomputeRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1])
        x2 = P.placeholder([1])
        y1 = P.add(x1, x2)
        y2 = P.add(x1, x2)
        return P, [x1, x2, y1, y2], []

    def _match_op(self, g_op: Operator, p_op: Operator):
        return p_op.tag == "placeholder" or len(g_op.inp_shapes) == len(p_op.inp_shapes)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        *gxs, gy1, gy2 = inp_ops
        if gy1.key != gy2.key:
            return False
        S = OpGraph()
        x1, x2 = S.add_nodes(gxs)
        y = S.add_op_with_args(gy1, [x1, x2])
        return S, [x1, x2, y, y], []
