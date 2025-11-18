from .base import *
import datetime  # 用于日志时间戳，方便定位执行顺序


def debug_log(msg, rule_name="UnknownRule"):
    """统一Debug日志格式，包含时间戳和规则名称"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"【DEBUG】[{timestamp}][{rule_name}] {msg}")


class SwappingRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.relu(x)
        return P, [x, y], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        
        return not g_op.is_mem_op()

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy = inp_ops
        debug_log(f"执行Swapping: 输入算子tag={gx.tag}→{gy.tag}", "SwappingRule")
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
    RULE_NAME = "RecomputeRule1(Unary)"  # 规则标识

    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.relu(x)
        z1 = P.relu(y)
        z2 = P.relu(y)
        self.y_op = P[y]
        debug_log("初始化重计算模式（一元算子）", self.RULE_NAME)
        return P, [x, y, z1, z2], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        debug_log(f"匹配边: g_etag={g_etag}, p_etag={p_etag}", self.RULE_NAME)
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        # 1. 打印当前匹配的算子基础信息
        op_info = f"tag={g_op.tag}, inp_shapes_len={len(g_op.inp_shapes)}, is_target={self.y_op is p_op}"
        debug_log(f"匹配算子: {op_info}", self.RULE_NAME)

        # 2. 原有基础条件判断
        base_condition = not (self.y_op is p_op and len(g_op.inp_shapes) != 1)
        debug_log(f"基础条件判断: base_condition={base_condition}", self.RULE_NAME)
        if not base_condition:
            return False

        # 3. 密集型算子过滤逻辑
        if self.y_op is p_op:
            fixed_heavy_tags = {
                "matmul", "flex_matmul",
                "linear", "layer_norm", "layer_norm_bwd",
                "attention", "feedforward",
                "conv2d", "conv2d_bwd_inp", "conv2d_bwd_wgt",
                "pool2d.avg", "pool2d.max", "pool2d_bwd.avg", "pool2d_bwd.max",
                "interpolate", "interpolate_bwd",
                "softmax", "softmax_bwd"
            }
            is_reduce_op = g_op.tag.startswith("reduce.")
            is_heavy = g_op.tag in fixed_heavy_tags or is_reduce_op

            # 打印过滤判断细节
            debug_log(
                f"过滤逻辑判断: tag_in_fixed={g_op.tag in fixed_heavy_tags}, is_reduce={is_reduce_op}, is_heavy={is_heavy}",
                self.RULE_NAME
            )
            if is_heavy:
                debug_log(f"✅ 过滤密集型算子: {g_op.tag}（不执行重计算）", self.RULE_NAME)
                return False

        debug_log(f"❌ 未过滤，允许重计算", self.RULE_NAME)
        return True

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy, gz1, gz2 = inp_ops
        # 打印重计算执行信息
        debug_log(
            f"执行重计算: 中间算子tag={gy.tag}, 下游算子tag={gz1.tag}/{gz2.tag}",
            self.RULE_NAME
        )
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
    RULE_NAME = "RecomputeRule2(Binary)"  # 规则标识

    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1])
        x2 = P.placeholder([1])
        self.y = y = P.add(x1, x2)
        self.z1 = z1 = P.relu(y)
        self.z2 = z2 = P.relu(y)
        self.y_op = P[y]
        debug_log("初始化重计算模式（二元算子）", self.RULE_NAME)
        return P, [x1, x2, y, z1, z2], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        debug_log(f"匹配边: g_etag={g_etag}, p_etag={p_etag}", self.RULE_NAME)
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        # 1. 打印当前匹配的算子基础信息
        op_info = f"tag={g_op.tag}, inp_shapes_len={len(g_op.inp_shapes)}, is_target={self.y_op is p_op}"
        debug_log(f"匹配算子: {op_info}", self.RULE_NAME)

        # 2. 原有基础条件判断
        base_condition = not (self.y_op is p_op and len(g_op.inp_shapes) != 2)
        debug_log(f"基础条件判断: base_condition={base_condition}", self.RULE_NAME)
        if not base_condition:
            return False

        # 3. 密集型算子过滤逻辑
        if self.y_op is p_op:
            fixed_heavy_tags = {
                "matmul", "flex_matmul",
                "linear", "layer_norm", "layer_norm_bwd",
                "attention", "feedforward",
                "conv2d", "conv2d_bwd_inp", "conv2d_bwd_wgt",
                "pool2d.avg", "pool2d.max", "pool2d_bwd.avg", "pool2d_bwd.max",
                "interpolate", "interpolate_bwd",
                "softmax", "softmax_bwd"
            }
            is_reduce_op = g_op.tag.startswith("reduce.")
            is_heavy = g_op.tag in fixed_heavy_tags or is_reduce_op

            # 打印过滤判断细节
            debug_log(
                f"过滤逻辑判断: tag_in_fixed={g_op.tag in fixed_heavy_tags}, is_reduce={is_reduce_op}, is_heavy={is_heavy}",
                self.RULE_NAME
            )
            if is_heavy:
                debug_log(f"✅ 过滤密集型算子: {g_op.tag}（不执行重计算）", self.RULE_NAME)
                return False

        debug_log(f"❌ 未过滤，允许重计算", self.RULE_NAME)
        return True

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        *gxs, gy, gz1, gz2 = inp_ops
        # 打印重计算执行信息
        gx_tags = [op.tag for op in gxs]
        debug_log(
            f"执行重计算: 中间算子tag={gy.tag}, 输入算子tag={gx_tags}, 下游算子tag={gz1.tag}/{gz2.tag}",
            self.RULE_NAME
        )
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


# 以下规则保持不变，仅添加基础触发日志
class DeSwappingRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log("初始化去Swapping模式（单load）", "DeSwappingRule1")
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y = P.load(y)
        return P, [x], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log(f"执行去Swapping: 输入算子tag={inp_ops[0].tag}", "DeSwappingRule1")
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x]


class DeSwappingRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log("初始化去Swapping模式（双load）", "DeSwappingRule2")
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y1 = P.load(y)
        y2 = P.load(y)
        return P, [x], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log(f"执行去Swapping: 输入算子tag={inp_ops[0].tag}", "DeSwappingRule2")
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x, x]


class DeSwappingRule3(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log("初始化去Swapping模式（三load）", "DeSwappingRule3")
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
        debug_log(f"执行去Swapping: 输入算子tag={inp_ops[0].tag}", "DeSwappingRule3")
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x, x, x]


class DeRecomputeRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log("初始化去重计算模式（一元）", "DeRecomputeRule1")
        P = OpGraph()
        x = P.placeholder([1])
        y1 = P.relu(x)
        y2 = P.relu(x)
        return P, [x, y1, y2], []

    def _match_op(self, g_op: Operator, p_op: Operator):
        debug_log(f"匹配算子: tag={g_op.tag}, p_tag={p_op.tag}", "DeRecomputeRule1")
        return p_op.tag == "placeholder" or len(g_op.inp_shapes) == len(p_op.inp_shapes)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy1, gy2 = inp_ops
        debug_log(f"执行去重计算: 算子tag={gy1.tag}", "DeRecomputeRule1")
        if gy1.key != gy2.key:
            return False
        S = OpGraph()
        x = S.add_node(gx)
        y = S.add_op_with_args(gy1, [x])
        return S, [x, y, y], []


class DeRecomputeRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        debug_log("初始化去重计算模式（二元）", "DeRecomputeRule2")
        P = OpGraph()
        x1 = P.placeholder([1])
        x2 = P.placeholder([1])
        y1 = P.add(x1, x2)
        y2 = P.add(x1, x2)
        return P, [x1, x2, y1, y2], []

    def _match_op(self, g_op: Operator, p_op: Operator):
        debug_log(f"匹配算子: tag={g_op.tag}, p_tag={p_op.tag}", "DeRecomputeRule2")
        return p_op.tag == "placeholder" or len(g_op.inp_shapes) == len(p_op.inp_shapes)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        *gxs, gy1, gy2 = inp_ops
        debug_log(f"执行去重计算: 算子tag={gy1.tag}", "DeRecomputeRule2")
        if gy1.key != gy2.key:
            return False
        S = OpGraph()
        x1, x2 = S.add_nodes(gxs)
        y = S.add_op_with_args(gy1, [x1, x2])
        return S, [x1, x2, y, y], []