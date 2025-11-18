import time
import os

import framework.scheduler as SCH
import framework.optimizer as OPT
import framework.backend as BAK
from framework.op_graph import OpGraph, OpId
from framework.utils import LOG
from framework import utils
from .configs import get_configured_mutator


def setup_training_graph(G: OpGraph, y: OpId, inplace=True, update_weight=True):
    """
    构建训练图（添加损失函数和反向传播）
    Args:
        G: 原始计算图
        y: 模型输出节点ID
        inplace: 是否原地修改图（默认True）
        update_weight: 是否更新权重（默认True）
    Returns:
        包含反向传播的完整训练图
    """
    G = G.may_copy(inplace=inplace)
    y_bar = G.placeholder(G[y].out_shape, "y_bar")  # 真实标签占位符
    loss = G.pow_const(G.sub(y, y_bar), 2)  # 平方损失函数
    return G.backward(loss, inplace=True, update_weight=update_weight)


def run_optimization(
    G: OpGraph,
    name,
    opt: OPT.BaseOptimizer = None,
    scheduler=None,
    mutator=None,
    dtype="float32",
    dump_file=None,
    save_graph=True,
    save_records=True,
    **kwargs,
):
    """
    执行内存优化流程（含图变换、调度、性能 profiling）
    关键修改：按原文档规则，将「元素个数」格式的内存指标，换算为 GB 单位（保留2位小数）
    """
    # 计算模型权重占用内存（原始单位：元素个数，已÷dtype字节数）
    wgt_mem = sum(G[v].out_memory for v in G.all_ids() if G[v].is_weight())
    
    with BAK.TorchCudaBackend(dtype=dtype) as bknd:
        # 初始化优化器（未指定时使用默认配置）
        if not isinstance(opt, OPT.BaseOptimizer):
            scheduler = scheduler or SCH.RefineMemOpRpoScheduler(adjust_load_op=False)
            mutator = mutator or get_configured_mutator(** kwargs)
            opt = (opt or OPT.RelaxOptimizer)(scheduler, bknd, mutator, **kwargs)
        
        # 执行优化（图变换 + 调度）
        G, sched = opt.run(G)
        
        # 获取优化后的模拟结果和实测结果
        sim_res = opt._best_state.sim_res
        opt._always_simulation = False
        run_res = opt._profile(state=opt._best_state, number=4, repeat=5, flag=True)
        
        # 获取原始图的实测结果
        opt._init_state.run_res = opt._profile(
            state=opt._init_state, number=4, repeat=5
        )

        # 提取优化配置参数（所有内存相关值均为「元素个数」，已÷dtype字节数）
        ll = opt._lat_limit          # 延迟上限（毫秒）
        ml = opt._mem_limit          # 内存限制（元素个数）
        dml = opt._dev_mem_limit     # 设备总内存（元素个数）
        llr = opt._lat_limit_ratio   # 延迟限制比例
        mlr = opt._mem_limit_ratio   # 内存限制比例

        # -------------------------- 核心修正：内存单位换算（元素个数 → GB）--------------------------
        # 1. 定义 dtype 对应的字节数（覆盖常用类型，可按需补充）
        dtype_to_bytes = {
            "float32": 4,
            "float16": 2,
            "float64": 8,
            "int32": 4,
            "int16": 2,
            "int64": 8,
            "uint8": 1,
            "bool": 1
        }
        bytes_per_elem = dtype_to_bytes.get(dtype.lower(), 4)  # 默认按float32处理（4字节）
        
        # 2. 定义 GB 换算系数（二进制：1GB=1024^3字节；十进制用1000**3）
        GB = 1024 ** 3
        decimal_places = 2  # 保留2位小数，可调整

        # 3. 换算逻辑：元素个数 → 字节数（×字节数）→ GB（÷GB系数）
        def to_gb(element_count):
            return round((element_count * bytes_per_elem) / GB, decimal_places)
        
        # 4. 批量转换所有内存字段
        dml_gb = to_gb(dml)                # 设备总内存（GB）
        ml_gb = to_gb(ml)                  # 设定的内存限制（GB）
        wgt_mem_gb = to_gb(wgt_mem)        # 模型权重内存（GB）
        opt_mem_gb = to_gb(run_res[2])     # 优化后实测峰值内存（GB）
        opt_sim_mem_gb = to_gb(sim_res.peak_memory)  # 优化后模拟峰值内存（GB）
        ori_mem_gb = to_gb(opt._init_state.run_res[2])  # 原始实测峰值内存（GB）
        ori_sim_mem_gb = to_gb(opt._init_state.sim_res.peak_memory)  # 原始模拟峰值内存（GB）
        # ---------------------------------------------------------------------------------------

        # 保存优化后的图和调度信息
        if save_graph:
            os.makedirs("./data", exist_ok=True)
            utils.save_pickle(os.path.join("./data", name + ".pkl"), (G, sched))
        
        # 保存优化过程记录
        if save_records:
            with open(os.path.join("./data", name + ".tmp.csv"), "w") as fp:
                for rec in opt._records:
                    print(*rec, sep=",", file=fp)

        # 组装最终结果（所有内存字段已转为 GB 单位）
        res = (
            name,
            dml_gb, ll, ml_gb, llr, mlr,  # 设备内存、延迟/内存限制（内存→GB）
            wgt_mem_gb,                    # 权重内存（→GB）
            run_res[0], run_res[1], opt_mem_gb,  # 优化后：是否实测、延迟、峰值内存（→GB）
            sim_res.latency, opt_sim_mem_gb,      # 优化后：模拟延迟、模拟峰值内存（→GB）
            opt._init_state.run_res[0], opt._init_state.run_res[1], ori_mem_gb,  # 原始：是否实测、延迟、峰值内存（→GB）
            opt._init_state.sim_res.latency, ori_sim_mem_gb,  # 原始：模拟延迟、模拟峰值内存（→GB）
        )

        # 日志输出和结果写入文件
        LOG.info(f"opt mem:{opt_mem_gb}.GB")
        LOG.info(f"opt latency:{run_res[1]}.ms")
        LOG.info(f"opi mem:{ori_mem_gb}.GB")
        LOG.info(f"ori latency:{opt._init_state.run_res[1]}.ms")

        if dump_file:
            print(*res, file=dump_file, flush=True, sep=",")
    
    return res  # 可选：返回结果供外部使用