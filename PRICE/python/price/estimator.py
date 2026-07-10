"""Performance / memory estimator for PRICE.

Given a scheduled :class:`ComputeGraph`, the estimator simulates the execution
timeline of every operator and derives the peak memory footprint.  It supports
both a *synchronous* timeline (single stream) and an *asynchronous* timeline
(compute / memory streams overlapping).
"""
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

from .graph import ComputeGraph, NodeId
from .nodes import FissionOp
from . import common as utils


@dataclass
class SimOpLife:
    """Lifetime of one operator's output tensor on the timeline."""

    start_time: float = 0
    end_time: float = float("inf")
    free_time: float = float("inf")

    def overlap(self, other: "SimOpLife") -> bool:
        if self.free_time is None or other.free_time is None:
            return False
        return not (self.free_time <= other.start_time or other.free_time <= self.start_time)

    def exec_in(self, t: float) -> bool:
        return self.start_time <= t < self.end_time

    def live_in(self, t: float) -> bool:
        return self.start_time <= t < self.free_time


@dataclass
class SimMemUsage:
    """Memory footprint sampled at discrete time points."""

    time_points: List[float]            # discrete-time-id -> real-time
    foot_prints: List[int]              # i-th time-point's memory footprint
    peak_op_ids: Set[NodeId] = None
    time_to_idx: Dict[float, int] = None  # real-time -> discrete-time-id
    peak_timeid: int = None
    peak_memory: int = None
    peak_timept: int = None
    live_op_ids: List[Set[NodeId]] = None

    def __post_init__(self):
        if self.peak_timeid is None or self.peak_memory is None:
            self.peak_timeid, self.peak_memory = max(
                enumerate(self.foot_prints), key=lambda t: t[1]
            )
        if self.peak_timept is None:
            self.peak_timept = self.time_points[self.peak_timeid]
        if self.time_to_idx is None:
            self.time_to_idx = {t: i for i, t in enumerate(self.time_points)}
        if self.peak_op_ids is None:
            assert self.live_op_ids is not None
            self.peak_op_ids = self.live_op_ids[self.peak_timeid]


@dataclass
class EstimateResult:
    """Result of one estimation pass over a scheduled graph."""

    graph: ComputeGraph
    op_lifes: Dict[NodeId, SimOpLife]
    latency: float = None
    _mem_cache: SimMemUsage = None

    def __post_init__(self):
        if self.latency is None:
            self.latency = max(self.op_lifes.values(), key=lambda life: life.end_time).end_time

    @property
    def mem_usage(self) -> SimMemUsage:
        if self._mem_cache is None:
            self._mem_cache = self._build_mem_usage()
        return self._mem_cache

    def _build_mem_usage(self) -> SimMemUsage:
        # collect every event time and build a piecewise-constant memory signal
        time_pts = set()
        for life in self.op_lifes.values():
            time_pts.update((life.start_time, life.end_time, life.free_time))
        time_pts = sorted(time_pts)
        t2i = {t: i for i, t in enumerate(time_pts)}
        delta = [0] * len(time_pts)
        for op_id, life in self.op_lifes.items():
            op = self.graph.fs_get_op(op_id)
            if op.is_store():
                continue
            delta[t2i[life.start_time]] += op.out_memory + op.extra_memory
            delta[t2i[life.end_time]] -= op.extra_memory
            delta[t2i[life.free_time]] -= op.out_memory
        # prefix sum -> actual footprint, track peak
        peak_idx, peak_val = 0, delta[0]
        for i in range(1, len(delta)):
            delta[i] += delta[i - 1]
            if peak_val < delta[i]:
                peak_val = delta[i]
                peak_idx = i
        peak_time = time_pts[peak_idx]
        peak_set = {oid for oid, life in self.op_lifes.items() if life.live_in(peak_time)}
        return SimMemUsage(time_pts, delta, peak_set, t2i, peak_idx, peak_val, peak_time)

    @property
    def peak_memory(self) -> int:
        return self.mem_usage.peak_memory

    def lifetime_overlap(self, a: NodeId, b: NodeId) -> bool:
        return self.op_lifes[a].overlap(self.op_lifes[b])

    def select_swap_remat_candidates(self) -> List[Tuple[NodeId, NodeId, float]]:
        """Pick (remat, successor) pairs to swap for memory reduction."""
        peak_time = self.mem_usage.peak_timept
        cands: List[Tuple[NodeId, NodeId, float]] = []
        for oid in self.mem_usage.peak_op_ids:
            op = self.graph.fs_get_op(oid)
            if op.is_mem_op():
                continue
            oid_life = self.op_lifes[oid]
            assert oid_life.live_in(peak_time)
            if oid_life.exec_in(peak_time):
                continue
            succ_lives = [(sid, self.op_lifes[sid]) for sid in self.graph.fs_suc_ids(oid)]
            if len(succ_lives) <= 0:
                continue
            if any(life.exec_in(peak_time) for _, life in succ_lives):
                continue
            succ, succ_life = min(
                [(sid, life) for sid, life in succ_lives if life.start_time > peak_time],
                key=lambda t: t[1].start_time,
            )
            if self.graph.fs_get_op(succ).is_mem_op():
                continue
            gain = min(succ_life.start_time - peak_time, peak_time - oid_life.end_time)
            cands.append((oid, succ, gain))
        return sorted(cands, key=lambda t: t[-1], reverse=True)


class BaseEstimator:
    def __init__(self, save_last_sim_res: bool = False) -> None:
        self._keep_last = save_last_sim_res

    def _reset(self, graph: ComputeGraph = None, init_time: float = 0, op_lifes=None):
        self._cg = graph
        self._t0 = init_time
        self._lifes: Dict[NodeId, SimOpLife] = op_lifes

    def _post_reset(self):
        pass

    def _run_one_step(self, step_id: int, op_id: NodeId):
        raise NotImplementedError()

    def run(self, graph: ComputeGraph, init_time: float = 0) -> EstimateResult:
        self._reset(graph, init_time, defaultdict(SimOpLife))
        self._post_reset()

        for idx, op_id in enumerate(self._cg.sched):
            self._run_one_step(idx, op_id)
        end_t = max(self._lifes.values(), key=lambda life: life.end_time).end_time

        for op_id, life in self._lifes.items():
            if isinstance(op_id, (list, tuple)):
                continue
            cur_op = self._cg[op_id]
            if not cur_op.in_mem:
                assert cur_op.is_store()
                life.free_time = life.end_time
                continue
            succs = self._cg.suc_ids(op_id)
            if len(succs) <= 0:
                life.free_time = end_t
            else:
                life.free_time = max(self._lifes[sid].end_time for sid in succs)

        result = EstimateResult(self._cg, self._lifes, end_t - init_time)
        self._reset()
        if self._keep_last:
            self.last_sim_res = result
        return result

    def _handle_fs_op(self, op_id: NodeId, op: FissionOp, cur_time: float):
        sub_res = self.__class__().run(op.fs_graph, cur_time)
        op._latency = sub_res.latency * (op.fs_length // op.fs_factor)
        self._lifes.update(
            {
                (op_id, *utils.to_tuple(ids)): life
                for ids, life in sub_res.op_lifes.items()
            }
        )


class SyncEstimator(BaseEstimator):
    def _post_reset(self):
        self._clock = self._t0

    def _run_one_step(self, step_id: int, op_id: NodeId):
        op = self._cg[op_id]
        if op.is_fission():
            self._handle_fs_op(op_id, op, self._clock)
        life = self._lifes[op_id]
        life.start_time = self._clock
        life.end_time = self._clock + op.latency
        self._clock = life.end_time


class AsyncEstimator(BaseEstimator):
    def _post_reset(self):
        self._clocks = [self._t0, self._t0]

    def _run_one_step(self, step_id: int, op_id: NodeId):
        op = self._cg[op_id]

        stream_id = int(op.is_mem_op())
        if stream_id == 1:
            self._clocks[0] = self._clocks[1] = max(self._clocks)
        cur_time = self._clocks[stream_id]
        cur_time = max(
            [cur_time, *(self._lifes[pre_id].end_time for pre_id in self._cg.pre_ids(op_id))]
        )
        if op.is_fission():
            self._handle_fs_op(op_id, op, cur_time)
        life = self._lifes[op_id]
        life.start_time = cur_time
        life.end_time = cur_time + op.latency
        self._clocks[stream_id] = life.end_time


DefaultEstimator = AsyncEstimator
