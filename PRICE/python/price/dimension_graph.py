"""Dynamic dimension graph for PRICE.

The dimension graph exposes how the *loop dimensions* of operators connect
across a :class:`ComputeGraph`.  It is derived from a computation graph and is
consumed by the scheduling / fission passes to reason about dimension-level
dependencies without touching the operator-level IR.
"""
from typing import List, Tuple, Dict, Union
from collections import namedtuple, defaultdict

import numpy as np
import rustworkx as rx

from .common.base_graph import BaseGraph, NodeId
from .graph import ComputeGraph, NodeId, ArgIndex
from . import common as utils

# A dimension is identified by the operator that owns it and its index.
DimIndex = int
Dimension = namedtuple("Dimension", ["op_id", "d_idx"])
DimId = NodeId


class DimGraph(BaseGraph["DimGraph", Dimension, ArgIndex]):
    """Dimension-level view of a computation graph."""

    def __init__(self, graph: ComputeGraph = None, rx_graph: rx.PyDiGraph = None) -> None:
        super().__init__(rx_graph)
        self.graph = None
        if graph is not None:
            self.bind_graph(graph)

    # ------------------------------------------------------------------ #
    # cached properties
    # ------------------------------------------------------------------ #
    @property
    def op_dom_table(self):
        if self._op_dom_cache is None:
            self._build_op_dom_table()
        return self._op_dom_cache

    def _reset_saved_props(self):
        super()._reset_saved_props()
        self._op_dom_cache: np.ndarray = None

    def _build_op_dom_table(self):
        G = self.graph
        dom_table = self._op_dom_cache = G.dom_table.copy()
        dim_op_ids = self.all_op_ids()
        for node in G.all_ids():
            if G[node].is_placeholder():
                continue
            if node not in dim_op_ids:
                dom_table[node, :] = 0
                dom_table[:, node] = 0

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def bind_graph(self, G: ComputeGraph):
        if G is None:
            return
        self.reset()
        self.graph = G

        dim_union = utils.UnionFindSet[Dimension]()
        pending_deps: Dict[Dimension, List[Tuple[Dimension, ArgIndex]]] = defaultdict(list)
        dim_node_ids = dict()

        dep_table = G.des_table

        for node in G.bwd_rpo_topo_order:
            for dim_idx in G[node].loop_dims:
                dim = Dimension(node, dim_idx)
                group_ids: List[int] = []
                linked: List[int] = []
                feasible = True
                for dep_dim, _ in pending_deps[dim]:
                    dep_root = dim_union.find_root(dep_dim)
                    if dep_root in linked:
                        continue
                    dep_ids = dim_union[dep_root]
                    if (
                        dep_table[np.ix_(group_ids, dep_ids)].any()
                        or dep_table[np.ix_(dep_ids, group_ids)].any()
                    ):
                        feasible = False
                        break
                    group_ids.extend(dep_ids)
                    linked.append(dep_root)
                if feasible:
                    group_ids.append(node)
                    dim_union[dim] = group_ids
                    for grp in linked:
                        dim_union.union_trees(grp, dim)
                    dim_node_ids[dim] = dim_node_id = self.add_node(dim)
                    for dep_dim, arg_idx in pending_deps[dim]:
                        self.add_edge(dim_node_id, dim_node_ids[dep_dim], arg_idx)
                else:
                    dim_union[dim] = [node]
                    dim_node_ids[dim] = self.add_node(dim)

            for (i, src), links in zip(enumerate(G.arg_ids(node)), G[node]._dim_links):
                for src_dim, dim_idx in links:
                    pending_deps[(src, src_dim)].append(((node, dim_idx), i))

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def get_dim_op(self, did: Union[DimId, Dimension]):
        dim = did if isinstance(did, Dimension) else self[did]
        return self.graph[dim.op_id]

    def get_dim_len(self, did: Union[DimId, Dimension]):
        dim = did if isinstance(did, Dimension) else self[did]
        return self.graph[dim.op_id].loop_lens[dim.d_idx]

    def get_connected_dim_len(self):
        assert self.is_connected()
        dim_lens = [self.get_dim_len(did) for did in self.all_ids()]
        result = next(length for length in dim_lens if length > 0)
        assert all(length == result for length in dim_lens if length > 0), dim_lens
        return result

    def all_op_ids(self):
        return {dim.op_id for dim in self.all_nodes()}

    def dim_map(self) -> Dict[NodeId, DimIndex]:
        return {dim.op_id: dim.d_idx for dim in self.all_nodes()}

    def graphviz(self, name, fmt="png", render=True):
        dot = self._rx_graph.to_dot(
            node_attr=lambda d: dict(fontname="consolas", label=str(tuple(d)), shape="box")
        )
        return utils.render_dot(dot, name=name, fmt=fmt, render=render)
