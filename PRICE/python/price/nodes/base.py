"""Base operator definitions for the PRICE graph IR.

An :class:`Operator` is a single node of a :class:`ComputeGraph`.  It caches
several derived quantities lazily (latency, output memory, digest, loop sizes)
so that repeated queries during the search are cheap.  The public attribute
names (``_dim_links``, ``_latency``, ``_out_memory`` ...) form an implicit
contract shared with the rest of the package and are intentionally preserved.
"""
from typing import List, Tuple, Set, Dict
from copy import copy as _shallow_copy

from .. import common as utils


class Operator:
    DEFAULT_IMAGE_LAYOUT = "nchw"
    # whether instances with different shapes may reuse the same profiling code
    REUSE_PROF_CODE = True
    ZERO_LATENCY = 1e-8

    def __init__(self, tag, out_shape, inp_shapes, in_mem, **attrs) -> None:
        self.tag: str = tag
        self.out_shape: List[int] = tuple(out_shape)
        self.inp_shapes: List[List[int]] = [tuple(s) for s in inp_shapes]
        self.in_mem = in_mem  # whether the output stays resident in memory
        self.attrs = attrs

        # dimension connections between the output and every input, e.g.
        # [[(a1, c1), (a2, -1)], [(b1, c2), (b2, -1)]] means:
        #   input A: a1<->c1 spatial, a2 is the 1st reduce axis
        #   input B: b1<->c2 spatial, b2 is the 1st reduce axis
        self._dim_links: List[List[Tuple[int, int]]] = []
        self._loop_dims: Set[int] = None

        # lazily computed, may vary with shape
        self._out_memory: int = -1
        self._extra_memory: int = -1
        self._latency: float = -1
        self._key = None
        self._digest = None
        self._loop_lens: Dict[int, int] = None

    # ------------------------------------------------------------------ #
    # copying
    # ------------------------------------------------------------------ #
    def copy(self):
        clone = _shallow_copy(self)
        clone._out_memory = -1
        clone._extra_memory = -1
        clone._latency = -1
        clone._key = None
        clone._digest = None
        clone._loop_lens = None
        clone.attrs = dict(clone.attrs)
        return clone

    def may_copy(self, inplace=False):
        return self if inplace else self.copy()

    # ------------------------------------------------------------------ #
    # attribute access
    # ------------------------------------------------------------------ #
    def __getitem__(self, k):
        return self.attrs[k]

    def __setitem__(self, k, v):
        self.attrs[k] = v

    # ------------------------------------------------------------------ #
    # cached cost metrics
    # ------------------------------------------------------------------ #
    @property
    def latency(self):
        if self._latency < 0:
            self._latency = self._measure_latency()
        return self._latency

    def _measure_latency(self):
        from ..runtime import BaseBackend

        backend = BaseBackend.CURRENT_BACKEND
        if backend is not None:
            return backend.measure_operator_latency(
                self,
                BaseBackend.DEFAULT_MEASURE_NUMBER,
                BaseBackend.DEFAULT_MEASURE_REPEAT,
            )
        return self.ZERO_LATENCY

    @property
    def out_memory(self):
        if self._out_memory < 0:
            self._out_memory = self._compute_out_mem()
        return self._out_memory

    def _compute_out_mem(self):
        return utils.prod(self.out_shape)

    @property
    def extra_memory(self):
        if self._extra_memory < 0:
            self._extra_memory = self._compute_extra_mem()
        return self._extra_memory

    def _compute_extra_mem(self):
        return 0

    @property
    def key(self):
        if self._key is None:
            self._key = self._compute_key()
        return self._key

    def _compute_key(self):
        return (
            self.tag + str(sorted(self.attrs.items())),
            str(self.out_shape) + str(self.inp_shapes),
        )

    @property
    def digest(self):
        if self._digest is None:
            self._digest = self._compute_digest()
        return self._digest

    def _compute_digest(self):
        return utils.digest(str(self.key).encode())

    def _merge_args_digest(self, args_digest):
        return utils.digest(b"".join([self.digest, *args_digest]))

    # ------------------------------------------------------------------ #
    # rendering / predicates
    # ------------------------------------------------------------------ #
    def dump(self, op_id, pre_ids, indent=0):
        return "{spaces}${dst} {{shape={shape}}} := {op}({srcs}) {attrs}".format(
            spaces="  " * indent,
            dst=op_id,
            shape=list(self.out_shape),
            op=self.tag,
            srcs=", ".join("$" + str(pi) for pi in pre_ids),
            attrs=self.attrs,
        )

    def is_weight(self):
        return self.attrs.get("is_weight", False)

    def is_mem_op(self):
        return self.tag in ("store", "load")

    def is_load(self):
        return self.tag == "load"

    def is_store(self):
        return self.tag == "store"

    def is_fission(self):
        return self.tag == "fission"

    def is_remat(self):
        return getattr(self, "_is_remat", False)

    def is_placeholder(self):
        return self.tag == "placeholder"

    def multi_out_shapes(self):
        return self.attrs.get("multi_out_shapes", None)

    # ------------------------------------------------------------------ #
    # autodiff
    # ------------------------------------------------------------------ #
    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        """Differentiate this operator.

        Parameters
        ----------
        out_grad: Operator
            gradient flowing into this operator's output.
        inps: List[Operator]
            the operator's input operators.

        Returns
        -------
        inp_grads: List[Operator]
            gradient of each input operator.
        links: List[Tuple[Operator, List[Operator]]]
            new operators produced during differentiation together with their
            inputs, e.g. ``[(c, [a, b]), (d, [a, c])]`` denotes
            ``c = f1(a, b); d = f2(a, c)``.
        """
        if len(inps) == 0:
            return [], []
        raise NotImplementedError(f"{type(self)} currently does not support backward")

    # ------------------------------------------------------------------ #
    # dimension connectivity
    # ------------------------------------------------------------------ #
    def infer_output_linked_dims(self, inp_dims: List[Set[int]]) -> Set[int]:
        """Given selected input dims, return the linked output dims."""
        linked = set()
        for dims, conns in zip(inp_dims, self._dim_links):
            linked.update(d1 for (d0, d1) in conns if d0 in dims)
        return linked

    def infer_input_linked_dims(self, out_dims: Set[int]) -> List[Set[int]]:
        """Given selected output dims, return the linked input dims per input."""
        result = []
        for conns in self._dim_links:
            result.append({d0 for (d0, d1) in conns if d1 in out_dims})
        return result

    @property
    def loop_dims(self):
        if self._loop_dims is None:
            self._loop_dims = self._collect_loop_dims()
        return self._loop_dims

    def _collect_loop_dims(self):
        return {d1 for conns in self._dim_links for (_, d1) in conns}

    @property
    def loop_lens(self):
        if self._loop_lens is None:
            self._loop_lens = self._collect_loop_lens()
        return self._loop_lens

    def _collect_loop_lens(self):
        sizes = dict()
        for shape, conns in zip(self.inp_shapes, self._dim_links):
            for d0, d1 in conns:
                known = sizes.get(d1, None)
                if d1 < 0:
                    src_len = shape[d0]
                    if known is None:
                        sizes[d1] = src_len
                    else:
                        assert known == src_len
                else:
                    if known is None:
                        sizes[d1] = self.out_shape[d1]
        return sizes


class PlaceholderOp(Operator):
    _USED_NAMES = set()

    def __init__(self, shape, name_hint="p", in_mem=True, is_weight=False) -> None:
        suffix = len(self._USED_NAMES)
        name = name_hint
        while name in self._USED_NAMES:
            name = name_hint + str(suffix)
            suffix += 1
        self._USED_NAMES.add(name)
        super().__init__("placeholder", shape, [], in_mem, name=name, is_weight=is_weight)

    def _measure_latency(self):
        return self.ZERO_LATENCY

    def _compute_key(self):
        return (
            self.tag
            + str(sorted(dict(in_mem=self.in_mem, is_weight=self["is_weight"]).items())),
            str(self.out_shape) + str(self.inp_shapes),
        )


class ComputeOp(Operator):
    def __init__(self, tag, out_shape, inps: List[Operator], **attrs) -> None:
        assert all(op.in_mem for op in inps)
        super().__init__(tag, out_shape, [op.out_shape for op in inps], True, **attrs)


class StoreOp(Operator):
    def __init__(self, inp: Operator, **attrs) -> None:
        assert inp.in_mem
        super().__init__("store", inp.out_shape, [inp.out_shape], False, **attrs)
        self._dim_links = [[(i, i) for i in range(len(inp.out_shape))]]


class LoadOp(Operator):
    def __init__(self, inp: Operator, **attrs) -> None:
        assert not inp.in_mem
        super().__init__("load", inp.out_shape, [inp.out_shape], True, **attrs)
        self._dim_links = [[(i, i) for i in range(len(inp.out_shape))]]


class IndexOutOp(ComputeOp):
    def __init__(self, inp: Operator, index, dim_links, **attrs) -> None:
        out_shapes = inp.multi_out_shapes()
        assert out_shapes is not None
        super().__init__(
            "index_out",
            out_shapes[index],
            [inp],
            index=index,
            dim_links=dim_links,
            **attrs,
        )
        self._dim_links = [list(dim_links)]

    def backward(
        self, out_grad: Operator, inps: List[Operator]
    ) -> Tuple[List[Operator], List[Tuple[Operator, List[Operator]]]]:
        out_grad.attrs["grad_index"] = self.attrs["index"]
        return [out_grad], []

    def _measure_latency(self):
        return self.ZERO_LATENCY
