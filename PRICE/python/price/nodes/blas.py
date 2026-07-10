from .base import Operator, ComputeOp
from typing import List, Tuple


class MatmulOp(ComputeOp):
    """General (batch) matrix multiplication.

    Shapes: ``[*b, m, k] x [*b, k, n] -> [*b, m, n]``.  The leading ``*b`` part
    is treated as batch dimensions that are passed through unchanged.
    """

    def __init__(
        self,
        a: Operator,
        b: Operator,
        trans_a=False,
        trans_b=False,
        trans_c=False,
    ) -> None:
        if trans_a:
            *batch_a, k_dim, m_dim = a.out_shape
        else:
            *batch_a, m_dim, k_dim = a.out_shape
        if trans_b:
            *batch_b, n_dim, k1 = b.out_shape
        else:
            *batch_b, k1, n_dim = b.out_shape
        assert k_dim == k1
        assert tuple(batch_a) == tuple(batch_b)
        out_shape = (*batch_a, n_dim, m_dim) if trans_c else (*batch_a, m_dim, n_dim)

        super().__init__(
            "matmul",
            out_shape,
            [a, b],
            trans_a=trans_a,
            trans_b=trans_b,
            trans_c=trans_c,
        )

        n_batch = len(batch_a)
        self._dim_links = [
            [
                *((i, i) for i in range(n_batch)),
                (n_batch + int(trans_a), n_batch + int(trans_c)),
                (n_batch + int(not trans_a), -1),
            ],
            [
                *((i, i) for i in range(n_batch)),
                (n_batch + int(not trans_b), n_batch + int(not trans_c)),
                (n_batch + int(trans_b), -1),
            ],
        ]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        a, b = inps
        ta, tb, tc = (self.attrs[f"trans_{x}"] for x in "abc")
        a_grad = MatmulOp(out_grad, b, tc, not tb, ta)
        b_grad = MatmulOp(a, out_grad, not ta, tc, tb)
        return [a_grad, b_grad], [(a_grad, [out_grad, b]), (b_grad, [a, out_grad])]


class FlexMatmulOp(ComputeOp):
    """A more flexible (batch) matmul.

    It allows several batch dimensions *and* several reduce dimensions, i.e.
    ``[*b, *m, *k] x [*b, *k, *n] -> [*b, *m, *n]``.  The split between batch,
    reduce and the two free axes is given by ``n_batch_dims`` / ``n_reduce_dims``.
    """

    REUSE_PROF_CODE = False

    def __init__(
        self,
        a: Operator,
        b: Operator,
        trans_a=False,
        trans_b=False,
        trans_c=False,
        n_batch_dims=0,
        n_reduce_dims=1,
        a_pivots=None,
        b_pivots=None,
        **attrs,
    ) -> None:
        assert n_reduce_dims > 0
        assert n_batch_dims + n_reduce_dims < len(a.out_shape)
        assert n_batch_dims + n_reduce_dims < len(b.out_shape)
        batch1 = a.out_shape[:n_batch_dims]
        batch2 = b.out_shape[:n_batch_dims]
        assert tuple(batch1) == tuple(batch2)
        if trans_a:
            ks1 = a.out_shape[n_batch_dims : n_batch_dims + n_reduce_dims]
            m_dims = a.out_shape[n_batch_dims + n_reduce_dims :]
        else:
            m_dims = a.out_shape[n_batch_dims:-n_reduce_dims]
            ks1 = a.out_shape[-n_reduce_dims:]
        if trans_b:
            n_dims = b.out_shape[n_batch_dims:-n_reduce_dims]
            ks2 = b.out_shape[-n_reduce_dims:]
        else:
            ks2 = b.out_shape[n_batch_dims : n_batch_dims + n_reduce_dims]
            n_dims = b.out_shape[n_batch_dims + n_reduce_dims :]
        assert tuple(ks1) == tuple(ks2)
        out_shape = (*batch1, *n_dims, *m_dims) if trans_c else (*batch1, *m_dims, *n_dims)

        n_batch = n_batch_dims
        n_reduce = n_reduce_dims
        n_m = len(a.out_shape) - n_batch - n_reduce
        n_n = len(b.out_shape) - n_batch - n_reduce

        a_pivot2 = n_batch + (n_reduce if trans_a else n_m)
        b_pivot2 = n_batch + (n_n if trans_b else n_reduce)
        a_pivots = a_pivots or (0, n_batch, a_pivot2, len(a.out_shape))
        b_pivots = b_pivots or (0, n_batch, b_pivot2, len(b.out_shape))

        am_base = n_batch + n_reduce * trans_a
        cm_base = n_batch + n_n * trans_c
        ak_base = n_batch + n_m * (not trans_a)
        bn_base = n_batch + n_reduce * (not trans_b)
        cn_base = n_batch + n_m * (not trans_c)
        bk_base = n_batch + n_n * trans_b

        super().__init__(
            "flex_matmul",
            out_shape,
            [a, b],
            trans_a=trans_a,
            trans_b=trans_b,
            trans_c=trans_c,
            n_batch_dims=n_batch_dims,
            n_reduce_dims=n_reduce_dims,
            a_pivots=a_pivots,
            b_pivots=b_pivots,
            **attrs,
        )

        self._dim_links = [
            [
                *((i, i) for i in range(n_batch)),
                *((am_base + i, cm_base + i) for i in range(n_m)),
                *((ak_base + i, -(i + 1)) for i in range(n_reduce)),
            ],
            [
                *((i, i) for i in range(n_batch)),
                *((bn_base + i, cn_base + i) for i in range(n_n)),
                *((bk_base + i, -(i + 1)) for i in range(n_reduce)),
            ],
        ]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        a, b = inps
        ta, tb, tc = (self.attrs[f"trans_{x}"] for x in "abc")
        n_batch = self.attrs["n_batch_dims"]
        n_reduce = self.attrs["n_reduce_dims"]
        n_m = len(a.out_shape) - n_batch - n_reduce
        n_n = len(b.out_shape) - n_batch - n_reduce
        a_grad = FlexMatmulOp(out_grad, b, tc, not tb, ta, n_batch, n_n)
        b_grad = FlexMatmulOp(a, out_grad, not ta, tc, tb, n_batch, n_m)
        return [a_grad, b_grad], [(a_grad, [out_grad, b]), (b_grad, [a, out_grad])]
