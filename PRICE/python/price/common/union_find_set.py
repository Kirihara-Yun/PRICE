"""A tiny disjoint-set (union-find) structure.

It is implemented directly on top of a :class:`dict` so that any hashable value
can act as a node.  Path compression is applied on every :meth:`find_root` call.
"""
from typing import Dict, TypeVar

_VT = TypeVar("_VT")


class UnionFindSet(Dict[_VT, _VT]):
    def find_root(self, nd: _VT) -> _VT:
        parent = self[nd]
        if parent is None or isinstance(parent, (set, list)):
            return nd
        root = self.find_root(parent)
        self[nd] = root
        return root

    def union_trees(self, a: _VT, b: _VT):
        root_a, root_b = self.find_root(a), self.find_root(b)
        if root_a == root_b:
            return
        self[root_a] = root_b
