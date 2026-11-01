"""Deterministic BVSE rate-limiting hop on the periodic Li-site network (Gate 1).

Inputs are covariates only: nebBVSE122k hops (``edge_id``, ``material_id``, BVSE
barrier ``em``).  As documented by the LiTraj loader,
``edge_id = <material_id>_<source>_<target>_<ox>_<oy>_<oz>``, i.e. the hop goes from
``source`` to ``target`` shifted by the cell offset (ox, oy, oz).

* Physical hops.  ``s -> t + o`` and ``t -> s - o`` are the same hop.  Each hop has
  one canonical key (smaller site first; for s == t the lexicographically positive
  offset) and keeps every stored edge_id.  LiTraj's barrier is max - min image
  energy, which does not depend on direction, so rows of one hop with different
  barriers mean inconsistent data.  The material then fails closed
  (``direction_conflict``), never resolved by choice.
* Critical barrier.  E*_d(S) is the smallest barrier E at which hops with barrier
  <= E contain a d-dimensional periodic percolating component.  Dimensionality is
  the rank of the cell-offset vectors of independent cycles (union-find with offsets).
* Rate-limiting hop.  h*(S) is the first hop, in ascending (barrier, canonical key)
  order, whose addition makes the component d-dimensional (the minimax bottleneck).
  ``tied_hops`` lists every hop with barrier within TIE_TOLERANCE_EV of E* that would
  also complete percolation if added to the hops strictly below E*.  More than one
  such hop is a recorded tie; the choice follows the ordering above.
No DFT outcome is read.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

EDGE_ID = re.compile(r"^(?P<material>.+)_(?P<source>\d+)_(?P<target>\d+)"
                     r"_(?P<ox>-?\d+)_(?P<oy>-?\d+)_(?P<oz>-?\d+)$")
TIE_TOLERANCE_EV = 1e-9


def parse_edge_id(edge_id: str) -> dict[str, Any]:
    match = EDGE_ID.match(edge_id)
    if match is None:
        raise ValueError(f"unparseable edge_id {edge_id!r}")
    g = match.groupdict()
    return {"material_id": g["material"], "source": int(g["source"]), "target": int(g["target"]),
            "offset": (int(g["ox"]), int(g["oy"]), int(g["oz"]))}


def canonical_hop(source: int, target: int, offset: tuple[int, int, int]
                  ) -> tuple[int, int, tuple[int, int, int]]:
    """Direction-independent key of the physical hop s -> t + o."""

    negated = (-offset[0], -offset[1], -offset[2])
    if source > target or (source == target and negated > offset):
        return target, source, negated
    return source, target, offset


class _PeriodicUnionFind:
    """Union-find over Li sites tracking each site's cell offset relative to its root."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.shift: dict[int, np.ndarray] = {}
        self.cycles: dict[int, list[np.ndarray]] = {}

    def _find(self, site: int) -> tuple[int, np.ndarray]:
        if site not in self.parent:
            self.parent[site], self.shift[site], self.cycles[site] = site, np.zeros(3, int), []
        path = []
        node = site
        while self.parent[node] != node:  # iterative: no recursion limit on long chains
            path.append(node)
            node = self.parent[node]
        root = node
        total = np.zeros(3, int)
        for node in reversed(path):  # shifts accumulate from the root downwards
            total = total + self.shift[node]
            self.shift[node] = total.copy()
            self.parent[node] = root
        return root, (self.shift[site] if site != root else np.zeros(3, int))

    def add_edge(self, a: int, b: int, offset: tuple[int, int, int]) -> int:
        """Add edge a -> b + offset; return the dimensionality of the merged component."""

        root_a, shift_a = self._find(a)
        root_b, shift_b = self._find(b)
        offset = np.asarray(offset, int)
        if root_a == root_b:
            cycle = shift_a + offset - shift_b
            if cycle.any():
                self.cycles[root_a].append(cycle)
            return self.dimension(root_a)
        # Attach b's root under a's root so that position(b) = position(a) + offset.
        self.parent[root_b] = root_a
        self.shift[root_b] = shift_a + offset - shift_b
        self.cycles[root_a].extend(self.cycles.pop(root_b))
        return self.dimension(root_a)

    def dimension(self, root: int) -> int:
        vectors = self.cycles.get(root, [])
        return int(np.linalg.matrix_rank(np.asarray(vectors))) if vectors else 0


def physical_hops(hops: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Merge stored rows into physical hops; return (hops, failure status or None)."""

    merged: dict[tuple, dict[str, Any]] = {}
    materials = set()
    for row in hops:
        parsed = parse_edge_id(row["edge_id"])
        materials.add(parsed["material_id"])
        barrier = float(row["em"])
        if not np.isfinite(barrier):
            return [], "non_finite_barrier"
        key = canonical_hop(parsed["source"], parsed["target"], parsed["offset"])
        entry = merged.setdefault(key, {"key": key, "em": barrier, "edge_ids": []})
        if abs(entry["em"] - barrier) > TIE_TOLERANCE_EV:
            return [], "direction_conflict"
        entry["edge_ids"].append(row["edge_id"])
    if len(materials) > 1:
        raise ValueError(f"hops of more than one material: {sorted(materials)}")
    for entry in merged.values():
        entry["edge_ids"].sort()
    return sorted(merged.values(), key=lambda h: (h["em"], h["key"])), None


def _reaches(hops: list[dict[str, Any]], dimension: int) -> bool:
    network = _PeriodicUnionFind()
    reached = 0
    for hop in hops:
        source, target, offset = hop["key"]
        reached = max(reached, network.add_edge(source, target, offset))
    return reached >= dimension


def rate_limiting_hop(hops: Iterable[Mapping[str, Any]], dimension: int) -> dict[str, Any]:
    """h*(S) and E*_d(S) for one structure's hops (each with ``edge_id`` and ``em``)."""

    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be 1, 2 or 3")
    rows = list(hops)
    if not rows:
        return {"status": "no_hops"}
    merged, failure = physical_hops(rows)
    if failure is not None:
        return {"status": failure}
    network = _PeriodicUnionFind()
    for index, hop in enumerate(merged):
        source, target, offset = hop["key"]
        if network.add_edge(source, target, offset) >= dimension:
            critical = hop["em"]
            below = [h for h in merged if h["em"] < critical - TIE_TOLERANCE_EV]
            at_critical = [h for h in merged if abs(h["em"] - critical) <= TIE_TOLERANCE_EV]
            tied = [h for h in at_critical if _reaches(below + [h], dimension)]
            return {"status": "percolating", "dimension": dimension,
                    "critical_barrier_eV": critical,
                    "rate_limiting_hop_key": list(hop["key"][:2]) + list(hop["key"][2]),
                    "rate_limiting_edge_ids": hop["edge_ids"],
                    "tied_hops": [h["edge_ids"][0] for h in tied],
                    "n_physical_hops": len(merged), "n_rows": len(rows),
                    "hops_added": index + 1}
    return {"status": "no_percolating_network", "dimension": dimension,
            "n_physical_hops": len(merged), "n_rows": len(rows)}
