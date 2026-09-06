"""Brute-force rulebook oracle -- the single source of truth for this project.

Enumerates in (output, kernel) space with an independently derived formula,
deliberately sharing no derivation with the implementation (which enumerates in
kernel space and solves back for the output position). Returns the mathematical
set of pairs only, never a physical rulebook layout.

Stays NDim-generic: pure Python, speed is irrelevant, and genericity lets a
hand-computed 2D example validate the 3D implementation.

See docs/design/oracle.md for the full rationale.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

# ---------------------------------------------------------------------------
# canonical form -- the shared interface for every comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True, slots=True)
class Pair:
    """One rulebook entry: an input contributing to an output via a kernel tap.

    Positions are absolute coordinates, not active-table indices, so that two
    implementations with different active-table orderings still compare equal.
    `frozen=True` keeps it hashable for set comparison; `order=True` keeps it
    sortable for the L3.d determinism mode.
    """

    batch: int
    tap: tuple[int, ...]  # kernel tap index, 0..ks-1 per axis
    in_pos: tuple[int, ...]  # input spatial coordinate (batch excluded)
    out_pos: tuple[int, ...]  # output spatial coordinate (batch excluded)

    def __post_init__(self) -> None:
        # A mismatched length means to_canonical() sliced the coordinate columns
        # wrong -- most likely it left the batch index inside in_pos / out_pos.
        if not (len(self.tap) == len(self.in_pos) == len(self.out_pos)):
            raise ValueError(
                f"ndim mismatch: tap={len(self.tap)}, "
                f"in_pos={len(self.in_pos)}, out_pos={len(self.out_pos)}"
            )


type Rulebook = frozenset[Pair]


def normalize(param: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Expand `int | Sequence[int]` into a tuple of length `ndim`.

    Idempotent, so multiple entry points may each call it without cost.
    Also used by conv.py.
    """
    # bool is a subclass of int; accepting it would silently mean 0 or 1
    if isinstance(param, bool):
        raise TypeError(f"expected int or sequence of int, got bool: {param!r}")
    if isinstance(param, int):
        return (param,) * ndim
    # str is a Sequence too -- without this guard "same" becomes ('s','a','m','e')
    if isinstance(param, str):
        raise TypeError(f"string parameter must be resolved before normalize(): {param!r}")
    if not isinstance(param, Sequence):
        raise TypeError(f"expected int or sequence of int, got {type(param).__name__}")
    if len(param) != ndim:
        raise ValueError(f"expected length {ndim}, got {len(param)}: {tuple(param)!r}")
    return tuple(int(v) for v in param)


def out_spatial_shape(
    spatial_shape: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> tuple[int, ...]:
    """Output spatial extent. Part of the spec, even though upstream has the
    caller compute it.

    All parameters must already be normalized to equal length. `ndim` is derived
    from `spatial_shape` rather than passed separately -- two sources of truth
    are two places to disagree.
    """
    lengths = {len(spatial_shape), len(kernel_size), len(stride), len(padding), len(dilation)}
    if len(lengths) != 1:
        raise ValueError(
            f"length mismatch: spatial_shape={len(spatial_shape)}, "
            f"kernel_size={len(kernel_size)}, stride={len(stride)}, "
            f"padding={len(padding)}, dilation={len(dilation)}"
        )

    return tuple(
        (x + 2 * p - d * (ks - 1) - 1) // s + 1
        for x, ks, s, p, d in zip(spatial_shape, kernel_size, stride, padding, dilation)
    )


def build_rulebook(
    indices: Sequence[Sequence[int]],
    spatial_shape: Sequence[int],
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    subm: bool = False,
) -> Rulebook:
    """Exhaustively enumerate (output, kernel) space into a canonical pair set.

    Args:
        indices: `[N, ndim+1]`, column 0 is the batch index (the whole batch is
            flattened into one coordinate table).
        subm: when True the output coordinate set is identical to the input set,
            keeping only pairs that land inside it. SubM and regular share one
            core loop; only where the output coordinate set comes from differs.
            SubM also pins the other parameters: stride must be 1, kernel_size
            must be odd, and padding is determined by dilation * (ks - 1) // 2.
            See docs/design/subm.md for the derivation -- this function does not
            enforce them, so a caller passing anything else gets a well-formed
            but meaningless pair set.
    """
    ndim = len(spatial_shape)
    shape = tuple(int(v) for v in spatial_shape)
    ks = normalize(kernel_size, ndim)
    st = normalize(stride, ndim)
    pd = normalize(padding, ndim)
    dl = normalize(dilation, ndim)

    # Active input set, converted to plain Python ints at the boundary so the
    # oracle has zero torch / numpy dependency downstream.
    active: set[tuple[int, tuple[int, ...]]] = {
        (int(row[0]), tuple(int(v) for v in row[1:])) for row in indices
    }
    if any(len(coord) != ndim for _, coord in active):
        raise ValueError(f"indices rows must have {ndim + 1} columns (batch + {ndim} coords)")

    out_shape = out_spatial_shape(shape, ks, st, pd, dl)
    batches = {b for b, _ in active}

    pairs: set[Pair] = set()
    for b in batches:
        if subm:
            out = [c for bb, c in active if bb == b]
        else:
            out = product(*(range(x) for x in out_shape))
        for o in out:
            for k in product(*(range(x) for x in ks)):
                i = tuple(o[d] * st[d] - pd[d] + k[d] * dl[d] for d in range(ndim))
                if (b, i) in active:
                    pairs.add(Pair(batch=b, tap=k, in_pos=i, out_pos=o))

    return frozenset(pairs)
