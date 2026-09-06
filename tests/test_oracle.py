"""Oracle tests: the hand-computed 2D case, plus the invariants that separate
SubM from regular conv.

Everything here is 2D on purpose -- the oracle is NDim-generic, so a case a
human can verify on paper also exercises the code path the 3D implementation
will be checked against.
"""

from itertools import product

import pytest

from minispconv.oracle import Pair, Rulebook, build_rulebook, out_spatial_shape


def _triples(rb: Rulebook) -> set[tuple]:
    """Rulebook -> the (tap, in_pos, out_pos) form the fixtures are written in.

    Drops the batch index, which every hand-computed case leaves at 0; the tests
    that care about batching check it separately.
    """
    return {(p.tap, p.in_pos, p.out_pos) for p in rb}


def test_regular_matches_hand_computed_pairs(hand_2d: dict):
    """Every pair, item by item -- not just the count (ROADMAP S0.1 acceptance)."""
    rb = build_rulebook(**hand_2d["build_rulebook_args"])
    assert _triples(rb) == hand_2d["pairs"]
    assert all(p.batch == 0 for p in rb)


def test_regular_output_sites_do_not_fill_the_grid(hand_2d: dict):
    """Regular conv dilates the active set; it does not densify the tensor.

    A rulebook whose outputs covered all 25 cells would mean the geometry had
    stopped depending on the input at all -- which is exactly what a bug that
    ignores the active set looks like.
    """
    rb = build_rulebook(**hand_2d["build_rulebook_args"])
    out_sites = {p.out_pos for p in rb}
    assert out_sites == hand_2d["active"]
    assert out_sites < set(product(range(5), range(5)))


def test_subm_matches_hand_computed_pairs(hand_2d: dict):
    """Every pair, item by item -- not just the count (ROADMAP S0.1 acceptance)."""
    rb = build_rulebook(**hand_2d["build_rulebook_args"] | {"subm": True})
    assert _triples(rb) == hand_2d["subm_pairs"]
    assert all(p.batch == 0 for p in rb)


def test_subm_center_tap_is_identity(hand_2d: dict):
    """The center tap must pair every active site with itself, exactly once.

    Catches an off-by-one in the in_pos formula: any shift moves the center tap
    off the diagonal.
    """
    rb = build_rulebook(**hand_2d["build_rulebook_args"] | {"subm": True})
    center = [p for p in rb if p.tap == (1, 1)]
    assert len(center) == len(hand_2d["subm_active"])
    assert all(p.in_pos == p.out_pos for p in center)


def test_subm_output_sites_are_the_input_sites(hand_2d: dict):
    """SubM's defining property, and the only one that separates it from regular.

    Without this, a build_rulebook that ignores `subm` still passes the count
    checks on small examples, because the two pair sets largely overlap.
    """
    rb = build_rulebook(**hand_2d["build_rulebook_args"] | {"subm": True})
    assert {p.out_pos for p in rb} <= hand_2d["subm_active"]


def test_subm_is_regular_restricted_to_active_sites(hand_2d: dict):
    """The exact relationship between the two modes, not merely that they differ.

    `subm < regular` alone would still pass for a build_rulebook that dropped
    the wrong pairs. Pinning the restriction says which ones: SubM keeps every
    regular pair whose output lands on an active site, and no others -- which is
    the whole of what the `subm` flag is allowed to change.
    """
    args = hand_2d["build_rulebook_args"]
    subm = build_rulebook(**args | {"subm": True})
    regular = build_rulebook(**args)

    assert subm < regular
    assert _triples(subm) == {t for t in _triples(regular) if t[2] in hand_2d["subm_active"]}


def test_batches_stay_independent():
    """The same coordinate in two batches must not produce cross-batch pairs."""
    rb = build_rulebook([[0, 1, 1], [1, 1, 1]], (5, 5), 3, padding=1, subm=True)
    assert len(rb) == 2
    assert {p.batch for p in rb} == {0, 1}
    assert all(p.tap == (1, 1) for p in rb)  # isolated points: center tap only


def test_stride_shrinks_the_output_grid(hand_2d: dict):
    rb = build_rulebook(**hand_2d["build_rulebook_args"] | {"stride": 2})
    out_shape = out_spatial_shape((5, 5), (3, 3), (2, 2), (1, 1), (1, 1))
    assert out_shape == (3, 3)
    assert all(0 <= c < 3 for p in rb for c in p.out_pos)


def test_dilation_changes_tap_reach():
    """Dilation spreads the taps out; it does not simply add pairs.

    Two points exactly 2 cells apart are neighbours only when d=2 -- with d=1
    the taps cannot reach that far, and with a denser cloud d=2 would find
    *fewer* pairs than d=1. Reach is the property, count is not.
    """
    idx = [[0, 1, 1], [0, 1, 3]]  # spaced 2 apart along the last axis
    near = build_rulebook(idx, (5, 5), 3, padding=1, dilation=1, subm=True)
    far = build_rulebook(idx, (5, 5), 3, padding=2, dilation=2, subm=True)

    assert {p for p in near if p.in_pos != p.out_pos} == set()
    assert len({p for p in far if p.in_pos != p.out_pos}) == 2


def test_padding_zero_drops_border_pairs(hand_2d: dict):
    """No padding shrinks the output grid, so border sites lose their pairs."""
    padded = build_rulebook(**hand_2d["build_rulebook_args"])
    bare = build_rulebook(**hand_2d["build_rulebook_args"] | {"padding": 0})
    assert len(bare) < len(padded)


def test_oracle_is_ndim_generic():
    """One code path, three dimensionalities -- the point of keeping it generic."""
    for shape, idx in (
        ((5, 5), [[0, 2, 2]]),
        ((5, 5, 5), [[0, 2, 2, 2]]),
        ((4, 4, 4, 4), [[0, 2, 2, 2, 2]]),
    ):
        rb = build_rulebook(idx, shape, 3, padding=1, subm=True)
        assert len(rb) == 1  # a lone point pairs with itself via the center tap
        (pair,) = rb
        assert len(pair.tap) == len(pair.in_pos) == len(pair.out_pos) == len(shape)


def test_empty_input_yields_empty_rulebook():
    assert build_rulebook([], (5, 5), 3, padding=1, subm=True) == frozenset()


class TestCanonicalForm:
    """Pair must stay hashable and orderable -- the whole comparison design
    rests on it (see docs/design/oracle.md)."""

    def test_pair_is_hashable(self):
        p = Pair(0, (1, 1), (2, 2), (3, 3))
        assert {p, Pair(0, (1, 1), (2, 2), (3, 3))} == {p}

    def test_pair_is_orderable(self):
        assert Pair(0, (0, 0), (0, 0), (0, 0)) < Pair(0, (0, 1), (0, 0), (0, 0))

    def test_pair_is_frozen(self):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            Pair(0, (1, 1), (2, 2), (3, 3)).batch = 1  # type: ignore[misc]

    def test_rulebook_is_order_independent(self):
        """Shuffling the input table must not change the canonical pair set."""
        a = build_rulebook([[0, 1, 1], [0, 1, 2]], (5, 5), 3, padding=1, subm=True)
        b = build_rulebook([[0, 1, 2], [0, 1, 1]], (5, 5), 3, padding=1, subm=True)
        assert a == b
