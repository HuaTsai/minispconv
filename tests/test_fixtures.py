"""Contract tests for the 3D point cloud generator.

These fail until random_point_cloud() is implemented -- they are the spec.
"""

import pytest
from conftest import random_point_cloud

SHAPE = (8, 8, 8)


def test_same_seed_is_reproducible():
    """ROADMAP S0.2 acceptance: fixtures must be reproducible."""
    a = random_point_cloud(SHAPE, density=0.1, seed=42)
    b = random_point_cloud(SHAPE, density=0.1, seed=42)
    assert a == b


def test_different_seeds_differ():
    a = random_point_cloud(SHAPE, density=0.1, seed=1)
    b = random_point_cloud(SHAPE, density=0.1, seed=2)
    assert a != b


def test_no_duplicate_sites():
    """A duplicated (batch, coord) row is not a sparse tensor.

    The oracle builds a set from the table, so duplicates vanish silently --
    the implementation may not be so forgiving, and the two would disagree.
    """
    rows = random_point_cloud(SHAPE, density=0.3, seed=7, batch_size=3)
    keys = [(r[0], tuple(r[1:])) for r in rows]
    assert len(keys) == len(set(keys))


def test_coordinates_are_in_bounds():
    rows = random_point_cloud(SHAPE, density=0.3, seed=7, batch_size=2)
    assert all(0 <= c < s for r in rows for c, s in zip(r[1:], SHAPE))


def test_every_batch_is_populated():
    """An empty batch would let a per-batch bug hide."""
    rows = random_point_cloud(SHAPE, density=0.1, seed=3, batch_size=4)
    assert {r[0] for r in rows} == {0, 1, 2, 3}


def test_row_layout_matches_build_rulebook():
    """Rows are [batch, *coords] -- the layout build_rulebook documents."""
    rows = random_point_cloud(SHAPE, density=0.1, seed=5)
    assert all(len(r) == len(SHAPE) + 1 for r in rows)


def test_density_controls_how_many_points():
    sparse = random_point_cloud(SHAPE, density=0.05, seed=11)
    dense = random_point_cloud(SHAPE, density=0.5, seed=11)
    assert len(sparse) < len(dense)


@pytest.mark.parametrize("density", [0.05, 0.25, 0.6])
def test_feeds_the_oracle(density: float):
    """The generated table must actually drive build_rulebook end to end."""
    from minispconv.oracle import build_rulebook

    rows = random_point_cloud(SHAPE, density=density, seed=17, batch_size=2)
    rb = build_rulebook(rows, SHAPE, 3, padding=1, subm=True)
    assert {p.out_pos for p in rb} <= {tuple(r[1:]) for r in rows}
