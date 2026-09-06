"""Shared fixtures for the oracle and, later, the implementation.

Two fixture families with opposite trust directions (see docs/ROADMAP.md S0.2):

- The 2D example is hand-verifiable and feeds the ORACLE. Here a human is the
  second oracle, so the expected pairs are written out by hand below.
- The 3D generator feeds the IMPLEMENTATION, with the oracle as ground truth.
  Random input is meaningless for checking the oracle itself.
"""

from collections.abc import Sequence

import pytest


# ---------------------------------------------------------------------------
# 2D hand-computed example -- 5x5 grid, 4 active points, 3x3 kernel
# ---------------------------------------------------------------------------
@pytest.fixture
def hand_2d() -> dict:
    """The hand-computed 2D case: one argument dict, two expected answers.

    `build_rulebook_args` carries `subm: False`, so a test spells out only its
    deliberate deviation: `build_rulebook(**args | {"subm": True})`. Passing
    `subm=` as a separate keyword is a TypeError -- the intended failure.

    These are the only parameter values the pairs below were computed for, and
    SubM pins three of the four: `stride` must be 1, `kernel_size` must be odd,
    and `padding` is not free -- centring fixes it at
    `dilation * (kernel_size - 1) // 2`, here 1. build_rulebook enforces none of
    it; see docs/design/subm.md.
    """
    return {
        "build_rulebook_args": {
            "indices": ((0, 1, 1), (0, 1, 2), (0, 3, 3), (0, 2, 3)),
            "spatial_shape": (5, 5),
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "subm": False,
        },
        # Output sites reached by at least one pair: 20 of the 25 grid cells.
        # A proper subset, which is the point -- regular conv dilates the active
        # set but does not fill the grid. The five it misses are (0,4), (3,0),
        # (3,1), (4,0), (4,1), each further than one cell from every input.
        # fmt: off
        "active": {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 2),
            (4, 3),
            (4, 4),
        },
        # fmt: on
        # 36 pairs, grouped by INPUT -- the opposite direction from subm_pairs
        # below, which is grouped by output. Deriving the two sets the two ways
        # round keeps them independent: an error in one grouping does not
        # reproduce itself in the other.
        #
        # With s=1, p=1, d=1 the rule reads `tap = input_coor - output_coor + 1`,
        # so every active input scatters to the full 3x3 neighbourhood of output
        # sites around it. None of the four sits close enough to a border to
        # lose a tap, hence 4 x 9 with no exceptions.
        "pairs": {
            # format: (kernel_tap, input_coor, output_coor)
            # input_coor = (1,1)
            ((2, 2), (1, 1), (0, 0)),
            ((2, 1), (1, 1), (0, 1)),
            ((2, 0), (1, 1), (0, 2)),
            ((1, 2), (1, 1), (1, 0)),
            ((1, 1), (1, 1), (1, 1)),
            ((1, 0), (1, 1), (1, 2)),
            ((0, 2), (1, 1), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 0), (1, 1), (2, 2)),
            # input_coor = (1,2)
            ((2, 2), (1, 2), (0, 1)),
            ((2, 1), (1, 2), (0, 2)),
            ((2, 0), (1, 2), (0, 3)),
            ((1, 2), (1, 2), (1, 1)),
            ((1, 1), (1, 2), (1, 2)),
            ((1, 0), (1, 2), (1, 3)),
            ((0, 2), (1, 2), (2, 1)),
            ((0, 1), (1, 2), (2, 2)),
            ((0, 0), (1, 2), (2, 3)),
            # input_coor = (2,3)
            ((2, 2), (2, 3), (1, 2)),
            ((2, 1), (2, 3), (1, 3)),
            ((2, 0), (2, 3), (1, 4)),
            ((1, 2), (2, 3), (2, 2)),
            ((1, 1), (2, 3), (2, 3)),
            ((1, 0), (2, 3), (2, 4)),
            ((0, 2), (2, 3), (3, 2)),
            ((0, 1), (2, 3), (3, 3)),
            ((0, 0), (2, 3), (3, 4)),
            # input_coor = (3,3)
            ((2, 2), (3, 3), (2, 2)),
            ((2, 1), (3, 3), (2, 3)),
            ((2, 0), (3, 3), (2, 4)),
            ((1, 2), (3, 3), (3, 2)),
            ((1, 1), (3, 3), (3, 3)),
            ((1, 0), (3, 3), (3, 4)),
            ((0, 2), (3, 3), (4, 2)),
            ((0, 1), (3, 3), (4, 3)),
            ((0, 0), (3, 3), (4, 4)),
        },
        # A set, not a tuple: SubM's output sites ARE the input sites, and the
        # canonical form is deliberately unordered (docs/design/oracle.md), so
        # ordering here would assert something the oracle does not promise.
        "subm_active": {(1, 1), (1, 2), (3, 3), (2, 3)},
        "subm_pairs": {
            # format: (kernel_tap, input_coor, output_coor)
            # output_coor = (1,1)
            ((1, 1), (1, 1), (1, 1)),
            ((1, 2), (1, 2), (1, 1)),
            # output_coor = (1,2)
            ((1, 0), (1, 1), (1, 2)),
            ((1, 1), (1, 2), (1, 2)),
            ((2, 2), (2, 3), (1, 2)),
            # output_coor = (2,3)
            ((0, 0), (1, 2), (2, 3)),
            ((1, 1), (2, 3), (2, 3)),
            ((2, 1), (3, 3), (2, 3)),
            # output_coor = (3,3)
            ((0, 1), (2, 3), (3, 3)),
            ((1, 1), (3, 3), (3, 3)),
        },
    }


# ---------------------------------------------------------------------------
# 3D random point cloud generator
# ---------------------------------------------------------------------------


def random_point_cloud(
    spatial_shape: Sequence[int],
    density: float,
    seed: int,
    batch_size: int = 1,
) -> list[list[int]]:
    """Generate a reproducible random active-coordinate table.

    Returns rows of `[batch, *coords]`, the same layout `build_rulebook` takes.

    Contract this must satisfy (tests/test_fixtures.py checks all four):
      1. Same seed -> identical output, across processes and runs.
      2. No duplicate (batch, coord) rows -- an active table with duplicates is
         not a sparse tensor, and the oracle would silently dedupe it.
      3. Every coordinate is inside `spatial_shape`.
      4. Every batch index in `range(batch_size)` appears at least once, so a
         per-batch bug cannot hide behind an empty batch.

    Args:
        spatial_shape: grid extent per axis.
        density: fraction of grid cells that are active, in (0, 1].
        seed: PRNG seed; the only source of randomness.
        batch_size: number of batches sharing one flattened table.
    """
    # TODO(human)
    raise NotImplementedError
