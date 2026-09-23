"""Smoke test for the spconv reference environment (ROADMAP S0.3).

Runs only in the `spconv` pixi environment; elsewhere the module skips.
"""

import pytest
import torch
from conftest import random_point_cloud

spconv = pytest.importorskip("spconv.pytorch", reason="spconv only in the `spconv` env")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="spconv needs a GPU")

SHAPE = (8, 8, 8)
IN_CH, OUT_CH = 4, 8


def test_subm_conv3d_forward_runs():
    rows = random_point_cloud(SHAPE, density=0.1, seed=0, batch_size=2)
    indices = torch.tensor(rows, dtype=torch.int32, device="cuda")
    features = torch.randn(len(rows), IN_CH, device="cuda")

    tensor = spconv.SparseConvTensor(features, indices, SHAPE, 2)
    net = spconv.SubMConv3d(IN_CH, OUT_CH, 3).cuda()
    out = net(tensor)

    assert out.features.shape == (len(rows), OUT_CH)
    assert torch.equal(out.indices, indices)
