import pytest
import torch

import minispconv


def test_import():
    assert minispconv.__version__


def test_torch_is_cuda_build():
    # this is a CUDA-kernel project: a CPU-only torch resolve is a broken env
    assert torch.version.cuda is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU / driver")
def test_gpu_tensor_roundtrip():
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    assert torch.equal((x + 1).cpu(), torch.arange(1, 9, dtype=torch.float32))
