"""Tests for `normalize`, the int | Sequence[int] parameter adapter.

Split out from test_oracle.py because `normalize` is not conv geometry -- it is
an interface adapter that `conv.py` needs as much as the oracle does, and it is
headed for `utils.py` (dependency direction `oracle -> utils <- conv`). When it
moves, only the import below changes.
"""

import pytest

from minispconv.oracle import normalize


class TestNormalize:
    def test_int_expands(self):
        assert normalize(3, 3) == (3, 3, 3)

    def test_sequence_passes_through(self):
        assert normalize([1, 2, 3], 3) == (1, 2, 3)

    def test_tuple_passes_through(self):
        assert normalize((1, 2, 3), 3) == (1, 2, 3)

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="bool"):
            normalize(True, 3)

    def test_rejects_str(self):
        with pytest.raises(TypeError, match="string"):
            normalize("same", 3)  # type: ignore[arg-type]

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="expected length 3"):
            normalize([1, 2], 3)


class TestNormalizeArrayTypes:
    """Geometry parameters mirror torch.nn.Conv3d's signature: int | tuple.

    Arrays and tensors are rejected on purpose. `collections.abc.Sequence`
    membership is granted by explicit ABC registration, and neither numpy nor
    torch ever registered theirs -- so the existing isinstance check already
    turns them away. A tensor reaching here almost always means a bug upstream,
    and failing at the boundary beats silently iterating it.

    torch / numpy are imported inside each test: the oracle promises zero
    torch/numpy dependency, and its test module should not undo that.
    """

    def test_rejects_numpy_array(self):
        np = pytest.importorskip("numpy")
        with pytest.raises(TypeError, match="got ndarray"):
            normalize(np.array([1, 2, 3]), 3)

    def test_rejects_torch_tensor(self):
        torch = pytest.importorskip("torch")
        with pytest.raises(TypeError, match="got Tensor"):
            normalize(torch.tensor([1, 2, 3]), 3)

    def test_rejects_zero_dim_tensor(self):
        """A 0-dim tensor reads as a scalar but is still not an int."""
        torch = pytest.importorskip("torch")
        with pytest.raises(TypeError, match="got Tensor"):
            normalize(torch.tensor(3), 3)

    def test_accepts_torch_size(self):
        """torch.Size subclasses tuple, so it IS a Sequence and passes.

        This is what actually reaches normalize in practice: `tensor.shape` is
        a torch.Size, not a tensor. The elements come back as plain Python ints,
        which is the boundary conversion the oracle relies on.
        """
        torch = pytest.importorskip("torch")
        got = normalize(torch.Size([1, 2, 3]), 3)
        assert got == (1, 2, 3)
        assert all(type(v) is int for v in got)

    def test_rejects_numpy_integer_scalar(self):
        """Known rough edge, not a settled decision.

        np.int64 is semantically an integer but does not subclass Python's int,
        so `isinstance(param, int)` misses it and it falls through to the
        Sequence check, failing with a confusing "got int64". Accepting it would
        mean switching that branch to the __index__ protocol.
        """
        np = pytest.importorskip("numpy")
        with pytest.raises(TypeError, match="got int64"):
            normalize(np.int64(3), 3)
