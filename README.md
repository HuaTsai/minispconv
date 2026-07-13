# minispconv

A from-scratch reimplementation of [spconv](https://github.com/traveller59/spconv)-style
sparse convolution for 3D point clouds — built to understand every layer of the stack,
from rulebook construction down to hand-written CUDA kernels.

> **Status**: early scaffold. Roadmap below; no implementation yet.

## Why

Sparse convolution is the workhorse of LiDAR 3D perception (SECOND, CenterPoint, ...),
yet most users treat it as a black box. This project rebuilds it in stages, keeping each
stage numerically aligned with `traveller59/spconv` as the reference:

1. **L1 — reference implementation (pure PyTorch)**: sparse indices hashing, rulebook
   generation, gather–GEMM–scatter forward.
2. **L2 — CUDA**: hash-table and rulebook-generation kernels, forward v0
   (gather → cuBLAS GEMM → scatter).
3. **L3 — optimization**: memory coalescing, shared memory, Nsight-profiled iterations,
   benchmarked against spconv.

Forward pass is the focus; backward is a stretch goal.

## Setup

Requires [pixi](https://pixi.sh) and an NVIDIA GPU. Two environments are provided —
pick by your **driver** and **GPU generation** (the CUDA toolchain itself is installed
by pixi; you don't need CUDA on the system):

| env       | CUDA | driver | note                                            |
| --------- | ---- | ------ | ----------------------------------------------- |
| `default` | 13.x | r580+  | development mainline (Turing / sm_75 and newer) |
| `cu12`    | 12.x | r525+  | older drivers, or Pascal/Volta GPUs             |

CUDA 11 is intentionally unsupported: the PyTorch ecosystem no longer ships cu11 builds,
and this project uses PyTorch as its correctness reference.

## License

MIT
