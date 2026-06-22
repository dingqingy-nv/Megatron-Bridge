# DeepSeek-V4-Pro performance recipe (Megatron-Bridge)

Reproduces the mcore-direct training performance of DeepSeek-V4-Pro on NVIDIA
GB200 / GB300, using the Megatron-Bridge perf recipe with a bind-mounted dev
Megatron-Core backbone. Config matches the mcore-direct reference (MLA + DSA/CSA
sparse attention, hash-MoE, MTP=1, mHC, full-iteration CUDA graph, fused DSA,
cuteDSL fused grouped MLP, FGO, MXFP8 compute).

## Prerequisites

- Bridge venv with `nemo_run` (Bridge code rides in via `PYTHONPATH`).
- Dev Megatron-Core worktree to bind-mount (set `MCORE_DEV` in the launcher).
- Container: `nemo_fht_dsa.sqsh` (provides fast_hadamard_transform + DSA kernels);
  set `CONT` in the launcher.
- `HF_TOKEN`, `WB_TOKEN` exported in the environment.

## How to run

Both launchers wrap `scripts.performance.setup_experiment` (nemo_run + SLURM).
Use `DRY=1` to print the command without submitting.

Full 61L on GB300 (256 GPU / 64 nodes):
```bash
DRY=1 bash submit_dsv4_pro_gb300.sh   # preview
bash     submit_dsv4_pro_gb300.sh     # submit
```

8L proxy on GB200 (64 GPU / 16 nodes) — fast pipeclean / perf check:
```bash
bash submit_dsv4_pro_proxy_gb200.sh
```

Override `CONT` / `MCORE_DEV` / `ACCOUNT` as needed (they have sensible defaults
in the launchers).

## Expected numbers (MXFP8, full-iter CG)

| config | GPUs | steady iter time | calibrated TF/s/GPU |
|--------|------|------------------|---------------------|
| 8L proxy (GB200)   | 64  | —          | ~950 (matches mcore-direct ~917) |
| Full 61L (GB300)   | 256 | ~21.5 s    | ~996 (matches mcore-direct ~967) |

Report **calibrated** TFLOP/s/GPU via `deepseek_next_tflops_calculator.py`
(`Calculated TFLOP/s per GPU` line). The recipe's in-log `MODEL_TFLOP/s/GPU`
(~974 at full scale) is a close DSv4-aware estimate; mcore's generic in-log
throughput over-reports and should not be used for DSv4.
