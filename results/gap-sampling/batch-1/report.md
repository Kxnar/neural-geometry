# Sampling batch 1

Cycling balanced 16384-point subsets through the same 65536-point dense pool preserves observed persistent structure while reducing point presentations fourfold and measured training time.

Frozen before execution: [manifest](manifest.json). Audit: [audit](audit.json). All results use ordinary MSE and the unchanged two-resolution structural gate. Fresh cases remain reserved.

| Arm | Persistent passes | Presentations per trajectory | Training seconds (six trajectories) | Evaluation seconds |
|---|---:|---:|---:|---:|
| dense | 6/6 | 655,360,000 | 664.94 | 227.59 |
| cycle-16384 | 6/6 | 163,840,000 | 256.55 | 236.94 |

| Sampler | Seed | Arm | Gate 23k / 26k / 30k | Persistent |
|---|---:|---|---|---|
| global_narrow | 31 | dense | pass / pass / pass | True |
| global_narrow | 31 | cycle-16384 | pass / pass / pass | True |
| global_narrow | 32 | dense | pass / pass / pass | True |
| global_narrow | 32 | cycle-16384 | pass / pass / pass | True |
| global_narrow | 33 | dense | fail / pass / pass | True |
| global_narrow | 33 | cycle-16384 | pass / pass / pass | True |
| local | 31 | dense | pass / pass / pass | True |
| local | 31 | cycle-16384 | pass / pass / pass | True |
| local | 32 | dense | pass / pass / pass | True |
| local | 32 | cycle-16384 | pass / pass / pass | True |
| local | 33 | dense | pass / pass / pass | True |
| local | 33 | cycle-16384 | fail / pass / pass | True |

## Frozen decision

- cycle-16384: qualifies=True; 6/6 persistent; training-time ratio 0.386; presentation ratio 0.2500. Guardrail values at both later checkpoints are in [decision.json](decision.json).

All comparisons to dense use the six batch-1 controls; later batches are sequential, not simultaneous timing measurements. Training times include synchronized updates and logging; evaluation is measured separately. Full-process budget charges additionally include preparation, hashing and I/O. Pool fit is evaluated on all 65,536 pool points even when only a subset is presented each update.

## Limits and reproduction

One axis-aligned development geometry, two original samplers and three paired seeds each. The same seeds are reused adaptively across batches: this is development evidence. Cycling changes gradient noise and order; it retains all dense labels and storage. Persistent means observed at 26k and 30k, not a guarantee between checkpoints or in continuous space. Small independent sign errors and strict pool-fit failures remain visible in the complete metrics.

From the original neural-geometry checkout, use a fresh output directory and the frozen source snapshot. Parent checkpoints and saved pools are indexed by SHA-256. The budget-run command enforces the campaign limit; run is available for a separately authorised reproduction.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv/Scripts/python.exe -m scripts.gap_sampling_controls run --manifest docs/gap-sampling/batch-1.json --out runs/reproduce-sampling-batch-1
.venv/Scripts/python.exe -m scripts.report_gap_sampling_controls --out runs/gap-sampling/batch-1 --dest docs/gap-sampling/batch-1
```
