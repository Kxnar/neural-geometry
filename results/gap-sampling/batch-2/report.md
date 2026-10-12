# Sampling batch 2

Cycling 4096-point balanced blocks through the exact dense pool retains persistent structure at one sixteenth of dense point presentations. Coverage over successive updates, rather than repeatedly fitting a fixed sparse pool, may explain the improvement over the historical static 4096-point failure.

Frozen before execution: [manifest](manifest.json). Audit: [audit](audit.json). All results use ordinary MSE and the unchanged two-resolution structural gate. Fresh cases remain reserved.

| Arm | Persistent passes | Presentations per trajectory | Training seconds (six trajectories) | Evaluation seconds |
|---|---:|---:|---:|---:|
| cycle-4096 | 5/6 | 40,960,000 | 198.79 | 198.06 |

| Sampler | Seed | Arm | Gate 23k / 26k / 30k | Persistent |
|---|---:|---|---|---|
| global_narrow | 31 | cycle-4096 | fail / pass / pass | True |
| global_narrow | 32 | cycle-4096 | fail / pass / pass | True |
| global_narrow | 33 | cycle-4096 | fail / pass / pass | True |
| local | 31 | cycle-4096 | pass / pass / pass | True |
| local | 32 | cycle-4096 | fail / pass / pass | True |
| local | 33 | cycle-4096 | fail / fail / pass | False |

## Frozen decision

- cycle-4096: qualifies=False; 5/6 persistent; training-time ratio 0.299; presentation ratio 0.0625. Guardrail values at both later checkpoints are in [decision.json](decision.json).

All comparisons to dense use the six batch-1 controls; later batches are sequential, not simultaneous timing measurements. Training times include synchronized updates and logging; evaluation is measured separately. Full-process budget charges additionally include preparation, hashing and I/O. Pool fit is evaluated on all 65,536 pool points even when only a subset is presented each update.

## Limits and reproduction

One axis-aligned development geometry, two original samplers and three paired seeds each. The same seeds are reused adaptively across batches: this is development evidence. Cycling changes gradient noise and order; it retains all dense labels and storage. Persistent means observed at 26k and 30k, not a guarantee between checkpoints or in continuous space. Small independent sign errors and strict pool-fit failures remain visible in the complete metrics.

From the original neural-geometry checkout, use a fresh output directory and the frozen source snapshot. Parent checkpoints and saved pools are indexed by SHA-256. The budget-run command enforces the campaign limit; run is available for a separately authorised reproduction.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv/Scripts/python.exe -m scripts.gap_sampling_controls run --manifest docs/gap-sampling/batch-2.json --out runs/reproduce-sampling-batch-2
.venv/Scripts/python.exe -m scripts.report_gap_sampling_controls --out runs/gap-sampling/batch-2 --dest docs/gap-sampling/batch-2
```
