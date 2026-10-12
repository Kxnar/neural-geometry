# Sampling batch 3

The intermediate 8192-point cycle reduces gradient noise enough to retain the dense baseline's persistent structure and error guardrails, while using one eighth of its point presentations.

Frozen before execution: [manifest](manifest.json). Audit: [audit](audit.json). All results use ordinary MSE and the unchanged two-resolution structural gate. Fresh cases remain reserved.

| Arm | Persistent passes | Presentations per trajectory | Training seconds (six trajectories) | Evaluation seconds |
|---|---:|---:|---:|---:|
| cycle-8192 | 5/6 | 81,920,000 | 200.70 | 197.10 |

| Sampler | Seed | Arm | Gate 23k / 26k / 30k | Persistent |
|---|---:|---|---|---|
| global_narrow | 31 | cycle-8192 | fail / pass / pass | True |
| global_narrow | 32 | cycle-8192 | fail / pass / pass | True |
| global_narrow | 33 | cycle-8192 | pass / pass / pass | True |
| local | 31 | cycle-8192 | pass / pass / pass | True |
| local | 32 | cycle-8192 | pass / pass / pass | True |
| local | 33 | cycle-8192 | fail / fail / pass | False |

## Frozen decision

- cycle-8192: qualifies=False; 5/6 persistent; training-time ratio 0.302; presentation ratio 0.1250. Guardrail values at both later checkpoints are in [decision.json](decision.json).

All comparisons to dense use the six batch-1 controls; later batches are sequential, not simultaneous timing measurements. Training times include synchronized updates and logging; evaluation is measured separately. Full-process budget charges additionally include preparation, hashing and I/O. Pool fit is evaluated on all 65,536 pool points even when only a subset is presented each update.

## Limits and reproduction

One axis-aligned development geometry, two original samplers and three paired seeds each. The same seeds are reused adaptively across batches: this is development evidence. Cycling changes gradient noise and order; it retains all dense labels and storage. Persistent means observed at 26k and 30k, not a guarantee between checkpoints or in continuous space. Small independent sign errors and strict pool-fit failures remain visible in the complete metrics.

From the original neural-geometry checkout, use a fresh output directory and the frozen source snapshot. Parent checkpoints and saved pools are indexed by SHA-256. The budget-run command enforces the campaign limit; run is available for a separately authorised reproduction.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv/Scripts/python.exe -m scripts.gap_sampling_controls run --manifest docs/gap-sampling/batch-3.json --out runs/reproduce-sampling-batch-3
.venv/Scripts/python.exe -m scripts.report_gap_sampling_controls --out runs/gap-sampling/batch-3 --dest docs/gap-sampling/batch-3
```
