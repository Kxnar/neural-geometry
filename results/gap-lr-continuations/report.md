# Does learning-rate decay stabilise gap geometry?

Development only: ordinary-MSE SIREN width 128, parallel plates with gap 0.02, both samplers and seeds 31–33. All 12 matched continuations and their failures are retained. Fresh cases remain untouched.

Each pair loads the same 10,000-update model, Adam moments/step counts and sampling RNG. Each receives 10,000 further updates on byte-identical coordinates and labels. Constant Adam uses 1e-4; cosine uses 1e-4 on update 1 and 1e-6 on update 10,000. No optimiser restart at evaluation checkpoints.

## Checkpoint results

| Sampler | Schedule | Updates | Full-gap signs | Near-surface signs (≤.001) | Outer-footprint signs | Distance 192 / 384 | Gate at both grids |
|---|---|---:|---:|---:|---:|---:|---:|
| global_narrow | constant | 13000 | 87.84% | 69.08% | 80.36% | 0.008300 / 0.008259 | 0/3 |
| global_narrow | constant | 16000 | 91.40% | 71.02% | 89.42% | 0.008166 / 0.008126 | 0/3 |
| global_narrow | constant | 20000 | 93.66% | 71.66% | 91.03% | 0.008024 / 0.007994 | 0/3 |
| global_narrow | cosine | 13000 | 90.29% | 73.65% | 83.56% | 0.008180 / 0.008184 | 0/3 |
| global_narrow | cosine | 16000 | 94.43% | 77.61% | 91.25% | 0.007915 / 0.007917 | 0/3 |
| global_narrow | cosine | 20000 | 98.36% | 93.49% | 95.48% | 0.007766 / 0.007762 | 0/3 |
| local | constant | 13000 | 95.13% | 84.59% | 94.09% | 0.008445 / 0.008432 | 0/3 |
| local | constant | 16000 | 93.73% | 78.93% | 94.35% | 0.008328 / 0.008329 | 0/3 |
| local | constant | 20000 | 93.49% | 73.09% | 92.08% | 0.008096 / 0.008069 | 0/3 |
| local | cosine | 13000 | 94.89% | 82.27% | 91.37% | 0.008253 / 0.008287 | 0/3 |
| local | cosine | 16000 | 96.20% | 80.72% | 95.25% | 0.008005 / 0.007985 | 0/3 |
| local | cosine | 20000 | 99.16% | 94.83% | 97.78% | 0.007840 / 0.007850 | 0/3 |

## Persistence by trajectory

The gate retains all original probe signs and requires two components, Euler characteristic 4, closed edges and a resolved analytic reference at both grids. Full-gap signs are a separate, stricter sampled diagnostic. A sustained observed pass requires the 16,000 and 20,000 checkpoints; observations do not certify what happened between them.

| Sampler | Seed | Schedule | Gate 13k / 16k / 20k | Full-gap pass 13k / 16k / 20k | Sustained observed gate |
|---|---:|---|---|---|---|
| global_narrow | 31 | constant | fail / fail / fail | fail / fail / fail | no |
| global_narrow | 31 | cosine | fail / fail / fail | fail / fail / fail | no |
| global_narrow | 32 | constant | fail / fail / fail | fail / fail / fail | no |
| global_narrow | 32 | cosine | fail / fail / fail | fail / fail / fail | no |
| global_narrow | 33 | constant | fail / fail / fail | fail / fail / fail | no |
| global_narrow | 33 | cosine | fail / fail / fail | fail / fail / fail | no |
| local | 31 | constant | fail / fail / fail | fail / fail / fail | no |
| local | 31 | cosine | fail / fail / fail | fail / fail / fail | no |
| local | 32 | constant | fail / fail / fail | fail / fail / fail | no |
| local | 32 | cosine | fail / fail / fail | fail / fail / fail | no |
| local | 33 | constant | fail / fail / fail | fail / fail / fail | no |
| local | 33 | cosine | fail / fail / fail | fail / fail / fail | no |

## Decision

Decay did not consistently repair the gap under the frozen criterion (all six trajectories passing both the structural gate and sampled full-gap signs at both later checkpoints). The fixed-data diagnostic below was therefore run. No new loss is advanced.

At 20,000 updates, decay improves full-gap sign accuracy in 6/6 pairs, with a mean paired change of +5.18 percentage points. Mean paired sampled-distance changes are -0.000257 at 192 and -0.000226 at 384 (negative is better). Across all 36 continuation checkpoints, 0 satisfy the structural gate; 0/12 trajectories show sustained observed structural success.

Missing meshes: 0 across 108 checkpoint-resolution evaluations. Missing distances are excluded from distance means and paired distance differences; missing meshes always fail the structural gate.

## Fixed-data overfitting diagnostic

All six decayed 20,000-update checkpoints were continued with their Adam state on a saved 4,096-point dataset: 2,048 points from the original sampler, 1,024 uniform full-gap points, and 1,024 points inside the plates within 0.01 of the facing surfaces. Every update uses the entire fixed dataset and ordinary equal-point MSE, with the same 10,000-update cosine schedule. This intentionally changes coverage and removes stochastic sampling; it does not isolate either change alone.

Training-fit gate: every subset RMSE ≤0.001 and no wrong training signs where |true SDF|>0.001. Independent geometry and full-footprint checks reuse the frozen evaluation points, never the training dataset.

| Sampler | Seed | Original RMSE | Gap RMSE | Interior RMSE | Strong-margin training errors | Training fit gate | Independent gap signs | Geometry gate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| global_narrow | 31 | 0.000131884 | 0.000411757 | 0.000230536 | 1 | False | 94.53% | False |
| global_narrow | 32 | 0.000116083 | 0.00039582 | 0.000235481 | 0 | True | 94.60% | False |
| global_narrow | 33 | 0.000124059 | 0.000354005 | 0.000226312 | 0 | True | 94.84% | False |
| local | 31 | 0.000130152 | 0.000389656 | 0.000227718 | 0 | True | 95.09% | False |
| local | 32 | 0.000114956 | 0.000380012 | 0.000244271 | 0 | True | 94.00% | False |
| local | 33 | 0.000115652 | 0.000328011 | 0.000221194 | 0 | True | 95.54% | False |

At the final diagnostic checkpoint, 5/6 satisfy the fixed-training-set fit criterion and 0/6 satisfy the independent structural gate. Training fit and geometry are distinct outcomes. Failure under this finite budget and Adam schedule does not prove representational impossibility; passing finite training points does not certify the continuous field.

Mean fixed gap-training RMSE changes from 0.0027023 before the diagnostic to 0.000376543 after 10,000 full-batch updates. Full per-subset errors, sign margins and independent geometry at all three diagnostic checkpoints are in [fixed-data measurements](fixed-checkpoints.csv).

## Evidence and limits

- [All per-checkpoint measurements](checkpoints.csv), [paired cosine-minus-constant differences](paired-differences.csv), [persistence](persistence.json), [pair stream audit](pair-audit.json), [artifact audit](audit.json).
- [Gap stability curves](gap-stability.png), [distance-stratified errors](margin-errors.png), [frozen manifest](manifest.json), [exact source snapshots](source/scripts/gap_lr_controls.py).
- Full-gap probes: 131,072 requested uniform points across the entire 1.2×1.2 footprint and 0.02 gap. Exclude only numerical true margins ≤1e-7. Every region is also split into 0–.001, .001–.0025, .0025–.005 and .005–.01 true-distance buckets. The outer footprint is outside the central 80% along either axis.
- Surface distance is sampled unsquared symmetric Chamfer, half the two directed means, with 32,768 samples per surface. Sampling floor is retained per checkpoint; distances are not exact surface distances. Mesh diagnostics do not certify manifoldness or isotopy.
- Three seeds per sampler are descriptive evidence. The two grids and three evaluation checkpoints bound what was observed; they do not prove continuous-time or continuum correctness. Timings include diagnostics/stream hashing and are not speed benchmarks.
- Checkpoints, Adam state, RNG state, fixed datasets, partial attempts and source snapshots remain under `runs/gap-lr-continuations` in the original neural-geometry checkout. This export contains compact evidence and hashes.
- The existing sign penalty remains unpromoted, and no fresh geometry, seed or reserved case was evaluated.

## Reproduction

From the original neural-geometry checkout, with the six hashed baseline checkpoints present. The runner resumes verified completed segments and retains interrupted attempts. Use a new output directory to rerun from scratch.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv/Scripts/python.exe -m scripts.gap_lr_controls run --manifest docs/gap-controls/lr-continuations.json --out runs/gap-lr-continuations
.venv/Scripts/python.exe -m scripts.report_gap_lr_controls --out runs/gap-lr-continuations --dest docs/gap-lr-continuations
```
