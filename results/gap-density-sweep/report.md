# Does sparse training coverage cause the remaining geometry failures?

Coverage is supported by persistent restoration across all six pairs. Pursue coverage and sampling efficiency with ordinary MSE; dense-data residual fit remains unresolved.

Development geometry only: parallel plates, gap 0.02; ordinary-MSE SIREN width 128/depth 3; global_narrow and local samplers; seeds 31, 32, 33. All 18 arms and 54 checkpoints are retained. No new loss is advanced; fresh cases remain untouched.

| Fixed points | Persistent global_narrow | Persistent local | Final training-fit passes | Mean independent gap signs |
|---:|---:|---:|---:|---:|
| 4,096 | 0/3 | 0/3 | 5/6 | 94.77% |
| 16,384 | 0/3 | 2/3 | 0/6 | 99.42% |
| 65,536 | 3/3 | 3/3 | 0/6 | 99.71% |

Mean gap RMSE (training / independent) changes from 0.000376543 / 0.00290131 at 4,096 points to 0.00105195 / 0.00108283 at 65,536. Training error rises across densities while independent error falls and the train–independent gap narrows. This supports sparse-data overfitting as a contributor. Structural recovery is bounded by the existing gate: dense training fit and perfect independent signs are not achieved.

## Controlled intervention

Each density starts from its corresponding exact 20,000-update decayed model and Adam state. The 4,096-point dataset is byte-identical to the preceding diagnostic. Larger arms are nested prefixes of balanced blocks, with 50% original-sampler, 25% uniform full-gap and 25% facing-interior points. All arms receive 10,000 updates on the same endpoint-inclusive cosine schedule (1e-4 to 1e-6), with no optimiser reset between evaluation checkpoints.

Gradients accumulate chunk sums divided by the total number of target elements, then Adam steps once. Chunk size is 16,384; the two smaller arms use one chunk and the largest uses four. This is exact full-dataset ordinary MSE in the mathematical sense; floating-point accumulation can differ from a monolithic reduction.

Independent matched-distribution data use 65,536 points per sampler, common across seeds/densities, with zero exact-coordinate overlap with training data. The existing 131,072 requested full-gap probes and the original structural evaluator are also reused. Errors are stratified by absolute true SDF, for each subset and central/outer footprint; empty bins are missing, not zero.

## Checkpoint outcomes

| Sampler | Fixed points | Updates | Training fit | Gap train RMSE | Gap independent RMSE | Independent gap signs | Chamfer 192 / 384 | Structural gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_narrow | 4,096 | 23,000 | 0/3 | 0.000837435 | 0.00246675 | 97.05% | 0.013925 / 0.014032 | 0/3 |
| global_narrow | 4,096 | 26,000 | 1/3 | 0.000562895 | 0.00271965 | 95.32% | 0.015846 / 0.015927 | 0/3 |
| global_narrow | 4,096 | 30,000 | 2/3 | 0.000387194 | 0.00291174 | 94.66% | 0.016977 / 0.017077 | 0/3 |
| global_narrow | 16,384 | 23,000 | 0/3 | 0.00136129 | 0.00158009 | 99.81% | 0.008087 / 0.008128 | 0/3 |
| global_narrow | 16,384 | 26,000 | 0/3 | 0.00104949 | 0.00127783 | 99.62% | 0.008342 / 0.008351 | 0/3 |
| global_narrow | 16,384 | 30,000 | 0/3 | 0.000892443 | 0.00112256 | 99.40% | 0.008669 / 0.008729 | 0/3 |
| global_narrow | 65,536 | 23,000 | 0/3 | 0.0015049 | 0.00154164 | 99.82% | 0.007777 / 0.007801 | 2/3 |
| global_narrow | 65,536 | 26,000 | 0/3 | 0.00122789 | 0.00125832 | 99.66% | 0.007754 / 0.007757 | 3/3 |
| global_narrow | 65,536 | 30,000 | 0/3 | 0.00104988 | 0.00108142 | 99.69% | 0.007725 / 0.007744 | 3/3 |
| local | 4,096 | 23,000 | 0/3 | 0.00102756 | 0.00263578 | 95.34% | 0.013861 / 0.013951 | 0/3 |
| local | 4,096 | 26,000 | 1/3 | 0.000502526 | 0.00269077 | 95.84% | 0.014858 / 0.014929 | 0/3 |
| local | 4,096 | 30,000 | 3/3 | 0.000365893 | 0.00289088 | 94.87% | 0.015594 / 0.015644 | 0/3 |
| local | 16,384 | 23,000 | 0/3 | 0.00144823 | 0.00163756 | 99.74% | 0.008625 / 0.008680 | 3/3 |
| local | 16,384 | 26,000 | 0/3 | 0.00103491 | 0.00125513 | 99.63% | 0.008775 / 0.008831 | 3/3 |
| local | 16,384 | 30,000 | 0/3 | 0.000887614 | 0.00111277 | 99.43% | 0.008904 / 0.008943 | 2/3 |
| local | 65,536 | 23,000 | 0/3 | 0.00187501 | 0.00191027 | 98.52% | 0.007952 / 0.007932 | 3/3 |
| local | 65,536 | 26,000 | 0/3 | 0.00117647 | 0.00121011 | 99.80% | 0.007851 / 0.007870 | 3/3 |
| local | 65,536 | 30,000 | 0/3 | 0.00105402 | 0.00108424 | 99.72% | 0.007826 / 0.007851 | 3/3 |

Training-fit gate is unchanged: every original/gap/interior subset RMSE ≤0.001 and no wrong training signs with |true SDF|>0.001. Larger sets give a zero-error gate more opportunities to fail, so compare RMSE and error counts/denominators as well as binary fit. Means are descriptive seed means, not confidence intervals.

## Persistence and individual decisions

The structural gate retains all original probe signs and requires two components, Euler characteristic four, closed edges and a resolved analytic reference at BOTH 192 and 384. A persistent observed pass requires 26k and 30k; observations say nothing definitive between checkpoints. Full-gap signs remain a separate diagnostic.

| Sampler | Seed | Points | Gate 23k / 26k / 30k | Fit 23k / 26k / 30k | Interpretation |
|---|---:|---:|---|---|---|
| global_narrow | 31 | 4,096 | fail / fail / fail | fail / fail / fail | training fit unresolved; optimisation/capacity remain open |
| global_narrow | 31 | 16,384 | fail / fail / fail | fail / fail / fail | training fit unresolved; optimisation/capacity remain open |
| global_narrow | 31 | 65,536 | pass / pass / pass | fail / fail / fail | persistent structural pass |
| global_narrow | 32 | 4,096 | fail / fail / fail | fail / pass / pass | training fit passes; geometry unresolved |
| global_narrow | 32 | 16,384 | fail / fail / fail | fail / fail / fail | training fit unresolved; optimisation/capacity remain open |
| global_narrow | 32 | 65,536 | pass / pass / pass | fail / fail / fail | persistent structural pass |
| global_narrow | 33 | 4,096 | fail / fail / fail | fail / fail / pass | training fit passes; geometry unresolved |
| global_narrow | 33 | 16,384 | fail / fail / fail | fail / fail / fail | training fit unresolved; optimisation/capacity remain open |
| global_narrow | 33 | 65,536 | fail / pass / pass | fail / fail / fail | persistent structural pass |
| local | 31 | 4,096 | fail / fail / fail | fail / fail / pass | training fit passes; geometry unresolved |
| local | 31 | 16,384 | pass / pass / pass | fail / fail / fail | persistent structural pass |
| local | 31 | 65,536 | pass / pass / pass | fail / fail / fail | persistent structural pass |
| local | 32 | 4,096 | fail / fail / fail | fail / pass / pass | training fit passes; geometry unresolved |
| local | 32 | 16,384 | pass / pass / pass | fail / fail / fail | persistent structural pass |
| local | 32 | 65,536 | pass / pass / pass | fail / fail / fail | persistent structural pass |
| local | 33 | 4,096 | fail / fail / fail | fail / fail / pass | training fit passes; geometry unresolved |
| local | 33 | 16,384 | pass / pass / fail | fail / fail / fail | training fit unresolved; optimisation/capacity remain open |
| local | 33 | 65,536 | pass / pass / pass | fail / fail / fail | persistent structural pass |

Observed passing-to-failing transitions: 1.

- local, seed 33, 16,384 points: pass at 26,000, fail at 30,000; r192: components=3, Euler=6; r384: components=4, Euler=8.

A failed fit gate alone is not proof that optimisation stalled. The per-subset errors and the training history show whether error decreased; failure under this schedule/budget leaves optimisation and capacity unresolved. A passing finite-data fit alongside failed structure motivates bridge localisation between samples, but does not by itself prove where bridges form.

## Training progress despite strict fit failures

| Points | Subset | Mean initial RMSE | Mean final RMSE | RMSE improvement pairs | Final strong-margin errors / tested |
|---:|---|---:|---:|---:|---:|
| 4,096 | original | 0.00145891 | 0.000122131 | 6/6 | 0 / 11563 |
| 4,096 | gap | 0.0027023 | 0.000376543 | 6/6 | 0 / 5546 |
| 4,096 | interior | 0.00180566 | 0.000230919 | 6/6 | 1 / 5454 |
| 16,384 | original | 0.0014799 | 0.000374032 | 6/6 | 52 / 46211 |
| 16,384 | gap | 0.00270433 | 0.000890029 | 6/6 | 0 / 22196 |
| 16,384 | interior | 0.00184123 | 0.000592776 | 6/6 | 93 / 21978 |
| 65,536 | original | 0.00146703 | 0.000843664 | 6/6 | 718 / 185151 |
| 65,536 | gap | 0.00270454 | 0.00105195 | 6/6 | 0 / 88422 |
| 65,536 | interior | 0.00183693 | 0.000714392 | 6/6 | 643 / 88042 |

## Matched density differences at 30k

| Dense arm vs 4,096 | Gap signs improvement pairs | Mean sign change (pp) | Mean independent gap RMSE change | Mean Chamfer 192 / 384 change |
|---|---:|---:|---:|---:|
| 16,384 | 6/6 | +4.649 | -0.00178364 | -0.007499 / -0.007525 |
| 65,536 | 6/6 | +4.940 | -0.00181848 | -0.008510 / -0.008563 |

A density increase changes empirical coverage while keeping proportions, starting state, objective and update count fixed. Compute is deliberately not matched. Better sampled signs/distances without persistent structure are partial coverage benefits, not a geometric repair.

## Extra compute

| Points | Updates per arm | Presentations per arm | Work ratio | Six-arm training seconds | Measured time ratio | Evaluation seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 10,000 | 40,960,000 | 1x | 197.9 | 1.00x | 217.5 |
| 16,384 | 10,000 | 163,840,000 | 4x | 198.3 | 1.00x | 201.5 |
| 65,536 | 10,000 | 655,360,000 | 16x | 573.7 | 2.90x | 205.2 |

Completed-trajectory training presentations: 5,160,960,000. Measured completed training: 16.16 minutes; evaluation: 10.40 minutes. Training timers include full-loss logging/synchronisation and exclude dataset generation, checkpoint I/O and evaluation; evaluation timers include metric I/O. These are workload costs on one RTX 2080 Super, not a speed benchmark.

Interrupted attempts retained: 1. Their logs record at least 98,304,000 additional discarded point presentations, excluded from the matched completed-trajectory totals above. Each may have an unlogged tail of fewer than 100 completed updates plus a partial update. Their exact elapsed compute is unavailable; see [interruption audit](interrupted-attempts.json). Verified complete segments were reused and only interrupted segments restarted from the previous saved Adam checkpoint.

## Evidence and limits

- [Checkpoint measurements](checkpoints.csv), [all surface-distance bins](surface-distance-errors.csv), [20k baseline bins](baseline-errors.csv), [starting geometry](starting-geometry.json), [paired differences](paired-density-differences.csv), [compute](compute.csv), [persistence](persistence.json), [audit](audit.json), [frozen manifest](manifest.json).
- [Density curves](density-curves.png), [training versus independent surface-distance errors](surface-distance-errors.png).
- Missing meshes: 0 of 108 grid evaluations. Every absent/incorrect mesh fails the structural gate; missing distances are excluded from means.
- Sampled surface distance is unsquared symmetric Chamfer, half the directed means, using 32,768 surface points. Reference sampling floors remain recorded. This is not an exact surface distance or a topology certificate.
- Three paired seeds per sampler, one development geometry, one model capacity and one update budget cannot establish a universal cause. A fitted sparse arm and an unfitted dense arm do not cleanly distinguish interpolation from optimisation.
- Legacy 4,096-point control reproduction: maximum absolute parameter difference 0; exact Adam state match at 18/18 checkpoints.
- Full model/Adam/RNG states, nested and independent datasets, baseline errors, attempt folders and frozen sources are retained under runs/gap-density-sweep in the original checkout. This export is compact; the audit records their hashes.
- The audit verifies the fresh-case reservation hash and exact legacy 4,096-point prefix. No fresh cases or training seeds were evaluated. No new objective is promoted.

## Reproduction

From the original neural-geometry checkout, with the six verified decayed checkpoints present. Resume uses verified complete segments and retains interrupted attempts. Use a new output folder for a fresh execution.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv/Scripts/python.exe -m scripts.gap_density_controls run --manifest docs/gap-controls/density-sweep.json --out runs/gap-density-sweep
.venv/Scripts/python.exe -m scripts.report_gap_density_controls --out runs/gap-density-sweep --dest docs/gap-density-sweep
```
