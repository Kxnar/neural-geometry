# Do longer training and wider networks preserve the empty gap?

Completed 8 October 2026. Development experiment: parallel plates, gap 0.02, seeds 31–33. All planned runs and failures retained. This report separates duration, width and objective controls.

## Results at 10,000 updates

| Width | Loss | Sampler | Original gap accuracy | Dense gap accuracy | Surface distance | Structural success |
|---:|---|---|---:|---:|---:|---:|
| 64 | mse | global_narrow | 93.36% | 87.49% | 0.009112 | 0/3 |
| 64 | mse | local | 94.27% | 87.09% | 0.009264 | 0/3 |
| 128 | mse | global_narrow | 96.09% | 88.71% | 0.008591 | 0/3 |
| 128 | mse | local | 92.45% | 85.65% | 0.008853 | 0/3 |
| 128 | sign_margin | global_narrow | 97.40% | 90.87% | 0.008508 | 0/3 |
| 128 | sign_margin | local | 100.00% | 96.13% | 0.008334 | 0/3 |
| 128 | weighted_band | global_narrow | 100.00% | 99.32% | 0.008161 | 0/3 |
| 128 | weighted_band | local | 100.00% | 98.33% | 0.008212 | 0/3 |
| 128 | weighted_margin | global_narrow | 99.35% | 93.95% | 0.008303 | 0/3 |
| 128 | weighted_margin | local | 99.35% | 95.35% | 0.008554 | 0/3 |

## Decision

The six width-64 duration controls and six width-128 capacity controls did not meet the preregistered structural gate. Longer training improves average distance and many probe scores, but does not eliminate the gap failure in these settings. Increasing width from 64 (8,641 parameters) to 128 (33,665 parameters) also fails to reliably repair structure.

The sign-margin loss does not beat both weighted-MSE controls and plain MSE on structural success. Do not advance it as a successful method. The reserved fresh cases remain unevaluated.

This does not prove a representational impossibility or that optimisation has converged. The learning rate remains fixed at 1e-4 and the probe curves fluctuate. A learning-rate schedule is a separate, untested explanation; first observed perfect probes are not stable convergence.

## What was controlled

- Duration: six continuous 10,000-update trajectories, saving weights, Adam state and sampling RNG at 1,000, 3,000 and 10,000. Every 100 updates, original gap/plate probes are logged. All six 1,000-update checkpoints exactly reproduce the original model tensors.
- Width: same two samplers and three seeds, same depth, batches, labels, learning rate and evaluations. Width alone changes; different width also changes optimisation and cannot isolate capacity by itself.
- Objective: 18 additional width-128 trajectories, three losses × two samplers × three seeds, each with 10,000 updates. Same model initialisation and point streams; full coordinate/label SHA256 agrees across loss arms and with deterministic replay of the plain-MSE sampling stream. Replay final RNG agrees with the saved baseline checkpoint.
- Weighted band: mean((1 + 8*[|d|<0.02])*(f-d)^2). Weighted margin: mean((1 + 8*(0.005/(0.005+|d|))^2)*(f-d)^2). Sign margin: MSE + 8*mean([|d|<0.02]*ReLU(0.5*|d|-sign(d)*f)^2). Hyperparameters were frozen before objective outcomes; no added gap points, resampling or loss-specific labels.
- Fresh cases: eight reserved cases × three new seeds (71–73), including unseen thicknesses, rotations, mixed-thickness boxes and three-plate stacks with multiple gaps. The reservation and its SHA256 predate the first long-control result. No method can advance without the frozen fresh-case gate.

## Measurement limits

Gap accuracy in the primary metric uses the original 256 mid-plane probes; dense accuracy uses 36,864 points through the central 80% of the gap and footprint. All plate-interior accuracies are in the per-seed table. Neither probe set certifies the full gap.

Mesh structure requires two components, Euler characteristic 4, closed edges and all original probe signs correct at grid 192, where the analytic reference resolves the gap. Grid 96 does not resolve this reference and is not used to blame the network. These diagnostics do not certify manifoldness or isotopy.

Surface distance is sampled, unsquared symmetric Chamfer (half the two directed means), with 32,768 samples per surface. The independent analytic sampling floor is approximately 0.007637; reported distances are not noise-free exact geometric distances. Three seeds provide descriptive evidence only.

Training timing includes periodic probes, and objective runs additionally hash points. Do not use these timings as a fair speed comparison. A loss improvement may also reflect loss scaling/Adam conditioning; this experiment does not uniquely identify a mechanism. Only one fixed setting of each loss was tested; a negative result does not reject every sign-aware objective.

## Evidence

- [Duration curves](long/convergence.png), [duration checkpoints](long/checkpoints.csv), [exact reproduction](long/decision.json).
- [Width curves](wide/convergence.png), [width checkpoints](wide/checkpoints.csv).
- [All final per-seed results](final-checkpoints.csv), [all-seed gap cross-sections](objective-gap-slices.png), [summary](final-summary.json), [frozen protocol](../gap-controls-protocol.md), [fresh-case reservation](fresh-cases.json).
- Full model/optimizer checkpoints, logs and source snapshots remain under the ignored `runs/gap-controls-*` and `runs/gap-objective-*` directories. Compact audits retain their hashes.

## Exploratory spatial diagnosis

Added after seeing perfect central probes coexist with connected plates; this is post-hoc explanation work, not a changed success endpoint. An independent seed (61,000) samples 131,072 points throughout the full gap footprint and width. Numerically ambiguous true margins <=1e-7 are excluded. The outer footprint is the region outside the central 80% of each plate axis.

| Width | Loss | Sampler | Full-gap sign accuracy | Central mid-plane | Outer mid-plane |
|---:|---|---|---:|---:|---:|
| 64 | mse | global_narrow | 76.82% | 93.43% | 68.38% |
| 64 | mse | local | 74.10% | 94.10% | 63.72% |
| 128 | mse | global_narrow | 84.15% | 95.63% | 86.42% |
| 128 | mse | local | 81.17% | 92.06% | 85.66% |
| 128 | sign_margin | global_narrow | 85.32% | 97.65% | 86.99% |
| 128 | sign_margin | local | 92.26% | 100.00% | 96.93% |
| 128 | weighted_band | global_narrow | 96.01% | 100.00% | 97.40% |
| 128 | weighted_band | local | 94.79% | 100.00% | 97.37% |
| 128 | weighted_margin | global_narrow | 84.96% | 99.38% | 82.89% |
| 128 | weighted_margin | local | 92.24% | 99.13% | 95.57% |

The full-gap test includes points close to the true surfaces, where the sign margin is small. Its score is therefore a stricter diagnostic than the central probe sets, not a directly interchangeable estimate. Region and margin-bucket errors are retained in [spatial diagnostics](spatial-diagnostics/summary.json), with source and checkpoint hashes. This localises errors but does not establish their optimisation cause.

## Post-hoc extraction sensitivity

All six weighted-band checkpoints were re-extracted at grid 384, because all six had perfect original gap probes but failed at 192. Analytic references resolved: 6/6. Expected mesh structure: 0/6. Single connected component: 6/6. Doubling extraction resolution therefore did not remove these failures. This supplementary check does not replace the frozen 192-grid endpoint or certify continuum topology. [Detailed evaluations](resolution-checks/summary.json).
