# Thin-feature campaign

**Target not achieved.** Completed 54/54 independent confirmation trajectories. 
Selected configuration: `{'model': 'siren', 'width': 128, 'depth': 3}`; development selection inconclusive (frozen SIREN fallback).

| Arm | Persistent successes /18 | Full pipeline seconds, all attempts | Seconds per persistent success |
|---|---:|---:|---:|
| dense | 12 | 4841.2 | 403.43048231666035 |
| random | 12 | 3909.8 | 325.81935986666457 |
| cycling | 12 | 3582.3 | 298.52857390832895 |

| Family | Dense successes | Random successes | Cycling successes |
|---|---:|---:|---:|
| parallel_plates | 3 | 3 | 3 |
| mixed_thickness | 9 | 9 | 9 |
| multiscale_plate_stack | 0 | 0 | 0 |

Cycling/dense actual pipeline ratio: 0.7399752547057367. Lost dense successes: 0. Narrow-stack six-pair gate with error guards: False.

Missing, failed, unresolved and nonfinite evaluations fail closed. Failed attempts remain in pipeline totals. Unstarted trajectories are explicitly censored in trajectories.csv and prevent a complete-evidence claim.

Persistent success requires both 26k and 30k, both grids 192 and 384, every original probe sign, closed mesh edges, expected components/Euler and resolved analytic references. Onset at23k requires all three checkpoints; onset26k requires26k+30k. These are retrospective checkpoint-limited observations, not online stopping.

Runtime includes independently repeated prefixes, pool/label generation, random schedules, evaluation, transfers and checkpoint/hash I/O. Worker cumulative onset time includes supervisor launch and interpreter startup; final closure overhead appears in full pipeline totals. Historical attributed savings are separate and do not establish this target. Historical error guards compare endpoint means across the full paired set; per-pair guards are separately reported, with strict narrow-stack preservation.

Scope: oracle analytic box unions in three families. Known regions and exact SDF labels are used. This is not arbitrary unseen-shape evidence or a topology guarantee. Fourier features test bandwidth; they do not promise topology repair.

Random is evaluated identically. If random matches cycling, these results do not establish balancing necessity or superiority.

See protocol.json, audit.json, budget.json, paired-results.csv and trajectories.csv for immutable setup, failures and measurements.
