# Gap-preservation experiment

The latest [thin-feature campaign](results/thin-feature-campaign/report.md) completed
**54 independent trajectories** across six held-out analytic shapes and three seeds.
Cycling reduced **full fixed-trajectory pipeline time by 26.00%** (59.71 vs 80.69 minutes)
and preserved all 12 dense persistent successes; random saved 19.24% and matched
the same 12/18 successes. All three arms failed all six narrow-stack pairs.
The reliability target was **not achieved**, despite passing aggregate error guards.

On the 12 jointly successful pairs, cycling's **first observed persistent checkpoint**
took 157.77s vs 177.87s for dense (**11.30% reduction**); random took 174.73s
(1.76% reduction). This retrospective checkpoint statistic is conditional on success,
separate from the full 30k trajectory saving, and does not demonstrate online stopping.
Full pipeline measurements include startup, independently repeated prefixes,
sampling/labels, training, evaluation and checkpoint/hash I/O, including failed attempts.

The 60-minute repair screen completed 13/16 trajectories, retained one file-access
failure and left two unstarted. All eight narrow development trajectories failed;
the model comparison is inconclusive and uses the predeclared SIREN fallback.
Confirmation charged 205.56/240 minutes, without extending the allowance. 46 distinct
targeted tests passed after retained repairs. The scope uses known analytic box-union
regions and exact SDF labels; it does not establish arbitrary unseen-shape reliability.
Mixed feature values are half-thickness, while plate and stack gaps are full widths.

- [Experiment protocol](results/thin-feature-campaign/plan.md)
- [Report, exact guards and conditional timing](results/thin-feature-campaign/report.md)
- [Execution, dimension conventions and artifact audit](results/thin-feature-campaign/supplement.md)
- [Summary figure](results/thin-feature-campaign/summary.png)
- [All trajectories and failure horizons](results/thin-feature-campaign/trajectories.csv)
- [Checkpoint gates](results/thin-feature-campaign/checkpoints.csv)
- [Per-gap and per-component measurements](results/thin-feature-campaign/regions.csv)
- [Immutable setup and 54-task inventory](results/thin-feature-campaign/confirmation-manifest.json)
- [Verification](results/thin-feature-campaign/audit.json)

Verify the public evidence with `python scripts/verify_thin_publication.py --git-ref HEAD`.
The [publication note](publication/README.md) distinguishes normalized public copies
from retained originals. Historical seals and hash tables are preserved as original
references; explicit mappings identify public changes to documentation and path metadata.
Full numerical replay and original-seal verification require the private original archive,
retained raw model/dataset files and recorded CUDA environment. Public-only scientific
`verify`, `report` and summarizer commands cannot validate those missing original bytes.
Existing reservations are consumed and seals should not be overwritten.

The preceding [frozen confirmation campaign](results/gap-confirmation/report.md) did
**not** meet its consistency criterion. Balanced **16,384-point cycling** achieved
observed persistent structural recovery in **3/6** held-out geometry/seed pairs
versus dense **2/6**, using **75% fewer recovery point presentations**, **66.1%
less measured recovery training time**, and **26.5% less attributed pipeline
runtime** including the shared prefix and evaluation. The scope is two rotated
parallel-plate cases and three paired seeds, tested at 26k/30k and grids 192/384.

All three seeds failed on gap **0.012 / 17°**, where dense also failed despite
resolved analytic references. Random sampling matched cycling's structural
success rate; balancing superiority is unsupported. Static sampling passed 0/6.
The order-sensitivity gate had passed all six development pairs for canonical,
phase-shifted and shuffled cycles, with exact historical model/Adam reproduction.

Exactly **three new scientific batches**, **96 recovery checkpoints** and **12
shared-prefix checkpoints** are retained, charging **71.15 minutes** of the
two-hour allowance including failed attempts and 20 passing targeted tests.
At the close of that historical campaign, six reserved cases were untouched;
the latest campaign above has now consumed them. No confirmation tuning or
additional training replay was performed.

- [Paper-style report and precise bounded claim](results/gap-confirmation/report.md)
- [All confirmation checkpoints and ordinary MSE](results/gap-confirmation/confirmation-checkpoints.csv)
- [Paired control differences](results/gap-confirmation/confirmation-paired-differences.csv)
- [Measured compute and attributed prefix costs](results/gap-confirmation/confirmation-compute.csv)
- [Frozen protocol](results/gap-confirmation/confirmation-manifest.json)
- [Budget and retained failures](results/gap-confirmation/budget.json)
- [Reservation access ledger](results/gap-confirmation/reservation-consumption.json)
- [Artifact inventory](results/gap-confirmation/run-artifacts.json)

The completed [sampling-efficiency campaign](results/gap-sampling/report.md)
retains the dense baseline's observed persistent structure in **6/6** development
trajectories using balanced **16,384-point cycling** through the same dense pool:
**75% fewer point presentations**, **61.4% less measured training time**, and
**44.7% less measured training-plus-evaluation time** for the additional recovery
stage. Ordinary MSE, paired starting model/Adam states and the structural gate
remain unchanged. All dense labels and storage are still required.

Exactly three new batches and 72 checkpoints were completed, charging **38.38
minutes** including process overhead and verification tests. Both smaller cycling
sizes (4,096 and 8,192) retained only 5/6 persistent passes; these negative results
are preserved. All 66 tests passed, and the dense rerun exactly reproduced all 18
historical model/Adam checkpoints. At that stage fresh cases remained reserved;
block-order robustness and frozen confirmation were the next research decisions.

- [Protocols, complete results and limitations](results/gap-sampling/report.md)
- [Compute comparison](results/gap-sampling/methods.csv)
- [Budget ledger](results/gap-sampling/budget.json)
- [Complete artifact inventory](results/gap-sampling/run-artifacts.json)

The preceding [fixed-data density sweep](results/gap-density-sweep/report.md) completes
all 18 arms and 54 checkpoints, starting from the six corresponding decayed
20,000-update checkpoints with Adam retained. Persistent structural passes at
both 26k and 30k, at both 192 and 384, are **0/6 at 4,096 points, 2/6 at 16,384,
and 6/6 at 65,536**. Datasets are nested with unchanged sampling proportions;
every update uses ordinary exact full-dataset MSE accumulated in chunks.

Mean independent gap RMSE falls from 0.00290 to 0.00108 as density increases,
while training gap RMSE rises from 0.000377 to 0.00105. This supports coverage
and sampling efficiency as the next direction. Dense arms still fail the strict
training-fit gate and retain small independent sign errors; the result is
structural recovery under the existing gate, not perfect continuous geometry.
At that historical stage, no new loss was advanced and fresh cases remained untouched.

- [Density protocol](results/gap-density-protocol.md)
- [All checkpoint measurements](results/gap-density-sweep/checkpoints.csv)
- [Training versus independent errors by surface distance](results/gap-density-sweep/surface-distance-errors.png)
- [All distance bins](results/gap-density-sweep/surface-distance-errors.csv)
- [Compute costs](results/gap-density-sweep/compute.csv)
- [Artifact audit](results/gap-density-sweep/audit.json)

The largest arm uses 16x the point presentations and 2.90x the measured training
time of the smallest. All completed trajectories used 5.161 billion point
presentations; one retained interrupted attempt adds at least 98.304 million
discarded presentations. All 59 tests passed; all 18 legacy 4,096-point
checkpoints reproduce the earlier model and Adam states exactly. Full datasets,
checkpoints and the interrupted attempt remain under `runs/gap-density-sweep`
in the original checkout below. The runner is `scripts/gap_density_controls.py`;
its exact source and reporter are included in the export.

The preceding [learning-rate continuation report](results/gap-lr-continuations/report.md)
contains all 12 matched width-128 ordinary-MSE continuations and the conditional
six-run fixed-data diagnostic. Cosine decay improves final full-gap signs in
6/6 pairs, but no continuation checkpoint passes the structural gate at 192 or
384. Fixed-data training passes its declared fit tolerance in 5/6 runs, while
independent geometry still fails in all six. At that historical stage, no new loss
was promoted and fresh cases remained untouched.

- [Learning-rate protocol](results/lr-continuations-protocol.md)
- [Gap stability curves](results/gap-lr-continuations/gap-stability.png)
- [Errors by true surface distance](results/gap-lr-continuations/margin-errors.png)
- [All continuation measurements](results/gap-lr-continuations/checkpoints.csv)
- [Fixed-data measurements](results/gap-lr-continuations/fixed-checkpoints.csv)
- [Pairing and artifact audit](results/gap-lr-continuations/audit.json)

The new runner is `scripts/gap_lr_controls.py` in the original checkout below;
all 54 new model/Adam/RNG checkpoints and six fixed datasets remain under
`runs/gap-lr-continuations`. Exact source snapshots are included in the new
report export. That stage passed 53 tests before the six density tests were added.

The completed [report](results/gap-controls/report.md) contains the longer-training
controls, wider-network follow-up and matched-point loss comparisons requested
on 8 October 2026. All results, including failures, are retained.

- [Training curves](results/gap-controls/long/convergence.png)
- [All-seed gap cross-sections](results/gap-controls/objective-gap-slices.png)
- [Final per-seed measurements](results/gap-controls/final-checkpoints.csv)
- [Protocol](results/gap-controls-protocol.md)
- [Reserved, unevaluated fresh cases](results/gap-controls/fresh-cases.json)

The existing implementation and full model/optimizer checkpoints remain in the
original checkout:

`retained-project`

From that checkout, use `.venv\Scripts\python.exe` with these modules:

```powershell
python -m scripts.gap_training_controls report --out runs/gap-controls-long
python -m scripts.gap_training_controls report --out runs/gap-controls-wide
python -m scripts.report_gap_research
```

The experiment code is in `scripts/gap_training_controls.py` and
`scripts/gap_objective_controls.py`. The new tests cover continuation, sign-loss
gradients, point-stream hashing and probe coverage. The full test suite passed
46 tests. Compact results here are exact copies of `docs/gap-controls/` from that
checkout; source snapshots and all 54 training checkpoints remain under `runs/`.
