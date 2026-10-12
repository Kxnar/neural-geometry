# Implementation and pilot notes

Historical implementation and measurements as of 4 October 2026. See the main README for the completed later campaigns.

## Scientific question

How well do small coordinate networks preserve thin structures and narrow gaps
under limited model/storage budgets? Can a carefully controlled change improve
feature preservation without hiding losses in global field accuracy or compute?

A network receives (x, y, z) and predicts signed distance: negative inside a
shape, zero on its surface, positive outside. Each model fits one shape.
This is not generalization to unseen shapes.

The immediate study compares training-point sampling. A possible later extension
is geometry-aware sampling or a loss change, chosen after examining repeatable
failures and checking related work. No original extension is implemented yet.

## Implementation map

| Path | Purpose |
|---|---|
| neural_geometry/geometry.py | Analytic SDFs, point sampling, feature probes |
| neural_geometry/models.py | SIREN-style network and Fourier-feature ReLU network |
| neural_geometry/run.py | Training CLI, metrics, metadata, plots and mesh export |
| tests/test_core.py | 11 geometry, probe and model correctness tests |
| .github/workflows/tests.yml | Python 3.11 CPU test workflow |
| requirements-windows-cu126.txt | Tested Windows environment snapshot |
| docs/plan.md | Research milestones |
| docs/pilot.md | Pilot methods, observations and caveats |
| docs/pilot/ | Tracked pilot configs, metrics, loss logs and figures |
| runs/ | Ignored local experiment artifacts, including checkpoints and meshes |

Geometry lives in [-1, 1]^3:
- Sphere: radius 0.55.
- Torus: major radius 0.55; `feature` is tube radius.
- Thin box: half-extents (0.6, 0.6, feature); thickness is twice feature.
- Double sphere: two radius-0.3 spheres; feature is the gap between them.

Sampling modes:
- `uniform`: uniformly sample the domain.
- `mixed`: half domain samples, half analytic surface samples perturbed by
  Gaussian noise with standard deviation feature/2, clipped to the domain.
- Torus surface points are uniform in the two angles, not uniform in surface area.

Default model/training settings: SIREN, width 64, three hidden layers,
Adam learning rate 1e-4, 1,000 steps, batch size 2,048, training seed 0.
The Fourier model uses 16 fixed random frequencies with scale 3.
Default architecture widths do not match parameter/storage budgets.

Evaluation uses separate RNG seed 10000, 32,768 points, and mesh resolution 96.
There is no learning-rate schedule, early stopping, sweep driver, or
checkpoint-only evaluation CLI yet.

Each run writes config.json, training.csv, checkpoint.pt, metrics.json,
slice.png, field.npz and mesh.obj (when there is a zero crossing). Metadata
includes source revision, dirty status, device, versions, parameters and tensor
bytes. The CLI refuses to overwrite an existing output directory.

Example new run (use an unused output path):

~~~powershell
.\.venv\Scripts\python.exe -m neural_geometry.run --model siren --shape torus --feature 0.04 --sampling mixed --steps 1000 --seed 1 --eval-seed 10000 --device cuda --out runs/torus-f004-mixed-s1
~~~

## What has actually been run

Two GPU pilot runs on 3 October 2026, both SIREN, torus tube radius 0.04,
width 64, depth 3, training seed 0, evaluation seed 10000, 1,000 steps,
batch size 2,048, learning rate 1e-4 and extraction resolution 96.
Each run uses 2,048,000 training labels. Only sampling policy differs.

| Metric | Uniform | Mixed |
|---|---:|---:|
| Uniform SDF MAE | 0.008604 | 0.009776 |
| Near-surface SDF MAE | 0.014646 | 0.003624 |
| Ground-truth surface residual | 0.014259 | 0.002198 |
| Interior probe retention | 0.312500 | 1.000000 |
| Exterior probe preservation | 0.953125 | 1.000000 |
| Uniform occupancy IoU | 0.200000 | 0.788732 |
| Training seconds | 3.707541 | 4.237915 |

The extracted uniform-sampling ring was fragmented. Mixed sampling retained a
continuous-looking ring in the rendered extraction and improved local diagnostics,
while slightly worsening uniform-domain error. This is a useful failure example,
not proof of general superiority.

The comparison figure is `docs/pilot/comparison.png`.
Full local artifacts are in `runs/pilot-uniform/` and `runs/pilot-mixed/`.
Those run folders, checkpoints and meshes are ignored by Git and will not exist
in a fresh clone. The tracked pilot evidence in docs/pilot remains available.
Regenerate missing artifacts using docs/pilot.md; choose fresh output paths if
the original directories already exist.

## Claims and measurement limits to preserve

1. Near-surface sampling is established practice, not a new method.
2. This is an independent supervised analytic-SDF implementation inspired by
   SIREN/Fourier Features, not a full reproduction of SIREN's geometry experiments.
3. Only one shape, seed, training duration and extraction resolution were piloted.
   A longer uniform run may repair the failure.
4. Only 57 of 32,768 uniform evaluation samples were inside the object, making IoU
   a noisy diagnostic here.
5. Surface residual is one-sided, not a symmetric surface-distance measure.
   Feature probes do not certify topology. Visual continuity is not a topology test.
6. Extraction resolution may itself erase features; test resolution sensitivity.
7. SIREN and Fourier defaults are not parameter-matched. Do not claim architectural
   superiority from equal-width comparisons.
8. Training timing excludes startup, evaluation, plotting and extraction. Reported
   peak CUDA allocation is tensor memory measured through evaluation, not total
   GPU process memory or full end-to-end peak.
9. Pilot cases are development data. Reserve separate validation/final evaluation
   settings before making final method comparisons.
10. No real-mesh results, multi-seed conclusions, original method, publication
    status, symmetric surface metric or mesh-viewing demo have been established.

