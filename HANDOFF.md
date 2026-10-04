# Session handoff: Neural Geometry

Updated: 4 October 2026. Read this first when continuing the project.

## New-session instruction

> Continue the neural-geometry project from this handoff. Read README.md,
> docs/plan.md and docs/pilot.md, inspect git status, and use the existing Python
> 3.11 environment. Take ownership of implementation and experiment execution.
> The next milestone is a repeatable thin-feature failure benchmark, starting
> with a small multi-seed thickness/sampling sweep and longer-training controls.
> Explain the results simply and involve me in the research decisions.
> Preserve existing work and distinguish measured results from proposed work.

## User intent and working agreement

- The user is new to ML and wants their own substantial neural-geometry research
  project. Their friend's https://github.com/Chr4st/Integrals is an ambition
  reference, not code to copy or the destination repository.
- The assistant handles implementation, setup, tests, debugging, experiment
  tooling, execution and documentation. The user learns, reasons about results,
  selects hypotheses and makes research decisions.
- User availability is approximately one hour per day. Target a credible GitHub
  project in 1-2 weeks from 3 October: approximately 10-17 October 2026.
- Prioritize a small, convincing experiment and one controlled extension.
  Do not block progress on a long course or require the user to hand-code the stack.
- The target is a strong, reproducible project. Publication, novelty and matching
  another project's quality are not guaranteed.
- Use local compute. No paid cloud compute is authorized. Keep the repository
  private until the user approves public release.
- Work continues during active sessions; no unattended daily agent or recurring
  automation has been configured.

## Repository and current state

- GitHub: https://github.com/Kxnar/neural-geometry (private).
- Branch: main. Remote: git@github.com:Kxnar/neural-geometry.git.
- Local project root (nested under the session workspace):
  `C:\Users\knara\Documents\Codex\2026-10-03\https-github-com-chr4st-integrals-https\neural-geometry`
- GitHub CLI authentication worked as Kxnar; Git uses SSH.
- Before this handoff, the working tree was clean and HEAD was
  `701c56f` -- Record GPU sampling pilot with measured metrics and visual comparison.
- Baseline implementation commit: `1a9b6ea`. Pilot configurations record that
  source commit and a clean tree.
- Last recorded validation: all 11 local tests passed; GitHub Actions succeeded:
  https://github.com/Kxnar/neural-geometry/actions/runs/37130694549
- No AGENTS.md was found in the project at handoff time. Check again on resuming.
- This handoff documents existing work; it does not imply any new sweep has run.

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

## Environment: reuse rather than rebuild

Verified machine: Alienware laptop, NVIDIA RTX 2080 Super, 8192 MiB VRAM.
Working environment: Python 3.11, PyTorch 2.7.1+cu126, CUDA available.
The NVIDIA driver observed during setup was 610.74.

Use `.\.venv\Scripts\python.exe` from the project root. Activation is unnecessary.
The system-default `python` is Python 3.14.6; do not accidentally use it for ML
commands or recreate the environment with it.

Existing Python 3.11 interpreter, if the venv genuinely needs rebuilding:
`C:\Users\knara\AppData\Roaming\uv\python\cpython-3.11.16-windows-x86_64-none\python.exe`.

Exact Windows dependency snapshot: `requirements-windows-cu126.txt`.
The project is installed editable with development dependencies.

~~~powershell
# Run from the project root.
git status --short
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -m pytest -q
~~~

If rebuilding is necessary, create the venv with an explicit Python 3.11
interpreter, then install the lock file and editable project:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-cu126.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
~~~

Tooling note for this Windows session: default sandbox shell execution failed
with "sandbox provisioning failed"; escalated execution passed automatic review.
This was an execution-environment issue, not a project error. Respect the active
session's permissions rather than assuming these settings persist.

## Implementation map

| Path | Purpose |
|---|---|
| neural_geometry/geometry.py | Analytic SDFs, point sampling, feature probes |
| neural_geometry/models.py | SIREN-style network and Fourier-feature ReLU network |
| neural_geometry/run.py | Training CLI, metrics, metadata, plots and mesh export |
| tests/test_core.py | 11 geometry, probe and model correctness tests |
| .github/workflows/tests.yml | Python 3.11 CPU test workflow |
| requirements-windows-cu126.txt | Tested Windows environment snapshot |
| docs/plan.md | Original 14-day milestones and daily collaboration |
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

## Recommended next session

Do the useful implementation work directly, then bring the user concrete results
and a small number of meaningful research decisions.

1. Read this handoff, README, plan and pilot. Inspect current changes and the CLI.
   Reuse the existing environment and confirm CUDA. Preserve user edits.
2. Implement a modest sweep/aggregation tool with explicit run configurations,
   resumable behavior that never overwrites completed runs, and a compact summary.
3. Proposed first matrix: torus feature radii 0.02, 0.04 and 0.08; training seeds
   0, 1 and 2; uniform and mixed sampling; otherwise matched pilot settings.
   This is 18 runs and is a proposal, not completed work.
4. Include longer uniform training at radius 0.04 (for example 10,000 steps) to
   test undertraining. Compare matched longer mixed runs where needed for a
   method claim. Record both label counts and elapsed time.
5. Check extraction resolution 96 versus 192 using the same trained field.
   Add checkpoint evaluation/export support rather than retraining solely to
   change extraction resolution.
6. Summarize run-to-run variation, global error and feature retention together.
   Add stronger geometry measurements when needed to resolve a claim, and
   extend the failure study to a thin plate or narrow gap.
7. Review closest literature before selecting one extension. Compare it with
   strong ordinary sampling baselines, including any extra supervision or
   geometry access it uses. Do not name an established trick as a novel result.
8. Define development/validation/final cases before tuning the extension.
   Match model/storage and training-label budgets; report compute trade-offs.
9. Package actual evidence into a concise research-style README and figures.
   Keep the release date realistic; an honest negative result is acceptable.

Do not build a broad framework or run a huge grid before seeing the small sweep.
No new experiments are required merely to deliver this handoff.

## Reading and learning

Primary starting points:
- SIREN: https://www.vincentsitzmann.com/siren/
- Official implementation: https://github.com/vsitzmann/siren
- Fourier Features: https://bmild.github.io/fourfeat/
- Official implementation: https://github.com/tancik/fourier-feature-networks
- Potential related work to examine, not a chosen dependency:
  https://www.computationalimaging.org/publications/acorn/

For the user's limited learning time, explain signed distance, coordinate
networks, training loss, sampling bias and fair comparisons using the project's
own plots. Basic neural-network videos can accompany this; they are not a gate
before the assistant continues implementation.
