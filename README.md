# Neural Geometry

A research workbench for studying how small neural implicit fields preserve thin
structures, narrow gaps and disconnected components under limited model capacity.

**Status: initial implementation and pilot stage.** No novel method or improvement
is claimed yet. The first release targets a reproducible baseline and a controlled
sampling study; research claims will depend on measured results and related work.

## What the model learns

Given a coordinate (x, y, z), predict signed distance to a shape: negative inside,
zero on the surface, positive outside. The surface can then be extracted as a mesh.

Current baselines:
- SIREN-style sine MLP with frequency-aware weight initialization.
- ReLU MLP with fixed random Fourier input features.

Current analytic shapes: sphere, thin torus, thin plate, and two spheres separated
by a controlled gap. Models fit each shape separately; this is not an unseen-shape
generalization model.

## Scope and attribution

The implementation is inspired by [SIREN, NeurIPS 2020](https://www.vincentsitzmann.com/siren/)
and [Fourier Features, NeurIPS 2020](https://bmild.github.io/fourfeat/).
It is an independent implementation using supervised analytic SDF values.
The original SIREN geometry experiments use a different supervision setup; this
repository does **not** claim a full reproduction of their published results.
Near-surface sampling is an established baseline, not our claimed invention.

## Setup (Windows, NVIDIA GPU)

Use Python 3.11 in a virtual environment. The tested CUDA wheel is pinned below;
it includes its runtime, so a separate CUDA toolkit is unnecessary for these scripts.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
~~~

For CPU-only machines, install the corresponding PyTorch CPU wheel first.

## First experiment

~~~powershell
.\.venv\Scripts\python -m neural_geometry.run --model siren --shape torus --feature 0.08 --sampling mixed --steps 1000 --out runs/siren-torus
~~~

Use a new output directory for every run. Existing experiments are never overwritten.
Each run produces configuration/environment metadata, training loss CSV, checkpoint,
metrics JSON, a cross-section plot, a sampled field and an OBJ mesh.

Compare `--sampling uniform` against `--sampling mixed`, keeping model, seed,
steps and batch size fixed. Mixed sampling uses half uniform and half noisy surface
samples. Both configurations get the same number of training labels; sampling
cost still affects elapsed time. This initial controlled ablation is not a new
research contribution.

## Evaluation and limitations

- Fresh evaluation samples from a separate seed; keep evaluation settings fixed.
- Uniform SDF error and occupancy IoU.
- Near-surface error and ground-truth-surface residual.
- Known interior/exterior probes for feature retention and gap preservation.
- Parameter count, tensor storage, training duration and CUDA allocated-memory peak.

Probe scores are diagnostics, not topological guarantees. Surface residual is
one-sided and not a symmetric surface-distance metric. Uniform IoU can have high
variance when the object occupies little volume; inside counts are reported.
The extraction grid may miss sub-voxel features: increase resolution and check
convergence before making topology claims. Torus sampling is uniform in angles,
not surface area. All distances are in normalized scene units.

The default SIREN/Fourier widths are **not parameter-matched**. Architecture claims
require matching budgets or presenting accuracy-storage curves. Multiple seeds,
validation for tuning, real meshes, topology metrics and inference benchmarks
remain future work. No test-set tuning should be used for final comparisons.

## Short release plan

See [the 14-day plan](docs/plan.md). Implementation and experiment assistance use
AI tools; the repository records what was actually run and distinguishes pilot
observations from validated conclusions.
