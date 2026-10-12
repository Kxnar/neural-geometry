# Gap preservation: duration before capacity before objective

Frozen before the longer-training results on 8 October 2026. This study continues
the local-scale development experiment without changing its source or evidence.

## Primary control

Six continuous trajectories: parallel plates with full gap 0.02, global_narrow
and local sampling, seeds 31, 32, 33. Keep SIREN width 64, depth 3, Adam 1e-4,
batch 2048, 50% uniform/50% perturbed surface points. Save weights, optimizer and
sample RNG at 1000, 3000 and 10000 updates. Every 100 updates, record the original
256 gap and 512 plate-interior probes. Evaluate each checkpoint with the original
32768 uniform and surface samples and grids 96/192. The analytic gap is unresolved
at 96: do not call failures at that resolution network failures.

The supplementary dense diagnostic uses 64x64x9 probes covering the central 80%
of the plate footprint and gap width. These expose failures away from the original
mid-plane but exclude near-surface margins and do not certify an open gap.

The operational primary success gate is correct signs on all original probes,
two components, Euler characteristic 4 and closed mesh edges at resolved grid 192,
in all three seeds for a policy. Report probe and mesh outcomes separately.
Mixed outcomes are not forced into a success/failure claim. A passing primary
gate with dense-probe failures is qualified, not full gap preservation.

## Conditional follow-up

If both policies pass after longer training, measure time to correct geometry
with a denser checkpoint schedule, retaining regressions (first observed success
is not stable convergence). If neither passes, repeat the same six trajectories
at width 128, keeping other settings fixed. Width increases capacity and changes
optimisation; it cannot distinguish those causes alone. If either policy has
mixed seed outcomes, disclose them and use the wider control to investigate.

If both duration and width controls still fail, freeze a development comparison
of MSE, weighted MSE and a sign-aware objective before inspecting its results.
Within each sampler/seed, all losses must receive exactly the same training
coordinates and initial weights. Retain local sampling as an ablation, compare
all arms, match updates and labels, and record sampled point-stream hashes.
Weighting/loss hyperparameters must be selected on development cases only.

## Reserved fresh cases and advancement gate

`gap-controls/fresh-cases.json` fixes previously unused feature values, rotations,
mixed-thickness scenes, and three-plate stacks with two gap scales. The file
specifies analytic geometry, training seeds, independent evaluation seed and the
promotion rule. It is a reservation, not completed evidence or an executable
manifest for the extra stack geometry. Freeze the chosen method, loss settings
and full evaluation implementation/source before evaluating these cases.

An extension advances only if its aggregate structural success rate improves
over every simple control on the reserved cases, without reducing success on any
geometry family, while mean surface distance is no more than 5% worse than the
best simple control. Missing meshes are failures and must be reported. Report
all seeds/cases and paired differences; three seeds are descriptive evidence.
If no extension is selected or the gate fails, retain the negative result and
keep fresh cases unevaluated where possible. Do not tune on their results.

## Commands and provenance

Run in the neural-geometry repository using `.venv/Scripts/python.exe`:

```powershell
python -m scripts.gap_training_controls freeze --manifest docs/gap-controls/long.json
python -m scripts.gap_training_controls run --manifest docs/gap-controls/long.json --out runs/gap-controls-long
python -m scripts.gap_training_controls report --out runs/gap-controls-long
```

The runner refuses changed source, snapshots exact source bytes, validates SHA256
for completed artifacts, and binds evaluations to checkpoint/configuration.
Interrupted segments are retained as numbered attempts, then repeated from the
last completed checkpoint with saved Adam/sample RNG state. Existing earlier
experiments and their source snapshots remain intact. Training timing includes
periodic probe diagnostics; it excludes mesh evaluation and checkpoint I/O.
