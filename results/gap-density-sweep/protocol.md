# Fixed-data density intervention

The machine-readable protocol is frozen at `docs/gap-controls/density-sweep.json`.
Only the existing development parallel-plate geometry (gap 0.02), width-128
depth-3 SIREN, original samplers `global_narrow` and `local`, and seeds 31–33
are included. Fresh cases and new objectives remain out of scope.

Each of the six decayed 20,000-update model/Adam checkpoints starts three arms:
4,096, 16,384 and 65,536 fixed points. The smallest dataset exactly reproduces
the saved preceding diagnostic. Each larger dataset contains the smaller as
a byte-identical prefix. Generate balanced blocks from the same CPU generator
(training seed + 91,000), each with 2,048 original-sampler points, 1,024 uniform
full-gap points and 1,024 facing-interior points within 0.01 of the surface.
Consequently the proportions are exactly 50% / 25% / 25% at every density.

Every arm receives 10,000 additional updates with the earlier fixed-data
cosine schedule, endpoint-inclusive from 1e-4 to 1e-6. Restore Adam moments and
step counts without alteration. Preserve the training sampling RNG state;
fixed-data updates do not consume it. Evaluate at total updates 23k, 26k and
30k without restarting the optimiser.

The objective is ordinary equal-point MSE. Accumulate each chunk's sum of
squared errors divided by the total target element count. Clear gradients once
before all chunks and take one Adam step after all chunks. Use chunks of at
most 16,384 points; no stochastic minibatch updates or subset reweighting.
Numerical reduction order is the only difference from a monolithic full-batch
gradient. Tests compare gradients with an independent monolithic reference,
including an unequal final chunk, and verify continuation/Adam step counts.

Measure errors on training points and on 65,536 independent points drawn with
the same sampler/composition (evaluation seed 61,002, shared across training
seeds and density arms). Assert zero exact-coordinate overlap with training.
Report per-subset RMSE, MAE and sign errors by absolute true SDF: 0–.001,
.001–.0025, .0025–.005, .005–.01, .01–.05 and >.05. For gap and interior also
split the footprint into central 80% and outer regions. Only sign diagnostics
exclude numerical margins ≤1e-7; report their denominators. Evaluate before
the intervention and at every checkpoint. Empty bins are missing, not zero.

Reuse the frozen independent full-gap probe set (131,072 requested points,
seed 61,000), original field probes and sampled symmetric surface distance.
At both 192 and 384 require the unchanged structural gate: all original
probe signs correct, two mesh components, Euler characteristic four, closed
edges, and an analytically resolved reference. Keep all missing/failed meshes.
Persistent observed structure requires both the 26k and 30k checkpoints;
report regressions and all individual seeds. Nothing certifies the intervals
between measurements or the continuous field.

The existing training-fit gate remains subset RMSE ≤.001 and no wrong signs
where |true SDF|>.001 for each of original/gap/interior subsets. Also report
continuous improvements: failing this strict gate alone is not evidence of a
training plateau.

Persistent restoration across all six pairs at a denser size supports coverage
and motivates sampling efficiency. Fitted data with failed structure motivates
localising bridges between samples. Unresolved dense-data fit leaves
optimisation/capacity open. Keep mixed outcomes mixed; no new loss is advanced.

Match updates, not compute: 40.96 million, 163.84 million and 655.36 million
training point presentations per arm (1x / 4x / 16x). Record measured training
and evaluation time separately. Preserve datasets, checkpoint parent hashes,
model/Adam/RNG states, source snapshots, partial attempts, and before/after
fresh-case reservation hashes. The export audit verifies all evidence.
