# Learning-rate decay before further objective changes

Frozen in `neural-geometry/docs/gap-controls/lr-continuations.json` before the
continuations ran. Development geometry only: two parallel plates, full gap
0.02; ordinary-MSE SIREN width 128/depth 3; samplers `global_narrow` and `local`;
seeds 31, 32 and 33; batch size 2,048.

Each of six 10,000-update checkpoints starts two 10,000-update continuations.
Both restore the exact model, Adam state and sampling-generator state. One
keeps learning rate 1e-4. The other uses
`1e-6 + (1e-4-1e-6)/2 * (1 + cos(pi*(j-1)/9999))`, where `j=1..10000`.
Thus the first and last update use the stated endpoints exactly. All coordinate
and target bytes are hashed per segment and compared within each pair.

Evaluate at total updates 13,000, 16,000 and 20,000. Save model, Adam state,
sampling RNG, training history and artifact hashes at each checkpoint. Evaluate
sampled surface distance and mesh structure at both 192 and 384. Retain the
existing original probe gate and require expected structure at both grids:
two components, Euler characteristic four, closed edges and a resolved analytic
reference. Report every failed/missing mesh.

Independently evaluate 131,072 requested points across the entire plate
footprint and gap width, using seed 61,000. Exclude only true distances ≤1e-7.
Report the whole footprint, central 80% footprint and outer footprint, each
stratified by distance to the true surfaces: 0–.001, .001–.0025, .0025–.005,
and .005–.01. These sampled signs are separate from the original structural gate.

Report success/failure sequences, including regressions. Sustained observed
success requires passing at both 16,000 and 20,000; a passing suffix beginning
at 13,000 gives an earlier observed bound. Three checks cannot certify what
happened between them. Consistent repair requires all six cosine trajectories
to pass both the structural gate and all sampled full-gap signs at both later
checkpoints. This conservative criterion does not redefine earlier results.

If that criterion fails, continue all six decayed checkpoints on fixed data:
4,096 saved points per run, comprising 2,048 points from the original sampler,
1,024 uniformly distributed in the full gap, and 1,024 inside the facing plate
surfaces within distance .01. Use seed `training_seed+91000` on CPU. Use every
point on every update with ordinary mean squared error, retaining Adam and
using the same endpoint-inclusive 10,000-update cosine schedule. Check at
23,000, 26,000 and 30,000 total updates.

Measure training fit separately from independent geometry. The fixed-data
fit criterion is subset RMSE ≤.001 and all training signs correct where
|true SDF|>.001, for each of original-sampler/gap/interior subsets. This removes
resampling noise and increases difficult-region coverage together; it does not
isolate their individual effects. Failure under this budget does not establish
representational impossibility. Finite training-set success does not establish
continuum geometry or generalisation.

No new objective is advanced. The sign penalty remains unpromoted. The reserved
fresh cases are not evaluated in this experiment; their existing file hash is
checked before and after execution.
