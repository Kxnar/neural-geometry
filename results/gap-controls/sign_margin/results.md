# Longer-training gap controls

Same 0.02 gap, seeds 31–33, fixed Adam learning rate, batch size and paired sampling streams as the original development study. Continuous trajectories with optimizer/RNG checkpoints. All seeds retained.

| Policy | Updates | Gap accuracy mean (range) | Dense gap mean | Chamfer mean | Structure at 192 |
|---|---:|---:|---:|---:|---:|
| global_narrow | 10000 | 97.40% (92.19%–100.00%) | 90.87% | 0.008508 | 0/3 |
| local | 10000 | 100.00% (100.00%–100.00%) | 96.13% | 0.008334 | 0/3 |

Structure requires correct original probes, two components, Euler characteristic 4 and closed edges, conditional on resolved analytic reference. Dense probes cover the central gap volume, not the full gap or a topology certificate. Chamfer is sampled unsquared symmetric distance (half-sum); original 96 and 192 extraction grids retained. The 96-grid analytic reference does not resolve the 0.02 gap. Training timing includes periodic probe diagnostics and excludes mesh evaluation/checkpoint I/O.
