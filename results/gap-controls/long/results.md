# Longer-training gap controls

Same 0.02 gap, seeds 31–33, fixed Adam learning rate, batch size and paired sampling streams as the original development study. Continuous trajectories with optimizer/RNG checkpoints. All seeds retained.

| Policy | Updates | Gap accuracy mean (range) | Dense gap mean | Chamfer mean | Structure at 192 |
|---|---:|---:|---:|---:|---:|
| global_narrow | 1000 | 67.45% (57.03%–79.30%) | 64.20% | 0.011097 | 0/3 |
| global_narrow | 3000 | 91.67% (87.11%–95.31%) | 87.88% | 0.009610 | 0/3 |
| global_narrow | 10000 | 93.36% (89.06%–98.83%) | 87.49% | 0.009112 | 0/3 |
| local | 1000 | 73.31% (65.62%–80.47%) | 70.04% | 0.011160 | 0/3 |
| local | 3000 | 79.82% (61.33%–94.92%) | 74.42% | 0.010696 | 0/3 |
| local | 10000 | 94.27% (87.11%–100.00%) | 87.09% | 0.009264 | 0/3 |

Structure requires correct original probes, two components, Euler characteristic 4 and closed edges, conditional on resolved analytic reference. Dense probes cover the central gap volume, not the full gap or a topology certificate. Chamfer is sampled unsquared symmetric distance (half-sum); original 96 and 192 extraction grids retained. The 96-grid analytic reference does not resolve the 0.02 gap. Training timing includes periodic probe diagnostics and excludes mesh evaluation/checkpoint I/O.
