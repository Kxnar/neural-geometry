# Longer-training gap controls

Same 0.02 gap, seeds 31–33, fixed Adam learning rate, batch size and paired sampling streams as the original development study. Continuous trajectories with optimizer/RNG checkpoints. All seeds retained.

| Policy | Updates | Gap accuracy mean (range) | Dense gap mean | Chamfer mean | Structure at 192 |
|---|---:|---:|---:|---:|---:|
| global_narrow | 1000 | 84.51% (70.70%–92.97%) | 76.35% | 0.009172 | 0/3 |
| global_narrow | 3000 | 76.95% (71.09%–85.55%) | 64.92% | 0.009816 | 0/3 |
| global_narrow | 10000 | 96.09% (91.80%–99.22%) | 88.71% | 0.008591 | 0/3 |
| local | 1000 | 89.71% (87.89%–92.19%) | 81.87% | 0.009288 | 0/3 |
| local | 3000 | 80.73% (66.02%–94.14%) | 68.46% | 0.009811 | 0/3 |
| local | 10000 | 92.45% (88.67%–97.27%) | 85.65% | 0.008853 | 0/3 |

Structure requires correct original probes, two components, Euler characteristic 4 and closed edges, conditional on resolved analytic reference. Dense probes cover the central gap volume, not the full gap or a topology certificate. Chamfer is sampled unsquared symmetric distance (half-sum); original 96 and 192 extraction grids retained. The 96-grid analytic reference does not resolve the 0.02 gap. Training timing includes periodic probe diagnostics and excludes mesh evaluation/checkpoint I/O.
