# Thin-feature reconstruction campaign

The goal is reliable thin-feature reconstruction on genuinely reserved geometries, with at least a 25% reduction in measured full reconstruction time. If the frozen models fail, the deliverable is a reproducible limitation benchmark with explicit failures. Evidence is limited to analytic box unions with known regions and exact SDF labels.

## Fixed budgets

- Development model work: 60 minutes, including preparation, training, evaluation and failed attempts.
- Reserved confirmation: 240 minutes, with the same accounting.
- Model tests and verification experiments: 30 minutes.
- Combined experiment subprocess cap: 330 minutes. Implementation time is separate.

An external supervisor starts and times each subprocess, writes an attempt ledger, kills its entire process tree at the remaining cap, and preserves failed or partial artifacts. Finishing early does not authorize extra configurations.

## Development and selection

Exactly two architectures are compared: the existing SIREN at width 128 and depth 3; and the existing Fourier-feature ReLU model at width 128 and depth 3, with 64 frequencies and scale 10 explicitly wired into its factory.

The screen crosses widths 0.012 and 0.03 with angles 17 and 37 degrees, using seed 71 and dense/cycling recovery. That is sixteen trajectories. Cases and arms interleave the two models. New development prefixes are independently repeated and included in measured runtime. Endpoints are 26k and 30k, with both grids 192 and 384.

Selection uses, in order: fewer dense persistent successes lost by cycling; more cycling persistent narrow pairs; more cycling persistent pairs across all four cells; fewer frozen aggregate error-guard violations; lower measured cycling pipeline time; SIREN first for a tie. An incomplete or invalid comparison selects the predeclared SIREN fallback and is labelled inconclusive. Missing measurements cannot improve a rank.

If time remains, the selected model receives the narrow 0.012/17-degree validation at seeds 72 and 73, dense and cycling. This cannot change the selected model. No optional random development arm is run. Reserved confirmation runs even if development fails to find a repair.

## Frozen confirmation

Original reservation indices 2 through 7 are evaluated with seeds 71, 72 and 73 and arms dense, genuinely unstratified random, and balanced cycling: 54 independent trajectories. The arms rotate in a fixed order across case/seed pairs. Every trajectory starts from initialization and repeats its complete prefix, preparation and reference evaluation. No prefix or reference from another arm is reused without charging it.

Before any reserved generation, an exclusive seal records the exact cases, original reservation and previous consumption hashes, selected model and development hashes, scientific source and test hashes, environment, deterministic/TF32 settings, RNG namespaces, schedules, checkpoints, grids, criteria, ordering and budgets. A detached seal hash and reservation inventory are checked before and after each worker. Scientific code is frozen after exposure; there is no silent reseal. Append-only events precede geometry, pool, full-gap, model, schedule, training and checkpoint-evaluation actions.

The family adapter retains exact legacy plate pool sampling. Mixed-thickness gaps are sampled over their true x-separation and y/z overlap, with interiors from both boxes. Stack sampling covers both adjacent z-gaps and every component/facing surface. Local coordinates are rotated to world coordinates. The pool contains 65,536 points in balanced 50% original sampler, 25% gap, 25% interior blocks; cycling selects 16,384 points per update. Random selects 16,384 points without replacement per update from a separately seeded saved CPU schedule.

All models use unweighted ordinary MSE and Adam. The prefix is 10k stochastic updates of batch 2048 at learning rate 1e-4, followed by 10k stochastic updates with cosine decay from 1e-4 to 1e-6. Recovery is another 10k updates with the same cosine endpoints. The complete model, Adam and sampling-RNG states at 20k establish exact pairing across independent arms.

## Gates, timing and reporting

Persistent primary success requires both 26k and 30k, both grids 192 and 384, every original probe sign, a present closed mesh, expected component count/Euler characteristic, and resolved analytic references. Plate/mixed targets are two components/Euler 4; stack targets are three/Euler 6. Every tiny component remains counted. Missing, unresolved or nonfinite measurements fail closed.

Historical error guards compare paired endpoint means: gap RMSE at most 1.10 times dense; full-gap sign-accuracy difference at least -0.005; Chamfer at each grid at most 1.05 times dense. Narrow-stack pairs must also individually satisfy these guards. Per-gap and per-component sign, RMSE and margin diagnostics accompany aggregate results.

The 23k checkpoint enables retrospective onset: all three passing implies onset 23k; 26k and 30k passing implies onset 26k; 30k alone has no persistent onset. This is checkpoint-limited evidence, not online stopping. Cumulative pipeline time through onset is reported separately from later verification time. Failed reconstructions have no finite observed time to correct and retain consumed time, reasons and censor horizon.

The target is achieved only with complete valid evidence, preservation of every dense persistent success, all six narrow-stack pairs successful with guards, all aggregate guards satisfied, and actual cycling pipeline time at most 75% of dense time including failures. Random is reported identically; a matching random result does not establish balancing superiority. Reports include every attempted/unstarted trajectory, family and seed results, development factor differences, joint-success conditional timing, total pipeline time per persistent success, failures and artifact/source/RNG/event audits.

Machine-readable protocol, immutable manifests, budget ledger, append-only exposure events, CSVs and report are saved beside this plan. Historical attributed runtime savings remain distinct from this campaign's actual independently measured pipeline totals.
