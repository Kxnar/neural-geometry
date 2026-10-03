# First release: 7-14 days from 3 October 2026

Target: a credible GitHub project by 10-17 October. This is an engineering and
experimental milestone, not a promise of publication or positive results.

User time: about 60 minutes per day. GPU elapsed time is additional and can run
unattended while the machine remains available. Work happens during active sessions;
there is no autonomous daily service implied.

## Days 1-2: baseline
Assistant: environment, analytic geometry, SIREN/Fourier implementations, tests,
training/evaluation, meshes and loss curves.
User: understand signed distance, loss and coordinate networks; inspect one run.

## Days 3-4: failure benchmark
Assistant: thickness/gap sweep; compare sampling strategies; independent validation
shapes and final evaluation seeds; inspect extraction-resolution sensitivity.
User: choose the most meaningful, repeatable failure.

## Days 5-7: one extension
Assistant: read closest prior work, implement one mechanism and its ablations.
Candidate: geometry-aware sampling allocation, compared against ordinary uniform
and near-surface sampling at matched label and compute budgets.
User: articulate the hypothesis and choose the trade-off to optimise.
The candidate may change with baseline evidence; novelty is not assumed.

## Days 8-10: validation
Assistant: at least three seeds for the final small matrix; parameter/storage
controls; a small real-mesh evaluation if data preparation permits.
User: review plots and alternative explanations; decide what can be claimed.
Additional work: symmetric surface distances and resolution-aware component tests.

## Days 11-14: release
Assistant: concise README, figures, reproducible commands, environment lock,
limitations, method explanation and a mesh-viewing demo.
User: explain the project in five minutes and review the proposed public release.

## Daily collaboration
10 minutes: explanation of the current experiment.
20 minutes: inspect results together.
20 minutes: decide on the next comparison.
10 minutes: record the reasoning in the experiment log.

## Release acceptance
- Runnable setup and automated correctness tests.
- At least one observed failure reproduced.
- A fair baseline/extension comparison, even if improvement is absent.
- Clear separation of implemented, run, and planned features.
- No invented speedups, novelty or publication claims.
