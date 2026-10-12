# Campaign evidence and execution audit

Development completed 13/16 screen trajectories. Selection is inconclusive; predeclared SIREN fallback. 
The fixed narrow screen independently completed 8/8 trajectories with 0 persistent successes. This observed narrow failure is distinct from an incomplete screen, a file-access failure, or an architectural impossibility claim.

| Stage | External attempts | Failed attempts | Timeouts | Charged minutes |
|---|---:|---:|---:|---:|
| development | 14 | 1 | 0 | 59.952 |
| confirmation | 54 | 0 | 0 | 205.556 |
| qa | 5 | 2 | 0 | 0.777 |

The development Windows file-access failure is retained with consumed time and partial files; it is not classified as a measured reconstruction gate failure or replaced. Confirmation uses the separately frozen I/O-only retry wrapper. Earlier failed QA attempts remain charged, followed by passing repairs.

| Retained reference evaluation | Total | Resolved | Unresolved |
|---|---:|---:|---:|
| development:r192 | 26 | 26 | 0 |
| development:r384 | 26 | 26 | 0 |
| confirmation:r192 | 162 | 162 | 0 |
| confirmation:r384 | 162 | 162 | 0 |

All 2897 retained run files, including partial attempts and source snapshots, are hashed in artifact-inventory.json. 1672 source snapshots match their exact stage inventories. Retry events are exported in supplement.json. Stage and global hard budgets were checked from every durable finished attempt.

The plot reads retained results only. No model is evaluated or trained to make it.

## Geometric dimension conventions

| Reserved family/cases | Reported feature parameter | Physical dimensions |
|---|---|---|
| Parallel plates, case 2 | 0.06 | Full gap 0.06 |
| Mixed thickness, cases 3/4/5 | 0.015 / 0.03 / 0.06 half-thickness | Full thin-component thickness 0.03 / 0.06 / 0.12; these values are not gaps |
| Multiscale stacks, cases 6/7 | Full gaps 0.012 and 0.05 | Component half-thicknesses 0.025 / 0.07 / 0.12; full thicknesses 0.05 / 0.14 / 0.24 |

Development narrow/wide parameters 0.012/0.03 are full parallel-plate gaps. All dimensions remain the immutable reservation values.
