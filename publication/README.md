# Public evidence and retained originals

The completed experiment saves 26.00% of full fixed-trajectory wall time with cycling, while first observed persistent correctness is 11.30% faster on 12 jointly successful pairs. Every arm succeeds on 12/18 pairs and fails all six narrow-stack pairs. The reliability target is not achieved.

Public copies normalize source documentation and machine-specific metadata. All measured numbers, model configurations, sampling schedules, gates, outcomes, times, CSVs and figures are preserved. Documentation and metadata normalization leaves training and evaluation calculations unchanged. Technical literature credits remain.

`thin-feature-manifest.json` retains original file hashes and deliberate raw omissions. `public-normalization.json` records original and public SHA-256/byte counts and deduplicated deterministic byte-offset edits. Recipes contain replacement bytes, not removed text. One archival document is deliberately omitted. Original seals and historical hash tables remain references to original bytes; edited public files are not silently resealed as originals.

Compact verification checks current public bytes, historical references through the explicit mapping, all 105 final output references and all 2,897 run artifact records:

```powershell
python scripts/verify_thin_publication.py
python scripts/verify_thin_publication.py --git-ref HEAD
```

The mapping file itself is compared with Git when `--git-ref` is used. As with any unsigned evidence manifest, the trusted reference is the reviewed commit; editing both files and their mapping changes that reference.

The optional `--original-archive path/to/private-originals.zip` checks original public-tree hashes and applies every transformation recipe against retained originals. `--local-artifacts path/to/retained-workspace` verifies deliberately omitted raw `.pt` and `.npy` files. The private originals and approximately 12.66GB of raw data remain outside this repository. Full numerical replay and original-seal verification require original sources, raw artifacts and the recorded environment. The public scientific verifier and reporter are not substitutes for those prerequisites. Consumed reservations should not be treated as new experiments.

CI uses `python -m pytest` with clone-local `.publication-test-temp`, followed by the public Git-blob verifier. Publication infrastructure is separate from the original scientific seals. Initial CI failures and repairs remain recorded in validation.json.
