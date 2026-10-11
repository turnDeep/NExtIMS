# Evidence Snapshot

As of2026-10-11. These are aggregate saved experiment CSVs, not new accuracy runs.
`manifest.json` records relative source names, original SHA256, export SHA256 and
row counts. Local Windows/container artifact paths are redacted when present.
No compound-level NIST records, SMILES lists, MOL/MSP inputs or model weights
are distributed. The Git snapshot predates this publication commit.

Read [HISTORY](../HISTORY.md) and [RESULTS](../RESULTS.md) before comparing rows.
V001 is imported RTX5090 evidence without migrated weights. V008 is a matched
small-model revalidation, not an RTX5090 rerun. V020 and V021 have different
internal Test cohorts. V022 is in progress and has no final accuracy CSV here.

Re-export from locally retained artifacts:

```bash
python scripts/export_research_history.py --verification-root /path/to/verification --paper-summary /path/to/experiments/test_evaluation_summary.csv
```

The allowlist is intentionally bounded. Missing aggregate artifacts are recorded;
this command is not a license to publish raw data or an automatic full study audit.
