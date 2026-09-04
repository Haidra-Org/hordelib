# Training the kudos cost model on data from several machines

This page is for the maintainer of the cost model: the one person who receives corpus archives from
contributors, holds the training environment, and produces the model. Contributors only run the corpus
and send an archive; their instructions are the worker repository's how-to "Run a pricing corpus on
your machine". They never run anything on this page.

Everything here runs from one hordelib checkout with the `kudos-training` extra installed
(`uv pip install -e ".[kudos-training]"`); `kudos-train` is its console script.

## 1. Unpack and ingest each bundle

A contributor sends one bundle directory per run, named `corpus-<machine>-<tier>-<UTC stamp>`, holding
the run's definition JSON, its stats parts and a `bundle.json` manifest with their hashes. Unpack it
under a data directory and ingest by pointing at the definition inside it:

```bash
tar xzf corpus-alice-l40s-census-20260101T093000Z.tgz -C data
kudos-train ingest --session data/corpus-alice-l40s-census-20260101T093000Z/pricing-corpus-census-*.json --out runs/snap-alice-l40s
```

`ingest` does three things: it adds the machine to `hordelib/kudos_training/machines.toml` from the
facts the run stamped into the artifact (a one-line diff to commit; an id already in the table is left
alone), it finds the stats session that started just after the artifact was written and reads its
rotated parts as one session, and it writes a snapshot parquet under `--out`.

Before assembling, `ingest` checks each file `bundle.json` lists against the hash and size recorded
for it and refuses the run if any of them differs, so a part damaged in transit is never read as
measurements. It also refuses a definition stamped with a feature manifest revision other than the one
this checkout ships, since those rows were encoded against different feature meanings
(`--allow-manifest-mismatch` downgrades that to a warning). An id already in the table whose stamped
`gpu_model`, `vram_mb` or `os` differs from the entry warns and is left alone: one id covering two
machines would pool their clocks as one.

Your own machine's runs ingest the same way. `kudos-train machines list` shows what is registered.

## 2. Merge and sanitize

Concatenate the snapshots you intend to train on into one parquet (the `machine_id` column keeps every
row attributable), then sanitize:

```bash
kudos-train sanitize --snapshot runs/merged.parquet --out runs/clean
```

## 3. Calibrate onto the reference machine

The model is trained in the seconds of the reference machine named in
`hordelib/kudos_training/kudos_policy_ledger_v1.json`. Every other machine's rows are mapped onto it:

```bash
kudos-train calibrate --data runs/clean/clean-<hash>.parquet --out runs/calibrated
```

For each non-reference machine the stage fits `log t = a + b * f(shape)` on the cells both machines
measured and prints offset, slope, overlap count and residual spread. A machine whose spread exceeds the
bar (default 1.5, `--bar`) fails: the report is written, no calibrated parquet is produced, and its data
is refused rather than averaged in. Cells the reference never ran (the heavy tier's models) are flagged
`out_of_regime`, mapped, and kept, since they are the only measurements of those cells.

Data from the reference machine alone passes through unchanged. `train` refuses a multi-machine frame
that has not been through this stage.

## 4. Train, evaluate, export

```bash
kudos-train train --data runs/calibrated/calibrated-<hash>.parquet --out runs/train
kudos-train evaluate --run runs/train/<run> --data runs/calibrated/calibrated-<hash>.parquet --ledger hordelib/kudos_training/kudos_policy_ledger_v1.json
kudos-train export --run runs/train/<run> --data runs/calibrated/calibrated-<hash>.parquet --ledger hordelib/kudos_training/kudos_policy_ledger_v1.json
```

The evaluation report prints the per-cell spread separately for in-regime and out-of-regime cells.
