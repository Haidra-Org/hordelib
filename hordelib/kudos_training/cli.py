"""The ``kudos-train`` command line: file-in/file-out wrappers over the pipeline stages.

Each stage is a plain function first (importable by an eventual scheduled runner) and a CLI second;
this module only parses arguments and prints where the artifacts landed. The ``verify`` stage is
not yet implemented and deliberately absent rather than stubbed.
"""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run one pipeline stage.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="kudos-train", description="Kudos model training pipeline stages.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    assemble_parser = subparsers.add_parser("assemble", help="Stats JSONL files to a labeled parquet snapshot.")
    assemble_parser.add_argument("--stats", type=Path, nargs="+", required=True, help="Stats JSONL file(s).")
    assemble_parser.add_argument("--machine", required=True, help="Machine id from machines.toml.")
    assemble_parser.add_argument("--out", type=Path, required=True, help="Snapshot output directory.")
    assemble_parser.add_argument(
        "--definition",
        type=Path,
        nargs="*",
        default=[],
        help="Pricing-corpus definition artifact(s) for corpus sessions.",
    )
    assemble_parser.add_argument(
        "--no-baseline-resolution",
        action="store_true",
        help="Keep worker-reported baselines instead of consulting the model reference (offline runs).",
    )

    sanitize_parser = subparsers.add_parser("sanitize", help="Apply sanitation rules to a snapshot.")
    sanitize_parser.add_argument("--snapshot", type=Path, required=True, help="Assembled snapshot parquet.")
    sanitize_parser.add_argument("--out", type=Path, required=True, help="Cleaned snapshot output directory.")

    train_parser = subparsers.add_parser("train", help="Train the LightGBM reference model.")
    train_parser.add_argument("--data", type=Path, required=True, help="Cleaned snapshot parquet.")
    train_parser.add_argument("--out", type=Path, required=True, help="Directory run directories land under.")
    train_parser.add_argument("--manifest", type=Path, default=None, help="Manifest file (default: shipped revision).")

    evaluate_parser = subparsers.add_parser("evaluate", help="Per-cell report, spread ratios, v21 comparison.")
    evaluate_parser.add_argument("--run", type=Path, required=True, help="A train-stage run directory.")
    evaluate_parser.add_argument("--data", type=Path, required=True, help="The cleaned snapshot the run trained on.")
    evaluate_parser.add_argument("--against", type=Path, default=None, help="The live v21 npz, for the comparison.")
    evaluate_parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Policy ledger file the composed prices are charged under (default: shipped revision).",
    )

    export_parser = subparsers.add_parser("export", help="Distil the reference model into the served npz MLP.")
    export_parser.add_argument("--run", type=Path, required=True, help="A train-stage run directory.")
    export_parser.add_argument("--data", type=Path, required=True, help="The cleaned snapshot the run trained on.")
    export_parser.add_argument("--trials", type=int, default=None, help="Hyperparameter trials (default: 50).")
    export_parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Policy ledger the golden vectors' prices compose under (default: shipped revision).",
    )

    args = parser.parse_args(argv)

    if args.stage == "assemble":
        from hordelib.kudos_training.assemble import assemble

        assembly = assemble(
            list(args.stats),
            machine_id=args.machine,
            out_dir=args.out,
            definition_paths=list(args.definition),
            resolve_baselines=not args.no_baseline_resolution,
        )
        print(f"snapshot: {assembly.snapshot_path}")
        print(f"rows: {assembly.total_rows} (hash {assembly.content_hash})")
        for session in assembly.sessions:
            print(
                f"  {session.stats_file}: {session.rows} rows ({session.source_kind}), "
                f"{session.faulted_rows} faulted, missing positions {list(session.missing_positions) or 'none'}",
            )
        return 0

    if args.stage == "sanitize":
        from hordelib.kudos_training.sanitize import sanitize

        cleaned = sanitize(args.snapshot, out_dir=args.out)
        print(f"clean snapshot: {cleaned.clean_path}")
        print(f"rows: {cleaned.rows_in} -> {cleaned.rows_out}")
        for rule, dropped in cleaned.dropped_by_rule.items():
            print(f"  {rule}: dropped {dropped}")
        return 0

    if args.stage == "train":
        from hordelib.kudos_training.manifest import load_manifest
        from hordelib.kudos_training.train import train

        manifest = load_manifest(args.manifest) if args.manifest is not None else None
        run = train(args.data, out_dir=args.out, manifest=manifest)
        print(f"run: {run.run_dir} (split: {run.split_mode}, rows: {run.rows})")
        for split_name, metrics in run.metrics.items():
            print(f"  {split_name}: median APE {metrics['median_ape']:.3f}, p90 APE {metrics['p90_ape']:.3f}")
        return 0

    from hordelib.kudos_training.ledger import load_ledger

    if args.stage == "export":
        from hordelib.kudos_training.export import ExportConfig, export

        export_ledger = load_ledger(args.ledger) if args.ledger is not None else None
        export_config = ExportConfig(trials=args.trials) if args.trials is not None else None
        exported = export(args.run, args.data, ledger=export_ledger, config=export_config)
        print(f"artifact: {exported.model_path}")
        print(f"bundle: {exported.metadata_path}, {exported.golden_vectors_path}")
        print(
            f"acceptance on {exported.held_out_rows} held-out rows: "
            f"median APE {exported.median_ape:.4f}, p90 APE {exported.p90_ape:.4f}",
        )
        print(f"basis job: {exported.basis_seconds} seconds")
        return 0

    from hordelib.kudos_training.evaluate import evaluate

    ledger = load_ledger(args.ledger) if args.ledger is not None else None
    evaluation = evaluate(args.run, args.data, against_npz=args.against, ledger=ledger)
    print(f"report: {evaluation.report_path}")
    print(f"candidate pay-per-second spread: {evaluation.candidate_spread}")
    if evaluation.v21_spread is not None:
        print(f"v21 pay-per-second spread:       {evaluation.v21_spread}")
    print(f"prices composed under ledger {evaluation.ledger_version}")
    if evaluation.rows_without_ledger_baseline:
        print(f"  {evaluation.rows_without_ledger_baseline} rows left unpriced (baseline absent from the ledger)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
