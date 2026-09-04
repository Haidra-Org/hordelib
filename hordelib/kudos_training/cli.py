"""The ``kudos-train`` command line: file-in/file-out wrappers over the pipeline stages.

Each stage is a plain function first (importable by an eventual scheduled runner) and a CLI second;
this module only parses arguments and prints where the artifacts landed. The ``verify`` stage is
not yet implemented and deliberately absent rather than stubbed.
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hordelib.kudos_training.assemble import AssemblyResult, MachineDescriptor


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
    assemble_parser.add_argument("--stats", type=Path, nargs="+", default=[], help="Stats JSONL file(s).")
    assemble_parser.add_argument(
        "--session",
        type=Path,
        nargs="+",
        default=[],
        help="Definition artifact(s) whose stats session and machine are discovered beside them.",
    )
    assemble_parser.add_argument("--machine", default=None, help="Machine id from machines.toml.")
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
    assemble_parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        dest="allow_manifest_mismatch",
        help="Assemble a definition encoded under another feature manifest revision, warning instead of refusing.",
    )

    machines_parser = subparsers.add_parser("machines", help="Inspect and extend the trusted-machines table.")
    machines_actions = machines_parser.add_subparsers(dest="machines_action", required=True)
    machines_actions.add_parser("list", help="Print the trusted-machines table.")
    machines_add_parser = machines_actions.add_parser("add", help="Register the machine a definition artifact names.")
    machines_add_parser.add_argument(
        "--from-definition",
        type=Path,
        required=True,
        dest="from_definition",
        help="Definition artifact carrying the machine descriptor.",
    )
    machines_add_parser.add_argument("--notes", default=None, help="Free-text note recorded beside the entry.")

    ingest_parser = subparsers.add_parser("ingest", help="Register a machine if needed, then assemble its session.")
    ingest_parser.add_argument(
        "--session",
        type=Path,
        nargs="+",
        required=True,
        help="Definition artifact(s) whose stats session and machine are discovered beside them.",
    )
    ingest_parser.add_argument("--out", type=Path, required=True, help="Snapshot output directory.")
    ingest_parser.add_argument("--notes", default=None, help="Free-text note recorded beside a new machine entry.")
    ingest_parser.add_argument(
        "--no-baseline-resolution",
        action="store_true",
        help="Keep worker-reported baselines instead of consulting the model reference (offline runs).",
    )
    ingest_parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        dest="allow_manifest_mismatch",
        help="Assemble a definition encoded under another feature manifest revision, warning instead of refusing.",
    )

    calibrate_parser = subparsers.add_parser("calibrate", help="Map other machines' seconds onto the reference.")
    calibrate_parser.add_argument("--data", type=Path, required=True, help="Cleaned snapshot parquet.")
    calibrate_parser.add_argument("--out", type=Path, required=True, help="Calibration output directory.")
    calibrate_parser.add_argument(
        "--reference",
        default=None,
        help="Reference machine id (default: the policy ledger's reference machine).",
    )
    calibrate_parser.add_argument(
        "--bar",
        type=float,
        default=None,
        help="Largest admissible residual spread per machine (default: the pricing fairness bar, 1.5).",
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

    if args.stage == "machines":
        return _run_machines(args)

    if args.stage == "ingest":
        return _run_ingest(args, parser)

    if args.stage == "assemble":
        return _run_assemble(args, parser)

    if args.stage == "calibrate":
        return _run_calibrate(args)

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
    if evaluation.in_regime_spread is not None:
        print(f"  in-regime cells:     {evaluation.in_regime_spread}")
    if evaluation.out_of_regime_spread is not None:
        print(f"  out-of-regime cells: {evaluation.out_of_regime_spread}")
    print(f"prices composed under ledger {evaluation.ledger_version}")
    if evaluation.rows_without_ledger_baseline:
        print(f"  {evaluation.rows_without_ledger_baseline} rows left unpriced (baseline absent from the ledger)")
    return 0


def _resolve_sessions(session_paths: list[Path]) -> tuple[list[Path], list[Path], list["MachineDescriptor"]]:
    """Resolve definition artifacts into their stats sessions.

    Args:
        session_paths: Definition artifacts naming their own sessions.

    Returns:
        The stats files to assemble, the definition artifacts to pair against, and the machine
        descriptors the artifacts carry.
    """
    from hordelib.kudos_training.assemble import resolve_session

    stats_paths: list[Path] = []
    definition_paths: list[Path] = []
    machines: list[MachineDescriptor] = []
    for session_path in session_paths:
        session = resolve_session(session_path)
        stats_paths.extend(session.stats_paths)
        definition_paths.append(session_path)
        if session.machine is not None:
            machines.append(session.machine)
    return stats_paths, definition_paths, machines


def _session_machine_id(
    machines: list["MachineDescriptor"],
    session_paths: list[Path],
    parser: argparse.ArgumentParser,
) -> str:
    """Return the single machine id the resolved sessions agree on."""
    if len(machines) != len(session_paths):
        parser.error(
            "a definition artifact carries no machine descriptor; pass --stats, --definition and --machine explicitly",
        )
    machine_ids = sorted({machine.machine_id for machine in machines})
    if len(machine_ids) > 1:
        parser.error(f"sessions span several machines ({', '.join(machine_ids)}); assemble them one machine at a time")
    return machine_ids[0]


def _print_assembly(assembly: "AssemblyResult") -> None:
    """Print where a snapshot landed and what each session contributed to it."""
    print(f"snapshot: {assembly.snapshot_path}")
    print(f"rows: {assembly.total_rows} (hash {assembly.content_hash})")
    for session in assembly.sessions:
        print(
            f"  {session.stats_file}: {session.rows} rows ({session.source_kind}), "
            f"{session.faulted_rows} faulted, missing positions {list(session.missing_positions) or 'none'}",
        )


def _run_assemble(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the assemble stage from explicit files or from session artifacts."""
    from hordelib.kudos_training.assemble import assemble

    if args.session and args.stats:
        parser.error("pass either --session or --stats, not both")
    if not args.session and not args.stats:
        parser.error("assemble needs --session, or --stats with --machine")

    if args.session:
        stats_paths, definition_paths, machines = _resolve_sessions(list(args.session))
        definition_paths.extend(args.definition)
        machine_id = args.machine if args.machine is not None else _session_machine_id(machines, args.session, parser)
    else:
        if args.machine is None:
            parser.error("--machine is required with --stats")
        stats_paths = list(args.stats)
        definition_paths = list(args.definition)
        machine_id = args.machine

    assembly = assemble(
        stats_paths,
        machine_id=machine_id,
        out_dir=args.out,
        definition_paths=definition_paths,
        resolve_baselines=not args.no_baseline_resolution,
        allow_manifest_mismatch=args.allow_manifest_mismatch,
    )
    _print_assembly(assembly)
    return 0


def _run_machines(args: argparse.Namespace) -> int:
    """List the trusted-machines table or register a machine from a definition artifact."""
    from hordelib.kudos_training.assemble import (
        DEFAULT_MACHINES_PATH,
        add_machine,
        format_machines_table,
        load_machines,
    )

    if args.machines_action == "list":
        print(format_machines_table(load_machines()))
        return 0

    from hordelib.kudos_training.assemble import CorpusDefinition

    definition = CorpusDefinition.model_validate_json(args.from_definition.read_text(encoding="utf-8"))
    if definition.machine is None:
        print(f"{args.from_definition} carries no machine descriptor; add the entry by hand", file=sys.stderr)
        return 1
    entry = add_machine(definition.machine, notes=args.notes)
    print(f"registered {definition.machine.machine_id} in {DEFAULT_MACHINES_PATH}")
    for key, value in entry.items():
        print(f"  {key} = {value!r}")
    return 0


def _run_ingest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Register a session's machine if it is new, then assemble the session."""
    from hordelib.kudos_training.assemble import (
        add_machine,
        assemble,
        load_machines,
        warn_on_machine_facts_mismatch,
    )

    stats_paths, definition_paths, machines = _resolve_sessions(list(args.session))
    machine_id = _session_machine_id(machines, args.session, parser)
    registered = load_machines()
    if machine_id not in registered:
        add_machine(machines[0], notes=args.notes)
        print(f"registered new machine {machine_id}")
    else:
        warn_on_machine_facts_mismatch(machines[0], registered[machine_id])

    assembly = assemble(
        stats_paths,
        machine_id=machine_id,
        out_dir=args.out,
        definition_paths=definition_paths,
        resolve_baselines=not args.no_baseline_resolution,
        allow_manifest_mismatch=args.allow_manifest_mismatch,
    )
    _print_assembly(assembly)
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    """Map every machine onto the reference machine, refusing the ones the bar rejects."""
    from hordelib.kudos_training.calibrate import DEFAULT_SPREAD_BAR, calibrate, format_calibration_table

    result = calibrate(
        args.data,
        out_dir=args.out,
        reference_machine=args.reference,
        bar=args.bar if args.bar is not None else DEFAULT_SPREAD_BAR,
    )
    print(f"report: {result.report_path}")
    print(f"reference machine: {result.reference_machine} (bar {result.bar})")
    print(format_calibration_table(result))
    if not result.passed:
        print("calibration failed the bar; no calibrated snapshot written", file=sys.stderr)
        return 1
    print(f"calibrated snapshot: {result.calibrated_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
