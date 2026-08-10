#!/usr/bin/env python3
"""
Progress report for exp09 Calvano-ladder runs.

Counts paired run_<id>.parquet + run_<id>_qtable.parquet files per
(cell, mu, N) against the expected grid.

Usage:
  python hpc/exp09_progress_report.py --rounds 30
  python hpc/exp09_progress_report.py --root /path/to/data/results_exp09 --rounds 30
  python hpc/exp09_progress_report.py --cell cal_full --rounds 30
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

RUN_FILE_RE = re.compile(r"^run_(\d+)\.parquet$")
QTABLE_FILE_RE = re.compile(r"^run_(\d+)_qtable\.parquet$")

# The five rungs and the K each one runs at. Longer tags must precede their
# prefixes so the scan-dir regex prefers "cal_both_k30" over "cal_both".
CELL_K: Dict[str, int] = {
    "cal_full": 1,
    "cal_smin": 1,
    "cal_arule": 1,
    "cal_both_k30": 30,
    "cal_both": 1,
}
SCAN_DIR_RE = re.compile(
    r"^scan_(" + "|".join(CELL_K.keys()) + r")_mu(.+)_k(.+)$"
)


def _default_root() -> Path:
    bigdata = Path("~/bigdata/pricing_marl/data/results_exp09").expanduser()
    if bigdata.exists():
        return bigdata
    return Path(__file__).resolve().parent.parent / "data" / "results_exp09"


def _collect_run_ids(n_dir: Path) -> Tuple[Set[int], Set[int]]:
    eval_ids: Set[int] = set()
    qtable_ids: Set[int] = set()
    for p in n_dir.glob("run_*.parquet"):
        m_eval = RUN_FILE_RE.match(p.name)
        if m_eval:
            eval_ids.add(int(m_eval.group(1)))
            continue
        m_q = QTABLE_FILE_RE.match(p.name)
        if m_q:
            qtable_ids.add(int(m_q.group(1)))
    return eval_ids, qtable_ids


def _list_recent(root: Path, limit: int, kind: str) -> List[Tuple[float, Path]]:
    files: List[Tuple[float, Path]] = []
    for p in root.rglob("run_*.parquet"):
        is_eval = RUN_FILE_RE.match(p.name) is not None
        is_qtable = QTABLE_FILE_RE.match(p.name) is not None
        if kind == "eval" and not is_eval:
            continue
        if kind == "qtable" and not is_qtable:
            continue
        if not is_eval and not is_qtable:
            continue
        try:
            files.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue
    files.sort(key=lambda x: x[0])
    return files[-limit:]


def _extract_list_values(tree: ast.AST, name: str) -> Optional[List]:
    last_values: Optional[List] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        try:
                            val = ast.literal_eval(node.value)
                        except Exception:
                            continue
                        if isinstance(val, (list, tuple)):
                            last_values = list(val)
    return last_values


def _infer_grid_values(grid_file: Path) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    if not grid_file.exists():
        return None, None
    try:
        tree = ast.parse(grid_file.read_text())
    except Exception:
        return None, None

    n_vals_raw = _extract_list_values(tree, "N_VALUES")
    mu_vals_raw = _extract_list_values(tree, "MU_VALUES")
    if n_vals_raw is None or mu_vals_raw is None:
        return None, None
    try:
        return [int(v) for v in n_vals_raw], [float(v) for v in mu_vals_raw]
    except Exception:
        return None, None


def _parse_int_csv(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    return vals or None


def _parse_float_csv(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    vals = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    return vals or None


def _float_in(values: Sequence[float], x: float, tol: float = 1e-12) -> bool:
    return any(abs(v - x) <= tol for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Progress report for exp09 Calvano-ladder runs.")
    parser.add_argument("--root", type=str, default=None, help="Path to data/results_exp09")
    parser.add_argument("--rounds", type=int, default=30, help="Runs per cell/mu/N")
    parser.add_argument("--recent", type=int, default=10, help="Show N most recent files")
    parser.add_argument(
        "--recent-kind", type=str, choices=["all", "eval", "qtable"], default="all",
        help="Recent file type filter (default: all)",
    )
    parser.add_argument(
        "--grid-file", type=str, default=None,
        help="Path to exp09_calvano_ladder.py (to infer N/Mu values)",
    )
    parser.add_argument("--n-values", type=str, default=None, help="Comma-separated N values")
    parser.add_argument("--mu-values", type=str, default=None, help="Comma-separated mu values")
    parser.add_argument(
        "--cell", type=str, default=None, choices=sorted(CELL_K.keys()),
        help="Report only one ladder cell",
    )
    parser.add_argument(
        "--include-extra-dirs", action="store_true",
        help="Include result folders outside inferred/requested values",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else _default_root()
    if not root.exists():
        print(f"[ERROR] Results root not found: {root}")
        return

    grid_file = Path(args.grid_file).expanduser() if args.grid_file else (
        Path(__file__).resolve().parent.parent / "experiments" / "exp09_calvano_ladder.py"
    )
    n_values, mu_values = _infer_grid_values(grid_file)

    cli_n_values = _parse_int_csv(args.n_values)
    cli_mu_values = _parse_float_csv(args.mu_values)
    if cli_n_values is not None:
        n_values = cli_n_values
    if cli_mu_values is not None:
        mu_values = cli_mu_values

    # Report in ladder order (C1..C5), not the regex-priority order.
    ladder = ["cal_full", "cal_smin", "cal_arule", "cal_both", "cal_both_k30"]
    cells = [args.cell] if args.cell else ladder

    print(f"Results root: {root}")
    print(f"Rounds per config: {args.rounds}")
    print(f"Grid file: {grid_file}")
    if n_values is not None:
        print(f"N values: {n_values}")
    if mu_values is not None:
        print(f"Mu values: {mu_values}")
    print("-" * 60)

    overall_eval = 0
    overall_qtable = 0
    overall_paired = 0
    overall_expected = 0

    for cell in cells:
        k_val = CELL_K[cell]
        scan_dirs_all = sorted(root.glob(f"scan_{cell}_mu*"))
        scan_dirs = []
        ignored_scan_dirs = 0
        for scan in scan_dirs_all:
            m = SCAN_DIR_RE.match(scan.name)
            # The glob for "cal_both" cannot pick up "cal_both_k30" (the "_mu"
            # anchor differs), but re-check the parsed tag to be safe.
            if m is None or m.group(1) != cell:
                ignored_scan_dirs += 1
                continue
            if not args.include_extra_dirs and mu_values is not None:
                try:
                    mu_val = float(m.group(2))
                except ValueError:
                    ignored_scan_dirs += 1
                    continue
                if not _float_in(mu_values, mu_val):
                    ignored_scan_dirs += 1
                    continue
            scan_dirs.append(scan)

        n_dirs = 0
        total_eval = 0
        total_qtable = 0
        total_paired = 0
        incomplete: List[Tuple[Path, int, int, int]] = []
        missing_qtable: List[Tuple[Path, int]] = []
        orphan_qtable: List[Tuple[Path, int]] = []

        for scan in scan_dirs:
            n_subdirs = sorted(p for p in scan.iterdir() if p.is_dir() and p.name.startswith("N_"))
            if not args.include_extra_dirs and n_values is not None:
                kept = []
                for n_dir in n_subdirs:
                    try:
                        if int(n_dir.name.split("_", 1)[1]) in n_values:
                            kept.append(n_dir)
                    except Exception:
                        continue
                n_subdirs = kept

            for n_dir in n_subdirs:
                n_dirs += 1
                eval_ids, qtable_ids = _collect_run_ids(n_dir)
                paired = eval_ids & qtable_ids
                total_eval += len(eval_ids)
                total_qtable += len(qtable_ids)
                total_paired += len(paired)

                if len(paired) < args.rounds:
                    incomplete.append((n_dir, len(eval_ids), len(qtable_ids), len(paired)))
                if eval_ids - qtable_ids:
                    missing_qtable.append((n_dir, len(eval_ids - qtable_ids)))
                if qtable_ids - eval_ids:
                    orphan_qtable.append((n_dir, len(qtable_ids - eval_ids)))

        missing_expected_dirs: List[Path] = []
        expected_by_grid = None
        if n_values is not None and mu_values is not None:
            expected_by_grid = len(n_values) * len(mu_values) * args.rounds
            for mu_val in mu_values:
                scan = root / f"scan_{cell}_mu{mu_val}_k{k_val}"
                for n_val in n_values:
                    n_dir = scan / f"N_{n_val}"
                    if not n_dir.exists():
                        missing_expected_dirs.append(n_dir)

        expected_by_dirs = n_dirs * args.rounds
        expected = expected_by_grid or expected_by_dirs

        eval_pct = (total_eval / expected * 100.0) if expected else 0.0
        q_pct = (total_qtable / expected * 100.0) if expected else 0.0
        paired_pct = (total_paired / expected * 100.0) if expected else 0.0

        print(f"[{cell}]  (K={k_val})")
        print(f"  scan dirs: {len(scan_dirs)}")
        print(f"  N dirs:    {n_dirs}")
        if ignored_scan_dirs:
            print(f"  ignored scan dirs (outside grid): {ignored_scan_dirs}")
        print(f"  eval found:    {total_eval}")
        print(f"  qtable found:  {total_qtable}")
        print(f"  paired found:  {total_paired}")
        if expected_by_grid is not None:
            print(f"  expected (grid): {expected_by_grid}")
        print(f"  expected (dirs): {expected_by_dirs}")
        print(f"  eval progress:   {eval_pct:.2f}%")
        print(f"  qtable progress: {q_pct:.2f}%")
        print(f"  paired progress: {paired_pct:.2f}%")

        if missing_expected_dirs:
            print("  missing expected N dirs:")
            for path in missing_expected_dirs[:20]:
                print(f"    {path}")
            if len(missing_expected_dirs) > 20:
                print(f"    ... {len(missing_expected_dirs) - 20} more")
        if incomplete:
            print("  incomplete (paired < rounds):")
            for path, e_c, q_c, p_c in incomplete[:20]:
                print(f"    {path}  eval={e_c}/{args.rounds}, "
                      f"qtable={q_c}/{args.rounds}, paired={p_c}/{args.rounds}")
            if len(incomplete) > 20:
                print(f"    ... {len(incomplete) - 20} more")
        if missing_qtable:
            print("  dirs with missing qtable for existing eval:")
            for path, miss in missing_qtable[:20]:
                print(f"    {path}  missing_qtable={miss}")
        if orphan_qtable:
            print("  dirs with orphan qtable (no matching eval):")
            for path, orph in orphan_qtable[:20]:
                print(f"    {path}  orphan_qtable={orph}")
        print("-" * 60)

        overall_eval += total_eval
        overall_qtable += total_qtable
        overall_paired += total_paired
        overall_expected += expected

    if overall_expected > 0:
        print("[ALL]")
        print(f"  expected total:  {overall_expected}")
        print(f"  eval progress:   {overall_eval / overall_expected * 100.0:.2f}%")
        print(f"  qtable progress: {overall_qtable / overall_expected * 100.0:.2f}%")
        print(f"  paired progress: {overall_paired / overall_expected * 100.0:.2f}%")
        print("-" * 60)

    if args.recent > 0:
        print(f"Most recent {args.recent} files ({args.recent_kind}):")
        for _, p in _list_recent(root, args.recent, args.recent_kind):
            print(f"  {p}")


if __name__ == "__main__":
    main()
