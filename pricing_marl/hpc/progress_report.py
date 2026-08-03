#!/usr/bin/env python3
"""
Simple progress report for pricing_marl heatmap runs.

Usage:
  python progress_report.py --rounds 100
  python progress_report.py --root /path/to/data/results --rounds 100
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


def _default_root() -> Path:
    """Prefer bigdata path if it exists; otherwise use repo-local data/results."""
    bigdata = Path("~/bigdata/pricing_marl/data/results").expanduser()
    if bigdata.exists():
        return bigdata
    return Path(__file__).resolve().parent.parent / "data" / "results"


RUN_FILE_RE = re.compile(r"^run_(\d+)\.parquet$")
QTABLE_FILE_RE = re.compile(r"^run_(\d+)_qtable\.parquet$")


def _collect_run_ids(n_dir: Path) -> Tuple[Set[int], Set[int]]:
    """Return (eval_run_ids, qtable_run_ids) for one N directory."""
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


def _extract_list_len(tree: ast.AST, name: str) -> Optional[int]:
    """Find the last assignment to `name` and return list length if literal."""
    last_len: Optional[int] = None
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
                            last_len = len(val)
    return last_len


def _extract_list_values(tree: ast.AST, name: str) -> Optional[List]:
    """Find the last literal list/tuple assignment to `name` and return values."""
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


def _infer_grid_sizes(grid_file: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not grid_file.exists():
        return None, None, None
    try:
        tree = ast.parse(grid_file.read_text())
    except Exception:
        return None, None, None
    n_count = _extract_list_len(tree, "N_VALUES")
    mu_count = _extract_list_len(tree, "MU_VALUES")
    k_count = _extract_list_len(tree, "K_VALUES")
    return n_count, mu_count, k_count


def _infer_grid_values(grid_file: Path) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[int]]]:
    if not grid_file.exists():
        return None, None, None
    try:
        tree = ast.parse(grid_file.read_text())
    except Exception:
        return None, None, None

    n_vals_raw = _extract_list_values(tree, "N_VALUES")
    mu_vals_raw = _extract_list_values(tree, "MU_VALUES")
    k_vals_raw = _extract_list_values(tree, "K_VALUES")
    if n_vals_raw is None or mu_vals_raw is None or k_vals_raw is None:
        return None, None, None

    try:
        n_vals = [int(v) for v in n_vals_raw]
        mu_vals = [float(v) for v in mu_vals_raw]
        k_vals = [int(v) for v in k_vals_raw]
    except Exception:
        return None, None, None
    return n_vals, mu_vals, k_vals


SCAN_DIR_RE = re.compile(r"^scan_(3strats|4strats)_mu(.+)_k(.+)$")


def _parse_scan_dir(scan_dir_name: str) -> Optional[Tuple[str, float, int]]:
    m = SCAN_DIR_RE.match(scan_dir_name)
    if not m:
        return None
    label = m.group(1)
    try:
        mu_val = float(m.group(2))
        k_val = int(float(m.group(3)))
    except ValueError:
        return None
    return label, mu_val, k_val


def _float_in(values: Sequence[float], x: float, tol: float = 1e-12) -> bool:
    return any(abs(v - x) <= tol for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report progress for 3strats/4strats runs.")
    parser.add_argument("--root", type=str, default=None, help="Path to data/results")
    parser.add_argument("--rounds", type=int, default=100, help="Runs per N (default: 100)")
    parser.add_argument("--recent", type=int, default=10, help="Show N most recent files")
    parser.add_argument(
        "--recent-kind",
        type=str,
        choices=["all", "eval", "qtable"],
        default="all",
        help="Recent file type filter (default: all)",
    )
    parser.add_argument(
        "--grid-file",
        type=str,
        default=None,
        help="Path to exp02_heatmap_scan.py (to infer N/Mu/K sizes)",
    )
    parser.add_argument("--n-count", type=int, default=None, help="Override N_VALUES length")
    parser.add_argument("--mu-count", type=int, default=None, help="Override MU_VALUES length")
    parser.add_argument("--k-count", type=int, default=None, help="Override K_VALUES length")
    parser.add_argument(
        "--include-extra-dirs",
        action="store_true",
        help="Include result folders outside current grid-file values (default: filtered out)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else _default_root()
    if not root.exists():
        print(f"[ERROR] Results root not found: {root}")
        return

    print(f"Results root: {root}")
    print(f"Rounds per config: {args.rounds}")
    # Infer grid sizes (optional)
    grid_file = Path(args.grid_file).expanduser() if args.grid_file else (
        Path(__file__).resolve().parent.parent / "experiments" / "exp02_heatmap_scan.py"
    )
    n_count, mu_count, k_count = _infer_grid_sizes(grid_file)
    n_values, mu_values, k_values = _infer_grid_values(grid_file)
    if args.n_count is not None:
        n_count = args.n_count
    if args.mu_count is not None:
        mu_count = args.mu_count
    if args.k_count is not None:
        k_count = args.k_count

    print("-" * 60)
    if n_count and mu_count and k_count:
        print(f"Grid sizes: N={n_count}, Mu={mu_count}, K={k_count}")
        print(f"Expected runs per strategy: {n_count * mu_count * k_count * args.rounds}")
        print("-" * 60)

    labels = ["4strats", "3strats"]
    overall_eval = 0
    overall_qtable = 0
    overall_paired = 0
    overall_expected = 0

    for label in labels:
        scan_dirs_all = sorted(root.glob(f"scan_{label}_*"))
        scan_dirs = scan_dirs_all
        ignored_scan_dirs = 0
        ignored_n_dirs = 0
        if not args.include_extra_dirs and mu_values is not None and k_values is not None:
            filtered_scan_dirs = []
            for scan in scan_dirs_all:
                parsed = _parse_scan_dir(scan.name)
                if parsed is None:
                    ignored_scan_dirs += 1
                    continue
                _, mu_val, k_val = parsed
                if _float_in(mu_values, mu_val) and (k_val in k_values):
                    filtered_scan_dirs.append(scan)
                else:
                    ignored_scan_dirs += 1
            scan_dirs = filtered_scan_dirs

        n_dirs = 0
        total_eval = 0
        total_qtable = 0
        total_paired = 0
        incomplete: List[Tuple[Path, int, int, int]] = []
        missing_qtable: List[Tuple[Path, int]] = []
        orphan_qtable: List[Tuple[Path, int]] = []

        for scan in scan_dirs:
            n_subdirs_all = sorted(p for p in scan.iterdir() if p.is_dir() and p.name.startswith("N_"))
            n_subdirs = n_subdirs_all
            if not args.include_extra_dirs and n_values is not None:
                filtered_n_subdirs = []
                for n_dir in n_subdirs_all:
                    try:
                        n_val = int(n_dir.name.split("_", 1)[1])
                    except Exception:
                        ignored_n_dirs += 1
                        continue
                    if n_val in n_values:
                        filtered_n_subdirs.append(n_dir)
                    else:
                        ignored_n_dirs += 1
                n_subdirs = filtered_n_subdirs

            for n_dir in n_subdirs:
                n_dirs += 1
                eval_ids, qtable_ids = _collect_run_ids(n_dir)
                paired = eval_ids & qtable_ids
                missing_q = eval_ids - qtable_ids
                orphan_q = qtable_ids - eval_ids

                eval_count = len(eval_ids)
                q_count = len(qtable_ids)
                paired_count = len(paired)

                total_eval += eval_count
                total_qtable += q_count
                total_paired += paired_count

                if paired_count < args.rounds:
                    incomplete.append((n_dir, eval_count, q_count, paired_count))
                if missing_q:
                    missing_qtable.append((n_dir, len(missing_q)))
                if orphan_q:
                    orphan_qtable.append((n_dir, len(orphan_q)))

        expected_by_dirs = n_dirs * args.rounds
        if n_count and mu_count and k_count:
            expected_by_grid = n_count * mu_count * k_count * args.rounds
        else:
            expected_by_grid = None
        expected = expected_by_grid or expected_by_dirs

        eval_pct = (total_eval / expected * 100.0) if expected else 0.0
        q_pct = (total_qtable / expected * 100.0) if expected else 0.0
        paired_pct = (total_paired / expected * 100.0) if expected else 0.0

        print(f"[{label}]")
        print(f"  scan dirs: {len(scan_dirs)}")
        print(f"  N dirs:    {n_dirs}")
        if ignored_scan_dirs:
            print(f"  ignored scan dirs (outside grid): {ignored_scan_dirs}")
        if ignored_n_dirs:
            print(f"  ignored N dirs (outside grid):    {ignored_n_dirs}")
        print(f"  eval found:    {total_eval}")
        print(f"  qtable found:  {total_qtable}")
        print(f"  paired found:  {total_paired}")
        if expected_by_grid is not None:
            print(f"  expected (grid):  {expected_by_grid}")
        print(f"  expected (dirs): {expected_by_dirs}")
        print(f"  eval progress:   {eval_pct:.2f}%")
        print(f"  qtable progress: {q_pct:.2f}%")
        print(f"  paired progress: {paired_pct:.2f}%")

        if incomplete:
            print("  incomplete (paired < rounds):")
            for path, eval_count, q_count, paired_count in incomplete[:20]:
                print(
                    f"    {path}  eval={eval_count}/{args.rounds}, "
                    f"qtable={q_count}/{args.rounds}, paired={paired_count}/{args.rounds}"
                )
            if len(incomplete) > 20:
                print(f"    ... {len(incomplete) - 20} more")
        if missing_qtable:
            print("  dirs with missing qtable for existing eval:")
            for path, miss_count in missing_qtable[:20]:
                print(f"    {path}  missing_qtable={miss_count}")
            if len(missing_qtable) > 20:
                print(f"    ... {len(missing_qtable) - 20} more")
        if orphan_qtable:
            print("  dirs with orphan qtable (no matching eval):")
            for path, orphan_count in orphan_qtable[:20]:
                print(f"    {path}  orphan_qtable={orphan_count}")
            if len(orphan_qtable) > 20:
                print(f"    ... {len(orphan_qtable) - 20} more")
        print("-" * 60)

        overall_eval += total_eval
        overall_qtable += total_qtable
        overall_paired += total_paired
        overall_expected += expected

    if overall_expected > 0:
        print("[ALL]")
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
