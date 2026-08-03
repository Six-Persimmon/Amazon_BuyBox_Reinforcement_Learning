#!/usr/bin/env python3
"""
Progress report for exp04 fixed-K choice runs.

Usage:
  python exp04_progress_report.py --rounds 30
  python exp04_progress_report.py --root /path/to/data/results_exp04 --rounds 30
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


RUN_FILE_RE = re.compile(r"^run_(\d+)\.parquet$")
QTABLE_FILE_RE = re.compile(r"^run_(\d+)_qtable\.parquet$")
SCAN_DIR_RE = re.compile(r"^scan_fixk_(3strats|4strats)_mu(.+)_K(.+)$")


def _default_root() -> Path:
    bigdata = Path("~/bigdata/pricing_marl/data/results_exp04").expanduser()
    if bigdata.exists():
        return bigdata
    return Path(__file__).resolve().parent.parent / "data" / "results_exp04"


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


def _infer_grid_values(
    grid_file: Path,
) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[Tuple[int, ...]]]]:
    if not grid_file.exists():
        return None, None, None
    try:
        tree = ast.parse(grid_file.read_text())
    except Exception:
        return None, None, None

    n_vals_raw = _extract_list_values(tree, "N_VALUES")
    mu_vals_raw = _extract_list_values(tree, "MU_VALUES")
    k_profiles_raw = _extract_list_values(tree, "K_PROFILES")
    if n_vals_raw is None or mu_vals_raw is None or k_profiles_raw is None:
        return None, None, None

    try:
        n_vals = [int(v) for v in n_vals_raw]
        mu_vals = [float(v) for v in mu_vals_raw]
        k_profiles = [tuple(int(x) for x in profile) for profile in k_profiles_raw]
    except Exception:
        return None, None, None
    return n_vals, mu_vals, k_profiles


def _parse_scan_dir(scan_dir_name: str) -> Optional[Tuple[str, float, str]]:
    m = SCAN_DIR_RE.match(scan_dir_name)
    if not m:
        return None
    try:
        mu_val = float(m.group(2))
    except ValueError:
        return None
    return m.group(1), mu_val, m.group(3)


def _float_in(values: Sequence[float], x: float, tol: float = 1e-12) -> bool:
    return any(abs(v - x) <= tol for v in values)


def _parse_int_csv(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    vals = [int(token.strip()) for token in raw.split(",") if token.strip()]
    return vals or None


def _parse_float_csv(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    vals = [float(token.strip()) for token in raw.split(",") if token.strip()]
    return vals or None


def _parse_profile_csv(raw: Optional[str]) -> Optional[List[Tuple[int, ...]]]:
    if raw is None:
        return None
    vals = [
        tuple(int(k.strip()) for k in token.split("-") if k.strip())
        for token in raw.split(",")
        if token.strip()
    ]
    return vals or None


def _profile_label(profile: Sequence[int]) -> str:
    return "-".join(str(int(v)) for v in profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report progress for exp04 fixed-K choice runs.")
    parser.add_argument("--root", type=str, default=None, help="Path to data/results_exp04")
    parser.add_argument("--rounds", type=int, default=30, help="Runs per N/mu/profile/set")
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
        help="Path to exp04_fix_k_choice.py (to infer N/Mu/K profiles)",
    )
    parser.add_argument("--n-values", type=str, default=None, help="Comma-separated N values")
    parser.add_argument("--mu-values", type=str, default=None, help="Comma-separated mu values")
    parser.add_argument(
        "--k-profiles",
        type=str,
        default=None,
        help="Comma-separated K profiles, e.g. 10-10-10,10-10-30",
    )
    parser.add_argument(
        "--experiment-set",
        type=str,
        choices=["3strats", "4strats"],
        default=None,
        help="Report only one strategy set",
    )
    parser.add_argument(
        "--include-extra-dirs",
        action="store_true",
        help="Include result folders outside inferred/requested values",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else _default_root()
    if not root.exists():
        print(f"[ERROR] Results root not found: {root}")
        return

    grid_file = Path(args.grid_file).expanduser() if args.grid_file else (
        Path(__file__).resolve().parent.parent / "experiments" / "exp04_fix_k_choice.py"
    )
    n_values, mu_values, k_profiles = _infer_grid_values(grid_file)

    cli_n_values = _parse_int_csv(args.n_values)
    cli_mu_values = _parse_float_csv(args.mu_values)
    cli_k_profiles = _parse_profile_csv(args.k_profiles)
    if cli_n_values is not None:
        n_values = cli_n_values
    if cli_mu_values is not None:
        mu_values = cli_mu_values
    if cli_k_profiles is not None:
        k_profiles = cli_k_profiles

    expected_profile_labels = [_profile_label(profile) for profile in k_profiles] if k_profiles else None
    labels = [args.experiment_set] if args.experiment_set else ["4strats", "3strats"]

    print(f"Results root: {root}")
    print(f"Rounds per config: {args.rounds}")
    print(f"Grid file: {grid_file}")
    if n_values is not None:
        print(f"N values: {n_values}")
    if mu_values is not None:
        print(f"Mu values: {mu_values}")
    if expected_profile_labels is not None:
        print(f"K profile labels: {expected_profile_labels}")
    print("-" * 60)

    overall_eval = 0
    overall_qtable = 0
    overall_paired = 0
    overall_expected = 0

    for label in labels:
        scan_dirs_all = sorted(root.glob(f"scan_fixk_{label}_*"))
        scan_dirs = scan_dirs_all
        ignored_scan_dirs = 0
        ignored_n_dirs = 0

        if not args.include_extra_dirs and (mu_values is not None or expected_profile_labels is not None):
            filtered_scan_dirs = []
            for scan in scan_dirs_all:
                parsed = _parse_scan_dir(scan.name)
                if parsed is None:
                    ignored_scan_dirs += 1
                    continue
                _, mu_val, profile_label = parsed
                mu_ok = mu_values is None or _float_in(mu_values, mu_val)
                profile_ok = expected_profile_labels is None or profile_label in expected_profile_labels
                if mu_ok and profile_ok:
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

        missing_expected_dirs: List[Path] = []
        expected_by_grid = None
        if n_values is not None and mu_values is not None and expected_profile_labels is not None:
            expected_by_grid = len(n_values) * len(mu_values) * len(expected_profile_labels) * args.rounds
            for mu_val in mu_values:
                for profile_label in expected_profile_labels:
                    scan = root / f"scan_fixk_{label}_mu{mu_val}_K{profile_label}"
                    for n_val in n_values:
                        n_dir = scan / f"N_{n_val}"
                        if not n_dir.exists():
                            missing_expected_dirs.append(n_dir)

        expected_by_dirs = n_dirs * args.rounds
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
