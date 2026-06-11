#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = [f"Case{i}" for i in range(1, 11)]
MODES = ["wl", "thermal"]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(case: str, mode: str, repeat: int, run_root: Path, python_bin: str) -> dict:
    out_dir = run_root / case / mode / f"rep{repeat:02d}"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHON"] = python_bin
    env["ATPLACE_OUT_DIR"] = str(out_dir)
    env.setdefault("LC_ALL", "C")
    cmd = ["bash", str(ROOT / "reproduce.sh"), case, mode]
    start = time.time()
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.time() - start
    summary = read_summary(out_dir / "summary.json")
    row = {
        "case": case,
        "mode": mode,
        "repeat": repeat,
        "status": "ok" if proc.returncode == 0 and summary else "error",
        "returncode": proc.returncode,
        "runtime_s": summary.get("runtime_s", elapsed),
        "hpwl": summary.get("hpwl", ""),
        "twl_m": summary.get("twl_m", ""),
        "has_best_fp": summary.get("has_best_fp", ""),
        "layout_json": summary.get("layout_json", ""),
        "out_dir": str(out_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if proc.returncode != 0:
        row["error"] = f"returncode={proc.returncode}"
    else:
        row["error"] = ""
    return row


def run_case(case: str, repeats: int, run_root: Path, python_bin: str) -> list[dict]:
    rows = []
    for mode in MODES:
        for repeat in range(repeats):
            rows.append(run_one(case, mode, repeat, run_root, python_bin))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "case",
        "mode",
        "repeat",
        "status",
        "returncode",
        "runtime_s",
        "hpwl",
        "twl_m",
        "has_best_fp",
        "layout_json",
        "out_dir",
        "stdout",
        "stderr",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_best(path: Path, rows: list[dict]) -> None:
    best = []
    for case in CASES:
        for mode in MODES:
            group = [row for row in rows if row.get("case") == case and row.get("mode") == mode and row.get("status") == "ok"]
            if not group:
                best.append({"case": case, "mode": mode, "status": "missing"})
                continue
            group.sort(key=lambda row: float(row["twl_m"]))
            best.append(group[0])
    write_rows(path, best)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-workers", type=int, default=5)
    parser.add_argument("--run-name", default=time.strftime("repro_matrix_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.case_workers < 1:
        raise SystemExit("--case-workers must be positive")

    run_root = ROOT / "repro_runs" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "commands.json", {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "repeats": args.repeats,
        "case_workers": args.case_workers,
        "cases": CASES,
        "modes": MODES,
        "python": args.python,
        "command": ["python", "reproduce_matrix.py", "--repeats", str(args.repeats), "--case-workers", str(args.case_workers)],
    })

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.case_workers, len(CASES))) as pool:
        futures = {pool.submit(run_case, case, args.repeats, run_root, args.python): case for case in CASES}
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                rows = [{
                    "case": case,
                    "mode": "",
                    "repeat": "",
                    "status": "error",
                    "returncode": "",
                    "runtime_s": "",
                    "hpwl": "",
                    "twl_m": "",
                    "has_best_fp": "",
                    "layout_json": "",
                    "out_dir": "",
                    "stdout": "",
                    "stderr": "",
                    "error": repr(exc),
                }]
            all_rows.extend(rows)
            write_rows(run_root / "summary.csv", all_rows)
            write_best(run_root / "best.csv", all_rows)

    write_rows(run_root / "summary.csv", all_rows)
    write_best(run_root / "best.csv", all_rows)
    print(json.dumps({
        "run_root": str(run_root),
        "summary_csv": str(run_root / "summary.csv"),
        "best_csv": str(run_root / "best.csv"),
        "rows": len(all_rows),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
