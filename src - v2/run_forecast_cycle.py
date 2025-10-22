# src/run_forecast_cycle.py v1
"""
One-click pipeline to update data, rebuild features, and produce next-day prediction.

What it does (in order):
  1) update_data.py      -> extend raw CSV from Yahoo
  2) features.py         -> compute base technical features
  3) features_context.py -> compute contextual/rolling features
  4) predict_context.py  -> produce next-day prediction (prints + saves JSON)
  5) collects logs, checks "freshness" of features, and writes a combined
     run summary JSON in artifacts/auto_runs/

Usage:
    python .\src\run_forecast_cycle.py

Notes:
  - Uses the same Python interpreter (sys.executable) that runs this script.
  - Captures stdout/stderr from each step and writes to a per-run log file.
  - If a step fails, the script aborts and saves partial logs & metadata.
  - After features_context.py completes, checks the latest `timestamp` in
    data/processed/features_context.parquet (or features_daily.parquet if missing)
    and warns if it's lagging by more than 3 calendar days.
  - Produces an artifacts/auto_runs/prediction_summary_<ts>.json that contains:
      - which steps ran, timings, success/failure
      - last data date used and "prediction_date" attempted
      - stdout snippets and path to the predict_context saved JSON (if any)
"""

from pathlib import Path
import subprocess
import sys
import json
import time
from datetime import datetime, timedelta
import traceback

import pandas as pd

# -----------------------------
# Config
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
AUTO_RUN_DIR = ARTIFACTS_DIR / "auto_runs"
LOGS_DIR = REPO_ROOT / "logs"

SCRIPTS = [
    ("update_data", SRC_DIR / "update_data.py"),
    ("features", SRC_DIR / "features.py"),
    ("features_context", SRC_DIR / "features_context.py"),
    ("predict_context", SRC_DIR / "predict_context.py"),
]

FRESHNESS_WARNING_DAYS = 3  # warn if last features date older than this (calendar days)

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
AUTO_RUN_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run_script(script_path: Path, args=None, timeout=None):
    """Run a Python script and capture output. Returns dict with status and outputs."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd += args
    start = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        elapsed = time.time() - start
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": elapsed
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        return {
            "ok": False,
            "returncode": None,
            "stdout": getattr(e, "stdout", "") or "",
            "stderr": getattr(e, "stderr", "") or f"TimeoutExpired after {timeout} s",
            "elapsed_sec": elapsed
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"Exception when launching script: {e}\n{traceback.format_exc()}",
            "elapsed_sec": 0
        }


def read_latest_date_from_features_context():
    """
    Try to read features_context.parquet first (preferred).
    If missing, fall back to features_daily.parquet.
    Returns (latest_date (pd.Timestamp or None), path_used (Path or None)).
    """
    candidates = [
        REPO_ROOT / "data" / "processed" / "features_context.parquet",
        REPO_ROOT / "data" / "processed" / "features_daily.parquet",
        REPO_ROOT / "data" / "processed" / "features_safe.parquet",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                # try multiple possible date columns
                for col in ("timestamp", "date", "Date"):
                    if col in df.columns:
                        ser = pd.to_datetime(df[col], errors="coerce")
                        if ser.dropna().shape[0] == 0:
                            continue
                        return ser.max(), p
                # If no named date column, try index if datetime-like
                if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                    return pd.to_datetime(df.index).max(), p
            except Exception:
                # fail silently and try next candidate
                continue
    return None, None


def find_latest_predict_json():
    """Find the most recent prediction_context_*.json in artifacts/"""
    files = sorted(ARTIFACTS_DIR.glob("prediction_context_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def save_run_summary(summary: dict, ts_str: str):
    out = AUTO_RUN_DIR / f"prediction_summary_{ts_str}.json"
    with out.open("w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return out


def save_log_file(ts_str: str, content: str):
    out = LOGS_DIR / f"run_forecast_cycle_{ts_str}.log"
    with out.open("w", encoding="utf8") as fh:
        fh.write(content)
    return out


def main():
    start_time = datetime.utcnow()
    ts_str = start_time.strftime("%Y%m%d_%H%M%S_utc")
    run_log_parts = []
    summary = {
        "run_ts_utc": start_time.isoformat(),
        "steps": [],
        "final": {},
    }

    try:
        # 1) Run update_data.py
        name, path = SCRIPTS[0]
        run_log_parts.append(f"=== RUNNING {name} ({path.name}) ===\n")
        res = run_script(path)
        run_log_parts.append(f"STDOUT:\n{res['stdout']}\n")
        run_log_parts.append(f"STDERR:\n{res['stderr']}\n")
        summary["steps"].append({"name": name, "script": str(path), **res})

        if not res["ok"]:
            raise RuntimeError(f"{name} failed (see logs). aborting.")

        # 2) Run features.py
        name, path = SCRIPTS[1]
        run_log_parts.append(f"\n=== RUNNING {name} ({path.name}) ===\n")
        res = run_script(path)
        run_log_parts.append(f"STDOUT:\n{res['stdout']}\n")
        run_log_parts.append(f"STDERR:\n{res['stderr']}\n")
        summary["steps"].append({"name": name, "script": str(path), **res})
        if not res["ok"]:
            raise RuntimeError(f"{name} failed (see logs). aborting.")

        # 3) Run features_context.py
        name, path = SCRIPTS[2]
        run_log_parts.append(f"\n=== RUNNING {name} ({path.name}) ===\n")
        res = run_script(path)
        run_log_parts.append(f"STDOUT:\n{res['stdout']}\n")
        run_log_parts.append(f"STDERR:\n{res['stderr']}\n")
        summary["steps"].append({"name": name, "script": str(path), **res})
        if not res["ok"]:
            raise RuntimeError(f"{name} failed (see logs). aborting.")

        # After features_context, inspect freshness
        latest_date, used_path = read_latest_date_from_features_context()
        if latest_date is None:
            freshness_warning = "Could not determine latest features date (no usable date column)."
            run_log_parts.append("\n" + freshness_warning + "\n")
            summary["final"]["latest_features_date"] = None
            summary["final"]["freshness_warning"] = freshness_warning
        else:
            now = pd.Timestamp.now(tz=None).normalize()
            # latest_date may have timezone or time component; normalize to date
            try:
                latest_date_norm = pd.to_datetime(latest_date).tz_localize(None).normalize()
            except Exception:
                latest_date_norm = pd.to_datetime(latest_date).normalize()
            days_behind = (now - latest_date_norm).days
            summary["final"]["latest_features_date"] = str(latest_date_norm.date())
            summary["final"]["days_behind_calendar"] = int(days_behind)
            run_log_parts.append(f"\nLatest features date (from {used_path.name}): {latest_date_norm.date()} (days behind: {days_behind})\n")
            if days_behind > FRESHNESS_WARNING_DAYS:
                warning = f"⚠️ WARNING: features are {days_behind} days behind today. Consider re-running update/features; predictions may be stale."
                run_log_parts.append(warning + "\n")
                summary["final"]["freshness_warning"] = warning

        # 4) Run predict_context.py
        name, path = SCRIPTS[3]
        run_log_parts.append(f"\n=== RUNNING {name} ({path.name}) ===\n")
        res = run_script(path)
        run_log_parts.append(f"STDOUT:\n{res['stdout']}\n")
        run_log_parts.append(f"STDERR:\n{res['stderr']}\n")
        summary["steps"].append({"name": name, "script": str(path), **res})
        if not res["ok"]:
            raise RuntimeError(f"{name} failed (see logs). aborting.")

        # Try to detect which prediction JSON was saved by predict_context
        latest_pred = find_latest_predict_json()
        summary["final"]["prediction_json"] = str(latest_pred) if latest_pred else None
        # Grab first 2000 chars of stdout as snippet
        summary["final"]["predict_stdout_snippet"] = (res["stdout"] or "")[:2000]

        # Determine prediction_date if available in stdout
        pred_date = None
        try:
            # heuristically parse "Prediction Date: YYYY-MM-DD" from stdout
            for line in (res["stdout"] or "").splitlines():
                if "Prediction Date:" in line:
                    part = line.split("Prediction Date:", 1)[1].strip()
                    # try parse date-like token
                    try:
                        cand = part.split()[0]
                        pd_dt = pd.to_datetime(cand, errors="coerce")
                        if pd.notna(pd_dt):
                            pred_date = pd_dt
                            break
                    except Exception:
                        continue
        except Exception:
            pred_date = None
        summary["final"]["prediction_date"] = str(pred_date.date()) if (pred_date is not None and pd.notna(pred_date)) else None

        # Compose final message
        end_time = datetime.utcnow()
        summary["run_duration_sec"] = (end_time - start_time).total_seconds()

        run_log = "\n".join(run_log_parts)
        log_path = save_log_file(ts_str, run_log)
        summary["final"]["log_path"] = str(log_path)

        # Save run summary JSON
        run_summary_path = save_run_summary(summary, ts_str)

        # Friendly console output
        print("\n=== Forecast Cycle Completed Successfully ===")
        print(f"Run summary: {run_summary_path}")
        if latest_date is not None:
            print(f"Latest features date: {latest_date_norm.date()} (days behind: {days_behind})")
        if summary["final"].get("prediction_date"):
            print(f"Prediction Date reported by predict_context.py: {summary['final']['prediction_date']}")
        if summary["final"].get("prediction_json"):
            print(f"Prediction JSON: {summary['final']['prediction_json']}")
        else:
            print("No prediction JSON found in artifacts/ (predict_context.py may save to a different location).")
        print(f"Log file: {log_path}")
        return 0

    except Exception as e:
        # save failure logs + summary
        run_log_parts.append("\n=== EXCEPTION ===\n")
        run_log_parts.append(str(e) + "\n")
        run_log_parts.append(traceback.format_exc())
        run_log = "\n".join(run_log_parts)
        log_path = save_log_file(ts_str, run_log)
        summary["final"]["error"] = str(e)
        summary["final"]["log_path"] = str(log_path)
        run_summary_path = save_run_summary(summary, ts_str)
        print("\n=== Forecast Cycle FAILED ===")
        print(f"Run summary (partial): {run_summary_path}")
        print(f"See log: {log_path}")
        return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
