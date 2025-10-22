# src/run_forecast_cycle.py v2.3
"""
Automated Forecast Cycle Runner for Nifty Trend Indicator

Changelog v2.3:
- Use sys.executable to ensure subprocesses run in the same Python interpreter (fixes ModuleNotFoundError when venv packages are installed).
- Set PYTHONIOENCODING=utf-8 for subprocess env to avoid UnicodeEncodeError in Windows consoles.
- Abort on error for child scripts and surface stderr for debugging.
- Improved manifest handling and clearer console messages.
"""

import os
import json
import subprocess
import sys
from datetime import datetime

__version__ = "2.3"

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = BASE_DIR
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Helper — Run a Python script using the same interpreter and stream output
# -------------------------------------------------------------------
def run_script(script_name):
    path = os.path.join(SRC_DIR, script_name)
    print(f"\n=== Running {script_name} ===")
    env = os.environ.copy()
    # Ensure subprocess uses UTF-8 I/O to avoid windows encoding issues
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
    except Exception as e:
        print(f"[WARN] Failed to start {script_name}: {e}")
        return None

    # Print stdout
    if result.stdout:
        print(result.stdout)
    # If failure, print stderr and return non-zero
    if result.returncode != 0:
        print(f"[WARN] Error running {script_name} (returncode={result.returncode}):")
        if result.stderr:
            print(result.stderr)
        else:
            print("No stderr available.")
        return result
    return result

# -------------------------------------------------------------------
# Helper — Get version manifest from version_manifest_loader.py
# -------------------------------------------------------------------
def get_version_manifest():
    path = os.path.join(SRC_DIR, "version_manifest_loader.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        # Use same interpreter
        result = subprocess.run([sys.executable, path, "--json"], capture_output=True, text=True, env=env, check=True)
        manifest = json.loads(result.stdout)
        print("\n=== Script Versions Used ===")
        for script, ver in manifest["scripts"].items():
            print(f"{script:<25} ... {ver}")
        return manifest
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to run version manifest loader: returncode={e.returncode}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return None
    except Exception as e:
        print(f"[WARN] Failed to get version manifest: {e}")
        return None

# -------------------------------------------------------------------
# Main automated forecast cycle
# -------------------------------------------------------------------
def main():
    print(f"\n[START] Starting Nifty Forecast Cycle (v{__version__})")
    start_time = datetime.now()

    # Step 1: Update data
    res = run_script("update_data.py")
    if res is None or (hasattr(res, "returncode") and res.returncode != 0):
        print("Aborting forecast cycle due to update_data.py failure.")
        return

    # Step 2: Generate base features
    res = run_script("features.py")
    if res is None or (hasattr(res, "returncode") and res.returncode != 0):
        print("Aborting forecast cycle due to features.py failure.")
        return

    # Step 3: Generate contextual features
    res = run_script("features_context.py")
    if res is None or (hasattr(res, "returncode") and res.returncode != 0):
        print("Aborting forecast cycle due to features_context.py failure.")
        return

    # Step 4: Run contextual prediction
    res = run_script("predict_context.py")
    if res is None or (hasattr(res, "returncode") and res.returncode != 0):
        print("Aborting forecast cycle due to predict_context.py failure.")
        return

    # Step 5: Record versions used
    manifest = get_version_manifest()

    # Step 6: Update today's artifact with version info
    today = datetime.now().strftime("%Y-%m-%d")
    json_path = os.path.join(ARTIFACTS_DIR, f"prediction_context_{today}.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["forecast_cycle_version"] = __version__
            data["script_versions"] = manifest["scripts"] if manifest else {}
            data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"\n[OK] Updated artifact with version metadata: {json_path}")
        except Exception as e:
            print(f"[WARN] Could not update artifact JSON: {e}")
    else:
        print(f"[WARN] Prediction artifact not found: {json_path}")

    # Step 7: Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n[TARGET] Forecast cycle completed in {duration:.1f} seconds.")
    print(f"Cycle version: {__version__}")
    print(f"Generated artifact: {json_path}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
