# src/version_manifest_loader.py v1
"""
Nifty Trend Indicator — Version Manifest Loader
------------------------------------------------
Reads version information from all canonical scripts in the project.

Usage:
  python src/version_manifest_loader.py
      → Prints human-readable version summary.

  python src/version_manifest_loader.py --json
      → Outputs version data as JSON.

Changelog v1:
- Scans canonical script list
- Extracts __version__ (if defined)
- Prints clean table and JSON mode
- Auto includes generation timestamp
"""

import os
import re
import json
from datetime import datetime

__version__ = "1.0"

# -----------------------------------------------------------
# Canonical script list (update if new scripts are added)
# -----------------------------------------------------------
CANONICAL_SCRIPTS = [
    "update_data.py",
    "features.py",
    "features_context.py",
    "labeling.py",
    "train_context.py",
    "predict_context.py",
    "run_forecast_cycle.py",
]

BASE_DIR = os.path.dirname(__file__)

# -----------------------------------------------------------
# Helper: extract __version__ from file content
# -----------------------------------------------------------
def extract_version(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
        else:
            # fallback to version tag in comments (e.g. "# v8")
            comment_match = re.search(r'#\s*v(\d+(?:\.\d+)*)', text)
            return comment_match.group(1) if comment_match else "N/A"
    except FileNotFoundError:
        return "Missing"
    except Exception as e:
        return f"Error: {e}"

# -----------------------------------------------------------
# Collect all versions
# -----------------------------------------------------------
def collect_versions():
    results = {}
    for script in CANONICAL_SCRIPTS:
        path = os.path.join(BASE_DIR, script)
        version = extract_version(path)
        results[script] = version
    return results

# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
def main():
    import sys
    json_mode = "--json" in sys.argv

    versions = collect_versions()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if json_mode:
        output = {
            "generated": timestamp,
            "manifest_version": __version__,
            "scripts": versions,
        }
        print(json.dumps(output, indent=2))
    else:
        print("=== Nifty Trend Indicator — Script Versions ===")
        for script, ver in versions.items():
            print(f"{script:<25} ... {ver}")
        print("-----------------------------------------------")
        print(f"Generated: {timestamp}")
        print(f"Manifest loader version: {__version__}")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()
