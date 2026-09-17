"""Capture high-fidelity dashboard screenshot using Playwright Node script."""

import subprocess
import sys
import time
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    port = 8888

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.dashboard.app:app", "--port", str(port), "--host", "127.0.0.1"],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print("Starting FastAPI server...")
        time.sleep(2.0)
        res = subprocess.run(["node", "scripts/capture_preview.js"], cwd=str(repo_root), capture_output=True, text=True)
        print("Node stdout:\n", res.stdout)
        if res.stderr:
            print("Node stderr:\n", res.stderr)
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
