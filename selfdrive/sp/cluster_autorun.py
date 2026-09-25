#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLUSTER_DIR = Path("/data/openpilot/selfdrive/dp/cluster")


def main() -> None:
    main_script = CLUSTER_DIR / "main.py"
    print("[cluster_autorun] Using camera-based automatic brightness", flush=True)
    print("[cluster_autorun] Starting cluster HUD with live openpilot data", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(main_script),
            "--input",
            "live",
            "--fps",
            "15",
            "--usb-codec",
            "jpeg",
            "--usb-jpeg-quality",
            "50",
            "--theme",
            "auto",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()