#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent / "cluster"
# comma-deps-raylib picks its DRM "comma" backend on device, which would fight the
# openpilot UI for /dev/dri/card0. The cluster renders offscreen for the USB panel,
# so default to the EGL-surfaceless "headless" backend (llvmpipe, CPU rendered).
# Override with CLUSTER_RAYLIB_BACKEND=comma|headless|desktop.
DEFAULT_RAYLIB_BACKEND = "headless"


def cluster_env() -> dict[str, str]:
    env = os.environ.copy()
    backend = env.get("CLUSTER_RAYLIB_BACKEND", DEFAULT_RAYLIB_BACKEND).strip().lower()
    if backend:
        env["RAYLIB_BACKEND"] = backend
    return env


def main() -> None:
    main_script = CLUSTER_DIR / "main.py"
    env = cluster_env()
    print("[cluster_autorun] Using camera-based automatic brightness", flush=True)
    print(f"[cluster_autorun] Starting cluster HUD with live openpilot data (raylib backend={env.get('RAYLIB_BACKEND')})", flush=True)
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
        env=env,
    )


if __name__ == "__main__":
    main()