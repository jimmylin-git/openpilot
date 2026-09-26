#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent / "cluster"
# comma-deps-raylib picks its DRM "comma" backend on device, which would fight the
# openpilot UI for /dev/dri/card0. The cluster renders offscreen for the USB panel,
# so default to the EGL-surfaceless "headless" backend (llvmpipe, CPU rendered).
# Override with CLUSTER_RAYLIB_BACKEND=comma|headless|desktop.
DEFAULT_RAYLIB_BACKEND = "headless"
# Logging is off by default; set CLUSTER_LOG_PATH=/tmp/cluster.log to debug.
LOG_MAX_BYTES = 5 * 1024 * 1024
RESTART_DELAY_S = 5.0

# The headless backend needs Mesa's EGL (surfaceless + llvmpipe). comma-deps-raylib
# bundles it since 6.0.0.1.post101, but older AGNOS venvs ship a wheel without it and
# fall back to the Qualcomm EGL, which fails with EGL_BAD_ALLOC. In that case fetch the
# Mesa libs once from the pinned wheel into /data so they survive reinstalls.
MESA_LIBS = ("libEGL.so.1", "libGLESv2.so.2", "libgallium-26.2.1.so", "libdrm.so.2")
MESA_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/a1/17/12977631f6d86d1daa4f67a310cfdf2783ba288f42d444d6a89b2297a0a4/"
    "comma_deps_raylib-6.0.0.1.post101-py3-none-manylinux_2_28_aarch64.whl"
)
MESA_WHEEL_SHA256 = "c93ff9b45df414620b3011280da77151764c027e2f95d384fe255615d78872cc"
MESA_CACHE_DIR = Path(os.environ.get("CLUSTER_MESA_DIR", "/data/cluster_mesa/raylib-6.0.0.1.post101"))
MESA_RETRY_S = 300.0


def _has_mesa(lib_dir: Path) -> bool:
    return all((lib_dir / name).is_file() for name in MESA_LIBS)


def _bundled_raylib_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("raylib")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / "install" / "lib"


def _download_mesa(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        wheel = Path(tmp) / "raylib.whl"
        digest = hashlib.sha256()
        with urllib.request.urlopen(MESA_WHEEL_URL, timeout=60) as resp, wheel.open("wb") as out:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != MESA_WHEEL_SHA256:
            raise RuntimeError("raylib wheel sha256 mismatch")
        staging = Path(tmp) / "lib"
        staging.mkdir()
        with zipfile.ZipFile(wheel) as zf:
            for member in zf.namelist():
                name = member.rsplit("/", 1)[-1]
                if name in MESA_LIBS and "/raylib/install/lib/" in member:
                    (staging / name).write_bytes(zf.read(member))
        if not _has_mesa(staging):
            raise RuntimeError("raylib wheel is missing Mesa libs")
        if dest.exists():
            shutil.rmtree(dest)
        staging.replace(dest)


def ensure_headless_mesa(log_file=None) -> Path | None:
    """Return a directory to prepend to LD_LIBRARY_PATH, or None if not needed/unavailable."""
    bundled = _bundled_raylib_lib_dir()
    if bundled is not None and _has_mesa(bundled):
        return None
    if _has_mesa(MESA_CACHE_DIR):
        return MESA_CACHE_DIR
    if not MESA_CACHE_DIR.parent.parent.is_dir():
        return None
    try:
        _log(log_file, f"installed raylib has no Mesa EGL; downloading it to {MESA_CACHE_DIR}")
        _download_mesa(MESA_CACHE_DIR)
        _log(log_file, "Mesa EGL ready")
        return MESA_CACHE_DIR
    except Exception as exc:
        _log(log_file, f"Mesa EGL download failed: {exc}")
        return None


def cluster_env(mesa_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    backend = env.get("CLUSTER_RAYLIB_BACKEND", DEFAULT_RAYLIB_BACKEND).strip().lower()
    if backend:
        env["RAYLIB_BACKEND"] = backend
    if mesa_dir is not None and backend == "headless":
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{mesa_dir}:{existing}" if existing else str(mesa_dir)
    return env


def cluster_cmd() -> list[str]:
    return [
        sys.executable,
        str(CLUSTER_DIR / "main.py"),
        "--input",
        "live",
        "--fps",
        "5",
        "--usb-codec",
        "h264",
        "--usb-h264-backend",
        "auto",
        "--theme",
        "auto",
    ]


def _open_log():
    log_path = os.environ.get("CLUSTER_LOG_PATH", "").strip()
    if not log_path:
        return None
    path = Path(log_path)
    try:
        if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
            path.replace(path.with_name(path.name + ".1"))
        return path.open("a", buffering=1)
    except OSError:
        return None


def _log(log_file, message: str) -> None:
    if log_file is None:
        return
    line = f"[cluster_autorun {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    log_file.write(line + "\n")


def main() -> None:
    # sunnypilot's manager never restarts a PythonProcess that exited, so supervise
    # the cluster here: restart on crash (e.g. screen not enumerated yet at boot).
    env = cluster_env()
    log_file = _open_log()
    child: subprocess.Popen | None = None
    stopping = False
    mesa_dir: Path | None = None
    mesa_next_try = 0.0

    def _stop(signum, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    _log(log_file, f"starting cluster HUD with live openpilot data (raylib backend={env.get('RAYLIB_BACKEND')})")
    while not stopping:
        if env.get("RAYLIB_BACKEND") == "headless" and mesa_dir is None and time.monotonic() >= mesa_next_try:
            mesa_dir = ensure_headless_mesa(log_file)
            if mesa_dir is None:
                mesa_next_try = time.monotonic() + MESA_RETRY_S
            env = cluster_env(mesa_dir)
        output = log_file if log_file is not None else subprocess.DEVNULL
        child = subprocess.Popen(cluster_cmd(), env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            returncode = child.wait()
        except KeyboardInterrupt:
            stopping = True
            child.terminate()
            returncode = child.wait()
        if stopping:
            break
        _log(log_file, f"cluster exited with code {returncode}, restarting in {RESTART_DELAY_S:.0f}s")
        deadline = time.monotonic() + RESTART_DELAY_S
        while not stopping and time.monotonic() < deadline:
            time.sleep(0.2)

    if child is not None and child.poll() is None:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
    _log(log_file, "stopped")


if __name__ == "__main__":
    main()