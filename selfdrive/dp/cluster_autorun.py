#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import time
from pathlib import Path

CLUSTER_DIR = Path("/data/openpilot/selfdrive/dp/cluster")

def main() -> None:
    main_script = CLUSTER_DIR / "main.py"
    print(f"[cluster_autorun] Starting lightweight cluster HUD (Live, 15FPS, Dark Theme)...", flush=True)

    while True:
        try:
            # 每次重試前先清理可能佔用 USB 的舊行程
            subprocess.run(["pkill", "-f", "cluster/main.py"], stderr=subprocess.DEVNULL)
            time.sleep(1.0)

            # 執行主要的 cluster 程式（降低 FPS 減輕負擔）
            subprocess.run([
                "python3",
                str(main_script),
                "--input", "live",
                "--fps", "15",              # 從 25 降到 15，大幅減少運算
                "--usb-codec", "jpeg",      # 改用較省資源的 JPEG 編碼 (視需求可開)
                "--usb-jpeg-quality", "50", # 降低畫質來減省傳輸與編碼開銷
                "--usb-brightness", "0",
                "--theme", "auto"
            ], check=True)
        except Exception as exc:
            print(f"[cluster_autorun] main.py exited: {exc}; retrying in 5s", flush=True)
            time.sleep(5.0)

if __name__ == "__main__":
    main()