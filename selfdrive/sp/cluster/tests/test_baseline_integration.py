from __future__ import annotations

import ast
import contextlib
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[4]
BASELINE = "330d3f634d34bc3055d2a3141268836fc8220208"
INTEGRATION_PATHS = {
    "openpilot/common/params_keys.h",
    "openpilot/common/usbgpu_bus_lock.py",
    "openpilot/common/tests/test_usbgpu_bus_lock.py",
    "openpilot/system/manager/process_config.py",
    "openpilot/system/updated/updated.py",
    "openpilot/selfdrive/ui/layouts/home.py",
    "openpilot/system/loggerd/SConscript",
    "openpilot/system/loggerd/encoder/cluster_h264_encoder.cc",
    "openpilot/system/loggerd/encoder/cluster_h264_encoder.h",
    "openpilot/system/loggerd/encoder/cluster_h264_encoder_bridge.cc",
    "openpilot/system/hardware/chestnut/monitoring.py",
    "tinygrad_repo/tinygrad/engine/realize.py",
    "tinygrad_repo/tinygrad/runtime/support/usb.py",
    "tinygrad_repo/test/unit/test_usb_lock.py",
}


class BaselineIntegrationTests(unittest.TestCase):
    def test_home_welcome_without_wechat_banner(self):
        path = ROOT / "openpilot" / "selfdrive" / "ui" / "layouts" / "home.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HomeLayout")
        render = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_render_home_content")
        rl = Mock()
        rl.Vector2.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)
        namespace = {
            "rl": rl, "gui_app": Mock(), "FontWeight": SimpleNamespace(BOLD=1, NORMAL=2),
            "measure_text_cached": lambda font, text, size: SimpleNamespace(x=len(text) * size / 2, y=size),
        }
        exec(compile(ast.Module(body=[render], type_ignores=[]), str(path), "exec"), namespace)
        logo = SimpleNamespace(width=220, height=220)
        home = SimpleNamespace(_home_logo=logo, content_rect=SimpleNamespace(x=0, y=0, width=1920, height=800))
        namespace["_render_home_content"](home)
        self.assertNotIn("_home_banner", source)
        self.assertNotIn("Welcome to MR.ONE", source)
        self.assertEqual(rl.draw_texture_ex.call_count, 1)
        self.assertIs(rl.draw_texture_ex.call_args.args[0], logo)
        self.assertEqual(rl.draw_text_ex.call_args_list[0].args[1], "Welcome to Openpilot")
        self.assertEqual(rl.draw_text_ex.call_args_list[1].args[1], "Drive smarter. Arrive safer.")
        self.assertEqual(rl.draw_texture_ex.call_args.args[1].y, 182)

    def test_updater_only_adds_missing_time_import(self):
        path = "openpilot/system/updated/updated.py"
        upstream = subprocess.run(
            ["git", "--no-pager", "show", f"{BASELINE}:{path}"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        current = (ROOT / Path(path)).read_text(encoding="utf-8")
        self.assertEqual(current, upstream.replace("import threading\n", "import threading\nimport time\n", 1))
        tree = ast.parse(current)
        fetch = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fetch_update")
        callback = next(node for node in fetch.body if isinstance(node, ast.FunctionDef) and node.name == "on_git_progress")
        namespace = {"last_progress": [-1, 0.0], "self": SimpleNamespace(params=Mock()),
                     "parse_git_progress": lambda _: (50, "1 MiB/s")}
        imports = [node for node in tree.body if isinstance(node, ast.Import)]
        time_import = next(node for node in imports if any(alias.name == "time" for alias in node.names))
        exec(compile(ast.Module(body=[time_import, callback], type_ignores=[]), path, "exec"), namespace)
        namespace["on_git_progress"]("Receiving objects: 50%")
        namespace["self"].params.put.assert_called_once_with("UpdaterState", "downloading... 50% 1 MiB/s", block=True)

    def test_only_cluster_paths_differ_from_pinned_upstream(self):
        result = subprocess.run(
            ["git", "--no-pager", "diff", "--name-only", BASELINE, "--"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        unexpected = [
            path for path in result.stdout.splitlines()
            if not path.startswith("selfdrive/sp/") and path not in INTEGRATION_PATHS
        ]
        self.assertEqual(unexpected, [], "Non-Cluster changes must not enter the MR.ONE baseline branch")

    def test_all_cluster_parameters_registered_without_changing_upstream_keys(self):
        config = ast.parse((ROOT / "selfdrive" / "sp" / "cluster" / "cluster_config.py").read_text(encoding="utf-8"))
        params = {
            node.value.value for node in config.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith("_PARAM") and isinstance(node.value, ast.Constant)
        }
        path = "openpilot/common/params_keys.h"
        current = (ROOT / Path(path)).read_text(encoding="utf-8")
        upstream = subprocess.run(
            ["git", "--no-pager", "show", f"{BASELINE}:{path}"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        key_pattern = r'^\s*(\{"[^"]+",.*)$'
        old_keys = set(re.findall(key_pattern, upstream, re.MULTILINE))
        new_keys = set(re.findall(key_pattern, current, re.MULTILINE))
        self.assertTrue(old_keys <= new_keys)
        added = new_keys - old_keys
        self.assertEqual({re.search(r'"([^"]+)"', line).group(1) for line in added}, params)

    def test_manager_registers_one_cluster_supervisor(self):
        tree = ast.parse((ROOT / "openpilot" / "system" / "manager" / "process_config.py").read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "PythonProcess" and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cluster_d"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[1].value, "selfdrive.sp.cluster_autorun")
        self.assertEqual(ast.unparse(calls[0].args[2]), "always_run")
        self.assertTrue((ROOT / "selfdrive" / "sp" / "cluster_autorun.py").is_file())

    def test_native_encoder_target_is_comma_only(self):
        path = ROOT / "openpilot" / "system" / "loggerd" / "SConscript"
        for arch in ("comma_arm64", "x86_64"):
            with self.subTest(arch=arch):
                env = Mock()
                namespace = {
                    "Import": lambda *args: None, "env": env, "arch": arch,
                    "common": "common", "messaging": "messaging", "visionipc": "visionipc", "ffmpeg_libs": [],
                }
                exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
                if arch == "comma_arm64":
                    env.SharedLibrary.assert_called_once()
                    target, sources = env.SharedLibrary.call_args.args
                    self.assertEqual(target, "cluster_h264_encoder_bridge")
                    for source in sources:
                        self.assertTrue((path.parent / source).is_file())
                else:
                    env.SharedLibrary.assert_not_called()

    def test_chestnut_monitor_reads_inside_cluster_bus_lock(self):
        path = ROOT / "openpilot" / "system" / "hardware" / "chestnut" / "monitoring.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_chestnut_state")
        import struct
        events = []

        @contextlib.contextmanager
        def lock():
            events.append("lock")
            try:
                yield
            finally:
                events.append("unlock")

        def read(*args, **kwargs):
            self.assertEqual(events, ["lock"])
            return struct.pack("<Hh?", 12000, 250, False) if args[1] == 0xC0 else bytes([0x78])

        message = SimpleNamespace(valid=False, chestnutState=SimpleNamespace())
        namespace = {
            "struct": struct, "usbgpu_bus_lock": lock,
            "messaging": SimpleNamespace(new_message=lambda _: message),
            "usb1": SimpleNamespace(USBError=RuntimeError),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        gpu = SimpleNamespace(tempC=45)
        result = namespace["read_chestnut_state"](SimpleNamespace(controlRead=read), gpu)
        self.assertTrue(result.valid)
        self.assertEqual(result.chestnutState.tempC, 45)
        self.assertEqual(result.chestnutState.supplyVoltage, 12000)
        self.assertEqual(result.chestnutState.pcieLtssm, 0x78)
        self.assertEqual(events, ["lock", "unlock"])


if __name__ == "__main__":
    unittest.main()
