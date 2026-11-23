"""Offline CPU tests of verify_upstream_gpu_burst.py: no GPU, no network, no clone; nvidia-smi and children mocked."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_upstream_gpu_burst as burst  # noqa: E402

T4 = "Tesla T4, 41, 300, 15360, 7.5\n"


def reading(temp=40.0, used=300.0, total=8188.0, cc=8.9):
    return burst.GpuReading("GPU", temp, used, total, cc)


def no_network(*_a, **_k):
    raise AssertionError("no network, clone or child process in offline tests")


class Offline(unittest.TestCase):
    def setUp(self):
        for target in ("subprocess.Popen", "subprocess.run", "urllib.request.urlopen"):
            patcher = mock.patch(target, side_effect=no_network)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def args(self, *extra):
        return burst.build_parser().parse_args(["--output", str(self.tmp / "receipt.json"), *extra])

    def inputs(self):
        for name in ("in24.npy", "in96.npy"):
            np.save(self.tmp / name, np.zeros((4, 8, 8), np.uint8))
        return ["--hecate-input-2p4", str(self.tmp / "in24.npy"), "--hecate-input-9p6", str(self.tmp / "in96.npy")]


class SafetyDefaultTests(Offline):
    def test_laptop_safety_defaults(self):
        args = self.args()
        self.assertEqual((args.vram_fraction, args.batch, args.max_gpu_temp_c), (0.35, 1, 68.0))
        self.assertEqual(args.stages, "studio,hecate")

    def test_validation_rejects_unsafe_or_incomplete_settings(self):
        base = self.inputs()
        for extra, message in ((["--vram-fraction", "0.95"], "vram-fraction"), (["--batch", "64"], "batch"),
                               (["--max-gpu-temp-c", "95"], "max-gpu-temp-c"), (["--stages", "gpu"], "stages")):
            with self.subTest(extra=extra):
                self.assertTrue(any(message in e for e in burst.validate(self.args(*base, *extra))))
        self.assertIn("the hecate stage needs --hecate-input-2p4 and/or --hecate-input-9p6",
                      burst.validate(self.args()))
        self.assertEqual(burst.validate(self.args("--stages", "studio")), [])
        (self.tmp / "receipt.json").write_text("{}")
        self.assertTrue(any("already exists" in e for e in burst.validate(self.args("--stages", "studio"))))

    def test_burst_refuses_without_nvidia_smi(self):
        with mock.patch("shutil.which", return_value=None), self.assertRaisesRegex(RuntimeError, "nvidia-smi"):
            burst.burst(self.args("--stages", "studio"))


class PlanTests(Offline):
    def dry_run(self, *extra):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(burst.main(["--output", str(self.tmp / "r.json"), "--dry-run", *extra]), 0)
        return json.loads(out.getvalue())

    def test_dry_run_covers_every_open_check_with_safe_settings(self):
        plan = self.dry_run(*self.inputs())
        self.assertEqual(plan["safety"], {"vram_fraction": 0.35, "batch": 1, "max_gpu_temp_c": 68.0})
        runs = {r["tag"]: r for r in plan["runs"]}
        self.assertEqual(sorted(t for t in runs if t.startswith("studio")),
                         ["studio_cpu_fp32", "studio_cuda_auto", "studio_cuda_bf16", "studio_cuda_fp16"])
        hecate = [r for r in plan["runs"] if r["script"] == "hecate.py"]
        self.assertEqual(len(hecate), 2 * 2 * 2)
        for run in plan["runs"]:
            argv = run["argv"]
            self.assertEqual(argv[argv.index("--batch-size") + 1], "1")
            self.assertFalse(any(Path.home().name in a or str(self.tmp) in a for a in argv))
        for um in ("2.4", "9.6"):
            for reverse in (False, True):
                runs_um = [r for r in hecate if r["argv"][r["argv"].index("--spacing-um") + 1] == um
                           and ("--reverse" in r["argv"]) == reverse]
                self.assertEqual(sorted(r["argv"][r["argv"].index("--precision") + 1] for r in runs_um),
                                 ["fp16", "fp32"])
                for r in runs_um:
                    self.assertIn("--output", r["argv"])
                    self.assertIn("--output-3d", r["argv"])
                    self.assertIn(f"hecate_{um}um.pth", r["argv"][r["argv"].index("--checkpoint") + 1])
        ref = runs["studio_cpu_fp32"]
        self.assertFalse(ref["gpu"])
        self.assertEqual(ref["argv"][ref["argv"].index("--device") + 1], "cpu")
        self.assertEqual(ref["script"], "infer_release.py")

    def test_bf16_identification_run_only_where_auto_can_pick_it(self):
        self.assertNotIn("studio_cuda_bf16", [r.tag for r in burst.studio_runs(1, 48, 7.5)])
        self.assertIn("studio_cuda_bf16", [r.tag for r in burst.studio_runs(1, 48, 8.9)])
        self.assertEqual((burst.expected_auto(7.5), burst.expected_auto(8.0), burst.expected_auto(None)),
                         ("fp16", "bf16", None))

    def test_patch_bases_are_the_pinned_commits(self):
        self.assertEqual(burst.base_commit(burst.STUDIO_PATCH), "bf66f753176792a966bca17822629fb8821f20c8")
        self.assertEqual(burst.base_commit(burst.HECATE_PATCHES["main"]), "9cb86e500e944b11a06a7020403cde5dffb5bcb2")
        self.assertEqual(burst.base_commit(burst.HECATE_PATCHES["pr3"]), "f4386b19e0639f1c3da79572309ecc3f4cec760e")


class GuardTests(Offline):
    def test_parse_nvidia_smi(self):
        gpu = burst.parse_nvidia_smi(T4)
        self.assertEqual((gpu.name, gpu.temp_c, gpu.used_mib, gpu.total_mib, gpu.compute_capability),
                         ("Tesla T4", 41.0, 300.0, 15360.0, 7.5))
        self.assertIsNone(burst.parse_nvidia_smi("RTX, 50, 1, 8188, [N/A]").compute_capability)
        self.assertIsNone(burst.parse_nvidia_smi("RTX, 50, 1, 8188").compute_capability)
        with self.assertRaises(ValueError):
            burst.parse_nvidia_smi("garbage")

    def test_guard_violations(self):
        self.assertIsNone(burst.guard_violation(reading(68), 300, 68, 0.35))
        self.assertIn("69 C", burst.guard_violation(reading(69), 300, 68, 0.35))
        budget = 8188 * 0.35 + burst.VRAM_CONTEXT_ALLOWANCE_MIB
        self.assertIsNone(burst.guard_violation(reading(used=300 + budget), 300, 68, 0.35))
        self.assertIn("VRAM", burst.guard_violation(reading(used=301 + budget), 300, 68, 0.35))

    def test_cooldown_waits_then_times_out(self):
        temps, clock = iter([70, 66, 62]), iter(range(0, 100, 10))
        sleeps = []
        self.assertEqual(burst.wait_for_cooldown(lambda: reading(next(temps)), 63, 600, sleeps.append,
                                                 lambda: next(clock)), 62)
        self.assertEqual(sleeps, [5, 5])
        clock = iter(range(0, 10000, 400))
        with self.assertRaisesRegex(RuntimeError, "stopping the burst"):
            burst.wait_for_cooldown(lambda: reading(75), 63, 600, lambda s: None, lambda: next(clock))

    def watch(self, read):
        proc = mock.Mock()
        proc.poll.return_value = None
        dog = burst.Watchdog(proc, read, 300, 68, 0.35, interval_s=0.001)
        dog.run()
        return dog, proc

    def test_watchdog_terminates_a_hot_child(self):
        temps = iter([55, 60, 70])
        dog, proc = self.watch(lambda: reading(next(temps)))
        proc.terminate.assert_called_once()
        self.assertIn("70 C", dog.violation)
        self.assertEqual(dog.peak_temp_c, 70)

    def test_watchdog_terminates_when_nvidia_smi_fails(self):
        dog, proc = self.watch(mock.Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 20)))
        proc.terminate.assert_called_once()
        self.assertIn("nvidia-smi failed", dog.violation)


class ChildLauncherTests(Offline):
    def test_child_applies_vram_cap_and_propagates_it_to_python_grandchildren(self):
        script = self.tmp / "cli.py"
        script.write_text("import subprocess, sys\n"
                          "subprocess.run([sys.executable, 'inner.py', '--x', '1'], check=True)\n"
                          "subprocess.run(['nvidia-smi'])\n", encoding="utf-8")
        fake_cuda = types.SimpleNamespace(is_available=lambda: True, set_per_process_memory_fraction=mock.Mock())
        calls = []
        with mock.patch.object(burst, "torch", types.SimpleNamespace(cuda=fake_cuda)), \
                mock.patch.object(subprocess, "run", side_effect=lambda cmd, *a, **k: calls.append(cmd)), \
                mock.patch.object(sys, "argv", list(sys.argv)), mock.patch.object(sys, "path", list(sys.path)):
            burst.main(["--child", "0.35", str(script), "--flag"])
            self.assertEqual(sys.argv, [str(script), "--flag"])
        fake_cuda.set_per_process_memory_fraction.assert_called_once_with(0.35, 0)
        self.assertEqual(calls[0][:4], [sys.executable, str(Path(burst.__file__).resolve()), "--child", "0.35"])
        self.assertEqual(calls[0][4:], ["inner.py", "--x", "1"])
        self.assertEqual(calls[1], ["nvidia-smi"])

    def test_cpu_reference_runs_with_cuda_hidden_and_receipt_has_no_local_paths(self):
        run = burst.studio_runs(1, 48, 7.5)[0]
        proc = mock.Mock(returncode=0)
        proc.communicate.return_value = (f"loaded {self.tmp}/x\n", None)
        args = self.args("--stages", "studio")
        log = []
        with mock.patch("subprocess.Popen", return_value=proc) as popen, \
                mock.patch.object(burst, "query_gpu", side_effect=AssertionError("CPU run must not poll the GPU")):
            record = burst.execute(run, self.tmp / "repo", self.tmp, args, log)
        cmd, kwargs = popen.call_args.args[0], popen.call_args.kwargs
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(cmd[2:4], ["--child", "0.35"])
        self.assertTrue(cmd[4].endswith("infer_release.py"))
        self.assertNotIn(str(self.tmp), json.dumps(record))
        self.assertEqual(log, [record])

    def test_gpu_run_failure_stops_with_redacted_log(self):
        run = burst.hecate_runs(["9.6"], 1)[0]
        proc = mock.Mock(returncode=1)
        proc.poll.return_value = 1
        proc.communicate.return_value = (f"OOM in {self.tmp}/hecate_input_9p6.npy", None)
        with mock.patch("subprocess.Popen", return_value=proc), \
                mock.patch.object(burst, "query_gpu", return_value=reading(40)):
            log = []
            with self.assertRaisesRegex(RuntimeError, "failed"):
                burst.execute(run, self.tmp, self.tmp, self.args("--stages", "studio"), log)
        self.assertIn("<work>", log[0]["log_tail"])
        self.assertNotIn(str(self.tmp), json.dumps(log))


class DataTests(Offline):
    def test_studio_stack_is_release_desc_order(self):
        st = np.broadcast_to(np.arange(24, dtype=np.uint8), (5, 7, 24))
        stack = burst.studio_stack(st)
        self.assertEqual(stack.shape, (16, 5, 7))
        self.assertEqual(stack[:, 0, 0].tolist(), list(range(20, 4, -1)))
        with self.assertRaises(ValueError):
            burst.studio_stack(st[:, :, :20])

    def test_center_crop(self):
        vol = np.arange(2 * 10 * 12, dtype=np.uint8).reshape(2, 10, 12)
        np.testing.assert_array_equal(burst.center_crop_yx(vol, 4), vol[:, 3:7, 4:8])
        self.assertEqual(burst.center_crop_yx(vol, 0).shape, vol.shape)
        self.assertEqual(burst.center_crop_yx(vol, 50).shape, vol.shape)

    def test_lfs_pointer(self):
        oid = "a" * 64
        text = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 123\n"
        self.assertEqual(burst.lfs_oid(text), oid)
        with self.assertRaises(ValueError):
            burst.lfs_oid("binary checkpoint bytes")

    def test_parity_metrics_and_verdicts(self):
        rng = np.random.default_rng(0)
        ref = rng.random((20, 20)).astype(np.float32)
        valid = np.ones_like(ref, bool)
        close = burst.compare_float(ref + 0.002, ref, valid)
        self.assertAlmostEqual(close["max_abs_diff"], 0.002, places=5)
        self.assertTrue(burst.studio_verdict(close))
        self.assertFalse(burst.studio_verdict(burst.compare_float(ref + 0.05, ref, valid)))
        levels = rng.integers(0, 250, (3, 9, 9), dtype=np.uint8)
        one = levels.copy()
        one[0, 0, 0] += 1
        metrics = burst.compare_levels(one, levels)
        self.assertEqual(metrics["max_level_diff"], 1)
        self.assertTrue(burst.hecate_verdict(metrics))
        self.assertFalse(burst.hecate_verdict(burst.compare_levels(levels + 5, levels)))


if __name__ == "__main__":
    unittest.main()
