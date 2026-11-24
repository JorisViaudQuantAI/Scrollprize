"""Offline, CPU-only checks of the upstream fp16 patch kit.

Always run: unified-diff structure of the three patches, the code the patches add (extracted from the hunks and
executed on CPU), equivalence of the two Hecate variants, and the draft guard rails (no PHerc0841 w00 AUC claim for
Studio, no local paths, no user name, no tokens).

With STUDIO_UPSTREAM_DIR and/or HECATE_UPSTREAM_DIR pointing to git clones of the upstream repositories (the Hecate
clone with refs/pr/3 fetched), each patch is also applied with `git apply --check` to the files of its pinned base
commit, the pre-image and post-image blob hashes are compared with the patch's `index` lines, and the patched
inference code runs a stand-in model on CPU.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # before torch is imported: the kit never touches a GPU

import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the structural tests
    torch = None

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import leak_guard  # noqa: E402  (shared public-repo privacy guard)

STUDIO = HERE / "0001-studio-upstream-fp16.patch"
HECATE = HERE / "0002-hecate-upstream-fp16.patch"
HECATE_PR3 = HERE / "0002b-hecate-upstream-fp16-after-pr3.patch"
PATCHES = (STUDIO, HECATE, HECATE_PR3)
DRAFTS = (HERE / "PR_STUDIO_DRAFT.md", HERE / "PR_HECATE_DRAFT.md", HERE / "PROGRESS_PRIZE_NOV_SUBMISSION_DRAFT.md")
STUDIO_INFERENCE = "studio-multiscroll-cpt-20261023/hf/inference.py"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TOKEN = re.compile(r"hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{20,}|sk-[A-Za-z0-9_-]{20,}"
                   r"|AKIA[0-9A-Z]{16}|xox[abpr]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY")
LOCAL_PATH = leak_guard.LOCAL_PATH   # always rejected, for every account name including "root"
needs_torch = unittest.skipIf(torch is None, "torch is not installed")


def parse_patch(path: Path) -> dict:
    """{file: [(old_start, old_lines, new_start, new_lines, [body lines])]}; checks every hunk's line counts."""
    text = path.read_bytes().decode("utf-8")
    files, current, hunk = {}, None, None
    for line in text.split("\n")[:-1]:
        if line.startswith("diff --git "):
            a, b = line.split()[2:4]
            assert a[2:] == b[2:], f"{path.name}: renames are not expected ({line})"
            current, hunk = files.setdefault(b[2:], []), None
        elif current is None or line.startswith(("index ", "--- ", "+++ ")):
            continue
        elif m := HUNK.match(line):
            os_, ol, ns, nl = (int(g) if g is not None else 1 for g in m.groups())
            hunk = [os_, ol, ns, nl, []]
            current.append(hunk)
        else:
            assert line[:1] in (" ", "+", "-", "\\"), f"{path.name}: stray line {line!r}"
            hunk[4].append(line)
    for name, hunks in files.items():
        for os_, ol, ns, nl, body in hunks:
            old = sum(1 for b in body if b[:1] in (" ", "-"))
            new = sum(1 for b in body if b[:1] in (" ", "+"))
            assert (old, new) == (ol, nl), f"{path.name} {name} @@ -{os_},{ol} +{ns},{nl}: counted {old}/{new}"
    return files


def preamble(path: Path) -> dict:
    head = path.read_text(encoding="utf-8").split("diff --git ", 1)[0]
    return dict(re.findall(r"^(\w+): (.+)$", head, re.M))


def base_commit(path: Path) -> str:
    return re.search(r"\b([0-9a-f]{40})\b", preamble(path)["Base"]).group(1)


def added_block(path: Path, name: str, first: str) -> str:
    """The run of consecutive added lines of `name` that starts with the line matching `first`."""
    for *_, body in parse_patch(path)[name]:
        for i, line in enumerate(body):
            if line.startswith("+") and re.search(first, line):
                block = []
                for nxt in body[i:]:
                    if not nxt.startswith("+"):
                        break
                    block.append(nxt[1:])
                return "\n".join(block) + "\n"
    raise AssertionError(f"{path.name}: no added line matching {first!r} in {name}")


def changes(path: Path, name: str, inside: str) -> list[str]:
    """Added and removed lines of the hunks of `name` whose header or body mentions `inside`."""
    out = []
    for *_, body in parse_patch(path)[name]:
        if any(inside in b for b in body):
            out += [b for b in body if b[:1] in "+-"]
    return out


class PatchStructureTests(unittest.TestCase):
    def test_lf_unified_diffs_with_pinned_base(self):
        for path in PATCHES:
            with self.subTest(patch=path.name):
                raw = path.read_bytes()
                self.assertNotIn(b"\r", raw, "patches must keep LF line endings (see .gitattributes)")
                self.assertTrue(raw.endswith(b"\n"))
                self.assertRegex(preamble(path)["Base"], r"\b[0-9a-f]{40}\b")
                files = parse_patch(path)
                self.assertTrue(files)
                for name in files:
                    self.assertIn(f"--- a/{name}\n+++ b/{name}\n", raw.decode())
                    self.assertRegex(raw.decode(), rf"diff --git a/{re.escape(name)} b/{re.escape(name)}\nindex [0-9a-f]{{7,}}\.\.[0-9a-f]{{7,}} 100644\n")

    def test_studio_patch_touches_only_inference_cli_docs_and_tests(self):
        self.assertEqual(set(parse_patch(STUDIO)), {"README.md", "infer_release.py", STUDIO_INFERENCE,
                                                    "test_infer_release.py", "test_inference_tiling.py"})

    def test_hecate_patches_touch_hecate_py_and_readme(self):
        for path in (HECATE, HECATE_PR3):
            self.assertEqual(set(parse_patch(path)), {"README.md", "hecate.py"})

    def test_hecate_variants_make_the_same_change(self):
        for inside in ("device = next(model.parameters()).device", "torch.autocast", "'--precision'"):
            with self.subTest(hunk=inside):
                self.assertEqual(changes(HECATE, "hecate.py", inside), changes(HECATE_PR3, "hecate.py", inside))
        self.assertEqual(changes(HECATE, "README.md", "--precision"), changes(HECATE_PR3, "README.md", "--precision"))

    def test_pr3_variant_keeps_resample_and_main_variant_does_not_know_it(self):
        pr3 = "\n".join(b for *_, body in parse_patch(HECATE_PR3)["hecate.py"] for b in body)
        main = "\n".join(b for *_, body in parse_patch(HECATE)["hecate.py"] for b in body)
        self.assertIn(" " + "    parser.add_argument('--resample', action='store_true',", pr3)
        self.assertNotIn("--resample", main)
        self.assertNotIn("+    parser.add_argument('--resample'", pr3)

    def test_defaults_are_unchanged_for_existing_users(self):
        studio = "".join(changes(STUDIO, STUDIO_INFERENCE, "--precision"))
        self.assertIn('"--precision", "--amp", dest="amp"', studio)   # old flag keeps working
        hecate = added_block(HECATE, "hecate.py", r"'--precision'")
        self.assertIn("default='fp32'", hecate)


@needs_torch
class StudioAddedCodeTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"torch": torch}
        exec(added_block(STUDIO, STUDIO_INFERENCE, r"^\+AMP_DTYPES"), self.ns)

    def test_auto_precision_by_compute_capability(self):
        resolve = self.ns["resolve_amp"]
        self.assertEqual(resolve("auto", "cpu"), "fp32")
        for capability, expected in (((6, 0), "fp16"), ((7, 0), "fp16"), ((7, 5), "fp16"),
                                     ((8, 0), "bf16"), ((8, 6), "bf16"), ((8, 9), "bf16"), ((9, 0), "bf16")):
            with self.subTest(capability=capability), \
                    mock.patch.object(torch.cuda, "get_device_capability", return_value=capability) as cap:
                self.assertEqual(resolve("auto", "cuda"), expected)
                cap.assert_called_once()

    def test_explicit_precision_is_never_overridden(self):
        with mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("not queried")):
            for amp in ("bf16", "fp16", "fp32"):
                self.assertEqual(self.ns["resolve_amp"](amp, "cuda"), amp)
                self.assertEqual(self.ns["resolve_amp"](amp, "cpu"), amp)

    def test_autocast_context_selection(self):
        line = added_block(STUDIO, STUDIO_INFERENCE, r"context = torch\.autocast").strip()
        expr = line.split("=", 1)[1]
        ns = dict(self.ns, nullcontext=nullcontext)
        for amp in ("fp32", "bf16", "fp16"):
            self.assertIsInstance(eval(expr, dict(ns, device="cpu", amp=amp)), nullcontext)
        with mock.patch.object(torch, "autocast", side_effect=lambda dev, dtype: (dev, dtype)):
            self.assertEqual(eval(expr, dict(ns, device="cuda", amp="fp16")), ("cuda", torch.float16))
            self.assertEqual(eval(expr, dict(ns, device="cuda:0", amp="bf16")), ("cuda", torch.bfloat16))
            self.assertIsInstance(eval(expr, dict(ns, device="cuda", amp="fp32")), nullcontext)

    def test_cli_flags(self):
        for path, name in ((STUDIO, STUDIO_INFERENCE), (STUDIO, "infer_release.py")):
            with self.subTest(file=name):
                parser = argparse.ArgumentParser()
                exec(textwrap.dedent(added_block(path, name, r"""add_argument\(['"]--precision""")), {"parser": parser})
                self.assertEqual(parser.parse_args([]).amp, "auto")
                self.assertEqual(parser.parse_args(["--precision", "fp16"]).amp, "fp16")
                self.assertEqual(parser.parse_args(["--amp", "fp32"]).amp, "fp32")
                with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                    parser.parse_args(["--precision", "fp8"])


@needs_torch
class HecateAddedCodeTests(unittest.TestCase):
    def check(self, path, precision, device):
        body = added_block(path, "hecate.py", r"if precision not in")
        ns = {"torch": torch}
        exec("def check(precision, device):\n" + body + "    return amp_dtype\n", ns)
        return ns["check"](precision, torch.device(device))

    def test_precision_validation_and_dtype(self):
        for path in (HECATE, HECATE_PR3):
            with self.subTest(patch=path.name):
                self.assertEqual(self.check(path, "fp16", "cuda"), torch.float16)
                self.assertEqual(self.check(path, "bf16", "cuda"), torch.bfloat16)
                self.check(path, "fp32", "cpu")
                for precision, device in (("fp16", "cpu"), ("bf16", "cpu"), ("fp8", "cuda"), ("auto", "cuda")):
                    with self.assertRaisesRegex(ValueError, "require CUDA"):
                        self.check(path, precision, device)

    def test_autocast_enabled_only_off_fp32(self):
        line = added_block(HECATE, "hecate.py", r"with torch\.autocast").strip()
        self.assertIn("dtype=amp_dtype", line)
        enabled = re.search(r"enabled=([^)]+)\)", line).group(1)
        self.assertEqual({p: eval(enabled, {"precision": p}) for p in ("fp32", "bf16", "fp16")},
                         {"fp32": False, "bf16": True, "fp16": True})

    def test_cli_flag(self):
        parser = argparse.ArgumentParser()
        exec(textwrap.dedent(added_block(HECATE, "hecate.py", r"'--precision'")), {"parser": parser})
        self.assertEqual(parser.parse_args([]).precision, "fp32")
        self.assertEqual(parser.parse_args(["--precision", "fp16"]).precision, "fp16")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parser.parse_args(["--precision", "auto"])


class DraftGuardRailTests(unittest.TestCase):
    def kit_files(self):
        return [p for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in (".md", ".patch", ".py", "")
                or p.name == ".gitattributes"]

    def test_drafts_exist(self):
        for draft in DRAFTS:
            self.assertTrue(draft.is_file(), draft.name)
            self.assertGreater(len(draft.read_text(encoding="utf-8")), 500)

    def test_no_studio_w00_auc_claim(self):
        for draft in DRAFTS:
            for n, line in enumerate(draft.read_text(encoding="utf-8").splitlines(), 1):
                with self.subTest(draft=draft.name, line=n):
                    self.assertFalse(re.search(r"w00\b", line, re.I) and re.search(r"\bAUC\b", line),
                                     f"PHerc0841 w00 is a Studio training donor; no AUC claim on it: {line!r}")
                    self.assertNotRegex(line, r"0\.91[56]|0\.924")

    def check_kit_privacy(self, user: str):
        for path in self.kit_files():
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name, user=user):
                self.assertIsNone(LOCAL_PATH.search(text))
                self.assertIsNone(TOKEN.search(text))
                self.assertEqual(leak_guard.find_user_leaks(text, user), [])

    def test_no_local_paths_user_names_or_tokens(self):
        self.check_kit_privacy(Path.home().name)

    def test_kit_passes_leak_guard_for_generic_root_account(self):
        # The English word "root" (repo_root, hecate_root) must not trip the guard when the account is "root".
        self.check_kit_privacy("root")

    def test_root_account_still_rejects_paths_and_identifying_contexts(self):
        secret = "/" + "root" + "/secret"                       # built here so this file holds no literal path
        win = "C:" + chr(92) + "Users" + chr(92) + "alice" + chr(92) + "data"
        for text in (secret, f"cp x {secret}", win, "/" + "home/alice/x", "/" + "Users/alice/x", "App" + "Data",
                     ".t3/" + "worktrees/x"):
            with self.subTest(text=text):
                self.assertIsNotNone(LOCAL_PATH.search(text))
        for text in ("ssh root" + "@host", "user=" + "root", "~" + "root/x"):
            with self.subTest(text=text):
                self.assertTrue(leak_guard.find_user_leaks(text, "root"))
        for text in ("repo_root = Path('.')", "hecate_root", "the root of the problem", "root_dir", "https://h.io/root-cause"):
            with self.subTest(text=text):
                self.assertIsNone(LOCAL_PATH.search(text))
                self.assertEqual(leak_guard.find_user_leaks(text, "root"), [])

    def test_distinctive_user_name_is_still_a_bare_substring_leak(self):
        self.assertTrue(leak_guard.find_user_leaks("written by Alice", "alice"))
        self.assertEqual(leak_guard.find_user_leaks("written by someone", "alice"), [])


def git(*args, cwd=None, data=None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, input=data, capture_output=True)


class UpstreamApplyTests(unittest.TestCase):
    """Real `git apply --check` against the pinned upstream blobs; skipped without a local upstream clone."""

    def stage(self, upstream: Path, patch: Path, tmp: Path, extra=()):
        index = dict(re.findall(r"diff --git a/(\S+) b/\S+\nindex ([0-9a-f]+)\.\.", patch.read_text(encoding="utf-8")))
        rev = base_commit(patch)
        self.assertEqual(git("init", "-q", str(tmp)).returncode, 0)
        for name in {*index, *extra}:
            blob = git("show", f"{rev}:{name}", cwd=upstream)
            self.assertEqual(blob.returncode, 0, f"{rev}:{name} not in the upstream clone: {blob.stderr.decode()}")
            (tmp / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp / name).write_bytes(blob.stdout)
            if name in index:
                sha = git("hash-object", str(tmp / name)).stdout.decode().strip()
                self.assertTrue(sha.startswith(index[name]), f"{name}: upstream blob {sha} != patch pre-image {index[name]}")
        check = git("apply", "--check", "-v", str(patch), cwd=tmp)
        self.assertEqual(check.returncode, 0, check.stderr.decode())
        self.assertEqual(git("apply", str(patch), cwd=tmp).returncode, 0)
        post = dict(re.findall(r"diff --git a/(\S+) b/\S+\nindex [0-9a-f]+\.\.([0-9a-f]+)", patch.read_text(encoding="utf-8")))
        for name, expected in post.items():
            sha = git("hash-object", str(tmp / name)).stdout.decode().strip()
            self.assertTrue(sha.startswith(expected), f"{name}: patched blob {sha} != patch post-image {expected}")

    @unittest.skipUnless(os.environ.get("STUDIO_UPSTREAM_DIR") and shutil.which("git"), "set STUDIO_UPSTREAM_DIR")
    def test_studio_patch_applies_and_runs_on_cpu(self):
        upstream = Path(os.environ["STUDIO_UPSTREAM_DIR"])
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self.stage(upstream, STUDIO, tmp, extra=("studio-multiscroll-cpt-20261023/hf/model.py",))
            if torch is None:
                return
            sys.path.insert(0, str(tmp / "studio-multiscroll-cpt-20261023/hf"))
            sys.modules.pop("inference", None)
            sys.modules.pop("model", None)
            try:
                inference = importlib.import_module("inference")
            finally:
                sys.path.pop(0)
                sys.modules.pop("inference", None)
                sys.modules.pop("model", None)
            stub = lambda x: x.mean(2) * 8 - 2 + x.std(2)   # noqa: E731 - per-pixel stand-in for the U-Net
            volume = np.random.default_rng(0).integers(1, 255, (16, 150, 170), dtype=np.uint8)
            out = {amp: np.array(inference.predict(stub, volume, tmp / f"{amp}.npy", device="cpu", amp=amp,
                                                   stride=48, blend="hann"))
                   for amp in ("fp32", "fp16", "bf16")}
            np.testing.assert_array_equal(out["fp32"], out["fp16"])
            np.testing.assert_array_equal(out["fp32"], out["bf16"])
            self.assertEqual(inference.resolve_amp("auto", "cpu"), "fp32")

    @unittest.skipUnless(os.environ.get("HECATE_UPSTREAM_DIR") and shutil.which("git"), "set HECATE_UPSTREAM_DIR")
    def test_hecate_patches_apply_and_validate_on_cpu(self):
        upstream = Path(os.environ["HECATE_UPSTREAM_DIR"])
        for patch in (HECATE, HECATE_PR3):
            with self.subTest(patch=patch.name), tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                self.stage(upstream, patch, tmp)
                if torch is None:
                    continue
                spec = importlib.util.spec_from_file_location(f"hecate_{patch.stem[:5]}", tmp / "hecate.py")
                hecate = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hecate)

                class Stub(torch.nn.Module):
                    patch_size, divisor = (4, 16, 16), 255.0

                    def __init__(self):
                        super().__init__()
                        self.w = torch.nn.Parameter(torch.ones(()))

                    def forward(self, image, valid):
                        return image.mean(2) * self.w

                volume = np.random.default_rng(1).integers(1, 255, (6, 40, 40), dtype=np.uint8)
                output = np.zeros((40, 40), np.uint8)
                with mock.patch("builtins.print"):
                    hecate.predict(Stub(), volume, output, precision="fp32")
                self.assertGreater(int(output.max()), 0)
                for precision in ("fp16", "bf16"):
                    with self.assertRaisesRegex(ValueError, "bf16 and fp16 require CUDA"):
                        hecate.predict(Stub(), volume, np.zeros((40, 40), np.uint8), precision=precision)


if __name__ == "__main__":
    unittest.main()
