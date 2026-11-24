"""Offline CPU-only tests for leak_guard.py (generic account names such as "root")."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leak_guard as lg  # noqa: E402

BS = chr(92)


class LeakGuardTests(unittest.TestCase):
    def test_local_paths_rejected_for_every_account_including_root(self):
        for text in ("C:" + BS + "Users" + BS + "alice" + BS + "f", "/" + "Users/alice/f", "/" + "home/alice/f",
                     "/" + "root" + "/secret", "cat /" + "root/", "App" + "Data", ".t3/" + "worktrees"):
            with self.subTest(text=text):
                self.assertTrue(lg.find_local_paths(text))
                for user in ("root", "alice", "runner"):
                    self.assertTrue(lg.find_leaks(text, user))

    def test_root_word_is_not_a_leak_for_root_account(self):
        for text in ("repo_root", "hecate_root = 1", "root of the tree", "tree/root-node", "rootfs",
                     "https://example.org/root-cause"):
            with self.subTest(text=text):
                self.assertEqual(lg.find_leaks(text, "root"), [])

    def test_root_account_identifying_contexts_rejected(self):
        for text in ("root" + "@host", "ssh -l x @" + "root", "~" + "root/x", "user=" + "root",
                     'username: "' + 'root"', BS + "root" + BS + "x", "/srv/" + "root/x"):
            with self.subTest(text=text):
                self.assertTrue(lg.find_user_leaks(text, "root"))

    def test_generic_names_set(self):
        for n in ("root", "user", "admin", "administrator", "runner", "ubuntu", "default", "owner", "ROOT"):
            self.assertTrue(lg.is_generic_account(n))
        self.assertFalse(lg.is_generic_account("alice"))

    def test_distinctive_name_bare_substring(self):
        self.assertTrue(lg.find_user_leaks("by Alice Smith", "alice"))
        self.assertEqual(lg.find_user_leaks("by someone", "alice"), [])

    def test_short_distinctive_name_skipped(self):
        self.assertEqual(lg.find_user_leaks("abc", "abc"), [])

    def test_other_generic_accounts_behave_like_root(self):
        self.assertEqual(lg.find_user_leaks("the user guide, default owner", "user"), [])
        self.assertTrue(lg.find_user_leaks("ssh admin" + "@host", "admin"))


if __name__ == "__main__":
    unittest.main()
