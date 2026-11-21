"""Local tests for worker_v002.py and the notebook bootstrap, using LocalStore in a scratch directory.

Run: python test_worker_v2.py [-v]   (CPU only; uses a local folder instead of Drive)
"""
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("WORKER_TEST_TMP", os.path.join(tempfile.gettempdir(), "worker_v2_tests"))
FAST = dict(poll=0.3, tick=0.05, log_every=0.5, out_every=1, status_every=0.5, update_every=0.4)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = load(os.path.join(HERE, "worker_v002.py"), "worker_v002")
V1 = load(os.path.join(HERE, "worker.py"), "worker_v1")
NB = load(os.path.join(HERE, "make_notebook_v2.py"), "make_notebook_v2")


def bootstrap_ns():
    ns = {"__name__": "bootstrap_under_test"}
    exec(compile(NB.BOOTSTRAP, "bootstrap", "exec"), ns)
    return ns


class CountingStore(W.LocalStore):
    def __init__(self):
        self.calls = []

    def ls(self, folder_id):
        self.calls.append(("ls", os.path.basename(folder_id)))
        return super().ls(folder_id)

    def download(self, file_id, path):
        self.calls.append(("download", os.path.basename(file_id)))
        return super().download(file_id, path)

    def mkdir(self, name, parent):
        self.calls.append(("mkdir", name))
        return super().mkdir(name, parent)

    def put(self, path, parent, name, mime=None, known=None):
        self.calls.append(("put", name))
        return super().put(path, parent, name, mime, known)


class Env:
    def __init__(self, tag, store=None):
        os.makedirs(SCRATCH, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix=tag + "_", dir=SCRATCH)
        self.store = store or W.LocalStore()
        self.jobs, self.results, self.inputs = (os.path.join(self.root, n) for n in ("jobs", "results", "inputs"))
        for d in (self.jobs, self.results, self.inputs):
            os.makedirs(d)
        self.work = os.path.join(self.root, "_work")
        self.trace = os.path.join(self.root, "trace.txt")
        self.result = None
        self.thread = None

    @property
    def ids(self):
        return (self.root, self.jobs, self.results, self.inputs)

    def job(self, name, body, header=""):
        with open(os.path.join(self.jobs, name + ".sh"), "w") as f:
            f.write(header + body)

    def timed(self, name, secs, header="", extra=""):
        """A job that records its start and end in the shared trace file."""
        self.job(name, f'echo "$JOB start $(date +%s.%N)" >> {self.trace}\n{extra}\nsleep {secs}\n'
                       f'echo "$JOB end $(date +%s.%N)" >> {self.trace}\n', header)

    def touch(self, rel, text=""):
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        return p

    def main(self, **kw):
        opts = dict(FAST)
        opts.update(kw)
        return W.main(*self.ids, store=self.store, work=self.work, **opts)

    def start(self, **kw):
        def go():
            self.result = self.main(**kw)
        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()

    def stop(self, timeout=20):
        self.touch("jobs/STOP")
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "worker did not stop"
        os.remove(os.path.join(self.jobs, "STOP"))
        return self.result

    def done(self, name):
        p = os.path.join(self.results, name, "DONE.txt")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return W.read_fields(f.read())

    def wait(self, cond, timeout=20, what="condition"):
        t = time.time()
        while time.time() - t < timeout:
            if cond():
                return
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}")

    def wait_done(self, *names, timeout=20):
        self.wait(lambda: all(self.done(n) for n in names), timeout, f"DONE of {names}")

    def status(self):
        with open(os.path.join(self.root, "status.txt")) as f:
            return f.read()

    def intervals(self):
        iv = {}
        if not os.path.exists(self.trace):
            return iv
        for line in rd(self.trace).splitlines():
            n, what, t = line.split()
            iv.setdefault(n, [None, None])[0 if what == "start" else 1] = float(t)
        return iv

    def log(self, name):
        with open(os.path.join(self.results, name, "log.txt")) as f:
            return f.read()


def rd(path):
    with open(path) as f:
        return f.read()


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def max_concurrent(iv):
    ev = sorted([(s, 1) for s, e in iv.values()] + [(e, -1) for s, e in iv.values()], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


class TestHeaders(unittest.TestCase):
    def test_parse_header(self):
        h = W.parse_header("# timeout: 120\n# slot: cpu\n# after: a_1, b_2.sh\n# after: c_3 d_4\necho\n")
        self.assertEqual(h, {"timeout": 120, "slot": "cpu", "after": ["a_1", "b_2", "c_3", "d_4"]})
        self.assertEqual(W.parse_header("echo hi\n"), {"timeout": 10800, "slot": "gpu", "after": []})
        late = "\n" * 40 + "# slot: cpu\n# after: x\n# timeout: 5\n"
        self.assertEqual(W.parse_header(late), {"timeout": 5, "slot": "gpu", "after": []})  # timeout: whole file, as v1

    def test_round_robin_pick(self):
        self.assertEqual(W.pick_round_robin(["w_1", "fl_2", "fl_1", "hr_1"], None), "fl_1")
        self.assertEqual(W.pick_round_robin(["w_1", "fl_2", "hr_1"], "fl"), "hr_1")
        self.assertEqual(W.pick_round_robin(["w_1", "fl_2", "hr_1"], "hr"), "w_1")
        self.assertEqual(W.pick_round_robin(["fl_2", "hr_1"], "w"), "fl_2")
        self.assertEqual(W.pick_round_robin(["fl_2", "w_1"], "hr"), "w_1")


class TestJobs(unittest.TestCase):
    def test_v1_job_unchanged(self):
        e = Env("v1")
        e.touch("inputs/data.txt", "input-content")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
        e.job("v_ok", '# timeout: 60\nset -x\necho "OUT=$OUT INPUTS=$INPUTS JOB=$JOB PWD=$(pwd)"\n'
                      'echo "cuda=${CUDA_VISIBLE_DEVICES-unset} unbuf=$PYTHONUNBUFFERED"\n'
                      'cat $INPUTS/data.txt; echo; mkdir -p $OUT/sub; echo hello > $OUT/sub/x.txt; echo top > $OUT/top.txt\n')
        e.job("v_fail", "echo failing; exit 3\n")
        e.job("v_timeout", "# timeout: 1\nsleep 30\n")
        t0 = time.time()
        self.assertEqual(e.main(max_idle=2), "idle")
        self.assertLess(time.time() - t0, 15)
        d = e.done("v_ok")
        self.assertEqual((d["status"], d["exit code"]), ("ok", "0"))
        log = e.log("v_ok")
        jdir = os.path.join(e.work, "jobs", "v_ok")
        self.assertIn(f"OUT={jdir}/out INPUTS={e.work}/inputs JOB=v_ok PWD={jdir}", log)
        self.assertIn("cuda=unset unbuf=1", log)
        self.assertIn("input-content", log)
        self.assertEqual(rd(os.path.join(e.results, "v_ok", "sub__x.txt")), "hello\n")
        self.assertEqual(rd(os.path.join(e.results, "v_ok", "top.txt")), "top\n")
        d = e.done("v_fail")
        self.assertEqual((d["status"], d["exit code"]), ("failed", "3"))
        self.assertIn("failing", e.log("v_fail"))
        d = e.done("v_timeout")
        self.assertEqual(d["status"], "timeout")
        self.assertLessEqual(int(d["seconds"]), 3)
        for k in ("status", "exit code", "seconds", "finished", "gpu"):
            self.assertIn(k, d)

    def test_cpu_gpu_overlap_gpu_exclusive(self):
        e = Env("slots")
        e.timed("g_1", 1.0)
        e.timed("g_2", 1.0)
        e.timed("c_1", 1.0, "# slot: cpu\n", 'echo "cuda=[${CUDA_VISIBLE_DEVICES-unset}]"')
        e.main(max_idle=2)
        iv = e.intervals()
        self.assertEqual(len(iv), 3)
        self.assertFalse(overlap(iv["g_1"], iv["g_2"]), "two gpu jobs overlapped")
        self.assertTrue(overlap(iv["c_1"], iv["g_1"]) or overlap(iv["c_1"], iv["g_2"]), "cpu and gpu did not overlap")
        self.assertIn("cuda=[]", e.log("c_1"))
        self.assertEqual(e.done("c_1")["slot"], "cpu")
        self.assertLessEqual(max_concurrent(iv), 2)

    def test_two_cpu_jobs_when_no_gpu_job(self):
        e = Env("twocpu")
        e.timed("c_a", 1.0, "# slot: cpu\n")
        e.timed("c_b", 1.0, "# slot: cpu\n")
        e.timed("c_c", 1.0, "# slot: cpu\n")
        e.main(max_idle=2)
        iv = e.intervals()
        self.assertTrue(overlap(iv["c_a"], iv["c_b"]))
        self.assertEqual(max_concurrent(iv), 2)

    def test_pending_gpu_job_keeps_the_gpu_slot(self):
        # p_gpu waits for p_prep (cpu); q_cpu must not take the second slot meanwhile.
        e = Env("reserve")
        e.timed("p_prep", 1.0, "# slot: cpu\n")
        e.timed("p_zgpu", 0.5, "# after: p_prep\n")
        e.timed("q_cpu", 1.0, "# slot: cpu\n")
        e.main(max_idle=2)
        iv = e.intervals()
        self.assertGreaterEqual(iv["p_zgpu"][0], iv["p_prep"][1])
        self.assertGreaterEqual(iv["q_cpu"][0], iv["p_prep"][1] - 0.05)
        self.assertTrue(overlap(iv["q_cpu"], iv["p_zgpu"]))

    def test_round_robin_three_lanes(self):
        e = Env("rr")
        for l in ("w", "fl", "hr"):
            for i in (1, 2, 3):
                e.timed(f"{l}_{i:03d}", 0.1)
        e.timed("fl_c_001", 0.1, "# slot: cpu\n")
        e.timed("w_c_001", 0.1, "# slot: cpu\n")
        e.timed("w_c_002", 0.1, "# slot: cpu\n")
        e.main(max_idle=2)
        iv = e.intervals()
        gpu = sorted((n for n in iv if "_c_" not in n), key=lambda n: iv[n][0])
        self.assertEqual(gpu, ["fl_001", "hr_001", "w_001", "fl_002", "hr_002", "w_002", "fl_003", "hr_003", "w_003"])
        cpu = sorted((n for n in iv if "_c_" in n), key=lambda n: iv[n][0])
        self.assertEqual(cpu, ["fl_c_001", "w_c_001", "w_c_002"])
        self.assertLessEqual(max_concurrent(iv), 2)

    def test_after_dependencies(self):
        e = Env("after")
        e.timed("d_1", 0.6)
        e.timed("d_2", 0.1, "# after: d_1\n")
        e.job("e_1", "exit 1\n")
        e.timed("e_2", 0.1, "# after: e_1\n")
        e.timed("e_3", 0.1, "# slot: cpu\n# after: d_2, e_2\n")
        e.timed("f_1", 0.1, "# after: ghost_1\n")
        e.start()
        e.wait_done("d_2", "e_3")
        iv = e.intervals()
        self.assertGreaterEqual(iv["d_2"][0], iv["d_1"][1])
        self.assertEqual(e.done("e_2")["status"], "skipped")
        self.assertIn("e_1", e.done("e_2")["reason"])
        self.assertEqual(e.done("e_3")["status"], "skipped")
        self.assertIn("e_2", e.done("e_3")["reason"])
        self.assertNotIn("e_2", iv)
        time.sleep(1.0)
        self.assertIsNone(e.done("f_1"))
        st = e.status()
        self.assertIn("pending gpu f_1 (after: ghost_1; not submitted yet: ghost_1)", st)
        self.assertIn("skipped e_2 (dependency e_1 ended with status failed)", st)
        e.timed("ghost_1", 0.3)
        e.wait_done("f_1")
        iv = e.intervals()
        self.assertGreaterEqual(iv["f_1"][0], iv["ghost_1"][1])
        e.stop()

    def test_cancel_pending_and_running(self):
        e = Env("cancel")
        pidf = os.path.join(e.root, "child.pid")
        e.job("h_run", f'sleep 60 & echo $! > {pidf}\necho started; wait\n')
        e.job("k_pend", f'touch {e.root}/k_pend_ran\n')
        e.start()
        e.wait(lambda: os.path.exists(pidf) and rd(pidf).strip(), what="h_run start")
        time.sleep(0.5)
        self.assertIsNone(e.done("k_pend"))   # gpu slot is taken by h_run
        e.touch("jobs/CANCEL_k_pend.txt")
        e.wait_done("k_pend")
        self.assertEqual(e.done("k_pend")["status"], "cancelled")
        self.assertIsNone(e.done("h_run"))
        e.touch("jobs/CANCEL_h_run")
        e.wait_done("h_run", timeout=10)
        d = e.done("h_run")
        self.assertEqual(d["status"], "cancelled")
        child = int(rd(pidf))
        time.sleep(0.2)
        self.assertFalse(os.path.exists(f"/proc/{child}") and "sleep" in rd(f"/proc/{child}/cmdline"),
                         "job child process still alive after cancel")
        self.assertFalse(os.path.exists(os.path.join(e.root, "k_pend_ran")))
        self.assertIn("started", e.log("h_run"))
        st = e.status()
        self.assertRegex(st, r"(?m)^cancelled \d+s h_run")
        self.assertRegex(st, r"(?m)^cancelled k_pend")
        e.stop()

    def test_restart_does_not_rerun_finished_jobs(self):
        e = Env("restart")
        cnt = os.path.join(e.root, "runs.txt")
        for n in ("r_done", "r_half", "r_new"):
            e.job(n, f"echo $JOB >> {cnt}\n")
        e.touch("results/r_done/DONE.txt", "status: ok\nexit code: 0\nseconds: 12\n")
        e.touch("results/r_half/log.txt", "partial log from a VM that died\n")
        e.main(max_idle=2)
        self.assertEqual(sorted(rd(cnt).split()), ["r_half", "r_new"])
        self.assertEqual(e.done("r_half")["status"], "ok")
        self.assertIn("done ok 12s r_done", e.status())
        e.main(max_idle=2)   # second worker run, same VM
        shutil.rmtree(e.work)  # and a fresh VM
        e.main(max_idle=2)
        self.assertEqual(sorted(rd(cnt).split()), ["r_half", "r_new"])

    def test_stop_kills_and_job_reruns(self):
        e = Env("stop")
        cnt = os.path.join(e.root, "cnt")
        e.job("s_long", f'n=$(cat {cnt} 2>/dev/null || echo 0); echo $((n+1)) > {cnt}\n'
                        f'if [ "$n" = 0 ]; then sleep 60; fi\necho finished\n')
        e.start()
        e.wait(lambda: os.path.exists(cnt), what="s_long start")
        time.sleep(0.3)
        t0 = time.time()
        self.assertEqual(e.stop(), "stopped")
        self.assertLess(time.time() - t0, 5)
        self.assertIsNone(e.done("s_long"))
        self.assertIn("state: stopped", e.status())
        self.assertEqual(e.main(max_idle=2), "idle")
        self.assertEqual(e.done("s_long")["status"], "ok")
        self.assertEqual(rd(cnt).strip(), "2")
        self.assertIn("finished", e.log("s_long"))

    def test_leftover_process_groups_killed_at_start(self):
        e = Env("leftover")
        os.makedirs(e.work)
        import subprocess
        p = subprocess.Popen(["sleep", "60"], start_new_session=True)
        with open(os.path.join(e.work, "pgids.json"), "w") as f:
            f.write(f"[{p.pid}]")
        e.main(max_idle=1)
        self.assertEqual(p.wait(timeout=5), -9)


class TestOrphansAndWatch(unittest.TestCase):
    def test_orphans_of_an_older_worker_killed_at_start(self):
        # v1 kept no pgid list: orphans are recognised by cwd under work/jobs or by OUT= in their environment
        e = Env("orphan")
        import subprocess
        jd = os.path.join(e.work, "jobs", "old_job")
        os.makedirs(os.path.join(jd, "out"))
        by_cwd = subprocess.Popen(["sleep", "60"], cwd=jd, start_new_session=True)
        by_env = subprocess.Popen(["sleep", "60"], cwd=e.root, start_new_session=True,
                                  env=dict(os.environ, OUT=os.path.join(jd, "out")))
        bystander = subprocess.Popen(["sleep", "60"], cwd=e.root, start_new_session=True)
        e.main(max_idle=1)
        self.assertEqual(by_cwd.wait(timeout=5), -9)
        self.assertEqual(by_env.wait(timeout=5), -9)
        self.assertIsNone(bystander.poll())
        bystander.kill()

    def status_text(self, age, wid="deadbeef"):
        t = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(time.time() - age))
        return f"time: {t}\nworker started: {t}\nstate: running x (gpu) 10s\ngpu: x\ndisk free: 1 GB\nworker id: {wid}\n"

    def test_watch_goes_stale_then_runs(self):
        # the other session was just stopped: its last heartbeat is fresh but never moves -> we start by ourselves
        e = Env("watch_stale")
        ran = os.path.join(e.root, "ran")
        e.job("g_1", f"echo $JOB >> {ran}\n")
        e.touch("status.txt", self.status_text(0))
        t = time.time()
        self.assertEqual(e.main(max_idle=2, fresh_seconds=3, guard_wait=10, guard_poll=0.5), "idle")
        self.assertGreaterEqual(time.time() - t, 2)
        self.assertEqual(rd(ran).split(), ["g_1"])

    def test_watch_sees_heartbeat_move_then_refuses(self):
        e = Env("watch_alive")
        ran = os.path.join(e.root, "ran")
        e.job("g_1", f"echo $JOB >> {ran}\n")
        e.touch("status.txt", self.status_text(0))
        stop = threading.Event()

        def beat():   # a live foreign worker keeps refreshing its heartbeat
            while not stop.is_set():
                time.sleep(1.1)
                e.touch("status.txt", self.status_text(0))
        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            self.assertEqual(e.main(max_idle=2, fresh_seconds=30, guard_wait=10, guard_poll=0.5), "refused")
        finally:
            stop.set()
            th.join()
        self.assertFalse(os.path.exists(ran))


class TestUpdateAndGuard(unittest.TestCase):
    def lister_reader(self, e):
        return (lambda: e.store.ls(e.root)), (lambda i: rd(i))

    def test_hot_update_through_bootstrap(self):
        e = Env("update")
        shutil.copyfile(os.path.join(HERE, "worker_v002.py"), os.path.join(e.root, "worker_v002.py"))
        mark = os.path.join(e.root, "v3_ran.txt")
        e.timed("u_1", 1.5)
        e.timed("u_2", 0.1)
        ns = bootstrap_ns()
        lister, reader = self.lister_reader(e)
        out = {}
        opts = dict(FAST, store=e.store, work=e.work, pause=0.05)
        th = threading.Thread(target=lambda: out.setdefault("r", ns["lancer_ouvrier"](lister, reader, e.ids, **opts)))
        th.start()
        e.wait(lambda: "u_1" in e.intervals(), what="u_1 start")
        e.touch("worker_v003.py",
                "import time\ndef main(root, jobs, results, inputs, store=None, **kw):\n"
                f"    open({mark!r}, 'w').write(f'{{LOADED_VERSION}} {{sorted(SKIP_VERSIONS)}} {{time.time()}}')\n"
                "    return 'v3-marker'\n")
        e.wait(lambda: "updating to worker_v003" in e.status(), what="draining status")
        th.join(20)
        self.assertEqual(out.get("r"), "v3-marker")
        self.assertEqual(e.done("u_1")["status"], "ok")
        self.assertIsNone(e.done("u_2"), "a new job started while draining")
        v, skip, t = rd(mark).split(" ", 2)
        self.assertEqual((v, skip), ("3", "[]"))
        self.assertGreater(float(t), e.intervals()["u_1"][1])
        self.assertIn("state: updating: handing over to worker_v003", e.status())

        # a broken higher version: tried 3 times, then the bootstrap falls back to v003
        e.touch("worker_v004.py", "def main(:\n")
        r = ns["lancer_ouvrier"](lister, reader, e.ids, **opts)
        self.assertEqual(r, "v3-marker")
        v, skip, _ = rd(mark).split(" ", 2)
        self.assertEqual((v, skip), ("3", "[4]"))

        # v002 told to skip v004 does not drain for it and runs u_2
        for n in ("worker_v003.py",):
            os.remove(os.path.join(e.root, n))
        ns2 = {"__name__": "x", "LOADED_VERSION": 2, "SKIP_VERSIONS": {4}}
        exec(compile(rd(os.path.join(HERE, "worker_v002.py")), "worker_v002.py", "exec"), ns2)
        self.assertEqual(ns2["main"](*e.ids, store=e.store, work=e.work, max_idle=2, **FAST), "idle")
        self.assertEqual(e.done("u_2")["status"], "ok")

    def test_bootstrap_without_worker_file(self):
        e = Env("noworker")
        ns = bootstrap_ns()
        lister, reader = self.lister_reader(e)
        self.assertIsNone(ns["lancer_ouvrier"](lister, reader, e.ids, pause=0.01))

    def foreign_status(self, e, age, wid="deadbeef", state="running fl_005 (gpu) 10s"):
        t = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(time.time() - age))
        idline = f"worker version: 002\nworker id: {wid}\n" if wid else ""
        e.touch("status.txt", f"time: {t}\nworker started: {t}\nstate: {state}\ngpu: x\ndisk free: 1 GB\n{idline}")

    def test_second_worker_guard(self):
        e = Env("guard")
        ran = os.path.join(e.root, "ran")
        e.job("g_1", f"echo $JOB >> {ran}\n")
        self.foreign_status(e, 20)
        before = e.status()
        self.assertEqual(e.main(max_idle=1), "refused")
        self.assertEqual(e.status(), before)
        self.assertFalse(os.path.exists(ran))
        self.foreign_status(e, 20, wid=None)   # a v1 worker
        self.assertEqual(e.main(max_idle=1), "refused")
        self.assertFalse(os.path.exists(ran))
        self.foreign_status(e, 600)             # stale
        self.assertEqual(e.main(max_idle=2), "idle")
        self.assertEqual(rd(ran).split(), ["g_1"])
        e.job("g_2", f"echo $JOB >> {ran}\n")
        own = rd(os.path.join(e.work, "worker_id")).strip()
        self.foreign_status(e, 5, wid=own)      # same VM, fresh
        self.assertEqual(e.main(max_idle=2), "idle")
        e.job("g_3", f"echo $JOB >> {ran}\n")
        self.foreign_status(e, 5, state="stopped by STOP file")  # fresh but stopped
        self.assertEqual(e.main(max_idle=2), "idle")
        self.assertEqual(rd(ran).split(), ["g_1", "g_2", "g_3"])


class TestStatusAndCalls(unittest.TestCase):
    def test_status_content_while_running(self):
        e = Env("status")
        e.touch("results/a_000_old/DONE.txt", "status: ok\nexit code: 0\nseconds: 1491\n")
        e.timed("fl_005_scan", 3)
        e.timed("w_011", 3, "# slot: cpu\n")
        e.timed("w_001_gpu", 0.1)
        e.timed("w_c_012", 0.1, "# slot: cpu\n# after: w_011\n")
        e.job("ab_bad", "exit 2\n", "# slot: cpu\n")
        e.timed("ab_next", 0.1, "# after: ab_bad\n")
        e.job("z_cancel", "echo never\n")
        e.touch("jobs/CANCEL_z_cancel.txt")
        e.job("aa_quick", "exit 0\n", "# slot: cpu\n")
        e.start()
        e.wait(lambda: "fl_005_scan" in e.intervals() and "w_011" in e.intervals(), what="two starts")
        time.sleep(1.2)
        st = e.status()
        lines = st.splitlines()
        self.assertEqual([l.split(":")[0] for l in lines[:7]],
                         ["time", "worker started", "state", "gpu", "disk free", "worker version", "worker id"])
        self.assertRegex(lines[2], r"^state: running fl_005_scan \(gpu\) \d+s; w_011 \(cpu\) \d+s$")
        self.assertEqual(lines[5], "worker version: 002")
        self.assertEqual(lines[6], "worker id: " + rd(os.path.join(e.work, "worker_id")).strip())
        self.assertRegex(st, r"(?m)^running \d+s gpu fl_005_scan$")
        self.assertRegex(st, r"(?m)^running \d+s cpu w_011$")
        self.assertIn("\npending gpu w_001_gpu\n", st)
        self.assertIn("\npending cpu w_c_012 (after: w_011)\n", st)
        self.assertIn("\nskipped ab_next (dependency ab_bad ended with status failed)\n", st)
        self.assertIn("\ncancelled z_cancel (cancelled by marker before it started)\n", st)
        self.assertRegex(st, r"(?m)^done failed \d+s ab_bad$")
        self.assertIn("\ndone ok 1491s a_000_old\n", st)
        fin = [l for l in lines if re.match(r"^(done|skipped|cancelled) ", l)]
        self.assertEqual(fin[-1], "done ok 1491s a_000_old")  # oldest last
        e.wait_done("w_c_012", "w_001_gpu")
        st = e.status()
        self.assertIn("state: idle, waiting for jobs", st)
        e.stop()

    def build(self, n):
        store = CountingStore()
        e = Env(f"calls{n}", store)
        base = time.time() - 10000
        for i in range(n):
            name = f"j_{i:03d}"
            e.job(name, "exit 0\n")
            p = e.touch(f"results/{name}/DONE.txt", f"status: ok\nexit code: 0\nseconds: {i}\n")
            os.utime(p, (base + i, base + i))
            e.touch(f"results/{name}/log.txt", "x")
        w = W.Worker(store, *e.ids, work=e.work, **FAST)
        self.assertTrue(w.startup())
        startup_calls = len(store.calls)
        store.calls.clear()
        for _ in range(3):   # idle passes, with every periodic action firing on each one
            self.assertIsNone(w.list_pass())
            w.schedule()
            w.check_update()
            w.write_status()
            w.tend_running()
        return e, w, startup_calls, list(store.calls)

    def v1_calls(self, n):
        store = CountingStore()
        e = Env(f"v1calls{n}", store)
        for i in range(n):
            e.job(f"j_{i:03d}", "exit 0\n")
            e.touch(f"results/j_{i:03d}/DONE.txt", "status: ok\n")
        V1.Worker(store, *e.ids, work=e.work, poll=0).loop(max_idle_loops=1)
        return len(store.calls)

    def test_store_calls_per_idle_pass(self):
        e5, w5, start5, c5 = self.build(5)
        e70, w70, start70, c70 = self.build(70)
        per5, per70 = len(c5) / 3, len(c70) / 3
        print(f"\n  v2 store calls per idle pass: {per5:g} with 5 finished jobs, {per70:g} with 70 "
              f"({sorted(set(c70))}); startup: {start5} vs {start70} calls")
        v1_5, v1_70 = self.v1_calls(5), self.v1_calls(70)
        print(f"  v1 store calls per idle pass: {v1_5} with 5 finished jobs, {v1_70} with 70")
        self.assertEqual(per5, per70)
        self.assertLessEqual(per70, 3)
        self.assertGreater(v1_70, v1_5)
        # status: capped at 60 finished, newest first, statuses read from DONE.txt
        st = e70.status()
        fin = [l for l in st.splitlines() if l.startswith("done ")]
        self.assertEqual(len(fin), 60)
        self.assertEqual(fin[0], "done ok 69s j_069")
        self.assertEqual(fin[-1], "done ok 10s j_010")
        self.assertIn("queue: 0 running, 0 pending, 70 finished (newest 60 shown)", st)
        # restart in the same VM: the finished cache avoids re-listing every results folder
        store = e70.store
        store.calls.clear()
        w = W.Worker(store, *e70.ids, work=e70.work, **FAST)
        self.assertTrue(w.startup())
        print(f"  v2 restart in the same VM with 70 finished jobs: {len(store.calls)} store calls")
        self.assertLess(len(store.calls), 10)


class TestNotebookAndEdgeCases(unittest.TestCase):
    def test_notebook_generator(self):
        nb = NB.notebook({"root": "R0", "jobs": "J0", "results": "S0", "inputs": "I0"})
        self.assertEqual([c["cell_type"] for c in nb["cells"]], ["markdown", "code", "code"])
        boot = "".join(nb["cells"][2]["source"])
        for var, val in (("ROOT_ID", "R0"), ("JOBS_ID", "J0"), ("RESULTS_ID", "S0"), ("INPUTS_ID", "I0")):
            self.assertEqual(re.search(var + r' = "([^"]+)"', boot).group(1), val)
            self.assertEqual(re.search(var + r' = "([^"]+)"', NB.BOOTSTRAP).group(1)[0], "<")
        self.assertLessEqual(len(boot.splitlines()), 45)
        compile(boot, "bootstrap", "exec")
        md = "".join(nb["cells"][0]["source"])
        self.assertIn("se met à jour toute seule", md)
        self.assertIn("le travail déjà terminé n'est pas refait", md)

    def test_unreadable_done_does_not_skip_dependents(self):
        class Flaky(W.LocalStore):
            def download(self, file_id, path):
                if file_id.endswith("DONE.txt"):
                    raise OSError("drive hiccup")
                return super().download(file_id, path)
        e = Env("flaky", Flaky())
        e.job("m_1", "exit 0\n")
        e.job("m_2", "exit 0\n", "# after: m_1\n")
        e.touch("results/m_1/DONE.txt", "status: ok\n")
        w = W.Worker(e.store, *e.ids, work=e.work, **FAST)
        self.assertTrue(w.startup())
        self.assertIsNone(w.list_pass())
        self.assertEqual(w.dep_state("m_2")[0], "waiting")
        self.assertIsNone(e.done("m_2"))
        e.store.__class__ = W.LocalStore   # Drive is back
        self.assertEqual(w.dep_state("m_2")[0], "eligible")


if __name__ == "__main__":
    unittest.main()
