"""GPU job worker v2: runs bash jobs dropped into a Google Drive folder and uploads their output.

Loaded by the notebook bootstrap (which execs the highest worker_vNNN.py of the Drive root and calls
main()), or run locally for testing:  python worker_v002.py <root_dir>   (LocalStore).

Protocol (see README_v2.md)
  jobs/<name>.sh        a bash script. Header lines (first 30 lines):
                          # timeout: <seconds>        default 10800
                          # slot: cpu                 does not use the GPU (run with CUDA_VISIBLE_DEVICES="")
                          # after: <job>[, <job>...]  start only once those jobs ended with status ok
  jobs/CANCEL_<name>[.*] cancels job <name> (pending or running).
  jobs/STOP             stops the worker; running jobs are killed without DONE.txt and re-run next time.
  inputs/*              files synced to $INPUTS before each job start.
  results/<name>/       log.txt (refreshed while running), every file the job writes under $OUT,
                        and DONE.txt (status ok|failed|timeout|cancelled|skipped) when it finishes.
  status.txt            heartbeat + full queue, refreshed about once a minute and at each job start/end.
  worker_vNNN.py        a higher number in the root makes this worker drain and hand over (hot update).
Up to 2 jobs run at once, at most one of them a GPU job. A job with results/<name>/DONE.txt never re-runs.
"""
import datetime
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

WORKER_VERSION = 2
MAX_UPLOAD_BYTES = 800 * 1024 * 1024
LOG_TAIL_BYTES = 400_000
FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_TIMEOUT = 10800
HEADER_LINES = 30
FINISHED_SHOWN = 60
FRESH_SECONDS = 150
WORKER_FILE_RE = re.compile(r"^worker_v(\d{3})\.py$")
TIME_FMT = "%Y-%m-%d %H:%M:%S UTC"


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(TIME_FMT)


def parse_now(s):
    try:
        return datetime.datetime.strptime(s.strip(), TIME_FMT).replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return None


def drive_ts(s):
    """modifiedTime as epoch seconds: RFC 3339 from Drive, a float string from LocalStore."""
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


class DriveStore:
    """Google Drive through the REST API, authenticated as the Colab user. (v1 code; only pageSize changed.)"""

    def __init__(self):
        from google.colab import auth
        from googleapiclient.discovery import build

        auth.authenticate_user()
        self.svc = build("drive", "v3", cache_discovery=False)

    def _retry(self, fn, tries=5):
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # network hiccups and rate limits
                if i == tries - 1:
                    raise
                print(f"[drive] {type(e).__name__}: {e}; retry {i + 1}", flush=True)
                time.sleep(3 * (i + 1))

    def ls(self, folder_id):
        out, token = [], None
        while True:
            r = self._retry(
                lambda: self.svc.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
                    pageSize=1000,
                    pageToken=token,
                )
                .execute()
            )
            out += r.get("files", [])
            token = r.get("nextPageToken")
            if not token:
                return out

    def download(self, file_id, path):
        from googleapiclient.http import MediaIoBaseDownload

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def go():
            with open(path, "wb") as f:
                dl = MediaIoBaseDownload(f, self.svc.files().get_media(fileId=file_id), chunksize=32 << 20)
                done = False
                while not done:
                    _, done = dl.next_chunk()

        self._retry(go)

    def mkdir(self, name, parent):
        for f in self.ls(parent):
            if f["name"] == name and f["mimeType"] == FOLDER_MIME:
                return f["id"]
        r = self._retry(
            lambda: self.svc.files()
            .create(body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent]}, fields="id")
            .execute()
        )
        return r["id"]

    def put(self, path, parent, name, mime="application/octet-stream", known=None):
        """Create or replace file `name` in folder `parent`. `known` caches name -> id."""
        from googleapiclient.http import MediaFileUpload

        known = known if known is not None else {}
        if name not in known:
            for f in self.ls(parent):
                known.setdefault(f["name"], f["id"])
        media = MediaFileUpload(path, mimetype=mime, resumable=os.path.getsize(path) > (4 << 20))
        if name in known:
            self._retry(lambda: self.svc.files().update(fileId=known[name], media_body=media).execute())
        else:
            r = self._retry(
                lambda: self.svc.files()
                .create(body={"name": name, "parents": [parent]}, media_body=media, fields="id")
                .execute()
            )
            known[name] = r["id"]
        return known[name]


class LocalStore:
    """Same interface on a local directory tree, for testing. Ids are paths."""

    def ls(self, folder_id):
        out = []
        for n in sorted(os.listdir(folder_id)):
            p = os.path.join(folder_id, n)
            out.append(
                {
                    "id": p,
                    "name": n,
                    "mimeType": FOLDER_MIME if os.path.isdir(p) else "file",
                    "modifiedTime": str(os.path.getmtime(p)),
                    "size": str(os.path.getsize(p)),
                }
            )
        return out

    def download(self, file_id, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        shutil.copyfile(file_id, path)

    def mkdir(self, name, parent):
        p = os.path.join(parent, name)
        os.makedirs(p, exist_ok=True)
        return p

    def put(self, path, parent, name, mime=None, known=None):
        dst = os.path.join(parent, name)
        shutil.copyfile(path, dst)
        return dst


def mime_for(name):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {
        "txt": "text/plain", "log": "text/plain", "json": "text/plain", "csv": "text/plain",
        "md": "text/plain", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")


def gpu_summary():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip() or "no GPU reported"
    except Exception as e:
        return f"no GPU ({type(e).__name__})"


def parse_header(text):
    """timeout (searched in the whole script, as v1 did), slot and after (first 30 lines)."""
    head = "\n".join(text.splitlines()[:HEADER_LINES])
    m = re.search(r"^#\s*timeout:\s*(\d+)", text, re.M)
    timeout = int(m.group(1)) if m else DEFAULT_TIMEOUT
    m = re.search(r"^#\s*slot:\s*(\S+)", head, re.M | re.I)
    slot = "cpu" if m and m.group(1).lower() == "cpu" else "gpu"
    after = []
    for m in re.finditer(r"^#\s*after:\s*(.*)$", head, re.M | re.I):
        for d in re.split(r"[,\s]+", m.group(1).strip()):
            d = re.sub(r"\.sh$", "", d)
            if d and d not in after:
                after.append(d)
    return {"timeout": timeout, "slot": slot, "after": after}


def lane(name):
    return name.split("_", 1)[0]


def pick_round_robin(names, last_lane):
    """Next lane after `last_lane` in sorted cyclic order; within a lane, the first name."""
    lanes = sorted({lane(n) for n in names})
    nxt = next((l for l in lanes if last_lane is not None and l > last_lane), lanes[0])
    return min(n for n in names if lane(n) == nxt)


def read_fields(text):
    out = {}
    for line in text.splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            out.setdefault(k.strip(), v.strip())
    return out


def say(msg):
    print(f"[worker] {now()} {msg}", flush=True)


class Worker:
    def __init__(self, store, root_id, jobs_id, results_id, inputs_id, work="/content/work", poll=30,
                 tick=5, log_every=60, out_every=300, status_every=60, update_every=120, max_running=2,
                 version=WORKER_VERSION, skip_versions=(), fresh_seconds=FRESH_SECONDS, guard_wait=0,
                 guard_poll=20):
        self.s, self.root, self.jobs, self.results, self.inputs = store, root_id, jobs_id, results_id, inputs_id
        self.work, self.poll, self.tick = os.path.abspath(work), poll, min(tick, poll)
        self.log_every, self.out_every, self.status_every, self.update_every = log_every, out_every, status_every, update_every
        self.max_running, self.version, self.skip_versions = max_running, int(version), set(skip_versions)
        self.fresh_seconds, self.guard_wait, self.guard_poll = fresh_seconds, guard_wait, guard_poll
        self.inputs_dir = os.path.join(self.work, "inputs")
        self.scripts_dir = os.path.join(self.work, "scripts")
        self.root_known, self.started = {}, now()
        self.scripts = {}        # file id -> {"path", "hdr"}
        self.job_files = {}      # job name -> Drive file of its script (latest listing)
        self.results_ids = {}    # job name -> results folder id
        self.finished = {}       # job name -> {"status", "secs", "t", "done_id", "folder", "reason"}
        self.running = {}        # job name -> run record
        self.markers = set()     # CANCEL_ suffixes from the latest listing
        self.last_lane = {"gpu": None, "cpu": None}
        self.draining = None     # higher worker version waiting to take over
        self.relist = True
        self.last_list = self.last_status = self.last_update = 0.0
        self.wid = None
        self.inputs_seen = {}
        self.leftover_pgids = set()  # finished jobs whose process group still had processes

    # ---------- small local files ----------
    def _p(self, name):
        return os.path.join(self.work, name)

    def _load(self, name, default):
        try:
            with open(self._p(name)) as f:
                return json.load(f)
        except Exception:
            return default

    def _save(self, name, obj):
        tmp = self._p(name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, self._p(name))

    def worker_id(self):
        p = self._p("worker_id")
        try:
            with open(p) as f:
                wid = f.read().strip()
            if wid:
                return wid
        except OSError:
            pass
        wid = secrets.token_hex(4)
        with open(p, "w") as f:
            f.write(wid + "\n")
        return wid

    def save_pgids(self):
        self._save("pgids.json", sorted({r["proc"].pid for r in self.running.values()} | self.leftover_pgids))

    def kill_leftovers(self):
        for pgid in self._load("pgids.json", []):
            if pgid in (os.getpgrp(), 0, 1):
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
                say(f"killed leftover job process group {pgid} from a previous run")
            except (ProcessLookupError, PermissionError):
                pass
        self._save("pgids.json", [])
        jobs_root = os.path.join(self.work, "jobs") + os.sep
        me = os.getpid()
        for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
            if not d.isdigit() or int(d) in (me, 1):
                continue
            try:   # a job process: started in its job directory, with OUT=<work>/jobs/<name>/out in its environment
                cwd = os.readlink(f"/proc/{d}/cwd")
                with open(f"/proc/{d}/environ", "rb") as f:
                    env_out = (b"\0OUT=" + os.fsencode(jobs_root)) in (b"\0" + f.read())
            except OSError:
                continue
            if env_out or (cwd + os.sep).startswith(jobs_root):
                try:
                    os.kill(int(d), signal.SIGKILL)
                    say(f"killed leftover process {d} (cwd {cwd}) from a previous run")
                except (ProcessLookupError, PermissionError):
                    pass

    def read_prev_status(self):
        if "status.txt" not in self.root_known:
            return ""
        p = self._p("_prev_status.txt")
        try:
            self.s.download(self.root_known["status.txt"], p)
            with open(p, errors="replace") as f:
                return f.read()
        except Exception as e:
            say(f"could not read the previous status.txt: {e}")
            return ""

    def other_worker_really_alive(self, prev):
        """A fresh foreign status may be the last word of a session that was just stopped: watch whether it
        still advances (alive) or goes stale (gone) for up to guard_wait seconds before refusing."""
        if not self.other_worker_alive(prev):
            return False
        if self.guard_wait <= 0:
            return True
        print("Un autre ouvrier GPU a donné signe de vie il y a moins de 3 minutes. Vérification en cours "
              "(3 minutes au plus) : s'il est bien arrêté, celui-ci démarrera tout seul.", flush=True)
        t0, deadline = read_fields(prev).get("time"), time.time() + self.guard_wait
        while time.time() < deadline:
            time.sleep(self.guard_poll)
            cur = self.read_prev_status()
            if not cur:
                continue
            if read_fields(cur).get("time") != t0:
                return self.other_worker_alive(cur)   # it moved: alive (unless it says stopped or is ours)
            if not self.other_worker_alive(cur):
                return False                          # went stale: the other session is gone
        return True

    # ---------- startup ----------
    def startup(self):
        for d in (self.work, self.inputs_dir, self.scripts_dir):
            os.makedirs(d, exist_ok=True)
        self.wid = self.worker_id()
        self.inputs_seen = self._load("inputs_seen.json", {})
        root_files = self.s.ls(self.root)
        self.note_root(root_files)
        prev = self.read_prev_status()
        if self.other_worker_really_alive(prev):
            print(
                "\n*** ARRÊT : un autre ouvrier GPU semble encore actif (une autre session Colab, ou un autre onglet).\n"
                "*** Cet ouvrier-ci ne démarre pas, pour ne pas faire le même travail deux fois.\n"
                "*** À faire : menu Exécution → Gérer les sessions, puis arrêtez l'autre session.\n"
                "*** Refaites ensuite ici Exécution → Tout exécuter.\n", flush=True)
            return False
        self.kill_leftovers()
        self.last_update = time.time()
        self.write_status("starting: reading the results folder")
        self.scan_results(self.seed_from_status(prev))
        print(f"Ouvrier GPU version {self.version:03d} démarré (identifiant {self.wid}). "
              f"Laissez cet onglet ouvert ; il n'y a rien d'autre à faire.", flush=True)
        say(f"ready, gpu: {gpu_summary()}; {len(self.finished)} finished jobs known")
        self.write_status()
        return True

    def other_worker_alive(self, text):
        if not text:
            return False
        f = read_fields(text)
        t = parse_now(f.get("time", ""))
        if t is None or time.time() - t > self.fresh_seconds:
            return False
        if f.get("worker id") == self.wid:
            return False
        if f.get("state", "").startswith("stopped"):
            return False
        say(f"status.txt is {time.time() - t:.0f}s old and belongs to worker id {f.get('worker id', '(v1, none)')}")
        return True

    @staticmethod
    def seed_from_status(text):
        """Statuses and seconds of finished jobs as listed by a previous v2 status.txt."""
        seed = {}
        for line in text.splitlines():
            m = re.match(r"^done (\S+) (\d+)s (\S+)$", line)
            if m:
                seed[m.group(3)] = {"status": m.group(1), "secs": int(m.group(2)), "reason": ""}
                continue
            m = re.match(r"^(skipped|cancelled)(?: (\d+)s)? (\S+)(?: \((.*)\))?$", line)
            if m:
                seed[m.group(3)] = {"status": m.group(1), "secs": int(m.group(2) or 0), "reason": m.group(4) or ""}
        return seed

    def scan_results(self, seed):
        cache = self._load("finished_cache.json", {})
        for f in self.s.ls(self.results):
            if f["mimeType"] != FOLDER_MIME:
                continue
            name = f["name"]
            self.results_ids.setdefault(name, f["id"])
            if name in self.finished:
                continue
            c = cache.get(name)
            if c and c.get("folder") == f["id"]:
                self.finished[name] = c
                continue
            done = [x for x in self.s.ls(f["id"]) if x["name"] == "DONE.txt"]
            if done:
                info = {"status": None, "secs": 0, "reason": "", "t": drive_ts(done[0]["modifiedTime"]),
                        "done_id": done[0]["id"], "folder": f["id"]}
                if name in seed:
                    info.update(seed[name])
                self.finished[name] = info
            self.maybe_status("starting: reading the results folder")
        for name in self.recent_finished():
            self.done_status(name)
        self._save("finished_cache.json", self.finished)

    def done_status(self, name):
        info = self.finished[name]
        if info.get("status") is None:
            p = self._p("_done.txt")
            try:
                self.s.download(info["done_id"], p)
                with open(p, errors="replace") as fh:
                    f = read_fields(fh.read())
                info["status"] = (f.get("status") or "unknown").split()[0]
                try:
                    info["secs"] = int(float(f.get("seconds", "0")))
                except ValueError:
                    info["secs"] = 0
                info["reason"] = f.get("reason", "")
            except Exception as e:   # None = not known yet; dependents keep waiting, they are not skipped
                say(f"could not read DONE.txt of {name}: {e}")
                return None
        return info["status"]

    def recent_finished(self):
        return sorted(self.finished, key=lambda n: -self.finished[n].get("t", 0))[:FINISHED_SHOWN]

    # ---------- Drive root: status and updates ----------
    def note_root(self, files):
        self.root_known = {}
        for f in files:
            self.root_known.setdefault(f["name"], f["id"])
        self.worker_files = {int(m.group(1)) for f in files if (m := WORKER_FILE_RE.match(f["name"]))}

    def check_update(self):
        self.last_update = time.time()
        self.note_root(self.s.ls(self.root))
        higher = [v for v in self.worker_files if v > self.version and v not in self.skip_versions]
        if higher and self.draining is None:
            self.draining = max(higher)
            print(f"Nouvelle version worker_v{self.draining:03d}.py trouvée : elle prendra le relais dès que les "
                  f"travaux en cours seront finis. Rien à faire.", flush=True)
            say(f"draining for worker_v{self.draining:03d}: no new job starts")
            self.write_status()

    def state_line(self):
        t = time.time()
        if self.running:
            s = "running " + "; ".join(f"{n} ({r['slot']}) {t - r['t0']:.0f}s" for n, r in sorted(self.running.items()))
        else:
            waiting = sum(1 for n in self.pending() if self.dep_state(n)[0] != "eligible")
            s = f"idle, {waiting} job(s) waiting for dependencies" if waiting else "idle, waiting for jobs"
        if self.draining is not None:
            s = f"updating to worker_v{self.draining:03d} once running jobs finish (no new job starts); " + s
        return s

    def queue_lines(self):
        t, lines = time.time(), []
        for n, r in sorted(self.running.items()):
            lines.append(f"running {t - r['t0']:.0f}s {r['slot']} {n}")
        for n in self.pending():
            hdr = self.header(n)
            if hdr is None:
                lines.append(f"pending ? {n} (script not read yet)")
                continue
            note = ""
            if hdr["after"]:
                st = self.dep_state(n)
                note = f" (after: {', '.join(hdr['after'])}"
                if st[0] == "unknown":
                    note += f"; not submitted yet: {', '.join(st[1])}"
                note += ")"
            lines.append(f"pending {hdr['slot']} {n}{note}")
        for n in self.recent_finished():
            i = self.finished[n]
            st = i.get("status") or "unknown"
            if st in ("skipped", "cancelled"):
                secs = f" {i.get('secs', 0)}s" if i.get("secs") else ""
                reason = f" ({i['reason']})" if i.get("reason") else ""
                lines.append(f"{st}{secs} {n}{reason}")
            else:
                lines.append(f"done {st} {i.get('secs', 0)}s {n}")
        return lines

    def write_status(self, state=None):
        self.last_status = time.time()
        p = self._p("status.txt")
        free = shutil.disk_usage(self.work).free / 1e9
        q = self.queue_lines()
        with open(p, "w") as f:
            f.write(f"time: {now()}\nworker started: {self.started}\nstate: {state or self.state_line()}\n"
                    f"gpu: {gpu_summary()}\ndisk free: {free:.0f} GB\n"
                    f"worker version: {self.version:03d}\nworker id: {self.wid}\n"
                    f"slots: {self.max_running} jobs at once, at most 1 gpu job; jobs list checked every {self.poll:g}s\n"
                    f"queue: {len(self.running)} running, {len(self.pending())} pending, "
                    f"{len(self.finished)} finished (newest {FINISHED_SHOWN} shown)\n")
            f.write("".join(line + "\n" for line in q))
        self.s.put(p, self.root, "status.txt", "text/plain", self.root_known)

    def maybe_status(self, state=None):
        if time.time() - self.last_status >= self.status_every:
            try:
                self.write_status(state)
            except Exception as e:
                say(f"status push failed: {e}")

    # ---------- the queue ----------
    def pending(self):
        return sorted(n for n in self.job_files if n not in self.finished and n not in self.running)

    def header(self, name):
        f = self.job_files.get(name)
        sc = self.scripts.get(f["id"]) if f else None
        return sc["hdr"] if sc else None

    def ensure_script(self, f):
        if f["id"] in self.scripts:
            return
        path = os.path.join(self.scripts_dir, re.sub(r"[^\w.-]", "_", f["id"]) + ".sh")
        if not os.path.exists(path):
            self.s.download(f["id"], path + ".part")
            os.replace(path + ".part", path)
        with open(path, errors="replace") as fh:
            self.scripts[f["id"]] = {"path": path, "hdr": parse_header(fh.read())}

    def dep_state(self, name):
        """('eligible'|'waiting'|'unknown', [names]) or ('skip', dep, status)."""
        hdr = self.header(name)
        if hdr is None:
            return ("waiting", [])
        waiting, unknown = [], []
        for d in hdr["after"]:
            if d in self.finished:
                st = self.done_status(d)
                if st is None:
                    waiting.append(d)
                elif st != "ok":
                    return ("skip", d, st)
            elif d in self.running or d in self.job_files:
                waiting.append(d)
            else:
                unknown.append(d)
        if unknown:
            return ("unknown", unknown)
        if waiting:
            return ("waiting", waiting)
        return ("eligible", [])

    def is_cancelled(self, name):
        return any(m == name or m.startswith(name + ".") for m in self.markers)

    def list_pass(self):
        """List jobs/ once; returns 'stop' if jobs/STOP exists."""
        self.last_list, self.relist = time.time(), False
        files = self.s.ls(self.jobs)
        if any(f["name"] == "STOP" for f in files):
            return "stop"
        self.markers = {f["name"][len("CANCEL_"):] for f in files if f["name"].startswith("CANCEL_")}
        jobs = {}
        for f in files:
            if f["mimeType"] == FOLDER_MIME or not f["name"].endswith(".sh") or f["name"].startswith("CANCEL_"):
                continue
            n = f["name"][:-3]
            if n not in jobs or drive_ts(f["modifiedTime"]) > drive_ts(jobs[n]["modifiedTime"]):
                jobs[n] = f   # same name uploaded twice: the newest copy wins
        self.job_files = jobs
        for n in self.pending():
            try:
                self.ensure_script(jobs[n])
            except Exception as e:
                say(f"could not download script of {n}: {e}")
        for n in list(self.running):
            if self.is_cancelled(n) and self.running[n]["status"] is None:
                say(f"cancel marker found: killing {n}")
                self.kill(self.running[n])
                self.running[n]["status"] = "cancelled"
                self.running[n]["reason"] = "cancelled by marker while running"
        for n in self.pending():
            if self.is_cancelled(n):
                self.write_terminal(n, "cancelled", "cancelled by marker before it started")
        self.apply_skips()
        return None

    def apply_skips(self):
        changed = True
        while changed:
            changed = False
            for n in self.pending():
                st = self.dep_state(n)
                if st[0] == "skip":
                    self.write_terminal(n, "skipped", f"dependency {st[1]} ended with status {st[2]}")
                    changed = True

    def write_terminal(self, name, status, reason):
        """DONE.txt for a job that never ran (cancelled while pending, or skipped)."""
        res_id = self.results_folder(name)
        p = self._p("_terminal_DONE.txt")
        with open(p, "w") as f:
            f.write(f"status: {status}\nreason: {reason}\nexit code: none\nseconds: 0\nfinished: {now()}\n")
        done_id = self.s.put(p, res_id, "DONE.txt", "text/plain", {})
        self.finished[name] = {"status": status, "secs": 0, "reason": reason, "t": time.time(),
                               "done_id": done_id, "folder": res_id}
        self._save("finished_cache.json", self.finished)
        say(f"{name}: {status} ({reason})")
        self.write_status()

    def results_folder(self, name):
        if name not in self.results_ids:
            self.results_ids[name] = self.s.mkdir(name, self.results)
        return self.results_ids[name]

    def schedule(self):
        excluded = set()
        while len(self.running) < self.max_running:
            pend = [n for n in self.pending() if n not in excluded and self.header(n) is not None]
            elig = {"gpu": [], "cpu": []}
            gpu_waiting = False
            for n in pend:
                st, slot = self.dep_state(n)[0], self.header(n)["slot"]
                if st == "eligible":
                    elig[slot].append(n)
                if slot == "gpu" and st in ("eligible", "waiting"):
                    gpu_waiting = True
            gpu_running = any(r["slot"] == "gpu" for r in self.running.values())
            cpu_running = sum(1 for r in self.running.values() if r["slot"] == "cpu")
            if not gpu_running and elig["gpu"]:
                slot = "gpu"
            elif elig["cpu"] and cpu_running < self.max_running - (1 if gpu_running or gpu_waiting else 0):
                slot = "cpu"
            else:
                return
            n = pick_round_robin(elig[slot], self.last_lane[slot])
            try:
                self.start_job(n)
                self.last_lane[slot] = lane(n)
            except Exception as e:
                say(f"could not start {n}: {type(e).__name__}: {e}")
                excluded.add(n)

    def can_start_more(self):
        if self.draining is not None or len(self.running) >= self.max_running:
            return False
        return any(self.dep_state(n)[0] == "eligible" for n in self.pending() if self.header(n))

    # ---------- running jobs ----------
    def sync_inputs(self):
        for f in self.s.ls(self.inputs):
            if f["mimeType"] == FOLDER_MIME:
                continue
            if self.inputs_seen.get(f["name"]) != f["modifiedTime"]:
                dst = os.path.join(self.inputs_dir, f["name"])
                self.s.download(f["id"], dst + ".part")
                os.replace(dst + ".part", dst)   # a running job keeps reading the old copy
                self.inputs_seen[f["name"]] = f["modifiedTime"]
                self._save("inputs_seen.json", self.inputs_seen)

    def start_job(self, name):
        sc = self.scripts[self.job_files[name]["id"]]
        hdr = sc["hdr"]
        res_id = self.results_folder(name)
        jdir = os.path.join(self.work, "jobs", name)
        out_dir = os.path.join(jdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        script = os.path.join(jdir, "job.sh")
        shutil.copyfile(sc["path"], script)
        self.sync_inputs()
        log_path = os.path.join(jdir, "log.txt")
        env = dict(os.environ, OUT=out_dir, INPUTS=self.inputs_dir, JOB=name, PYTHONUNBUFFERED="1")
        if hdr["slot"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        lf = open(log_path, "wb")
        proc = subprocess.Popen(["bash", script], cwd=jdir, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.running[name] = {"name": name, "slot": hdr["slot"], "timeout": hdr["timeout"], "proc": proc,
                              "lf": lf, "t0": time.time(), "end": None, "code": None, "status": None,
                              "reason": "", "res_id": res_id, "known": {}, "sent": {}, "last_log": 0.0,
                              "last_out": time.time(), "jdir": jdir, "out_dir": out_dir, "log_path": log_path}
        self.save_pgids()
        say(f"start {name} ({hdr['slot']}, timeout {hdr['timeout']}s)")
        self.write_status()

    @staticmethod
    def kill(r):
        try:
            os.killpg(r["proc"].pid, signal.SIGKILL)
        except Exception:
            try:
                r["proc"].kill()
            except Exception:
                pass
        r["proc"].wait()

    def push_log(self, r):
        tail = os.path.join(r["jdir"], "_log_tail.txt")
        if not r["lf"].closed:
            r["lf"].flush()
        size = os.path.getsize(r["log_path"])
        with open(r["log_path"], "rb") as f:
            if size > LOG_TAIL_BYTES:
                f.seek(size - LOG_TAIL_BYTES)
            data = f.read()
        with open(tail, "wb") as f:
            if size > LOG_TAIL_BYTES:
                f.write(f"[... first {size - LOG_TAIL_BYTES} bytes omitted ...]\n".encode())
            f.write(data)
        self.s.put(tail, r["res_id"], "log.txt", "text/plain", r["known"])

    def push_outputs(self, r, final=False):
        for dp, _, fns in os.walk(r["out_dir"]):
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                key = [st.st_mtime, st.st_size]
                if r["sent"].get(p) == key or (not final and time.time() - st.st_mtime < 5):
                    continue
                if st.st_size > MAX_UPLOAD_BYTES:
                    say(f"skip {p}: {st.st_size / 1e6:.0f} MB over upload cap")
                    r["sent"][p] = key
                    continue
                name = os.path.relpath(p, r["out_dir"]).replace(os.sep, "__")
                self.s.put(p, r["res_id"], name, mime_for(name), r["known"])
                r["sent"][p] = key
                self.maybe_status()

    def tend_running(self):
        for n in list(self.running):
            r = self.running[n]
            try:
                if r["end"] is None:
                    t = time.time()
                    if r["proc"].poll() is None:
                        if t - r["t0"] > r["timeout"]:
                            say(f"{n}: timeout after {r['timeout']}s, killing it")
                            self.kill(r)
                            r["status"] = "timeout"
                        else:
                            if t - r["last_log"] > self.log_every:
                                try:
                                    self.push_log(r)
                                except Exception as e:
                                    say(f"log push failed for {n}: {e}")
                                r["last_log"] = time.time()
                            if t - r["last_out"] > self.out_every:
                                try:
                                    self.push_outputs(r)
                                except Exception as e:
                                    say(f"output push failed for {n}: {e}")
                                r["last_out"] = time.time()
                            continue
                    r["end"], r["code"] = time.time(), r["proc"].returncode
                    if r["status"] is None:
                        r["status"] = "ok" if r["code"] == 0 else "failed"
                self.finalize(n)
            except Exception as e:
                say(f"error while finishing {n} (will retry): {type(e).__name__}: {e}")

    def finalize(self, name):
        r = self.running[name]
        if not r["lf"].closed:
            r["lf"].close()
        self.push_log(r)
        self.push_outputs(r, final=True)
        secs = r["end"] - r["t0"]
        done = os.path.join(r["jdir"], "DONE.txt")
        with open(done, "w") as f:
            f.write(f"status: {r['status']}\nexit code: {r['code']}\nseconds: {secs:.0f}\nfinished: {now()}\n"
                    f"slot: {r['slot']}\ngpu: {gpu_summary()}\n" + (f"reason: {r['reason']}\n" if r["reason"] else ""))
        done_id = self.s.put(done, r["res_id"], "DONE.txt", "text/plain", r["known"])
        self.finished[name] = {"status": r["status"], "secs": int(round(secs)), "reason": r["reason"],
                               "t": time.time(), "done_id": done_id, "folder": r["res_id"]}
        del self.running[name]
        try:   # background processes left by the job are killed at the next worker start, as leftovers
            os.killpg(r["proc"].pid, 0)
            self.leftover_pgids.add(r["proc"].pid)
        except (ProcessLookupError, PermissionError):
            pass
        self.save_pgids()
        self._save("finished_cache.json", self.finished)
        say(f"end {name}: {r['status']}, exit {r['code']}, {secs:.0f}s")
        self.relist = True
        self.write_status()

    def stop_all(self, reason, push=True):
        for n, r in list(self.running.items()):
            if r["end"] is not None and push:   # already ended, only its upload was pending: finish it
                try:
                    self.finalize(n)
                    continue
                except Exception as e:
                    say(f"could not finish {n}: {e}")
            if r["end"] is None:
                self.kill(r)
            if not r["lf"].closed:
                r["lf"].write(f"\n[worker] {now()} job killed: worker {reason}; it will run again next time\n".encode())
                r["lf"].close()
            if push:
                try:
                    self.push_log(r)
                except Exception as e:
                    say(f"log push failed for {n}: {e}")
        self.running = {}
        self.save_pgids()
        try:
            self.write_status(f"stopped: {reason}")
        except Exception as e:
            say(f"status push failed: {e}")

    # ---------- main loop ----------
    def run(self, max_idle=None):
        idle = 0
        try:
            while True:
                try:
                    self.tend_running()
                    if self.relist or time.time() - self.last_list >= self.poll:
                        if self.list_pass() == "stop":
                            say("STOP file found, exiting")
                            print("Arrêt demandé (fichier STOP). L'ouvrier s'est arrêté.", flush=True)
                            self.stop_all("stopped by STOP file")
                            return "stopped"
                        if self.draining is None:
                            self.schedule()
                        idle = idle + 1 if not self.running and not self.can_start_more() else 0
                    if time.time() - self.last_update >= self.update_every:
                        self.check_update()
                    if self.draining is not None and not self.running:
                        self.write_status(f"updating: handing over to worker_v{self.draining:03d}")
                        say(f"handing over to worker_v{self.draining:03d}")
                        return "update"
                    self.maybe_status()
                    if max_idle and idle >= max_idle:
                        self.write_status("stopped: local test run finished (idle)")
                        return "idle"
                    time.sleep(self.tick)
                except Exception as e:
                    say(f"error: {type(e).__name__}: {e}")
                    time.sleep(self.poll)
        except KeyboardInterrupt:
            say("interrupted: killing running jobs (they will run again next time)")
            self.stop_all("interrupted by the user", push=False)
            raise


def main(root_id, jobs_id, results_id, inputs_id, store=None, work="/content/work", poll=30, max_idle=None, **opts):
    """Entry point used by the notebook bootstrap. Returns 'update', 'stopped', 'refused' or 'idle'."""
    g = globals()
    version = int(g.get("LOADED_VERSION") or WORKER_VERSION)
    skip = set(g.get("SKIP_VERSIONS") or ())
    os.makedirs(work, exist_ok=True)
    if store is None:
        opts.setdefault("guard_wait", 200)   # real Colab run: watch a fresh foreign heartbeat before refusing
        opts.setdefault("max_running", 3 if (os.cpu_count() or 2) >= 4 else 2)   # high-RAM runtimes have 8 cores
    w = Worker(store or DriveStore(), root_id, jobs_id, results_id, inputs_id, work=work, poll=poll,
               version=version, skip_versions=skip, **opts)
    if not w.startup():
        return "refused"
    return w.run(max_idle=max_idle)


if __name__ == "__main__":  # local test: python worker_v002.py <root_dir>
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        print(__doc__)
        print("local test: python worker_v002.py <root_dir>  (runs jobs from <root_dir>/jobs until idle; on Colab the "
              "notebook written by make_notebook_v2.py starts it on Google Drive)")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    root = sys.argv[1]
    st = LocalStore()
    ids = [st.mkdir(n, root) for n in ("jobs", "results", "inputs")]
    print(main(root, *ids, store=st, work=os.path.join(root, "_work"), poll=1, tick=0.5, max_idle=2))
