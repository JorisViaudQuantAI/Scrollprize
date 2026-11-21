"""GPU job worker: runs bash jobs dropped into a Google Drive folder and uploads their output.

Runs inside a Colab notebook (DriveStore) or locally for testing (LocalStore).

Protocol
  jobs/<name>.sh        a bash script. Optional header line "# timeout: <seconds>" (default 10800).
  inputs/*              files synced to $INPUTS before each job.
  results/<name>/       log.txt (refreshed while running), every file the job writes under $OUT,
                        and DONE.txt (exit code, seconds) when it finishes.
  status.txt            heartbeat, refreshed about once a minute.
  jobs/STOP             any file with this name stops the worker.
A job is skipped when results/<name>/DONE.txt already exists, so restarting the worker is safe.
"""
import datetime
import io
import os
import re
import shutil
import subprocess
import sys
import time

MAX_UPLOAD_BYTES = 800 * 1024 * 1024
LOG_TAIL_BYTES = 400_000
FOLDER_MIME = "application/vnd.google-apps.folder"


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class DriveStore:
    """Google Drive through the REST API, authenticated as the Colab user."""

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
                    pageSize=200,
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


class Worker:
    def __init__(self, store, root_id, jobs_id, results_id, inputs_id, work="/content/work", poll=30):
        self.s, self.root, self.jobs, self.results, self.inputs = store, root_id, jobs_id, results_id, inputs_id
        self.work, self.poll = work, poll
        self.inputs_dir = os.path.join(work, "inputs")
        self.root_known, self.inputs_seen, self.started = {}, {}, now()
        os.makedirs(self.inputs_dir, exist_ok=True)

    def heartbeat(self, state):
        p = os.path.join(self.work, "status.txt")
        free = shutil.disk_usage(self.work).free / 1e9
        with open(p, "w") as f:
            f.write(f"time: {now()}\nworker started: {self.started}\nstate: {state}\n"
                    f"gpu: {gpu_summary()}\ndisk free: {free:.0f} GB\n")
        self.s.put(p, self.root, "status.txt", "text/plain", self.root_known)

    def sync_inputs(self):
        for f in self.s.ls(self.inputs):
            if f["mimeType"] == FOLDER_MIME:
                continue
            if self.inputs_seen.get(f["name"]) != f["modifiedTime"]:
                self.s.download(f["id"], os.path.join(self.inputs_dir, f["name"]))
                self.inputs_seen[f["name"]] = f["modifiedTime"]

    def push_log(self, log_path, res_id, known):
        tail = os.path.join(self.work, "_log_tail.txt")
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            if size > LOG_TAIL_BYTES:
                f.seek(size - LOG_TAIL_BYTES)
            data = f.read()
        with open(tail, "wb") as f:
            if size > LOG_TAIL_BYTES:
                f.write(f"[... first {size - LOG_TAIL_BYTES} bytes omitted ...]\n".encode())
            f.write(data)
        self.s.put(tail, res_id, "log.txt", "text/plain", known)

    def push_outputs(self, out_dir, res_id, known, sent):
        for dp, _, fns in os.walk(out_dir):
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                key = (st.st_mtime, st.st_size)
                if sent.get(p) == key or time.time() - st.st_mtime < 5:
                    continue
                if st.st_size > MAX_UPLOAD_BYTES:
                    print(f"[worker] skip {p}: {st.st_size / 1e6:.0f} MB over upload cap", flush=True)
                    sent[p] = key
                    continue
                name = os.path.relpath(p, out_dir).replace(os.sep, "__")
                self.s.put(p, res_id, name, mime_for(name), known)
                sent[p] = key

    def run_job(self, job):
        name = re.sub(r"\.sh$", "", job["name"])
        res_id = self.s.mkdir(name, self.results)
        if any(f["name"] == "DONE.txt" for f in self.s.ls(res_id)):
            return False
        jdir = os.path.join(self.work, "jobs", name)
        out_dir = os.path.join(jdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        script = os.path.join(jdir, "job.sh")
        self.s.download(job["id"], script)
        self.sync_inputs()
        with open(script) as f:
            m = re.search(r"^#\s*timeout:\s*(\d+)", f.read(), re.M)
        timeout = int(m.group(1)) if m else 10800
        log_path = os.path.join(jdir, "log.txt")
        env = dict(os.environ, OUT=out_dir, INPUTS=self.inputs_dir, JOB=name, PYTHONUNBUFFERED="1")
        print(f"[worker] {now()} start {name} (timeout {timeout}s)", flush=True)
        t0, known, sent, last_log, last_out = time.time(), {}, {}, 0, time.time()
        status = "ok"
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(["bash", script], cwd=jdir, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                time.sleep(5)
                if time.time() - t0 > timeout:
                    status = "timeout"
                    try:
                        os.killpg(proc.pid, 9)
                    except Exception:
                        proc.kill()
                    proc.wait()
                    break
                if time.time() - last_log > 60:
                    try:
                        self.push_log(log_path, res_id, known)
                        self.heartbeat(f"running {name} for {time.time() - t0:.0f}s")
                    except Exception as e:
                        print(f"[worker] log push failed: {e}", flush=True)
                    last_log = time.time()
                if time.time() - last_out > 300:
                    try:
                        self.push_outputs(out_dir, res_id, known, sent)
                    except Exception as e:
                        print(f"[worker] output push failed: {e}", flush=True)
                    last_out = time.time()
        code, secs = proc.returncode, time.time() - t0
        if status == "ok" and code != 0:
            status = "failed"
        self.push_log(log_path, res_id, known)
        time.sleep(6)
        self.push_outputs(out_dir, res_id, known, sent)
        done = os.path.join(jdir, "DONE.txt")
        with open(done, "w") as f:
            f.write(f"status: {status}\nexit code: {code}\nseconds: {secs:.0f}\nfinished: {now()}\n"
                    f"gpu: {gpu_summary()}\n")
        self.s.put(done, res_id, "DONE.txt", "text/plain", known)
        print(f"[worker] {now()} end {name}: {status}, exit {code}, {secs:.0f}s", flush=True)
        return True

    def loop(self, max_idle_loops=None):
        idle = 0
        print(f"[worker] {now()} ready, gpu: {gpu_summary()}", flush=True)
        while True:
            try:
                files = self.s.ls(self.jobs)
                if any(f["name"] == "STOP" for f in files):
                    self.heartbeat("stopped by STOP file")
                    print("[worker] STOP file found, exiting", flush=True)
                    return
                ran = False
                for job in sorted((f for f in files if f["name"].endswith(".sh")), key=lambda f: f["name"]):
                    if self.run_job(job):
                        ran = True
                        break  # re-list so newer jobs and STOP are seen
                if not ran:
                    idle += 1
                    self.heartbeat("idle, waiting for jobs")
                    if max_idle_loops and idle >= max_idle_loops:
                        return
                    time.sleep(self.poll)
                else:
                    idle = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[worker] error: {type(e).__name__}: {e}", flush=True)
                time.sleep(self.poll)


if __name__ == "__main__":  # local test: python worker.py <root_dir>
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        print(__doc__)
        print("local test: python worker.py <root_dir>  (runs jobs from <root_dir>/jobs until idle; on Colab the "
              "notebook written by make_notebook_v2.py starts it on Google Drive)")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    root = sys.argv[1]
    st = LocalStore()
    ids = [st.mkdir(n, root) for n in ("jobs", "results", "inputs")]
    Worker(st, root, *ids, work=os.path.join(root, "_work"), poll=1).loop(max_idle_loops=2)
