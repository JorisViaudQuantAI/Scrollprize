"""Private single-host reference control plane; no worker execution or submission.

SQLite transactions fence receipts and reserve integer resource vectors.
Byte integrity is checked, not the scientific truth of a producer's attestation.
Only synthetic/public fixtures may run in public CI. See RFC-001 for rollout.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
import time

LANES = ("exploit", "repair", "explore")
RESOURCES = ("cpu_ms", "gpu_ms", "human_ms", "io_bytes")
PREDECESSOR = {"REGISTER":None, "CONTROL":None, "GEOMETRY":"REGISTER",
               "COARSE":"GEOMETRY", "RANK":"COARSE", "DEEP":"RANK", "REVIEW":"DEEP"}
INFERENCE = {"COARSE","DEEP"}
FAILURES = {"TIMEOUT","OOM","NETWORK","MISSING_SUPPORT","BAD_GEOMETRY",
            "BAD_CONTROL","BAD_INPUT","REVIEW_INCOMPLETE","WORKER_STOPPED"}
RETRYABLE = {"TIMEOUT","OOM","NETWORK","WORKER_STOPPED"}
REVIEW = {"REVIEWED_NO_QUALIFYING_READING","INCONCLUSIVE","PRIVATE_REVIEW_REQUIRED"}
SHA = re.compile(r"[a-f0-9]{64}\Z")
SCROLL = re.compile(r"PHerc(?:[0-9]{3,4}[AB]?|Paris4)\Z")

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def require_sha(x):
    if not isinstance(x,str) or not SHA.fullmatch(x):raise ValueError("Expected SHA256")
    return x

def vector(x):
    if not isinstance(x,dict) or set(x)!=set(RESOURCES) or any(type(v)!=int or v<0 for v in x.values()):
        raise ValueError("Resource vector must use all four nonnegative integer units")
    return {k:x[k] for k in RESOURCES}

def zeros():
    return dict.fromkeys(RESOURCES,0)

def add(a,b):
    return {k:a[k]+b[k] for k in RESOURCES}

def within(a,b):
    return all(a[k]<=b[k] for k in RESOURCES)

def private_root(path):
    p=Path(path).resolve()
    if any((a/".git").exists() for a in (p,*p.parents)):
        raise ValueError("Private state cannot be inside a Git worktree")
    p.mkdir(parents=True,exist_ok=True,mode=0o700)
    if p.stat().st_mode & 0o077:raise ValueError("Private root must have mode 0700")
    return p

def file_sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

class Store:
    """Minimal local CAS. Do not expose this directory through public CI."""
    def __init__(self,root):
        self.root=private_root(root)
    def verify(self,h):
        p=self.root/require_sha(h)
        if p.is_symlink() or not p.is_file() or file_sha(p)!=h:
            raise ValueError("Missing or corrupt CAS artifact")
        return p
    def put(self,body):
        if not isinstance(body,bytes):raise TypeError("Expected bytes")
        h=hashlib.sha256(body).hexdigest();p=self.root/h
        if p.exists() or p.is_symlink():
            self.verify(h);return h
        fd,tmp=tempfile.mkstemp(dir=self.root)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(body);f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
            self.verify(h)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return h

def hashes(obj):
    if not isinstance(obj,dict) or any(not isinstance(k,str) or not k for k in obj):
        raise ValueError("Expected named artifact hashes")
    for v in obj.values():require_sha(v)
    return obj

def validate_spec(s):
    keys={"scroll","unit","frame","stage","recipe","inputs","deps","calibration"}
    if not isinstance(s,dict) or set(s)!=keys:raise ValueError("Job spec schema mismatch")
    if not isinstance(s["scroll"],str) or not SCROLL.fullmatch(s["scroll"]):raise ValueError("Invalid scroll code")
    if not isinstance(s["unit"],str) or not s["unit"] or len(s["unit"])>128:raise ValueError("Invalid private unit")
    require_sha(s["frame"]);require_sha(s["recipe"])
    if s["stage"] not in PREDECESSOR:raise ValueError("Invalid stage")
    hashes(s["inputs"])
    if not isinstance(s["deps"],list) or len(s["deps"])!=len(set(s["deps"])):raise ValueError("Invalid dependencies")
    for h in s["deps"]:require_sha(h)
    if s["stage"] in INFERENCE|{"CONTROL"}:require_sha(s["calibration"])
    elif s["calibration"] is not None:raise ValueError("Unexpected calibration key")
    return {**s,"deps":sorted(s["deps"])}

class Coordinator:
    """One private epoch, immutable caps, transactional claims; no network API."""
    def __init__(self,root,epoch,caps,eligible,lease_seconds=300,max_attempts=2):
        self.root=private_root(root)
        self.store=Store(self.root/"cas");self.db=self.root/"queue.sqlite3"
        if not isinstance(epoch,str) or not epoch:raise ValueError("Epoch required")
        if set(caps)!=set(LANES):raise ValueError("All lanes required")
        if not eligible or any(not isinstance(x,str) or not SCROLL.fullmatch(x) for x in eligible):raise ValueError("Invalid dated eligibility set")
        if type(lease_seconds)!=int or lease_seconds<=0 or type(max_attempts)!=int or max_attempts<=0:
            raise ValueError("Invalid lease/retry policy")
        self.policy={"epoch":epoch,"caps":{k:vector(v) for k,v in caps.items()},
                     "eligible":sorted(set(eligible)),"lease_seconds":lease_seconds,"max_attempts":max_attempts}
        con=self.connect()
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
            CREATE TABLE IF NOT EXISTS config(k TEXT PRIMARY KEY,v TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY,spec TEXT NOT NULL,lane TEXT NOT NULL,priority INTEGER NOT NULL,
              grant TEXT NOT NULL,state TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,
              reason TEXT,created INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(
              token TEXT PRIMARY KEY,job TEXT NOT NULL REFERENCES jobs(id),worker TEXT NOT NULL,
              started INTEGER NOT NULL,expires INTEGER NOT NULL,status TEXT NOT NULL,
              grant TEXT NOT NULL,actual TEXT,receipt TEXT,receipt_hash TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_lease ON attempts(job) WHERE status='LEASED';
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,at INTEGER NOT NULL,
              job TEXT NOT NULL,kind TEXT NOT NULL,details TEXT NOT NULL);
            """)
            con.execute("BEGIN IMMEDIATE")
            old=con.execute("SELECT v FROM config WHERE k='policy'").fetchone()
            if old and json.loads(old[0])!=self.policy:raise ValueError("Immutable epoch configuration mismatch")
            if not old:
                con.execute("INSERT INTO config VALUES('policy',?)",(canonical(self.policy),))
                con.execute("INSERT INTO config VALUES('halted','false')")
            con.commit()
        except BaseException:
            con.rollback();raise
        finally:con.close()
    def connect(self):
        c=sqlite3.connect(self.db,timeout=15,isolation_level=None)
        c.row_factory=sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON");c.execute("PRAGMA synchronous=FULL")
        return c
    @contextmanager
    def tx(self):
        c=self.connect()
        try:
            c.execute("BEGIN IMMEDIATE");yield c;c.commit()
        except BaseException:
            c.rollback();raise
        finally:c.close()
    @staticmethod
    def clock(now):
        return int(time.time()) if now is None else int(now)
    @staticmethod
    def event(c,now,j,kind,details=None):
        c.execute("INSERT INTO events(at,job,kind,details) VALUES(?,?,?,?)",(now,j,kind,canonical(details or {})))
    def enqueue(self,spec,lane,priority,grant,now=None):
        s=validate_spec(spec);j=digest(s);g=vector(grant);now=self.clock(now)
        if lane not in LANES or type(priority)!=int or priority<0 or not any(g.values()):
            raise ValueError("Invalid scheduling policy")
        if s["stage"]!="CONTROL" and s["scroll"] not in self.policy["eligible"]:
            raise ValueError("Scroll outside frozen eligibility")
        with self.tx() as c:
            found=c.execute("SELECT * FROM jobs WHERE id=?",(j,)).fetchone()
            if found:
                if (found["spec"],found["lane"],found["priority"],found["grant"])!=(canonical(s),lane,priority,canonical(g)):
                    raise ValueError("Duplicate spec with conflicting scheduling policy")
                return j
            parents=[]
            for dep in s["deps"]:
                p=c.execute("SELECT spec FROM jobs WHERE id=?",(dep,)).fetchone()
                if not p:raise ValueError("Unknown parent; register parent first")
                ps=json.loads(p["spec"]);parents.append(ps)
                if ps["stage"]!="CONTROL" and any(ps[k]!=s[k] for k in ("scroll","frame","unit")):
                    raise ValueError("Cross-unit lineage without registered transform")
            stages=[p["stage"] for p in parents]
            expected=PREDECESSOR[s["stage"]]
            required=([expected] if expected else [])+(["CONTROL"] if s["stage"] in INFERENCE else [])
            if sorted(stages)!=sorted(required):raise ValueError("Stage dependency contract mismatch")
            if s["stage"] in INFERENCE and not any(p["stage"]=="CONTROL" and p["calibration"]==s["calibration"] for p in parents):
                raise ValueError("Control recipe/domain mismatch")
            admissible=within(g,self.policy["caps"][lane])
            state="PENDING" if admissible else "BLOCKED"
            reason=None if admissible else "GRANT_EXCEEDS_LANE_CAP"
            c.execute("INSERT INTO jobs(id,spec,lane,priority,grant,state,reason,created) VALUES(?,?,?,?,?,?,?,?)",
                      (j,canonical(s),lane,priority,canonical(g),state,reason,now))
            self.event(c,now,j,"ENQUEUED",{"state":state,"reason":reason})
        return j
    def balances(self,c):
        spent={k:zeros() for k in LANES};reserved={k:zeros() for k in LANES}
        for a in c.execute("SELECT a.*,j.lane FROM attempts a JOIN jobs j ON j.id=a.job"):
            lane=a["lane"]
            if a["status"]=="LEASED":reserved[lane]=add(reserved[lane],json.loads(a["grant"]))
            else:spent[lane]=add(spent[lane],json.loads(a["actual"]))
        return spent,reserved
    def _expire(self,c,now):
        rows=c.execute("SELECT * FROM attempts WHERE status='LEASED' AND expires<=?",(now,)).fetchall()
        for a in rows:
            c.execute("UPDATE attempts SET status='LOST',actual=grant WHERE token=?",(a["token"],))
            c.execute("UPDATE jobs SET state='BLOCKED',reason='LEASE_EXPIRED_RECONCILE' WHERE id=?",(a["job"],))
            self.event(c,now,a["job"],"LEASE_EXPIRED",{"charged_full_grant":True})
        return len(rows)
    def expire(self,now=None):
        with self.tx() as c:return self._expire(c,self.clock(now))
    def _ready(self,c,row,now):
        s=json.loads(row["spec"]);inputs=set(s["inputs"].values())
        for dep in s["deps"]:
            p=c.execute("SELECT state,spec FROM jobs WHERE id=?",(dep,)).fetchone()
            if p["state"]=="BLOCKED":
                c.execute("UPDATE jobs SET state='BLOCKED',reason='UPSTREAM_BLOCKED' WHERE id=?",(row["id"],))
                self.event(c,now,row["id"],"BLOCKED",{"reason":"UPSTREAM_BLOCKED"});return False
            if p["state"]!="SUCCEEDED":return False
            a=c.execute("SELECT receipt FROM attempts WHERE job=? AND status='SUCCEEDED'",(dep,)).fetchone()
            if not a:raise RuntimeError("Parent state without receipt")
            receipt=json.loads(a["receipt"]);ps=json.loads(p["spec"])
            if not set(receipt["outputs"].values())<=inputs:
                c.execute("UPDATE jobs SET state='BLOCKED',reason='LINEAGE_HASH_MISMATCH' WHERE id=?",(row["id"],))
                self.event(c,now,row["id"],"BLOCKED",{"reason":"LINEAGE_HASH_MISMATCH"});return False
            if ps["stage"]=="CONTROL" and (receipt.get("control_pass") is not True or receipt.get("calibration")!=s["calibration"] or receipt["valid_until"]<=now):
                c.execute("UPDATE jobs SET state='BLOCKED',reason='CONTROL_STALE_OR_INVALID' WHERE id=?",(row["id"],))
                self.event(c,now,row["id"],"BLOCKED",{"reason":"CONTROL_STALE_OR_INVALID"});return False
        try:
            for h in inputs:self.store.verify(h)
        except ValueError:
            c.execute("UPDATE jobs SET state='BLOCKED',reason='INPUT_MISSING_OR_CORRUPT' WHERE id=?",(row["id"],))
            self.event(c,now,row["id"],"BLOCKED",{"reason":"INPUT_MISSING_OR_CORRUPT"});return False
        return True
    def lease(self,lane,worker,now=None):
        if lane not in LANES or not isinstance(worker,str) or not worker:raise ValueError("Invalid worker/lane")
        now=self.clock(now)
        with self.tx() as c:
            self._expire(c,now)
            if c.execute("SELECT v FROM config WHERE k='halted'").fetchone()[0]=="true":return None
            spent,reserved=self.balances(c);used=add(spent[lane],reserved[lane])
            counts={}
            for a in c.execute("SELECT j.spec FROM attempts a JOIN jobs j ON j.id=a.job WHERE j.lane=?",(lane,)):
                scroll=json.loads(a["spec"])["scroll"];counts[scroll]=counts.get(scroll,0)+1
            rows=c.execute("SELECT * FROM jobs WHERE lane=? AND state='PENDING'",(lane,)).fetchall()
            rows=sorted(rows,key=lambda r:(counts.get(json.loads(r["spec"])["scroll"],0),-r["priority"],r["created"],r["id"]))
            for row in rows:
                g=json.loads(row["grant"])
                if not within(add(used,g),self.policy["caps"][lane]):continue
                if not self._ready(c,row,now):continue
                token=secrets.token_hex(32);end=now+self.policy["lease_seconds"]
                c.execute("INSERT INTO attempts(token,job,worker,started,expires,status,grant) VALUES(?,?,?,?,?,'LEASED',?)",
                          (token,row["id"],worker,now,end,row["grant"]))
                c.execute("UPDATE jobs SET state='LEASED',attempts=attempts+1,reason=NULL WHERE id=?",(row["id"],))
                self.event(c,now,row["id"],"LEASED")
                return {"job":row["id"],"token":token,"spec":json.loads(row["spec"]),"grant":g,"expires":end}
        return None
    def finish(self,token,receipt,now=None):
        now=self.clock(now);r=dict(receipt)
        if not isinstance(r.get("job"),str):raise ValueError("Receipt job required")
        r["actual"]=vector(r.get("actual"));rh=digest(r)
        # Expire in its own committed transaction so stale receipt rejection cannot undo charges.
        self.expire(now)
        with self.tx() as c:
            a=c.execute("SELECT * FROM attempts WHERE token=?",(token,)).fetchone()
            if not a or a["job"]!=r["job"]:raise ValueError("Unknown or mismatched lease")
            if a["receipt_hash"] is not None:
                if a["receipt_hash"]!=rh:raise ValueError("Conflicting replay")
                return a["status"]
            if a["status"]!="LEASED" or a["expires"]<=now:raise ValueError("Stale/fenced receipt")
            j=c.execute("SELECT * FROM jobs WHERE id=?",(a["job"],)).fetchone();s=json.loads(j["spec"])
            if r.get("worker_stopped") is not True:raise ValueError("Completion requires stopped worker")
            if r.get("outcome") not in ("COMPLETE","FAILED"):raise ValueError("Invalid outcome")
            outputs=hashes(r.get("outputs"))
            for h in outputs.values():self.store.verify(h)
            # Failures may carry partial outputs, never unlock downstream work.
            over=not within(r["actual"],json.loads(a["grant"]))
            reason=None
            if over:
                status="OVERRUN";state="BLOCKED";reason="RESOURCE_OVERRUN"
                c.execute("UPDATE config SET v='true' WHERE k='halted'")
            elif r["outcome"]=="FAILED":
                reason=r.get("failure")
                if reason not in FAILURES:raise ValueError("Unknown failure classification")
                status="FAILED";state="PENDING" if reason in RETRYABLE and j["attempts"]<self.policy["max_attempts"] else "BLOCKED"
            else:
                if not outputs:raise ValueError("Complete stage needs byte-verified evidence")
                if r.get("failure") is not None:raise ValueError("Complete receipt contains failure")
                if s["stage"]=="CONTROL":
                    if r.get("calibration")!=s["calibration"] or type(r.get("control_pass"))!=bool:
                        raise ValueError("Control receipt key/verdict mismatch")
                    if type(r.get("valid_until"))!=int or r["valid_until"]<=now:
                        raise ValueError("Control needs a future validity limit")
                    if not r["control_pass"]:reason="BAD_CONTROL"
                if s["stage"] in INFERENCE:
                    if not {"asc_map","desc_map","asc_support","desc_support","metadata"}<=set(outputs) or r.get("orders")!=["asc","desc"]:
                        raise ValueError("Incomplete paired inference evidence")
                    for dep in s["deps"]:
                        ps=json.loads(c.execute("SELECT spec FROM jobs WHERE id=?",(dep,)).fetchone()[0])
                        if ps["stage"]=="CONTROL":
                            control=json.loads(c.execute("SELECT receipt FROM attempts WHERE job=? AND status='SUCCEEDED'",(dep,)).fetchone()[0])
                            if control["valid_until"]<=now:reason="CONTROL_STALE_OR_INVALID"
                if s["stage"]=="REVIEW" and r.get("review_status") not in REVIEW:
                    raise ValueError("Missing scoped review outcome")
                try:
                    for h in s["inputs"].values():self.store.verify(h)
                except ValueError:
                    reason="INPUT_MISSING_OR_CORRUPT"
                status="FAILED" if reason else "SUCCEEDED";state="BLOCKED" if reason else "SUCCEEDED"
            c.execute("UPDATE attempts SET status=?,actual=?,receipt=?,receipt_hash=? WHERE token=?",
                      (status,canonical(r["actual"]),canonical(r),rh,token))
            c.execute("UPDATE jobs SET state=?,reason=? WHERE id=?",(state,reason,j["id"]))
            self.event(c,now,j["id"],status,{"reason":reason})
            return status
    def reconcile_lost(self,job,token,worker_stopped,now=None):
        """Explicit local operator acknowledgement; spent reservation is NOT refunded."""
        if worker_stopped is not True:raise ValueError("Confirm worker has stopped")
        now=self.clock(now)
        with self.tx() as c:
            a=c.execute("SELECT * FROM attempts WHERE job=? AND token=?",(job,token)).fetchone()
            j=c.execute("SELECT state,reason,attempts FROM jobs WHERE id=?",(job,)).fetchone()
            if not a or not j or a["status"]!="LOST" or j["state"]!="BLOCKED" or j["reason"]!="LEASE_EXPIRED_RECONCILE":
                raise ValueError("No unreconciled expired lease")
            state="PENDING" if j["attempts"]<self.policy["max_attempts"] else "BLOCKED"
            c.execute("UPDATE jobs SET state=?,reason=? WHERE id=?",(state,None if state=="PENDING" else "RETRY_LIMIT",job))
            self.event(c,now,job,"LOST_RECONCILED",{"state":state,"reservation_refunded":False})
            return state
    def summary(self):
        """Operational aggregate only; no private unit/worker/hash/receipt exports."""
        with self.tx() as c:
            out={};reasons={};spent,reserved=self.balances(c)
            for j in c.execute("SELECT spec,state,reason,lane,grant FROM jobs"):
                s=json.loads(j["spec"]);scroll=s["scroll"]
                key=(scroll,s["stage"],j["state"]);out[key]=out.get(key,0)+1
                if j["state"]=="BLOCKED":
                    k=j["reason"] or "UNCLASSIFIED_BLOCK";reasons[k]=reasons.get(k,0)+1
                elif j["state"]=="PENDING":
                    if not within(add(add(spent[j["lane"]],reserved[j["lane"]]),json.loads(j["grant"])),self.policy["caps"][j["lane"]]):
                        k="PENDING_RESOURCE_CAP";reasons[k]=reasons.get(k,0)+1
                    elif any(c.execute("SELECT state FROM jobs WHERE id=?",(d,)).fetchone()[0]!="SUCCEEDED" for d in s["deps"]):
                        k="PENDING_DEPENDENCY";reasons[k]=reasons.get(k,0)+1
            return {"counts":[{"scroll":k[0],"stage":k[1],"state":k[2],"n":v} for k,v in sorted(out.items())],
                    "operational_reasons":reasons,"resources":{"spent":spent,"reserved":reserved},
                    "halted":c.execute("SELECT v FROM config WHERE k='halted'").fetchone()[0]=="true"}
