"""Lean file-only receipt adapter for R1. No scheduler, worker or external dispatch."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from order_audit import is_experiment, validate_outputs
from control_plane import canonical,digest,hashes,private_root,require_sha,validate_spec,vector,within,Store,INFERENCE,FAILURES,REVIEW

def job_key(spec):
    return digest(validate_spec(spec))

def validate(spec,attempt,receipt,store):
    s=validate_spec(spec);j=job_key(s)
    if type(attempt)!=int or attempt<1:raise ValueError("Invalid attempt number")
    base={"job","attempt","outcome","worker_stopped","grant","actual","outputs","failure"}
    extra=set()
    if s["stage"]=="CONTROL" and receipt.get("outcome")=="COMPLETE":extra={"calibration","control_pass","valid_until"}
    if s["stage"] in INFERENCE and receipt.get("outcome")=="COMPLETE":extra={"orders"}
    if s["stage"]=="REVIEW" and receipt.get("outcome")=="COMPLETE":extra={"review_status"}
    if set(receipt)!=base|extra or receipt["job"]!=j or receipt["attempt"]!=attempt:
        raise ValueError("Receipt schema/job/attempt mismatch")
    if receipt["worker_stopped"] is not True:raise ValueError("Worker not confirmed stopped")
    grant=vector(receipt["grant"]);actual=vector(receipt["actual"])
    out=hashes(receipt["outputs"])
    for h in out.values():store.verify(h)
    complete=receipt["outcome"]=="COMPLETE"
    if receipt["outcome"] not in ("COMPLETE","FAILED","PARTIAL"):raise ValueError("Invalid outcome")
    if complete:
        if not within(actual,grant):raise ValueError("Overrun cannot be COMPLETE")
        if not out or receipt["failure"] is not None:raise ValueError("Incomplete completed receipt")
        for h in s["inputs"].values():store.verify(h)
        if s["stage"] in INFERENCE:
            validate_outputs(s,receipt,store)
        if s["stage"]=="CONTROL":
            if receipt["calibration"]!=s["calibration"] or type(receipt["control_pass"])!=bool or type(receipt["valid_until"])!=int:
                raise ValueError("Control key or evidence invalid")
        if s["stage"]=="REVIEW" and receipt["review_status"] not in REVIEW:raise ValueError("Invalid review scope")
    elif receipt["failure"] not in FAILURES|{"RESOURCE_OVERRUN","SESSION_LOST","ACCOUNT_LIMIT","GPU_UNAVAILABLE"}:
        raise ValueError("Failed/partial work needs explicit reason")
    return s

class FileReceipts:
    """Atomic no-overwrite publication of attempt receipts into an owner-private folder.

    The operator owns budget and dependency approval. This verifies bytes and
    immutable attempt identity, not worker liveness or scientific correctness.
    """
    def __init__(self,root):
        self.root=private_root(root)
        self.store=Store(self.root/"cas")
        self.receipts=private_root(self.root/"receipts")
    def _path(self,spec,attempt):
        return self.receipts/(job_key(spec)+"."+str(attempt)+".json")
    def publish(self,spec,attempt,receipt):
        s=validate(spec,attempt,receipt,self.store)
        envelope={"schema":"funnel.file_receipt.v1","spec":s,"receipt":receipt}
        obj={**envelope,"sha256":digest(envelope)}
        body=(canonical(obj)+"\n").encode();p=self._path(s,attempt)
        fd,tmp=tempfile.mkstemp(prefix=".partial-",dir=self.receipts)
        try:
            with os.fdopen(fd,"wb") as f:f.write(body);f.flush();os.fsync(f.fileno())
            try:os.link(tmp,p)
            except FileExistsError:
                if p.is_symlink() or p.read_bytes()!=body:raise ValueError("Conflicting immutable attempt")
            d=os.open(self.receipts,os.O_RDONLY)
            try:os.fsync(d)
            finally:os.close(d)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return obj["sha256"]
    def read(self,spec,attempt):
        p=self._path(spec,attempt)
        if p.is_symlink() or not p.is_file():raise ValueError("Missing completed file receipt")
        obj=json.loads(p.read_text())
        if set(obj)!={"schema","spec","receipt","sha256"} or obj["schema"]!="funnel.file_receipt.v1":
            raise ValueError("Envelope schema mismatch")
        if digest({k:v for k,v in obj.items() if k!="sha256"})!=obj["sha256"] or obj["spec"]!=validate_spec(spec):
            raise ValueError("Receipt hash/spec mismatch")
        validate(spec,attempt,obj["receipt"],self.store)
        return obj
    def resume(self,spec,now=None):
        """List immutable attempts. Never silently pick competing successful results."""
        import time
        now=int(time.time()) if now is None else int(now)
        j=job_key(spec);rows=[]
        for p in sorted(self.receipts.glob(j+".*.json")):
            t=p.name.split(".")
            if len(t)!=3 or not t[1].isdigit():raise ValueError("Bad attempt filename")
            rows.append(self.read(spec,int(t[1])))
        good=[]
        for q in rows:
            r=q["receipt"]
            if r["outcome"]!="COMPLETE":continue
            if spec["stage"]=="CONTROL" and (not r["control_pass"] or r["valid_until"]<=now):continue
            good.append(q)
        if len(good)>1:raise ValueError("Multiple complete attempts need explicit operator reconciliation")
        return {"attempts":len(rows),"completed":len(good)==1,"receipt":good[0] if good else None,
                "experiment_only":is_experiment(spec),"eligible_for_standard_chain":len(good)==1 and not is_experiment(spec),
                "next_attempt":max([q["receipt"]["attempt"] for q in rows],default=0)+1}
