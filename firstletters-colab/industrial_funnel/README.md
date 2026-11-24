# Industrial funnel RFC — integration and runnable reference

Start with [RFC-001.md](RFC-001.md). This branch is a **review proposal**, derived from the completed 071 public-map replay. It is not an all-scroll deployed search, not an automatic discovery engine and not authorized GPU dispatch. The included control plane and survey are deliberately small reference implementations with standard-library dependencies.

## Run the actual synthetic suite

```bash
python firstletters-colab/industrial_funnel/verify.py --output /tmp/funnel-check.json
```

No network, scans, model weights, GPU or unread data required. The receipt contains test count, failures, source SHA256 fingerprints, versions and time. [test_control_plane.py](test_control_plane.py) includes a complete synthetic stage graph and two concurrent-claim checks, resource exhaustion, restart, expiry and retry, stale calibration, input/output corruption, receipt replay, scope and survey checks. This is not a fault injection into the real GPU worker or a production reliability claim.

## Reference components

- [control_plane.py](control_plane.py): single-host SQLite WAL, CAS byte hashing, dependency and calibration-key checks, explicit lane resource caps, immutable epoch, transactional lease/reservation, expiry fencing, non-refunded lost reservation, explicit stopped-worker reconciliation, overrun circuit breaker, replay-idempotent receipts, no automatic blank state. Counts and resource aggregates only in summary. All state and artifacts outside Git, mode 0700.
- [survey.py](survey.py): versioned fixed stratum population, deterministic draw independent of input ordering, without replacement, known inclusion probabilities, no substitution when reserve is infeasible. Complete adjudication required for HT estimates, variance estimate and zero-event bound. Exact finite-population zero-event bound with simultaneous Bonferroni adjustment; no confidence interval from positive-event survey, no inference of latent ink.
- [Adapter contract](ADAPTER_CONTRACT.md): proposed complete metadata interface. **Producer-level semantics and actual array shape/physical support validation remain adapter work.** The reference verifies byte hashes and named roles, not geometry correctness or authenticity of a calibration statement. There is no worker shell runner, multi-host service, encrypted object storage, authentication, live gate revocation, automatic migration or physical-area union implementation. Those are explicit R1/R2 review blockers.

## API sketch — synthetic only

```python
from control_plane import Coordinator, digest
from survey import freeze_draw, summarize

def cap(cpu=0, gpu=0, human=0, io=0):
    return dict(cpu_ms=cpu, gpu_ms=gpu, human_ms=human, io_bytes=io)

c = Coordinator("/tmp/private-funnel-example", "fixture-epoch",
                {"exploit": cap(cpu=10000), "repair": cap(cpu=3000),
                 "explore": cap(cpu=1500)}, ["PHerc0125"])
source = c.store.put(b"synthetic source manifest")
j = c.enqueue(dict(scroll="PHerc0125", unit="synthetic-chart",
                   frame=digest("synthetic frame"), stage="REGISTER",
                   recipe=digest("register-v1"), inputs={"source": source},
                   deps=[], calibration=None), "exploit", 1, cap(cpu=1000))
lease = c.lease("exploit", "fixture-worker")
# Operator runs only the separately authorized, bounded actual job.
if lease:
    evidence = c.store.put(b"synthetic result")
    c.finish(lease["token"], dict(job=j, outcome="COMPLETE",
             worker_stopped=True, actual=cap(cpu=20),
             outputs={"evidence": evidence}))
print(c.summary())  # operational aggregate, still review before public export
```

The proof-of-concept uses one stage predecessor and, for inference, one matched CONTROL parent. Parent output hashes must appear in child inputs; independent method branches require different recipe fingerprints. It intentionally does not merge multiple carrier/recipe branches. Multi-family aggregation and per-tile splits/joins need typed transforms and a verified physical-lineage graph in R1. Retry only after a stopped worker attestation; expired leases are blocked and fully charged until a local operator explicitly reconciles them. A stale result never unlocks a successor. SQLite reservations are not real hardware quotas; workers must enforce their caps and termination before production. Controls have a finite validity time and must be valid at dispatch and completion. To change epoch caps or eligibility, use a new private epoch and explicitly reconcile previous resource/coverage state; no silent reset.

## Review process

Keep all changes on this RFC branch or separate reviewer branches. Reply in the RFC PR or issue #1, using blocker/evidence/proposed change/acceptance test/owner. No merge into main without the owner. Share only sanitized aggregate evidence and public controls. The previous 071 failure is preserved; the proposed 70/20/10 allocation and R2 targets are decisions to review, not proven optimal. The already frozen 070 anchor-corruption study remains separate and unexecuted in this RFC. 
