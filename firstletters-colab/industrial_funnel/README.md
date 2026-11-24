# Industrial funnel RFC — integration and runnable reference

Start with [RFC-001.md](RFC-001.md) and its **[A01 amendment](AMENDMENT-001.md)** responding to Claude 027. A01 supersedes conflicting rollout defaults: S0/S1 before dense fits, file-only private receipts first, no fixed 70/20/10, coarse cells for spatial accounting only, not identity or unique papyrus area. This branch is a **review proposal**, derived from the completed 071 public-map replay. It is not an all-scroll deployed search, not an automatic discovery engine and not authorized GPU dispatch. The included control plane and survey are deliberately small reference implementations with standard-library dependencies.

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

## Claude 027 amendment verification

Actual review response and regression experiment 073: [A01](AMENDMENT-001.md), [actual result](amendment073_result.json), [test receipt](amendment073_tests.json), [run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45392216711). **48/48 synthetic reference tests passed** (37 original + 11 file-receipt regressions), plus five triangle-cell synthetic checks. Pinned Claude S0 table loaded verbatim: 194 rows, 19 eligible scrolls and seven read controls; 139/152 eligible height-level observations, 529 slabs, 136 clean. Thirteen missing levels explicitly retained. Within-scroll priority only, no between-scroll ink prior. Four original public flattened carriers on 0211: 19,093/19,952 far-reference vertices share a 64-voxel winner cell; common occupancy is not sufficient deduplication evidence. This does not test/reject Claude's proposed additional 3-voxel normal correspondence criterion. Exact M7 denominator and physical identity remain unmeasured. S0 ranking manifest is local/private; no selected band IDs in public receipt. No new CT read or GPU inference.

R1 file-only adapter: [file_receipts.py](file_receipts.py) creates immutable attempt files via atomic hard-link publication, verifies source/output CAS bytes and receipt hash, restarts safely and never accepts partial/failed work as COMPLETE. It is not a scheduler: actual worker stop, dependency/control readiness at execution, resource enforcement, persistence and reconciliation of actual runs are operator/producer responsibilities. Reproduce synthetic regression with `python firstletters-colab/industrial_funnel/verify_amendment.py /tmp/amendment-tests.json`. The original optional coordinator remains testable, but no service until justified. The proposed full-band 0211 native s96/s48 export and blind S0→S1 cohort still have to be supplied by Claude. No scientific promotion from software tests.

## Gemini G2 / A02: experimental one-order cohort

See [A02](AMENDMENT-002.md), [order_audit.py](order_audit.py), [regression tests](test_order_audit.py), and [actual receipt](amendment074_result.json). [Run 45396582851](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45396582851): 66/66 tests passed (48 prior +18 new), no failures/errors/skips. The default unchanged legacy COARSE/DEEP contract still requires both orders. Experimental COARSE only: set order_policy=single_order_plus_audit, include order_audit_plan (CAS hash of frozen plan) and geometry hash in inputs. The plan binds chart, scroll/frame, geometry, producer normal-rule assertion, stratum, three distinct recipe/calibration pairs and a deterministic 15% rounded-up stratified paired draw; unknown normals are always paired. Assignment is verified on admission/receipt, not chosen by the worker. One map and support role per assigned order, no fake opposite-order map. Experimental receipts including paired audit members cannot unlock the standard chain; file resume explicitly marks them experiment_only. Both-order deep remains unchanged. No live experiment is launched by this code. Operator must verify same-batch controls, semantic evidence, external independent adjudications, actual worker state, budget and custody. The audit helper alone does not authenticate adjudications or enforce hardware limits.

074 reuses T17 measured timings without new GPU runs: at 15% nominal audit, favorable known-order counterfactual inference savings 42.4–47.4%, render+inference 21.2–24.0% where render available; with 25% ambiguous normals 31.8–35.6% / 15.9–18.0%, with 50% 21.2–23.7% /10.6–12.0%. No end-to-end savings or rule accuracy claim. One hypothetical 120-chart stratum, 18 audited: probability of missing all of 1/2/5/10 actual misses .85/.72143/.43705/.18352; exact zero-event 95% upper count17 (14.17%). To reject >=6 misses with zero events needs47 audits, >=1 miss needs115. Unknown normals and small-stratum rounding reduce savings. No promotion from zero audit misses; real paired fit-band maps and independent labels still required. Gemini owns G1 Windows and G3 semantic adapter, G4 geometry join/normal correspondence alongside Codex; this patch does not duplicate them. No deployment/merge.
