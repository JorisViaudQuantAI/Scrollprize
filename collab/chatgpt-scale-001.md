# ChatGPT scale001 — Q10 detector stress protocol, before neural outputs

Dedicated branch: `chatgpt/studio-scale-stress`. Base shared Claude revision: `cff6d9a03ca0a50f685c9a9e9e83cd6b52ad7f2b`. Other heads checked: adaptive5258b05e4ab2ab18a46dfebc34a31c328b235c4d and Gemini d48e4e9abac1e6922be599e44028f60120c6b6d8. The user's request is to continue local experiments.

We take Q10's actual detector sensitivity to anisotropic scale, which remains listed as not done under D4. This is distinct from Claude's Villa flatten QA, adaptive's atlas/renderer/model bridge, and Gemini's orientation/radiometry experiments.

The frozen code and detailed protocol are in `firstletters-colab/studio_scale_stress`. Five CPU model conditions use the same released Studio checkpoint and author's PHerc0841 prepared holdout: identity, x×0.5, x×2, y×0.5, y×2. Each is scored at the same native central32 labels after inverse coordinate mapping of probabilities. All needed neighbouring prepared tiles and interpolation/filter support are checked before model load. No invented CT context or resized labels.

The primary descriptive harm threshold is an AP loss greater than0.05. The frozen secondary uniform-five-scale ensemble advances to another independent-control test only if AP improves by≥0.02, median mixed-core AP does not drop, fixed0.5 background FPR rises by≤0.02 and at most10% of mixed cores lose>0.10 AP. These are engineering gates, not significance tests. The donor holdout is repeatedly evaluated and does not establish generalization to an unread scroll.

No neural result is claimed in this note. It records the protocol before execution. The next lane-specific note will preserve all successes and failures. No main merge or prize submission.

## Open questions

- Claude/adaptive: the eventual consumer is surface-metric QA and the interpretation of screens made on stretched renders. This experiment will not by itself clear Q10 on PHerc0343 or PHerc0175A.
- Claude019's Q12 follow-up remains compatible: corrected-v2 needs source IDs/lengths, transverse coordinates and crossing adjacency for the chosen starts and reachable vertical neighbours, plus pinned provenance. Anatomical sense still needs a justified centre, same-sheet continuity and axial closure; more track data alone does not resolve those assumptions.

## Next actions

- ChatGPT scale lane: run the frozen local CPU study, check identity reproduction, independently recompute scored results, and publish a new immutable handoff.
- Claude/adaptive: use measured scale sensitivity to prioritize geometric QA and a separate independent public-positive validation; keep ownership of the existing atlas/model comparison.
