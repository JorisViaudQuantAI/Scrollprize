# ChatGPT B5/B6 — independent public audit continuation

Branch `chatgpt/b5-winding-coverage-audit`, based on integration `1770a7b`. Public already-awarded PHerc0211 only.

Actual B5: 40 fine-review windows touch 30/60 winding charts; 30 entirely untouched, holding 63 windows and 122.129 cm² summed valid-window area. 25 of those windows abstain, including three wholly abstaining untouched windings. No anatomical identity or ink recall inferred. Six synthetic checks pass.

B6: on the 109 remaining windows, exact four-stratum minimax prospective allocation for 45 reviews is (9,10,16,10) in order touched scored, touched abstain, untouched scored, untouched abstain. Conditional on zero flags and the frozen sampling design, worst probability under >=6 flags is 0.041648. 1316 independent small exhaustive oracle comparisons pass. No new reviews or GPU runs; compare SRS 42 for target six. No stopping gate.

Open questions
- Claude: are the 30 previously untouched charts actually reviewable under the same native fine recipe? Which failure/unknown labels can be kept distinct from negative?
- Codex: independently replay the chart-union and exact sampling oracle, check lineage and chart-vs-physical-sheet distinction.
- Gemini: provide actual per-job costs and native paired masks before any comparison of GPU efficiency.

Next actions
- Owner/Claude can choose a frozen review budget and independently draw within the strata before seeing any new outcome.
- Keep the all-winding coverage queue separate from statistical random review; do not apply B6 confidence bounds to deterministic coverage picks.
- No main merge, upstream submission, unread candidate or automatic blank decision.
