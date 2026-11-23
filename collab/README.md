# collab/ — notes between the agents working in this repository

Two AI coding agents work for the repository owner on the Vesuvius Challenge: Codex (the flat-ROI occupancy work at
the repository root) and Claude (the `firstletters-colab/` folder). This folder is how they talk.

Rules
- One file per message: `claude-NNN.md` or `codex-NNN.md`, numbered in order, never edited after it is pushed
  (corrections go in a new file).
- Each message ends with **Open questions** (for the other agent or the owner) and **Next actions** (who does what).
- This repository is public. Do not write here: personal data, private ids or tokens, or the location of anything
  that might be ink in an unread scroll. The First Letters rules forbid making a find public before the organisers
  announce it; a candidate is described here only as "a candidate exists on scroll X, under review".
- Claude works on the branch `claude/firstletters-colab` and reads `main` and every `codex/*` or `collab` branch at
  each check. Codex may answer on `main` or on any branch; a file in `collab/` is enough.
- Nothing is merged into `main`, submitted to the organisers, or posted elsewhere without the owner's go-ahead.

Integration (since 23 November 2026)
- Four agents now write here: Claude, Codex, ChatGPT and Gemini. Note files are `claude-NNN.md`, `codex-NNN.md`,
  `chatgpt-NNN.md`, `chatgpt-adaptive-NNN.md` and `gemini-NNN.md`.
- The branch `claude/firstletters-colab` is the integration branch: every agent branch is merged into it at each
  check (see `claude-022.md` for the first merge and its test results). Start new branches from it.
- Where things are: `firstletters-colab/` (Claude's and ChatGPT's tools, the methods ledger in
  `methods_ledger/`), `gemini_lane/` (Gemini), `registration-checks/`, `public-controls/`, `mesh-overlap/`,
  `spiral-replay/` and `codex/` (Codex), the flat-ROI work at the repository root.
