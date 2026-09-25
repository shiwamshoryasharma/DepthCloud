# DepthCloud agent instructions

The canonical project is `D:\10-SEP-2026\DepthCloud`. Work here even if the task initially opens in a different directory.

## Resume before acting
On a fresh clone, `.agents/` is intentionally absent. Read `README.md` and `docs/DEVELOPMENT.md` instead; do not invent missing private history or redirect a clone to another workstation path.
1. Read [.agents/README.md](.agents/README.md), [.agents/SUMMARY.md](.agents/SUMMARY.md), and [.agents/PLAN.md](.agents/PLAN.md).
2. Read [.agents/BUGS.md](.agents/BUGS.md) and [.agents/REGRESSION.md](.agents/REGRESSION.md) before changing behavior or dependencies.
3. Read [.agents/USER_PREFERENCES.md](.agents/USER_PREFERENCES.md); use the architecture, environment, records, and history documents as needed.
4. Recheck the actual files and Git status. Memory is a dated snapshot, not authority over current code.

## Workspace and memory rules
- Never copy, stage, generate, or back up this project under OneDrive. Keep edits, temporary files, logs, caches, build output, and memory in this existing project. Set the working directory explicitly.
- Use the local `backend\python.bat` launcher for backend work. Do not silently substitute the system Python.
- Preserve existing functionality; make the smallest practical change. Explain flawed approaches directly.
- Never copy secrets, tokens, or complete `.env` contents into memory or logs.
- Update the relevant `.agents` documents after each meaningful task: current state, completed work, evidence, open issues, and exact next step. Append history/changelog; distinguish facts, hypotheses, and proposals.
- Do not mark checks as passing unless actually run. Document skipped checks and pre-existing failures.
- All detailed project memory lives under `.agents\`. This root file is only the discovery entry point.
