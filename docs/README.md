# Engineering record

`docs/` is the permanent engineering record for this football-event detection project. It records what the code actually does, what is missing, and the phased plan for action spotting.

The root `README.md` is the operator guide for the detector, dataset, and training scripts. It is not this record. Where the two disagree, the audit under `docs/audit/` follows the code.

## How to use this directory

Work proceeds in this order:

**AUDIT → PLAN → IMPLEMENT → TEST → RECORD RESULT → NEXT PHASE**

Future work updates this directory when an architecture decision, a test result, an important finding, or a phase completion changes the plan. Do not silently change the architecture. If a later measurement disproves an audit conclusion, mark that conclusion as superseded and leave the original text in place.

## Audit

- [Current architecture](audit/01-current-architecture-audit.md) — what the repository implements today, with files and functions.
- [Gap and risk analysis](audit/02-gap-and-risk-analysis.md) — why the known failures happen, and whether they need new labels.
- [Audit summary](audit/03-audit-summary.md) — keep / modify / replace / create, and the labeling decision.

## Planning

- [Master roadmap](planning/00-master-roadmap.md) — target architecture, built on the components that already exist.
- [Phase plan](planning/phase-plan.md) — phased implementation order, dependencies, event matrix, and success criteria.

No phase in the plan is approved for implementation until that approval is explicit.
