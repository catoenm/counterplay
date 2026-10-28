# Counterplay

A research workbench for verified language-model self-play: planning puzzles,
competitive games, programmable agents, and bounded training experiments.

This is a one-time archival snapshot dated October 28, 2026. All recorded studies
are closed. The original README and research files are preserved as historical
documents; their descriptions of active work and launch commands are not current
instructions or authorization to resume experiments.

## Start here

- [Research overview and historical setup](README.md)
- [Research plan](docs/research-plan.md) and [literature review](docs/literature-review.md)
- [Game environments](docs/adding-arena-environments.md)
- [Scalar training diagnosis](docs/scalar-training-diagnosis-2026-10-25.md)
- [Scalar evaluation results](runs/arena-mixed-scalar-evaluation-gpu-v1/scientific-summary-v1.json)
- [Planning admission results](runs/planning-curriculum-gpu-v1/scientific-summary-v1.json)
- [Interactive replays](visualizations/)

No overall improvement was established by the scalar evaluation. The planning
admission screen passed its signal gate but failed its frozen timing and cost
gate, so no comparison training or held-out evaluation was admitted. Its original
failed final audit is preserved alongside the separate scientific reviews; those
reviews do not replace or change that audit result.

## Snapshot contents

Included: source, tests, research documentation, dependency specifications,
experiment definitions, small datasets, selected existing scientific reports and
figures, and copies of the existing interactive visualizations.

Excluded: credentials, model weights, adapters and checkpoints, environments and
caches, large raw trajectories and evaluation records, live accounting, and
operational deployment receipts and logs. All excluded evidence remains in the
original local workspace. Selected reports and manifests can reference files
outside this snapshot, so this is not a complete evidence bundle for rerunning
historical audits. Some tests also depend on locally retained run records or
specialized model dependencies.

No experiments, training workers, provider resources, or final audits were run to
create this snapshot. Git commits are not scheduled; this snapshot is a single
manual checkpoint. Existing billing and unknown-resource reconciliation remains
separate from the archived research.
