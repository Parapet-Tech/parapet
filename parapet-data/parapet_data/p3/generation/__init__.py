"""P3 generation-batch validation plumbing.

A generation worker (untrusted) emits `p3-generated-trajectory/1` records into a
git-ignored batch dir; this package validates them over a ROUND-TRIPPED carrier
backbone before any survivor is promoted into the corpus pool. The worker chooses
graft positions and writes event text; liveness, origin, label, and detector scores
are decided HERE, never trusted from the worker.

See local-llm/.local/multiturn/generation_worker_output_contract.md.
"""
