"""P3 carrier corpus-prep package (Paper 3, multi-turn injection sensing).

Recurring infrastructure for turning AgentDojo / AgentDyn pre-executed runs into
normalized carrier artifacts with a liveness label. The load-bearing logic (the
reviewed action allowlist and the linear-downstream liveness rule) lives here as
tested package code; scripts/ holds thin CLIs only.

Deliberately separate from parapet-runner, which is classifier-training
orchestration. Generated/staged carrier data is git-ignored; this package is not.
"""
