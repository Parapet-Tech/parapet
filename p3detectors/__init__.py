"""P3 per-event detector package (Paper 3, multi-turn injection sensing).

Per-event surface_signal scorers for the detector ensemble (D_gen + cross-family
D_eval). Step 1 ships only D_gen: an MLX-native LLM judge over the local
OpenAI-compatible endpoint, scoring tool-call argument strings from staged carriers.

Design: local-llm/.local/multiturn/detector_ensemble_spec.md. Peer to p3carriers
(which provides the normalized carriers + liveness labels this scores against).
Generated score data is git-ignored; this package is not.
"""
