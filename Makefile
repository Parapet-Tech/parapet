.PHONY: build dev test eval run clean

# Debug build
build:
	cd parapet && cargo build

# Release build
release:
	cd parapet && cargo build --release

# Run engine on :9800 (debug, parapet.yaml in repo root)
dev:
	cd parapet && cargo run -- --config ../parapet.yaml --port 9800

# Run engine on :9800 (release)
run:
	cd parapet && cargo run --release -- --config ../parapet.yaml --port 9800

# All tests (unit + integration)
test:
	cd parapet && cargo test

# Eval harness (all datasets, human-readable)
eval:
	cd parapet && cargo run --bin parapet-eval -- \
		--config ../schema/eval/eval_config.yaml \
		--dataset ../schema/eval/

# Eval harness (JSON output)
eval-json:
	cd parapet && cargo run --bin parapet-eval -- \
		--config ../schema/eval/eval_config.yaml \
		--dataset ../schema/eval/ \
		--json

clean:
	cd parapet && cargo clean
