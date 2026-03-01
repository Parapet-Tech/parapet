PYTHON ?= python

.PHONY: test semantic-hash

test:
	$(PYTHON) -m pytest -q

semantic-hash:
	$(PYTHON) -m parapet_runner.runner semantic-hash $(ARGS)
