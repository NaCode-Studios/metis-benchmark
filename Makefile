.PHONY: test status status-a status-b lint reproduce-g0

test:
	python -m pytest -q

# Dataset-availability reports: which raw files are present in data/raw/.
# These do NOT run any experiment — the runners live in scripts/ (see
# reproduce-g0 below). Named honestly for what they do.
status:
	python -m metis_benchmark.backtest

status-a:
	python -m metis_benchmark.backtest --track A

status-b:
	python -m metis_benchmark.backtest --track B

lint:
	ruff check src tests scripts

# Reproduce the full G0 evidence chain end to end, in the order documented in
# reports/REPORT.md sec. 8: download the public datasets, run the tabular
# channel and its honest ceiling, cache the Track B embeddings, run the
# semantic channel and its ceiling, then the test suite. Slow: the embedding
# step downloads a sentence model and embeds ~56k texts (once — cached).
reproduce-g0:
	python scripts/download.py
	python scripts/run_track_a_rolling.py
	python scripts/run_ceiling.py
	python scripts/embed_track_b.py
	python scripts/run_track_b.py
	python scripts/run_ceiling_b.py
	python -m pytest -q
