.PHONY: test status backtest-a backtest-b lint

test:
	python -m pytest -q

status:
	python -m metis_benchmark.backtest

backtest-a:
	python -m metis_benchmark.backtest --track A

backtest-b:
	python -m metis_benchmark.backtest --track B

lint:
	ruff check src tests scripts
