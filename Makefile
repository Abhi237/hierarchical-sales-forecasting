# Hierarchical and probabilistic revenue forecasting on the M5 Walmart data - reproducible pipeline
PY      ?= .venv/bin/python
HF      ?= .venv/bin/hier-forecast

.PHONY: setup data sql backtest r evaluate export tableau-web figures report test lint all determinism clean

setup:            ## create venv and install pinned deps + package
	uv venv --python 3.11 .venv || python3.11 -m venv .venv
	uv pip install --python .venv/bin/python -r requirements.txt || .venv/bin/pip install -r requirements.txt
	uv pip install --python .venv/bin/python -e . --no-deps || .venv/bin/pip install -e . --no-deps

data:             ## download the M5 files from public mirrors (no credentials), verify shapes, write data/raw/manifest.json
	$(PY) scripts/download_data.py

sql:              ## numbered DuckDB scripts: staging, weeks, item-store weekly fact, hierarchy, item layer, cleaning log; R inputs
	$(HF) sql

backtest:         ## rolling-origin backtest: baselines, ETS, SARIMA, LightGBM quantile + conformal (hierarchy and item layers)
	$(HF) backtest

r:                ## fable ETS/ARIMA, reconciliation, Ljung-Box (Rscript; package versions in R/sessionInfo.txt)
	$(HF) r

evaluate:         ## one metric code path over every model; Python reconciliation; artifacts/
	$(HF) evaluate

export:           ## Tableau CSVs in tableau/data/
	$(HF) export

tableau-web:      ## 5 flattened CSVs for Tableau Public web authoring in tableau/web/ (from tableau/data/)
	$(PY) scripts/build_tableau_web.py

figures:
	$(HF) figures

report:           ## Quarto/HTML report, memo PDF, README
	cd reports && R_LIBS=$${R_LIBS_USER_EXTRA:-$$HOME/R/lib} quarto render forecast_report.qmd --to html
	$(PY) scripts/build_memo.py
	$(PY) scripts/build_readme.py

lint:
	$(PY) -m ruff check .

test: lint        ## lint + pytest (committed fixture; no raw data needed)
	$(PY) -m pytest

determinism:      ## run sql + backtest + evaluate + export twice and compare SHA-256 of every output
	$(PY) scripts/check_determinism.py

all: data sql backtest r evaluate export tableau-web figures report test

clean:
	rm -rf data/processed reports/*_files
