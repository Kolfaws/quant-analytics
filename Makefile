# =============================================================================
# quant-analytics — 常用命令
# =============================================================================

.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= $(UV) run python
PYTEST ?= $(UV) run --with pytest python -m pytest
RUFF ?= $(UV) run --with ruff ruff

.PHONY: help install run test lint format clean

help:
	@echo "quant-analytics"
	@echo ""
	@echo "  install    uv sync"
	@echo "  run        streamlit run app.py"
	@echo "  test       pytest tests"
	@echo "  lint       ruff check ."
	@echo "  format     ruff format ."
	@echo "  clean      清理缓存与构建产物"

install:
	$(UV) sync

run:
	$(UV) run streamlit run app.py

test:
	$(PYTEST) tests -q

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for r in (pathlib.Path('src'), pathlib.Path('pages'), pathlib.Path('tests')) if r.exists() for p in r.rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in map(pathlib.Path, ('.pytest_cache','.ruff_cache','.mypy_cache','build','dist','htmlcov','.coverage'))]"
