.PHONY: format lint
format:
	ruff format . && ruff check . --fix
lint:
	ruff check .