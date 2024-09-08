format:
	@ruff format .
	@ruff check . --fix
	@mypy --config-file "pyproject.toml"



