.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:
INSTALL_STAMP := .install.stamp
POETRY := ${HOME}/poetry/bin/poetry

install_poetry:
	@echo "Installing poetry..."
	@curl -sSL https://install.python-poetry.org | POETRY_HOME=${HOME}/poetry python3 - 

install: 
	@echo "Installing..."
	$(POETRY) install
	$(POETRY) run pre-commit install

activate:
	@echo "Activating virtual environment"
	$(POETRY) shell

pull_data:
	$(POETRY) run dvc pull

setup: install_poetry activate

test:
	pytest

docs_view:
	@echo View API documentation... 
	pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	pdoc src -o docs

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache