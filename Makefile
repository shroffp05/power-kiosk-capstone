.PHONY: install
PY3 := $(shell command -v python3 2> /dev/null)
CONDA := $(shell conda info --base)

SITE_PACKAGES := $(shell pip show pip | grep '^Location' | cut -f2 -d':')

conda_create:
	@echo "Checking if python3 is installed"
	@if [ -z $(PY3) ]; then echo "Python 3 could not be found."; exit 2; fi
	@echo "Creating a conda environment"
	conda create --name power_kiosk python=3.9 pip

install: $(SITE_PACKAGES) requirements.txt
	@echo "Installing all packages"
	conda install -c conda-forge -c pytorch u8darts-all
	$(CONDA)/envs/power_kiosk/bin/pip install -r requirements.txt 

setup_git:
	@echo "Setting up git"
	git init 
	pre-commit install

pull_data:
	dvc pull

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