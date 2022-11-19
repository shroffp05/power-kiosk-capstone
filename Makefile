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
	pre-commit install 