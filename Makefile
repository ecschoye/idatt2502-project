VENV_NAME = venv
REQUIREMENTS_FILE = requirements.txt
PYTHON ?= python3
ifeq ($(OS),Windows_NT)
    VENV_PATH = $(VENV_NAME)\Scripts\actiavte
    RMDIR = rmdir /s /q
    PIP = pip
else
    VENV_PATH = source $(VENV_NAME)/bin/activate
    RMDIR = rm -rf
	PIP = pip3
endif

.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV_NAME) && \
	$(PIP) install -r $(REQUIREMENTS_FILE)

.PHONY: clean
clean:
	$(RMDIR) $(VENV_NAME)

.PHONY: format
format: ## Format code and imports
	@black src
	@isort src

.PHONY: check
check: ## Check formatting, imports, and linting
	@black --check src
	@isort --check-only src
	@flake8 src

.PHONY: black
black: ## Format code only
	@black src

.PHONY: isort
isort: ## Format imports only
	@isort src

.PHONY: flake8
flake8: ## Check code style
	@flake8 src
