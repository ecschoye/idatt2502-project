VENV_NAME = venv
REQUIREMENTS_FILE = requirements.txt
PYTHON ?= python3
ifeq ($(OS),Windows_NT)
    VENV_PATH = $(VENV_NAME)\Scripts
    RMDIR = rmdir /s /q
    PIP = pip
else
    VENV_PATH = source $(VENV_NAME)/bin
    RMDIR = rm -rf
	PIP = pip3
endif

.PHONY: setup
setup: # Makes virtual envoirement and installs requirements
	$(PYTHON) -m venv $(VENV_NAME) && \
	cd $(VENV_PATH) && \
	activate && \
	$(PIP) install -r $(REQUIREMENTS_FILE)
ifeq ($(OS),Windows_NT)
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endif
ifneq ($(OS),Windows_NT)
ifneq ($(findstring Darwin,$(shell uname)),Darwin)
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endif
endif

.PHONY: clean
clean: # Removes Virtual envoirement
	$(RMDIR) $(VENV_NAME)

.PHONY: ddqn
ddqn: # To train ddqn
	cd src && \
    $(PYTHON) main.py ddqn ${args}

.PHONY: ppo
ppo: # To train ppo
	cd src && \
    $(PYTHON) main.py ppo ${args}

.PHONY: render-ddqn
render-ddqn: # To render trained ddqn
	cd src && \
    $(PYTHON) main.py render-ddqn

.PHONY: render-ppo
render-ppo: # To render trained ppo
	cd src && \
    $(PYTHON) main.py render-ddqn

.PHONY: imports
imports: # To fix import issue
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
    $(PIP) install -e .

.PHONY: format
format: ## Format code and imports
	make black
	make isort

.PHONY: check
check: ## Check formatting, imports, and linting
	@black --check src
	@isort --check-only src
	@flake8 --max-line-length 88 src

.PHONY: black
black: ## Format code only
	@black src

.PHONY: isort
isort: ## Format imports only
	@isort src

.PHONY: flake8
flake8: ## Check code style
	@flake8 src
