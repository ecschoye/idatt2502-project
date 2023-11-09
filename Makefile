VENV_NAME = venv
REQUIREMENTS_FILE = requirements.txt
PYTHON ?= python3
DARWIN = Darwin
ifeq ($(OS),Windows_NT)
    VENV_PATH = $(VENV_NAME)\Scripts
    RMDIR = rmdir /s /q
    PIP = pip
else
    UNAME_S := $(shell uname -s)
    VENV_PATH = source $(VENV_NAME)/bin/activate
    RMDIR = rm -rf
    PIP = pip3
endif

ifeq ($(UNAME_S),$(DARWIN))
    TORCH_INSTALL_CMD := $(PIP) install torch torchvision torchaudio
else
    TORCH_INSTALL_CMD := $(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endif

.PHONY: setup
setup: # Makes virtual envoirement and installs requirements
	$(PYTHON) -m venv $(VENV_NAME)
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	$(PIP) install -r $(REQUIREMENTS_FILE) && \
	$(TORCH_INSTALL_CMD)
else
	$(VENV_PATH) && \
	$(PIP) install -r $(REQUIREMENTS_FILE) && \
	$(TORCH_INSTALL_CMD)
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
