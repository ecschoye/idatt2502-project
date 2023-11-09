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
setup: ## Makes virtual environment and installs requirements
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
clean: ## Removes Virtual environment
	$(RMDIR) $(VENV_NAME)

.PHONY: ddqn
ddqn: ## To train ddqn
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	cd src && \
	$(PYTHON) main.py ddqn ${args} && \
	deactivate
else
	$(VENV_PATH) && \
	cd src && \
	$(PYTHON) main.py ddqn ${args} && \
	deactivate
endif

.PHONY: ppo
ppo: ## To train ppo
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	cd src && \
	$(PYTHON) main.py ppo ${args} && \
	deactivate
else
	$(VENV_PATH) && \
	cd src && \
	$(PYTHON) main.py ppo ${args} && \
	deactivate
endif
.PHONY: render-ddqn
render-ddqn: ## To render trained ddqn
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	cd src && \
	$(PYTHON) main.py render-ddqn && \
	deactivate
else
	$(VENV_PATH) && \
	cd src && \
	$(PYTHON) main.py render-ddqn && \
	deactivate
endif

.PHONY: render-ppo
render-ppo: ## To render trained ppo
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	cd src && \
	$(PYTHON) main.py render-ppo && \
	deactivate
else
	$(VENV_PATH) && \
	cd src && \
	$(PYTHON) main.py render-ppo && \
	deactivate
endif

.PHONY: format
format: black isort ## Format code and imports

.PHONY: check
check: ## Check formatting, imports, and linting
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	black --check src && \
	isort --check-only src && \
	flake8 --max-line-length 88 src && \
	deactivate
else
	$(VENV_PATH) && \
	black --check src && \
	isort --check-only src && \
	flake8 --max-line-length 88 src && \
	deactivate
endif

.PHONY: black
black: ## Format code only
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	black src && \
	deactivate
else
	$(VENV_PATH) && \
	black src && \
	deactivate
endif

.PHONY: isort
isort: ## Format imports only
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	isort --check src && \
	deactivate
else
	$(VENV_PATH) && \
	isort --check-only src && \
	deactivate
endif

.PHONY: flake8
flake8: ## Check code style
ifeq ($(OS),Windows_NT)
	cd $(VENV_PATH) && \
	activate && \
	cd .. && cd .. && \
	flake8 --max-line-length 88 src && \
	deactivate
else
	$(VENV_PATH) && \
	flake8 src && \
	deactivate
endif

# List all available make commands
help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'