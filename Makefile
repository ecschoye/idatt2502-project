VENV_NAME = venv
REQUIREMENTS_FILE = requirements.txt
PYTHON ?= python3
MAC = Darwin
ifeq ($(OS),Windows_NT)
    VENV_PATH = .\$(VENV_NAME)\Scripts
    ACTIVATE_VENV = $(VENV_PATH)\activate
    RMDIR = rmdir /s /q
    PIP = pip
else
    UNAME_S := $(shell uname -s)
    VENV_PATH = $(VENV_NAME)/bin/activate
    ACTIVATE_VENV = . $(VENV_PATH)
    RMDIR = rm -rf
    PIP = pip3
endif

ifeq ($(UNAME_S),$(MAC))
    ACTIVATE_VENV = source $(VENV_PATH)
    TORCH_INSTALL_CMD := $(PIP) install torch torchvision torchaudio
else
    TORCH_INSTALL_CMD := $(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endif

.PHONY: setup
setup: ## Makes virtual environment and installs requirements
	$(PYTHON) -m venv $(VENV_NAME)
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	$(PIP) install -r $(REQUIREMENTS_FILE) && \
	$(TORCH_INSTALL_CMD)
else
	$(ACTIVATE_VENV) && \
	$(PIP) install -r $(REQUIREMENTS_FILE) && \
	$(TORCH_INSTALL_CMD)
endif

.PHONY: clean
clean: ## Removes Virtual environment
ifeq ($(OS),Windows_NT)
	@if exist "$(VENV_NAME)" ($(RMDIR) $(VENV_NAME))
	@if exist "src\__pycache__" (cd src && $(RMDIR) __pycache__)
else
	@if [ -d "$(VENV_NAME)" ]; then \
		$(RMDIR) $(VENV_NAME); \
	fi
	@if [ -d "src/__pycache__" ]; then \
		cd src && $(RMDIR) __pycache__; \
	fi
endif


.PHONY: ddqn
ddqn: ## To train ddqn. Use args "--log" to log graphs of training, and "--log-model" to push the model to Neptune. They can be used together.
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py ddqn ${args} && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py ddqn ${args} && \
	deactivate
endif

.PHONY: ppo
ppo: ## To train ppo
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py ppo ${args} && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py ppo ${args} && \
	deactivate
endif

.PHONY: render-ddqn
render-ddqn: ## To render trained ddqn
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py render-ddqn && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py render-ddqn && \
	deactivate
endif

.PHONY: log-ddqn
log-ddqn: ## To log trained ddqn model
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py log-ddqn && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py log-ddqn && \
	deactivate
endif

.PHONY: render-ppo
render-ppo: ## To render trained ppo
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py render-ppo && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py render-ppo && \
	deactivate
endif

.PHONY: log-ppo
log-ppo: ## To log trained ppo model
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py log-ppo && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	cd src && \
	$(PYTHON) main.py log-ppo && \
	deactivate
endif

.PHONY: format
format: black isort ## Format code and imports

.PHONY: check
check: ## Check formatting, imports, and linting
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	black --check src && \
	isort --check-only src && \
	flake8 --max-line-length 88 src && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	black --check src && \
	isort --check-only src && \
	flake8 --max-line-length 88 src && \
	deactivate
endif

.PHONY: black
black: ## Format code only
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	black src && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	black src && \
	deactivate
endif

.PHONY: isort
isort: ## Format imports only
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	isort src && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	isort src && \
	deactivate
endif

.PHONY: flake8
flake8: ## Check code style
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	flake8 --max-line-length 88 src && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	flake8 --max-line-length 88 src && \
	deactivate
endif

.PHONY: help
help: ## List all available make commands
ifeq ($(OS),Windows_NT)
	$(ACTIVATE_VENV) && \
	$(PYTHON) list_make_commands.py && \
	deactivate
else
	$(ACTIVATE_VENV) && \
	$(PYTHON) list_make_commands.py && \
	deactivate
endif