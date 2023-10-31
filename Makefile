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