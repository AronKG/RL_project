.PHONY: all train play analysis mc interface clean

PYTHON ?= python3

all: train play

train:
	@echo "Initiating training..."
	@$(MAKE) analysis
	@$(MAKE) mc
	@echo "Training complete. Agent strategies saved to models directory."

play:
	@echo "Starting interactive interface..."
	@$(PYTHON) interface.py

analysis:
	@echo "Running analysis.py..."
	$(PYTHON) analysis.py
	@echo "Q-learning agent training complete."

mc:
	@echo "Running mc.py..."
	$(PYTHON) mc.py
	@echo "Monte Carlo agent training complete."

clean:
	@echo "No build artifacts to remove."