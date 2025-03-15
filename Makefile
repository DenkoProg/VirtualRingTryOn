.PHONY: setup install clean run-image run-video

# Default Python interpreter
PYTHON = python

# Virtual environment directory
VENV = .venv

# Main script
MAIN_SCRIPT = src/cli.py

# Default paths
IMAGES_FOLDER = data/images
LANDMARKS_FOLDER = data/results
RGB_PATH = data/images/original_1.png
DEPTH_LOG_PATH = data/depth_logs/depth_logs_1.txt
RING_MODEL_PATH = data/models/ring.glb

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip

install:
	. $(VENV)/bin/activate && pip install -r requirements.txt

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-image:
	. $(VENV)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) render-ring-on-image \
		--images-folder "$(IMAGES_FOLDER)" \
		--landmarks-folder "$(LANDMARKS_FOLDER)" \
		--rgb-path "$(RGB_PATH)" \
		--depth-log-path "$(DEPTH_LOG_PATH)" \
		--ring-model-path "$(RING_MODEL_PATH)"

run-video:
	. $(VENV)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) render-ring-on-video \
		--images-folder "$(IMAGES_FOLDER)" \
		--landmarks-folder "$(LANDMARKS_FOLDER)" \
		--rgb-path "$(RGB_PATH)" \
		--depth-log-path "$(DEPTH_LOG_PATH)" \
		--ring-model-path "$(RING_MODEL_PATH)"

# Help target
help:
	@echo "Available targets:"
	@echo "  setup    - Create virtual environment"
	@echo "  install  - Install dependencies"
	@echo "  clean    - Clean up virtual environment and cache files"
	@echo "  run-image - Run ring rendering on an image"
	@echo "  run-video - Run ring rendering on a video"
	@echo "  help     - Show this help message"