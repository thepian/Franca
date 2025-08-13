UV := uv
PY := $(UV) run -q python

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  uv           - install deps via uv sync"
	@echo "  convert      - download checkpoint and export Core ML"
	@echo "  validate     - compare PyTorch vs Core ML outputs"
	@echo "  compile      - compile .mlmodel to .mlmodelc"
	@echo "  clean        - remove artifacts"

uv:
	$(UV) sync

convert: uv
	$(PY) scripts/download_checkpoint.py
	$(PY) scripts/export_coreml.py

validate: uv
	$(PY) scripts/validate_conversion.py

compile:
	@mkdir -p coreml_modelsc
	xcrun coremlcompiler compile coreml_models/franca_fp16.mlmodel coreml_modelsc/

clean:
	rm -rf coreml_models coreml_modelsc checkpoints
