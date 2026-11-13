ENV_PREFIX := $(shell if [ "$$CONDA_DEFAULT_ENV" = "s2delta" ]; then echo ""; elif conda env list 2>/dev/null | grep -q "^s2delta "; then echo "conda run -n s2delta --no-capture-output "; fi)

.PHONY: help
help:             ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: env
env:              ## Create or update the conda environment
	conda env update -n s2delta -f environment.yml --prune

.PHONY: show
show:             ## Show Python and key package versions
	@$(ENV_PREFIX)python -c "import sys, platform; print('python', sys.version.split()[0], platform.platform())"
	@$(ENV_PREFIX)python -c "import numpy, pandas, matplotlib, sklearn, rdkit; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib', matplotlib.__version__); print('scikit-learn', sklearn.__version__); print('rdkit', rdkit.__version__)"

.PHONY: clean
clean:            ## Remove Python cache artefacts
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
