#!/usr/bin/env bash
set -euo pipefail

echo "=== TruFor Detection Server Setup ==="

# ── Parse flags ──────────────────────────────────────────────────────
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
    esac
done

# ── Check for mamba/conda ────────────────────────────────────────────
if command -v mamba &> /dev/null; then
    CONDA=mamba
elif command -v conda &> /dev/null; then
    CONDA=conda
else
    echo "ERROR: Neither mamba nor conda found. Install Miniforge first."
    echo "  See: https://github.com/conda-forge/miniforge"
    exit 1
fi
echo "Using: ${CONDA}"

ENV_NAME="trufor_server"

# Helper: run a command inside the conda environment
run_in_env() {
    conda run -n "${ENV_NAME}" --live-stream "$@" 2>/dev/null \
        || conda run -n "${ENV_NAME}" "$@"
}

# ── Remove existing env if --clean ───────────────────────────────────
if $CLEAN; then
    echo ""
    echo "--- --clean: removing existing environment ---"
    ${CONDA} env remove -n "${ENV_NAME}" -y 2>/dev/null || true
fi

# ── Create or update environment from environment.yaml ───────────────
echo ""
if ${CONDA} env list | grep -q "^${ENV_NAME} "; then
    echo "--- Updating environment '${ENV_NAME}' ---"
    ${CONDA} env update -n "${ENV_NAME}" -f environment.yaml --prune
else
    echo "--- Creating environment '${ENV_NAME}' ---"
    ${CONDA} env create -f environment.yaml
fi

# ── Verify torch + CUDA ─────────────────────────────────────────────
echo ""
echo "--- Checking GPU ---"
run_in_env python -c "
import torch, numpy
print(f'numpy:  {numpy.__version__}')
print(f'torch:  {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: No CUDA GPU detected.')
"

# ── Clone TruFor repo ───────────────────────────────────────────────
echo ""
echo "--- Setting up TruFor model code ---"
if [ ! -d "TruFor" ]; then
    git clone https://github.com/grip-unina/TruFor.git
else
    echo "TruFor repo already cloned."
fi

# ── Download model weights ───────────────────────────────────────────
echo ""
echo "--- Downloading TruFor weights ---"
echo "This will download ~200MB of model weights."
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p weights
    WEIGHTS_ZIP="weights/TruFor_weights.zip"
    if [ ! -f "weights/trufor.pth.tar" ]; then
        curl -L -o "${WEIGHTS_ZIP}" "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
        unzip -o "${WEIGHTS_ZIP}" -d weights/
        rm -f "${WEIGHTS_ZIP}"
        echo "Weights extracted to weights/"
    else
        echo "Weights already downloaded."
    fi
else
    echo "Skipped weight download. Download manually:"
    echo "  curl -L -o weights/TruFor_weights.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
    echo "  unzip weights/TruFor_weights.zip -d weights/"
fi

mkdir -p weights

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the server:"
echo "  conda activate ${ENV_NAME}"
echo "  export PYTHONPATH=\"\${PWD}/TruFor/test_docker/src:\${PYTHONPATH:-}\""
echo "  uvicorn server:app --host 0.0.0.0 --port 8000"
