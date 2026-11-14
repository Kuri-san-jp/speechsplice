# Environment for the generation and scoring scripts. Source it from the repository root:
#   source scripts/env.sh
# Inside the containers of docker/ the repository is mounted at /mnt (as in the original
# runs) and REPO_ROOT defaults to /mnt; outside a container, set REPO_ROOT to the checkout.
export REPO_ROOT="${REPO_ROOT:-/mnt}"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"          # models come from scripts/download_models.py
export TORCH_HOME="$REPO_ROOT/.cache/torch"            # UTMOS22 (torch.hub tarepan/SpeechMOS v1.2.0)
export NUMBA_CACHE_DIR="$REPO_ROOT/.cache/numba"
# Offline loading of the T5Gemma-TTS and Qwen3-TTS tokenizers (see scripts/hf_offline/sitecustomize.py)
export PYTHONPATH="$REPO_ROOT/scripts/hf_offline${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
