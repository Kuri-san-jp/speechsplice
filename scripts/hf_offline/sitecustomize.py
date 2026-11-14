"""Run the generation scripts with HF_HUB_OFFLINE=1 (models already in the local cache).

Put this directory on PYTHONPATH:  PYTHONPATH=scripts/hf_offline python3 scripts/...

Why: when transformers 4.57.3 loads a tokenizer by repo id, it asks the Hub for the
repo's tags (PreTrainedTokenizerBase._patch_mistral_regex -> huggingface_hub.model_info)
to decide whether a Mistral regex fix applies. Under HF_HUB_OFFLINE=1 that call raises
OfflineModeIsEnabled, so T5Gemma-TTS (google/t5gemma-2b-2b-ul2 tokenizer) and
Qwen3-TTS stop. The experiments ran online, where the call returned tags without
"mistralai" and the tokenizer was left unchanged.

What it does, only when HF_HUB_OFFLINE is set: huggingface_hub.model_info returns a
record with no tags instead of raising OfflineModeIsEnabled, which gives the same
answer as online for the non-Mistral tokenizers used here. Every other offline error is
unchanged, and transformers itself is not imported here (scripts/longcat/longcat_splice.py
must patch transformers before importing it).
"""
import os
import types


def _offline():
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")


if _offline():
    import huggingface_hub
    from huggingface_hub.errors import OfflineModeIsEnabled

    _model_info = huggingface_hub.model_info

    def model_info(repo_id, *args, **kwargs):
        try:
            return _model_info(repo_id, *args, **kwargs)
        except OfflineModeIsEnabled:
            return types.SimpleNamespace(id=repo_id, tags=None)

    huggingface_hub.model_info = model_info
