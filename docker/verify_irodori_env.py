#!/usr/bin/env python3
"""Irodori 用イメージの自己検証。

Dockerfile.irodori のビルド最終段で走る（ここで落ちればイメージは作られない）。
コンテナの中で後から診断用に単体実行してもよい:

    IMAGE_NAME=irodori-tts:latest NUM_GPUS=0 SLURM_TIME=00:10:00 \\
      slurm_run.sh python3 scripts/irodori/verify_irodori_env.py

確認すること
  1. torch が 2.10 系のまま、かつ CUDA 付きであること
     （dacvae の install_requires に torch があるので、依存解決の拍子に
       PyPI の CPU 版へ差し替わる事故が起きうる。それを必ず検出する）
  2. transformers が 5.x であること
  3. 4.57.3 で ModuleNotFoundError だった transformers.initialization.no_init_weights
     が import できること（model.py:691 が要求する）
  4. ModernBERT の実装が layer_types / rope_parameters を **実際に参照** すること。
     4.57.3 はこれらを未知 kwargs として黙って飲むだけで参照せず、
     global_rope_theta / local_rope_theta を見る。落ちずに別の RoPE 設定で
     動いてしまうので、属性の有無ではなく実装側の参照を確認する。
  5. dacvae が import できること（codec.py:60 が遅延 import する）
  6. silentcipher が **入っていない** こと。入っていると生成音声に電子透かしの
     摂動が載り、MCD の比較が汚れる（watermark.py:35-49）。
"""
import inspect
import sys


def main() -> int:
    failures = []

    def check(name, fn):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}" + (f"  [{detail}]" if detail else ""))

    import numpy
    import safetensors
    import soundfile
    import tokenizers
    import torch
    import transformers

    print("versions:")
    print(f"  python       {sys.version.split()[0]}")
    print(f"  torch        {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"  transformers {transformers.__version__}")
    print(f"  tokenizers   {tokenizers.__version__}")
    print(f"  numpy        {numpy.__version__}")
    print(f"  safetensors  {safetensors.__version__}")
    print(f"  soundfile    {soundfile.__version__}")
    print("checks:")

    def _torch_version():
        if not torch.__version__.startswith("2.10"):
            raise AssertionError(
                f"torch is {torch.__version__}, expected 2.10.x. A dependency "
                "resolution probably replaced it (dacvae lists torch in "
                "install_requires; install it with --no-deps).")
        return torch.__version__

    def _torch_cuda():
        if torch.version.cuda is None:
            raise AssertionError(
                "a CPU-only torch build is installed; the cu128 wheel was replaced")
        return f"cuda {torch.version.cuda}"

    def _transformers_major():
        major = transformers.__version__.split(".")[0]
        if major != "5":
            raise AssertionError(
                f"transformers is {transformers.__version__}; Irodori requires >=5.12.1,<6")
        return transformers.__version__

    def _no_init_weights():
        from transformers.initialization import no_init_weights  # noqa: F401
        return "importable"

    def _modernbert_impl():
        from transformers.models.modernbert import modeling_modernbert

        src = inspect.getsource(modeling_modernbert)
        missing = [n for n in ("layer_types", "rope_parameters") if n not in src]
        if missing:
            raise AssertionError(
                f"the installed ModernBERT implementation never references {missing}; "
                "it would silently use different RoPE settings than the checkpoint's "
                "text_encoder_config_json (written by transformers 5.12.1)")
        return "references layer_types and rope_parameters"

    def _dacvae():
        import dacvae
        from dacvae import DACVAE  # noqa: F401

        return f"version {getattr(dacvae, '__version__', 'unknown')}"

    def _no_silentcipher():
        try:
            import silentcipher  # noqa: F401
        except ImportError:
            return "absent as intended; audio will not be watermarked"
        raise AssertionError(
            "silentcipher is installed; it would add a watermark perturbation to "
            "every generated wav and contaminate the MCD comparison")

    check("torch is 2.10.x", _torch_version)
    check("torch has CUDA", _torch_cuda)
    check("transformers is 5.x", _transformers_major)
    check("transformers.initialization.no_init_weights", _no_init_weights)
    check("ModernBERT implements layer_types/rope_parameters", _modernbert_impl)
    check("dacvae importable", _dacvae)
    check("silentcipher absent", _no_silentcipher)

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL ENVIRONMENT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
