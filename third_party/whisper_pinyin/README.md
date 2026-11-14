# whisper-pinyin (vendored subset)

The `whisper/` package of the Whisper-Pinyin repository by Xintong Wang
(`https://github.com/oshindow/whisper_pinyin`, branch `otc`, commit
`bacacb9e7a96d0456e6c82161d5d476eb1e674f2`, MIT License; the package itself is based on
OpenAI Whisper, MIT License). It is copied here unmodified because the upstream
repository is no longer reachable. `scripts/analysis/zh_ctc_lr_judge.py` (the Chinese judge
of Table A10) imports it for audio loading and log-mel features; the model weights are the
Hugging Face checkpoint `walston/whisper-pinyin`. See `LICENSE`, `LICENSES/` and
`THIRD_PARTY_NOTICES.md`.
