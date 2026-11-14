#!/bin/bash
# Fetch the third-party code that the generation scripts import, at the versions the
# experiments used. Run once from the repository root (needs git, pip and network).
#
#   scripts/irodori/_repo         Irodori-TTS inference package          (Table 3, Irodori-TTS)
#   scripts/kiseki/_vendor        qwen-tts 0.1.1 and sox 1.5.0 (--target) (Table 3, Kiseki-TTS codec)
#   scripts/xtts/_ref/src         coqui-tts 0.27.5 wheel, unpacked        (Table 3, XTTS-v2 tokenizer)
#   third_party/EdiTTS            EdiTTS + scripts/editts/edit_content.patch (Table 2)
#   third_party/VoiceCraft        VoiceCraft (also cloned inside docker/Dockerfile.voicecraft) (Table 2)
#
# third_party/longcat_audiodit_survey (LongCat-AudioDiT, Table 3), third_party/whisper_pinyin
# (Table A10, ZH judge) and src/_vendor/t5gemma_tts (T5Gemma-TTS) are shipped in this
# repository; see NOTICE.
set -euo pipefail
cd "$(dirname "$0")/.."

clone_at() {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone "$1" "$2"; fi
  git -C "$2" checkout -q "$3"
  echo "[setup] $2 at $(git -C "$2" rev-parse --short HEAD)"
}

clone_at https://github.com/Aratako/Irodori-TTS.git scripts/irodori/_repo 89f9d8fbd4d51ea019867ee1197725ede1df13c5
clone_at https://github.com/neosapience/EdiTTS.git third_party/EdiTTS 77255e5304f052abe54ac457a6da00a6e9319f87
if ! grep -q -- '--offset' third_party/EdiTTS/edit_content.py; then
  git -C third_party/EdiTTS apply "$PWD/scripts/editts/edit_content.patch"
  echo "[setup] applied scripts/editts/edit_content.patch"
fi
clone_at https://github.com/jasonppy/VoiceCraft.git third_party/VoiceCraft f0a7a971363905883a21c466a3aa4ea2d4c0b690

if [ ! -d scripts/kiseki/_vendor/qwen_tts ]; then
  pip install --no-deps --target scripts/kiseki/_vendor qwen-tts==0.1.1 sox==1.5.0
fi
if [ ! -d scripts/xtts/_ref/src/TTS ]; then
  mkdir -p scripts/xtts/_ref/src
  pip download --no-deps -d scripts/xtts/_ref coqui-tts==0.27.5
  python3 -m zipfile -e scripts/xtts/_ref/coqui_tts-0.27.5-py3-none-any.whl scripts/xtts/_ref/src
fi
echo "[setup] done"
