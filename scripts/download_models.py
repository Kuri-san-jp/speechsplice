"""Download every model and dataset the experiments load, at the revisions they used.

The scripts load Hugging Face repos by id from $HF_HOME (default
$REPO_ROOT/.cache/huggingface, i.e. /mnt/.cache/huggingface in the container) and most of
them run with HF_HUB_OFFLINE=1, which resolves a repo id through refs/main. This script
downloads each pinned revision and points refs/main at it, so that offline loading picks
exactly that snapshot.

Revisions are the snapshots that refs/main pointed to in the experiment cache. For five
repos the cache held two snapshots (wav2vec2-xlsr-53-espeak-cv-ft, speecht5_tts,
speecht5_hifigan, wavlm-base-plus-sv, cmu-arctic-xvectors); the one pinned here is the
refs/main one, which is what loading by repo id used.

Gated repos (T5Gemma-TTS, google/t5gemma, CSM-1B, ...) need an accepted license and a
token: export HF_TOKEN=... before running.

Usage:
  python3 scripts/download_models.py                 # everything (about 150 GB)
  python3 scripts/download_models.py --only t5gemma  # a subset by tag (see MODELS)
Also fetches the two non-Hub checkpoints: UTMOS22 strong (torch.hub tarepan/SpeechMOS
v1.2.0) into $TORCH_HOME and openai-whisper medium.en into $REPO_ROOT/.cache/whisper.
"""
import argparse
import os

ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (tag, repo_type, repo_id, revision)
MODELS = [
    ("t5gemma", "model", "Aratako/T5Gemma-TTS-2b-2b", "e548f8358891975e61d2107e3d7ccc47b1b7294e"),
    ("t5gemma", "model", "google/t5gemma-2b-2b-ul2", "f7593b977d03307f371ad1e45d23a4ef83321435"),
    ("t5gemma", "model", "NandemoGHS/Anime-XCodec2-44.1kHz-v2", "58a5080a103abd861052b323279ccabb2333b636"),
    ("t5gemma", "model", "facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b"),
    ("speecht5", "model", "microsoft/speecht5_tts", "30fcde30f19b87502b8435427b5f5068e401d5f6"),
    ("speecht5", "model", "microsoft/speecht5_hifigan", "bb6f429406e86a9992357a972c0698b22043307d"),
    ("speecht5", "dataset", "Matthijs/cmu-arctic-xvectors", "5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5"),
    ("longcat", "model", "meituan-longcat/LongCat-AudioDiT-1B", "4f4c19434f2e3b991998d0ead6ab2cd056d947b7"),
    ("longcat", "model", "google/umt5-base", "0de9394d54f8975e71838d309de1cb496c894ab9"),
    ("irodori", "model", "Aratako/Irodori-TTS-v4.1-Small", "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"),
    ("irodori", "model", "Aratako/Semantic-DACVAE-Japanese-32dim", "47376ee24834d7a05a48ebabfe3cde29b3c5e214"),
    ("irodori", "model", "sbintuitions/modernbert-ja-310m", "77675fc96a7e445e982e2ba90246b816efc74ec6"),
    ("kiseki", "model", "telecomadm1145/Kiseki-TTS", "ceb563aff4a052173d4c11e5b6541be7e9992a45"),
    ("kiseki", "model", "Qwen/Qwen3-TTS-Tokenizer-12Hz", "7dd38ad4e9bad454aae9cd937d0cd577604fe229"),
    ("qwen3", "model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
    ("qwen3", "model", "Qwen/Qwen3-TTS-Tokenizer-12Hz", "7dd38ad4e9bad454aae9cd937d0cd577604fe229"),
    ("xtts", "model", "coqui/XTTS-v2", "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"),
    ("llasa", "model", "HKUSTAudio/Llasa-3B", "740b08c956fdde4a5a33f2caffbf004ff76e2748"),
    ("llasa", "model", "HKUSTAudio/xcodec2", "e412427ed30f0cf9d5e3c95562113deb10a32d03"),
    ("llasa", "model", "facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b"),
    ("csm", "model", "sesame/csm-1b", "c92a71e1c419772e25be7dc14d952c2521a740ab"),
    ("bark", "model", "suno/bark", "70a8a7d34168586dc5d028fa9666aceade177992"),
    ("voicecraft", "model", "pyp1/VoiceCraft_830M_TTSEnhanced", "15ff81999431f57897a945a0d261a561a02ea342"),
    ("voicecraft", "model", "pyp1/VoiceCraft", "52b0f909d652ea016966804767320f13ede39abd"),
    ("scoring", "model", "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    ("scoring", "model", "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    ("scoring", "model", "Parakeet-Inc/furigana_whisper_small_jsut", "5b4002702605b4ce352f88a7fbc51e02b65ccc6b"),
    ("scoring", "model", "facebook/wav2vec2-xlsr-53-espeak-cv-ft", "2c733782da5604684829819a5eb744c193fe9398"),
    ("scoring", "model", "microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"),
    ("scoring", "model", "walston/whisper-pinyin", "7840ab2afafecd7b03aa66c7517d756c180dbb43"),
    ("prompts", "dataset", "yvonne66/BZNSYP_new", "1a97fc04e129ead20f42fdd14d7c61a40e7c8d0b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="tags to fetch (default: all)")
    a = ap.parse_args()
    hf_home = os.environ.get("HF_HOME", os.path.join(ROOT, ".cache", "huggingface"))
    hub = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    from huggingface_hub import snapshot_download
    done = set()
    for tag, rtype, repo, rev in MODELS:
        if a.only and tag not in a.only or (repo, rev) in done:
            continue
        done.add((repo, rev))
        path = snapshot_download(repo_id=repo, repo_type=rtype, revision=rev, cache_dir=hub)
        folder = os.path.join(hub, ("datasets--" if rtype == "dataset" else "models--") + repo.replace("/", "--"))
        os.makedirs(os.path.join(folder, "refs"), exist_ok=True)
        with open(os.path.join(folder, "refs", "main"), "w") as f:
            f.write(rev)
        print(f"[ok] {repo}@{rev[:8]} -> {path}")
    if not a.only or "scoring" in a.only:
        import torch
        os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, ".cache", "torch"))
        torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        import whisper
        whisper.load_model("medium.en", download_root=os.path.join(ROOT, ".cache", "whisper"))
        print("[ok] UTMOS22 strong, openai-whisper medium.en")


if __name__ == "__main__":
    main()
