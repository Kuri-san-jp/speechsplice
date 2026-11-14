#!/usr/bin/env python3
"""CJK N=300 accuracy + quality evaluation.

Same design as eval_eu_quality_accuracy.py:
- N=300 unique pairs per language, seed=42
- BL synthesis + OEOI injection + Whisper ASR + UTMOS22
- Checkpoint/resume support

Languages:
  JA: kanji entity → katakana reading injection (from entities_10000.jsonl)
  ZH: polyphone character, reading_a context → reading_b context injection
  KO: Hangul word pair injection (word_a → word_b)

Usage:
  SLURM_TIME=16:00:00 slurm_run.sh \
    python scripts/projection/eval_cjk_n300.py [--lang ja|zh|ko|all]
"""
import sys, os, json, gc, time, random, argparse, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)
import torch
import torchaudio

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

# コンテナ内でも TTS26 ローカルの HF キャッシュを使う（他スクリプトと同じ対処）
_hf_home = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SEED = 42
N_ACCURACY = 300
N_QUALITY = 100

# ============================================================
# JA data: load from entities_10000.jsonl
# ============================================================

def load_ja_pairs(n):
    """Load N unique (kanji, katakana) entity pairs for JA."""
    path = os.path.join(PROJECT_ROOT, "data/entities_10000.jsonl")
    entities = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("reading_kata") and e["surface"] != e["reading_kata"]:
                entities.append(e)
    rng = random.Random(SEED)
    rng.shuffle(entities)
    pairs = []
    for e in entities[:n]:
        pairs.append({
            "word_a": e["surface"],      # kanji
            "word_b": e["reading_kata"],  # katakana
            "template_a": "{word}は、日本で広く知られています。",
            "template_b": "{word}は、日本で広く知られています。",
        })
    assert len(pairs) >= n, f"JA: need {n} pairs, got {len(pairs)}"
    return pairs[:n]


# ============================================================
# ZH data: polyphone characters
# ============================================================

import json as _json
ZH_POLYPHONES = _json.load(open(os.path.join(
    PROJECT_ROOT, "data", "zh_polyphones_v2.json")))
assert len(ZH_POLYPHONES) == 25, f"Expected 25, got {len(ZH_POLYPHONES)}"


def load_zh_pairs(n):
    """Load N unique ZH polyphone context pairs.

    Each pair: (context_a_sentence, context_b_sentence, polyphone_char).
    We generate cross-product pairs from ctxs_a × ctxs_b for each character,
    then sample n_per_char pairs per character.
    """
    rng = random.Random(SEED)
    n_chars = len(ZH_POLYPHONES)
    n_per_char = n // n_chars  # 300 // 25 = 12
    remainder = n % n_chars

    pairs = []
    for ci, poly in enumerate(ZH_POLYPHONES):
        char = poly["char"]
        cross = [(a, b) for a in poly["ctxs_a"] for b in poly["ctxs_b"]]  # 25 combos
        rng.shuffle(cross)
        take = n_per_char + (1 if ci < remainder else 0)
        for ctx_a, ctx_b in cross[:take]:
            pairs.append({
                "word_a": char,  # the polyphone character (entity in context_a)
                "word_b": char,  # same character, different reading in context_b
                "text_a": ctx_a,
                "text_b": ctx_b,
            })
    rng.shuffle(pairs)
    assert len(pairs) == n, f"ZH: expected {n}, got {len(pairs)}"
    return pairs


# ============================================================
# KO data: Hangul word pairs (from eval_ko_hangul_n1000.py)
# ============================================================

KO_WORDS_A = [
    "사람", "뛰어가기", "나라", "물", "바람", "달", "해", "구름", "비", "눈",
    "안개", "서리", "우박", "번개", "무지개", "산", "강", "바다", "호수", "섬",
    "숲", "초원", "사막", "계곡", "폭포", "동굴", "절벽", "언덕", "평야", "해안",
    "밤", "아침", "저녁", "새벽", "황혼", "자정", "정오", "오후", "오전", "일출",
    "일몰", "봄", "여름", "가을", "겨울", "꽃", "나무", "풀", "잎", "뿌리",
    "열매", "씨앗", "줄기", "가지", "꽃잎", "이슬", "파도", "조류", "해류", "안개",
    "소나기", "태풍", "가뭄", "홍수", "지진", "화산", "토양", "암석", "모래", "진흙",
    "돌", "쇠", "금", "은", "구리", "사과", "배", "감", "밤", "포도",
    "딸기", "수박", "참외", "복숭아", "자두", "살구", "앵두", "석류", "귤", "레몬",
    "고양이", "강아지", "토끼", "거북이", "다람쥐", "참새", "까치", "비둘기", "독수리", "백조",
]
KO_WORDS_B = [
    "과학", "역사", "수학", "경제", "정치", "종교", "도서관", "박물관", "미술관", "편의점",
    "주유소", "병원", "약국", "학교", "대학", "유치원", "경찰서", "소방서", "우체국", "은행",
    "시장", "공원", "극장", "체육관", "수영장", "도로", "철도", "항구", "공항", "터널",
    "교량", "댐", "탑", "성", "궁전", "사찰", "교회", "모스크", "성당", "신사",
    "농업", "어업", "공업", "상업", "무역", "관광", "교육", "의료", "법률", "건축",
    "음악", "미술", "문학", "영화", "연극", "무용", "사진", "조각", "도자기", "서예",
    "축구", "야구", "농구", "배구", "테니스", "탁구", "골프", "수영", "달리기", "체조",
    "컴퓨터", "인터넷", "스마트폰", "텔레비전", "라디오", "신문", "잡지", "편지", "이메일", "전화",
    "지도자", "철학자", "과학자", "예술가", "음악가", "작가", "시인", "화가", "건축가", "의사",
    "선생님", "간호사", "요리사", "농부", "어부", "군인", "경찰관", "소방관", "기자", "변호사",
]

KO_TEMPLATE = "{word}는 한국에서 흔히 사용됩니다."


def load_ko_pairs(n):
    """Load N unique Hangul word pairs."""
    rng = random.Random(SEED)
    pairs = []
    used = set()
    # Sequential
    for i in range(min(n, len(KO_WORDS_A), len(KO_WORDS_B))):
        a, b = KO_WORDS_A[i], KO_WORDS_B[i]
        if a != b and (a, b) not in used:
            pairs.append({"word_a": a, "word_b": b})
            used.add((a, b))
    # Random fill
    while len(pairs) < n:
        a = rng.choice(KO_WORDS_A)
        b = rng.choice(KO_WORDS_B)
        if a != b and (a, b) not in used:
            pairs.append({"word_a": a, "word_b": b})
            used.add((a, b))
    return pairs[:n]


# ============================================================
# Shared functions
# ============================================================

def load_whisper(device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    model_name = "openai/whisper-large-v3"
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
    repo_dir = os.path.join(hub, f"models--{model_name.replace('/', '--')}", "snapshots")
    if os.path.isdir(repo_dir):
        snaps = sorted(os.listdir(repo_dir))
        if snaps:
            model_name = os.path.join(repo_dir, snaps[-1])
    print(f"[Info] Loading Whisper ({model_name})...")
    processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
    wm = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, local_files_only=True).to(device)
    wm.eval()
    return wm, processor


def transcribe(audio_tensor, src_sr, whisper_model, processor, lang):
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze(0)
    if src_sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
    feats = processor(audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = feats.input_features.to(
        whisper_model.device, dtype=next(whisper_model.parameters()).dtype)
    lang_map = {"ja": "japanese", "zh": "chinese", "ko": "korean"}
    forced = processor.get_decoder_prompt_ids(language=lang_map.get(lang, lang), task="transcribe")
    with torch.no_grad():
        pred = whisper_model.generate(input_features, forced_decoder_ids=forced, max_new_tokens=256)
    return processor.batch_decode(pred, skip_special_tokens=True)[0].strip()


def load_utmos(device):
    print("[Info] Loading UTMOS22...")
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    model.eval()
    return model


def compute_utmos(audio_tensor, src_sr, utmos_model, device):
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if src_sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
    with torch.no_grad():
        score = utmos_model(audio_tensor.to(device), 16000)
    return score.item()


def check_word_in_transcript(word, transcript, lang):
    w = word.lower()
    t = transcript.lower()
    return w in t


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CJK N=300 Accuracy + Quality")
    parser.add_argument("--lang", default="all", choices=["ja", "zh", "ko", "all"])
    parser.add_argument("--output_dir", default="eval_results/cjk_n300")
    parser.add_argument("--n_accuracy", type=int, default=N_ACCURACY)
    parser.add_argument("--n_quality", type=int, default=N_QUALITY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_wavs", action="store_true",
                        help="BL/OEOI 音声を output_dir に保存する（既定は保存しない=従来動作）")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    languages = ["ja", "zh", "ko"] if args.lang == "all" else [args.lang]

    print(f"[Config] Languages: {languages}")
    print(f"[Config] N_accuracy={args.n_accuracy}, N_quality={args.n_quality}, Seed={SEED}")

    # Load models
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from data.tokenizer import AudioTokenizer
    from encoder_injection_t5gemma import (
        get_encoder_output, generate_with_modified_encoder,
        find_entity_token_range, seed_everything,
    )

    def _local_snapshot(hub_id):
        # 型番名のままだと通信を試みるライブラリ経路があるため、保存済みの
        # 実体フォルダを直接指す（他スクリプトと同じ対処）
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
        repo_dir = os.path.join(hub, f"models--{hub_id.replace('/', '--')}", "snapshots")
        if os.path.isdir(repo_dir):
            snaps = sorted(os.listdir(repo_dir))
            if snaps:
                return os.path.join(repo_dir, snaps[-1])
        return hub_id

    print("[Info] Loading TTS model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        _local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(_local_snapshot(tok_name), local_files_only=True)
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_sr = audio_tokenizer.sample_rate
    codec_fps = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0

    whisper_model, whisper_proc = load_whisper(device)
    utmos_model = load_utmos(device)

    # --------------------------------------------------------
    # Per-language evaluation
    # --------------------------------------------------------

    for lang in languages:
        print(f"\n{'='*70}")
        print(f"LANGUAGE: {lang.upper()} (N={args.n_accuracy})")
        print(f"{'='*70}")

        # Load pairs
        if lang == "ja":
            raw_pairs = load_ja_pairs(args.n_accuracy)
        elif lang == "zh":
            raw_pairs = load_zh_pairs(args.n_accuracy)
        elif lang == "ko":
            raw_pairs = load_ko_pairs(args.n_accuracy)

        print(f"[Info] {len(raw_pairs)} pairs loaded")

        # Checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_{lang}.json")
        ckpt = load_checkpoint(ckpt_path)
        results = []
        done_indices = set()
        if ckpt:
            results = ckpt.get("results", [])
            done_indices = {r["pair_idx"] for r in results}
            print(f"[Resume] {len(done_indices)} done")

        lang_start = time.time()

        for pair_idx, pair in enumerate(raw_pairs):
            if pair_idx in done_indices:
                continue

            compute_quality = (pair_idx < args.n_quality)
            word_a = pair["word_a"]
            word_b = pair["word_b"]

            result = {
                "pair_idx": pair_idx, "lang": lang,
                "word_a": word_a, "word_b": word_b,
                "status": "pending",
            }

            try:
                # === Determine texts ===
                if lang == "ja":
                    tmpl = pair["template_a"]
                    text_a = tmpl.format(word=word_a)
                    text_b = tmpl.format(word=word_b)
                elif lang == "zh":
                    text_a = pair["text_a"]
                    text_b = pair["text_b"]
                elif lang == "ko":
                    text_a = KO_TEMPLATE.format(word=word_a)
                    text_b = KO_TEMPLATE.format(word=word_b)

                result["text_a"] = text_a
                result["text_b"] = text_b

                # === BL synthesis ===
                seed_everything(SEED)
                enc_a = get_encoder_output(model, text_tokenizer, text_a, device)
                bl_gen = generate_with_modified_encoder(
                    model, text_tokenizer, audio_tokenizer,
                    enc_a["last_hidden_state"], enc_a["attention_mask"],
                    enc_a.get("position_ids"), text_a, device, codec_sr, codec_fps)
                if bl_gen is None:
                    result["status"] = "error"
                    result["error"] = "bl_no_audio"
                    results.append(result)
                    done_indices.add(pair_idx)
                    continue

                bl_audio = bl_gen[0].cpu()
                if args.save_wavs:
                    _w = bl_audio.detach().float()
                    _w = _w if _w.dim() > 1 else _w.unsqueeze(0)
                    torchaudio.save(os.path.join(
                        args.output_dir, f"s{pair_idx:04d}_bl.wav"), _w, codec_sr)
                bl_transcript = transcribe(bl_audio, codec_sr, whisper_model, whisper_proc, lang)

                # BL accuracy: does word_a appear in transcript?
                bl_correct = check_word_in_transcript(word_a, bl_transcript, lang)
                result["bl_transcript"] = bl_transcript
                result["bl_correct"] = bl_correct

                if compute_quality:
                    result["bl_utmos22"] = round(compute_utmos(bl_audio, codec_sr, utmos_model, device), 4)

                del bl_audio, bl_gen

                # === OEOI synthesis ===
                seed_everything(SEED)
                enc_b = get_encoder_output(model, text_tokenizer, text_b, device)

                # Find entity token ranges
                src_range = find_entity_token_range(
                    text_tokenizer, enc_a["text"], word_a, add_bos=add_bos)
                ref_range = find_entity_token_range(
                    text_tokenizer, enc_b["text"], word_b, add_bos=add_bos)

                if src_range is None or ref_range is None:
                    result["status"] = "error"
                    result["error"] = f"token_range_not_found src={src_range} ref={ref_range}"
                    results.append(result)
                    done_indices.add(pair_idx)
                    del enc_a, enc_b
                    continue

                src_start, src_end = src_range
                ref_start, ref_end = ref_range

                # Cosine similarity
                src_h = enc_a["last_hidden_state"][0, src_start:src_end].float().mean(0)
                ref_h = enc_b["last_hidden_state"][0, ref_start:ref_end].float().mean(0)
                cos_sim = torch.nn.functional.cosine_similarity(
                    src_h.unsqueeze(0), ref_h.unsqueeze(0), dim=-1).item()
                result["cos_sim"] = round(cos_sim, 4)

                # Splice
                modified = enc_a["last_hidden_state"].clone()
                ref_entity = enc_b["last_hidden_state"][0, ref_start:ref_end]
                src_len = src_end - src_start
                ref_len = ref_end - ref_start

                if src_len == ref_len:
                    modified[0, src_start:src_end] = ref_entity
                    mod_attn = enc_a["attention_mask"]
                else:
                    prefix = modified[0, :src_start]
                    suffix = modified[0, src_end:]
                    modified = torch.cat([prefix, ref_entity, suffix], dim=0).unsqueeze(0)
                    new_len = modified.shape[1]
                    mod_attn = torch.ones((1, new_len), dtype=torch.long, device=device)

                pos_ids = None
                if int(getattr(model.args, "use_pm_rope", 1)):
                    new_len = modified.shape[1]
                    x_lens = torch.LongTensor([new_len]).to(device)
                    pos_ids = model._build_position_ids(x_lens, new_len, device)

                seed_everything(SEED)
                oeoi_gen = generate_with_modified_encoder(
                    model, text_tokenizer, audio_tokenizer,
                    modified, mod_attn, pos_ids,
                    text_a, device, codec_sr, codec_fps)

                if oeoi_gen is None:
                    result["status"] = "error"
                    result["error"] = "oeoi_no_audio"
                    results.append(result)
                    done_indices.add(pair_idx)
                    del enc_a, enc_b, modified
                    continue

                oeoi_audio = oeoi_gen[0].cpu()
                if args.save_wavs:
                    _w = oeoi_audio.detach().float()
                    _w = _w if _w.dim() > 1 else _w.unsqueeze(0)
                    torchaudio.save(os.path.join(
                        args.output_dir, f"s{pair_idx:04d}_oeoi.wav"), _w, codec_sr)
                oeoi_transcript = transcribe(oeoi_audio, codec_sr, whisper_model, whisper_proc, lang)

                # OEOI accuracy: does word_b appear in OEOI transcript?
                oeoi_correct = check_word_in_transcript(word_b, oeoi_transcript, lang)
                result["oeoi_transcript"] = oeoi_transcript
                result["oeoi_correct"] = oeoi_correct
                result["changed"] = (bl_transcript != oeoi_transcript)

                # ASR gating
                if bl_correct:
                    gated_correct = check_word_in_transcript(word_b, bl_transcript, lang)
                    result["gated_source"] = "bl"
                else:
                    gated_correct = oeoi_correct
                    result["gated_source"] = "oeoi"
                result["gated_correct"] = gated_correct

                if compute_quality:
                    result["oeoi_utmos22"] = round(
                        compute_utmos(oeoi_audio, codec_sr, utmos_model, device), 4)

                del oeoi_audio, oeoi_gen, enc_a, enc_b, modified
                result["status"] = "success"

            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                print(f"  [{pair_idx}] ERROR: {e}")
                traceback.print_exc()

            results.append(result)
            done_indices.add(pair_idx)

            # Progress
            n_done = len(done_indices)
            elapsed = time.time() - lang_start
            if n_done % 10 == 0 or n_done <= 5:
                rate = n_done / max(elapsed, 1) * 3600
                print(f"  [{n_done}/{len(raw_pairs)}] {lang} "
                      f"a={word_a} b={word_b} "
                      f"BL={result.get('bl_correct','?')} "
                      f"OEOI={result.get('oeoi_correct','?')} "
                      f"cos={result.get('cos_sim','?')} "
                      f"({elapsed/60:.1f}min, {rate:.0f}/h)")

            if n_done % 20 == 0:
                save_checkpoint(ckpt_path, {"results": results})

            gc.collect()
            torch.cuda.empty_cache()

        # Final checkpoint
        save_checkpoint(ckpt_path, {"results": results})

        # Summary
        ok = [r for r in results if r["status"] == "success"]
        n_ok = len(ok)
        if n_ok > 0:
            bl_acc = sum(1 for r in ok if r.get("bl_correct")) / n_ok * 100
            oeoi_acc = sum(1 for r in ok if r.get("oeoi_correct")) / n_ok * 100
            gated_acc = sum(1 for r in ok if r.get("gated_correct")) / n_ok * 100
            changed = sum(1 for r in ok if r.get("changed")) / n_ok * 100
            cos_vals = [r["cos_sim"] for r in ok if "cos_sim" in r]
            cos_mean = sum(cos_vals) / len(cos_vals) if cos_vals else 0

            quality = [r for r in ok if "bl_utmos22" in r]
            bl_utmos = sum(r["bl_utmos22"] for r in quality) / len(quality) if quality else 0
            oeoi_utmos = sum(r["oeoi_utmos22"] for r in quality) / len(quality) if quality else 0

            print(f"\n--- {lang.upper()} SUMMARY (N={n_ok}) ---")
            print(f"  BL accuracy:    {bl_acc:.2f}%")
            print(f"  OEOI accuracy:  {oeoi_acc:.2f}%")
            print(f"  Gated accuracy: {gated_acc:.2f}%")
            print(f"  Changed:        {changed:.1f}%")
            print(f"  Cos sim:        {cos_mean:.4f}")
            print(f"  BL UTMOS22:     {bl_utmos:.4f} (N={len(quality)})")
            print(f"  OEOI UTMOS22:   {oeoi_utmos:.4f} (N={len(quality)})")
        else:
            print(f"\n--- {lang.upper()} SUMMARY: No successful results ---")

        elapsed = time.time() - lang_start
        print(f"  Time: {elapsed/60:.1f} min")

    print(f"\nTotal time: {(time.time() - lang_start)/60:.1f} min" if languages else "")
    print("DONE")


if __name__ == "__main__":
    main()
