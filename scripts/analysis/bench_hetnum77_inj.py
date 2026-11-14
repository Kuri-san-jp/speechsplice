#!/usr/bin/env python3
"""furigana判定で誤読となったbench項目のみ、正読カナをencoder注入してINJ音声を生成.

対象: eval_results/bench_hetnum77_gen/furigana_verify.json の label=="wrong" 項目。
注入は eval_utmos_speaker_fixed.py と同一実装（entity範囲のencoder出力置換、
長さ不一致は adaptive_avg_pool1d、JSUT参照＋Whisper prefix、seed=42）。
出力: eval_results/bench_hetnum77_gen/s{idx:04d}_inj.wav（対象20本のみ）

Usage:
  SLURM_TIME=01:00:00 slurm_run.sh \
      python scripts/analysis/bench_hetnum77_inj.py
"""
import argparse
import json
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--script", default="kata", choices=["kata", "hira"],
                 help="注入する読みの表記（kata=カタカナ / hira=ひらがな）")
_ap.add_argument("--module", default="hetnum77", choices=["hetnum77", "ext83"],
                 help="対象セット: hetnum77=77項目（既定・従来動作） / ext83=拡張83項目")
_ap.add_argument("--align", default="splice_clone",
                 choices=["pool", "splice", "splice_clone", "nn"],
                 help="注入のアライメント方式。splice_clone=正式実装（2026-07-23〜、"
                      "クローン経路・系列長ごと差し替え・duration=text_a 固定） / "
                      "pool=旧実装（adaptive pooling圧縮、比較用） / "
                      "splice=plain経路の系列長差し替え（crossculture実装、比較用） / "
                      "nn=nearest-neighbor index mapping（長さ不変、比較用）")
_cli = _ap.parse_args()

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f:
                os.environ.setdefault("HF_TOKEN", _f.read().strip())
            break
        except Exception:
            pass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch

GEN_DIR = os.path.join(PROJECT_ROOT, "eval_results",
                       "bench_hetnum77_gen" if _cli.module == "hetnum77"
                       else "bench_ext83_gen")


def _run_splice(targets, suffix):
    """crossculture実装（eval_ja_crossculture_t5gemma_only.py）と同一のsplicing注入.

    plain合成経路（クローンなし・zero-length prompt）。長さ不一致時は
    prefix + target読み表現(全長) + suffix を連結し、attention mask と
    pm-RoPE position ids を新しい系列長で再構築する。
    """
    from ja_numerals_v2_verify import local_snapshot
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from data.tokenizer import AudioTokenizer
    from encoder_injection_t5gemma import (
        seed_everything, get_encoder_output, generate_with_modified_encoder,
        find_entity_token_range,
    )
    from inference_tts_utils import save_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tok = AutoTokenizer.from_pretrained(local_snapshot(tok_name), local_files_only=True)
    audio_tok = AudioTokenizer(backend="xcodec2",
                               model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0

    done = 0
    for i, p in targets:
        out_fn = os.path.join(GEN_DIR, f"s{i:04d}_{suffix}.wav")
        # 空ファイル（過去の失敗残骸）は未生成扱いで上書き
        if os.path.exists(out_fn) and os.path.getsize(out_fn) > 1000:
            done += 1
            continue
        seed_everything(42)
        try:
            src_enc = get_encoder_output(model, text_tok, p["text_a"], device, lang="ja")
            ref_enc = get_encoder_output(model, text_tok, p["text_b"], device, lang="ja")
            s_r = find_entity_token_range(text_tok, src_enc["text"], p["word_a"], add_bos=add_bos)
            r_r = find_entity_token_range(text_tok, ref_enc["text"], p["word_b"], add_bos=add_bos)
            if s_r is None or r_r is None:
                print(f"  [{i}] entity range None, skip", flush=True)
                continue
            s_s, s_e = s_r
            r_s, r_e = r_r
            modified = src_enc["last_hidden_state"].clone()
            if (s_e - s_s) == (r_e - r_s):
                modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
            else:
                prefix = modified[0, :s_s]
                ref_ent = ref_enc["last_hidden_state"][0, r_s:r_e]
                suffix_t = modified[0, s_e:]
                modified = torch.cat([prefix, ref_ent, suffix_t], dim=0).unsqueeze(0)
            attn = torch.ones((1, modified.shape[1]), dtype=torch.long, device=device)
            pos_ids = None
            if int(getattr(model.args, "use_pm_rope", 1)):
                x_lens = torch.LongTensor([modified.shape[1]]).to(device)
                pos_ids = model._build_position_ids(x_lens, modified.shape[1], device)
            audio = generate_with_modified_encoder(
                model, text_tok, audio_tok, modified, attn, pos_ids,
                p["text_a"], device, codec_audio_sr, codec_sr)
            audio = audio.detach().float().cpu()
            if audio.dim() == 3:      # [1, 1, T] -> [1, T]（crossculture実装の[0]と同じ）
                audio = audio[0]
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)
            save_audio(out_fn, audio, codec_audio_sr)
            done += 1
            print(f"  [{i}] {p['word_a']}→{p['word_b']} spliced (Δtok={ (r_e-r_s)-(s_e-s_s):+d}) saved",
                  flush=True)
        except Exception as e:
            print(f"  [{i}] fail: {str(e)[:120]}", flush=True)
    print(f"[Done] splice INJ {done}/{len(targets)}", flush=True)


def main():
    from ja_numerals_v2_verify import local_snapshot
    import eval_utmos_speaker_fixed as ev
    if _cli.module == "hetnum77":
        from bench_hetnum77 import build_pairs
    else:
        from bench_ext83 import build_pairs

    verify = json.load(open(os.path.join(GEN_DIR, "furigana_verify.json")))
    wrong_idx = [r["idx"] for r in verify["rows"] if r["label"] == "wrong"]
    pairs = build_pairs()

    def to_hira(s):
        return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
                       for c in s)

    if _cli.script == "hira":
        # word_b をひらがなに変換し text_b を作り直す
        for p in pairs:
            p["word_b"] = to_hira(p["word_b"])
            p["text_b"] = p["text_a"].replace(p["word_a"], p["word_b"], 1)
    SUFFIX = {"kata": "inj", "hira": "injh"}[_cli.script]
    if _cli.align == "splice":
        SUFFIX += "s"
    elif _cli.align == "splice_clone":
        SUFFIX += "sc"
    elif _cli.align == "nn":
        SUFFIX += "nn"
    targets = [(i, pairs[i]) for i in wrong_idx]
    print(f"[Info] 誤読 {len(targets)} 件に正読を注入（表記={_cli.script}, align={_cli.align}, "
          f"出力=_{SUFFIX}.wav）", flush=True)

    if _cli.align == "splice":
        _run_splice(targets, SUFFIX)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from data.tokenizer import AudioTokenizer
    from inference_tts_utils import save_audio, transcribe_audio

    ev.seed_all(ev.SEED)
    ref_wav = ev.DEFAULT_REF
    ref_prefix = transcribe_audio(ref_wav, device)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tok = AutoTokenizer.from_pretrained(local_snapshot(tok_name), local_files_only=True)
    audio_tok = AudioTokenizer(backend="xcodec2",
                               model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    done = 0
    for i, p in targets:
        out_fn = os.path.join(GEN_DIR, f"s{i:04d}_{SUFFIX}.wav")
        # 空ファイル（過去の失敗残骸）は未生成扱い（_run_spliceと同一ガード）
        if os.path.exists(out_fn) and os.path.getsize(out_fn) > 1000:
            done += 1
            continue
        ev.seed_all(ev.SEED)
        try:
            src_enc = ev.get_encoder_output_with_prefix(
                model, text_tok, ref_prefix, p["text_a"], device, lang="ja")
            ref_enc = ev.get_encoder_output_with_prefix(
                model, text_tok, ref_prefix, p["text_b"], device, lang="ja")
            s_sub = ev.find_entity_token_range(text_tok, src_enc["target_text"], p["word_a"], add_bos=0)
            r_sub = ev.find_entity_token_range(text_tok, ref_enc["target_text"], p["word_b"], add_bos=0)
            if s_sub is None or r_sub is None:
                print(f"  [{i}] entity range None ({p['word_a']}→{p['word_b']}), skip", flush=True)
                continue
            s_s = s_sub[0] + src_enc["target_start_in_full"]
            s_e = s_sub[1] + src_enc["target_start_in_full"]
            r_s = r_sub[0] + ref_enc["target_start_in_full"]
            r_e = r_sub[1] + ref_enc["target_start_in_full"]
            inj_target = p["text_a"]
            inj_dur_text = None
            if _cli.align == "splice_clone":
                from splice_clone import prepare_clone_splice
                modified, _inj_t, inj_dur_text, delta_tok = prepare_clone_splice(
                    text_tok, src_enc, ref_enc, (s_s, s_e), (r_s, r_e), "ja",
                    duration_source_text=p["text_a"],
                    natural_candidates=(p["text_b"],))
                if _inj_t is not None:
                    inj_target = _inj_t
            elif _cli.align == "nn":
                # nearest-neighbor index mapping（長さ不変・クローン経路）:
                # sourceスパンの各位置 j に、refスパン内の最近傍インデックスの行を割当
                modified = src_enc["last_hidden_state"].clone()
                s_len = s_e - s_s
                r_len = r_e - r_s
                if s_len == r_len:
                    modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
                else:
                    if s_len == 1:
                        nn_idx = torch.tensor([round((r_len - 1) / 2)])
                    else:
                        nn_idx = torch.round(
                            torch.linspace(0, r_len - 1, steps=s_len)).long()
                    modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e][nn_idx]
            else:
                modified = src_enc["last_hidden_state"].clone()
                s_len = s_e - s_s
                r_len = r_e - r_s
                if s_len == r_len:
                    modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
                else:
                    import torch.nn.functional as F
                    ref_seg = ref_enc["last_hidden_state"][0, r_s:r_e].unsqueeze(0).transpose(1, 2)
                    ref_resampled = F.adaptive_avg_pool1d(ref_seg.float(), s_len).transpose(1, 2)
                    modified[0, s_s:s_e] = ref_resampled[0].to(modified.dtype)

            inj = ev.voice_clone_synth(
                model, text_tok, audio_tok,
                target_text=inj_target, lang="ja", device=device,
                codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                ref_wav=ref_wav, ref_text=ref_prefix,
                ref_prompt_sec=100.0, modified_hidden=modified,
                duration_text=inj_dur_text)
            save_audio(out_fn, inj if inj.dim() > 1 else inj.unsqueeze(0), codec_audio_sr)
            done += 1
            print(f"  [{i}] {p['word_a']}→{p['word_b']} saved", flush=True)
        except Exception as e:
            print(f"  [{i}] fail: {str(e)[:120]}", flush=True)
    print(f"[Done] INJ {done}/{len(targets)}", flush=True)


if __name__ == "__main__":
    main()
