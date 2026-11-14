#!/usr/bin/env python3
"""T5Gemma-TTS: EN cross-culture name injection at an ARBITRARY encoder layer.

査読者 5eZ1 の「最終層のみに注入しており層別 ablation が無い」への数値回答用。

既存スクリプト `scripts/projection/eval_en_crossculture_t5gemma_fixed.py` は
encoder の **最終出力** (= layer 25 の出力に final RMSNorm をかけた last_hidden_state)
に注入する。本スクリプトは注入点を `--inject_layer L` で選べるようにし、
L+1 以降の層を置換後の隠れ状態から走らせて最終出力を作る。

出力レイアウト (既存採点器 scripts/per_fix/phoneme_per_v3.py がそのまま読める):
    <wav_dir>/s{pair_idx:04d}_inj.wav   ← 本スクリプトが生成
    <wav_dir>/s{pair_idx:04d}_bl.wav    ← --link_bl_tgt_from から hard link
    <wav_dir>/s{pair_idx:04d}_tgt.wav   ← --link_bl_tgt_from から hard link
BL/TGT は注入層に依存しないので既存の音声を流用する。

層番号の規約 (重要):
    --inject_layer L  (0 <= L <= n_layers-1, n_layers = encoder の層数 = 26)
      L <  n_layers-1 : encoder.layers[L] の出力 (pre-final-norm) を置換し、
                        layers[L+1..] と final norm を走らせる。
      L == n_layers-1 : 「残りの層が 0 層」の場合。既存スクリプトと **同一の関数を
                        同一引数で** 呼ぶ経路 (get_encoder_output で得た
                        last_hidden_state = post-final-norm を置換) にディスパッチする。
                        これは既存の最終層注入と同じ計算であり、md5 が一致するべき。
                        --final_via_hook を付けると L == n_layers-1 でも hook 経路
                        (pre-norm 置換 → final norm) を通す。final norm は位置ごとの
                        RMSNorm なので数学的には同値だが bit 一致は保証されない。

GPU 実行 (CPU 専用処理ではないので slurm_run.sh 必須):
  SLURM_TIME=02:00:00 slurm_run.sh \
    python3 -u /mnt/scripts/layer_ablation/eval_en_crossculture_t5gemma_layer.py \
      --inject_layer 25 --n 5 --lang en \
      --wav_dir /mnt/eval_results/_layerabl/t5gemma_en_L25_n5 \
      --out_dir /mnt/eval_results/_layerabl/meta_L25_n5
"""
import os as _os
# slurm_run.sh は HF_HOME=/tmp/.cache/huggingface を渡すが実体が無く書き込めない。
# setdefault だとコンテナ側の値が残るので、明示的に上書きする。
_hf = "/mnt/.cache/huggingface" if _os.path.isdir("/mnt/scripts") else \
      os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
_os.environ["HF_HOME"] = _hf
_os.environ["HF_HUB_CACHE"] = _os.path.join(_hf, "hub")
_os.environ["HF_MODULES_CACHE"] = _os.path.join(_hf, "modules")

# **スレッド上限。torch を読む前に置く必要がある。**
# 2026-09-18、この種のジョブを 3 本並べたら 1 本あたり 128 スレッドを取り、
# ログインノードの CPU を埋めて他の利用者に迷惑をかけた。
# **上限は 128 スレッドの 1/4 = 32**（2026-09-18 ユーザ指示）。
# GPU を 3 枚まで同時に使うので、1 本あたり 10 にして合計 30 に収める。
# 実測では GPU 律速で 1 コアしか使わないが、上限として置く。
_thr = _os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = _thr

import argparse, json, os, sys, time, random

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f:
                os.environ.setdefault("HF_TOKEN", _f.read().strip())
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
            break
        except Exception: pass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
# 既存スクリプトと同じ順序で挿入する (import 解決を揃えるため)
sys.path.insert(0, T5GEMMA_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, numpy as np

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

# --- backend: 既存実装をそのまま使う (差し替え可能にするためモジュール属性で保持) ---
from encoder_injection_t5gemma import (         # noqa: E402
    seed_everything,
    get_encoder_output,
    get_encoder_layer_output,
    generate_with_modified_encoder,
    find_entity_token_range,
)
from inference_tts_utils import normalize_text_with_lang  # noqa: E402

SEED = 42
LANG = "en"


def seed_all(s):
    """既存スクリプト (eval_en_crossculture_t5gemma_fixed.py L53-55) と同一。"""
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


# ----------------------------------------------------------------------------
# 注入点の構築
# ----------------------------------------------------------------------------
def build_final_layer_memory(model, text_tokenizer, text_a, text_b, word_a, word_b,
                             device, add_bos):
    """既存 t5_synth_inj (eval_en_crossculture_t5gemma_fixed.py L126-144) の逐語移植。

    呼ぶ関数と引数を1対1で対応させてある:
      L126  get_encoder_output(model, tok, text_a, device, lang=LANG)
      L127  get_encoder_output(model, tok, text_b, device, lang=LANG)
      L128  find_entity_token_range(tok, src_enc["text"], word_a, add_bos=add_bos)
      L129  find_entity_token_range(tok, ref_enc["text"], word_b, add_bos=add_bos)
      L132-139 span splice (同長は in-place / 異長は concat)
      L140  attn = ones(1, T)
      L142-144 pos_ids = model._build_position_ids(x_lens, T, device)
    """
    src_enc = get_encoder_output(model, text_tokenizer, text_a, device, lang=LANG)
    ref_enc = get_encoder_output(model, text_tokenizer, text_b, device, lang=LANG)
    s_r = find_entity_token_range(text_tokenizer, src_enc["text"], word_a, add_bos=add_bos)
    r_r = find_entity_token_range(text_tokenizer, ref_enc["text"], word_b, add_bos=add_bos)
    if s_r is None or r_r is None:
        return None
    s_s, s_e = s_r; r_s, r_e = r_r
    modified = src_enc["last_hidden_state"].clone()
    if (s_e - s_s) == (r_e - r_s):
        modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
        route = "same_len_inplace"
    else:
        prefix = modified[0, :s_s]
        ref_ent = ref_enc["last_hidden_state"][0, r_s:r_e]
        suffix = modified[0, s_e:]
        modified = torch.cat([prefix, ref_ent, suffix], dim=0).unsqueeze(0)
        route = "diff_len_concat"
    attn = torch.ones((1, modified.shape[1]), dtype=torch.long, device=device)
    pos_ids = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([modified.shape[1]]).to(device)
        pos_ids = model._build_position_ids(x_lens, modified.shape[1], device)
    meta = {
        "point": "encoder_final_output",
        "route": route,
        "span_a": [int(s_s), int(s_e)], "span_b": [int(r_s), int(r_e)],
        "T_src": int(src_enc["last_hidden_state"].shape[1]),
        "T_ref": int(ref_enc["last_hidden_state"].shape[1]),
        "T_out": int(modified.shape[1]),
    }
    return modified, attn, pos_ids, meta


def _run_from_layer(model, encoder, inject_layer, modified_h, carrier_ids, device):
    """layers[inject_layer] の出力を modified_h に差し替え、以降の層 + final norm を走らせる。

    carrier_ids は「系列長を決めるための入力」でしかない (layers[inject_layer] の
    出力は hook で完全に置き換わるので中身は結果に影響しない)。異長 splice で
    系列長が変わる場合に、attention_mask / position_ids を新しい長さで正しく
    作るために必要。
    """
    seq_len = int(modified_h.shape[1])
    assert int(carrier_ids.shape[1]) == seq_len, \
        f"carrier length {carrier_ids.shape[1]} != modified hidden length {seq_len}"
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)
    pos_ids = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([seq_len]).to(device)
        pos_ids = model._build_position_ids(x_lens, seq_len, device)

    fired = {"n": 0}

    def replacement_hook(module, inputs, output):
        fired["n"] += 1
        if fired["n"] > 1:
            return output
        if isinstance(output, tuple):
            return (modified_h,) + tuple(output[1:])
        return modified_h

    handle = encoder.layers[inject_layer].register_forward_hook(replacement_hook)
    try:
        with torch.no_grad():
            enc_out = encoder(input_ids=carrier_ids, attention_mask=attn,
                              position_ids=pos_ids)
    finally:
        handle.remove()
    assert fired["n"] == 1, f"replacement hook fired {fired['n']} times (expected 1)"
    final_hidden = enc_out.last_hidden_state
    assert int(final_hidden.shape[1]) == seq_len, \
        f"encoder output length {final_hidden.shape[1]} != {seq_len}"
    return final_hidden, attn, pos_ids


def build_intermediate_layer_memory(model, text_tokenizer, text_a, text_b, word_a, word_b,
                                    device, add_bos, inject_layer, selfcheck=False):
    """中間層注入。encoder_injection_t5gemma.py の記述どおり:
      1. 元文を符号化器に通し layer L の隠れ状態を取る (get_encoder_layer_output)
      2. 目標文でも同じ
      3. layer L の実体語位置を目標側で置き換える
      4. 残りの層 (L+1 以降) と final norm を走らせる (_run_from_layer)
      5. その最終出力を復号器へ
    """
    encoder = model.encoder_module
    src = get_encoder_layer_output(model, text_tokenizer, text_a, device, inject_layer, lang=LANG)
    ref = get_encoder_layer_output(model, text_tokenizer, text_b, device, inject_layer, lang=LANG)

    # find_entity_token_range に渡すテキストは get_encoder_output が返す 'text' と同じもの
    # (= normalize_text_with_lang(text, LANG)[0])。EN では恒等写像。
    text_a_norm, _ = normalize_text_with_lang(text_a, LANG)
    text_b_norm, _ = normalize_text_with_lang(text_b, LANG)
    s_r = find_entity_token_range(text_tokenizer, text_a_norm, word_a, add_bos=add_bos)
    r_r = find_entity_token_range(text_tokenizer, text_b_norm, word_b, add_bos=add_bos)
    if s_r is None or r_r is None:
        return None
    s_s, s_e = s_r; r_s, r_e = r_r

    src_h = src["layer_hidden"]      # [1, T_a, D]  layers[L] の出力 (pre-final-norm)
    ref_h = ref["layer_hidden"]      # [1, T_b, D]
    src_tokens = list(src["tokens"])
    ref_tokens = list(ref["tokens"])

    if (s_e - s_s) == (r_e - r_s):
        modified_h = src_h.clone()
        modified_h[0, s_s:s_e] = ref_h[0, r_s:r_e]
        carrier_tokens = src_tokens
        route = "same_len_inplace"
    else:
        prefix = src_h[0, :s_s]
        ref_ent = ref_h[0, r_s:r_e]
        suffix = src_h[0, s_e:]
        modified_h = torch.cat([prefix, ref_ent, suffix], dim=0).unsqueeze(0)
        # 系列長が変わるので carrier も同じ長さで作る (中身は結果に影響しない)
        carrier_tokens = src_tokens[:s_s] + ref_tokens[r_s:r_e] + src_tokens[s_e:]
        route = "diff_len_concat"
    carrier_ids = torch.LongTensor([carrier_tokens]).to(device)

    final_hidden, attn, pos_ids = _run_from_layer(
        model, encoder, inject_layer, modified_h, carrier_ids, device)

    meta = {
        "point": f"encoder_layer_{inject_layer}_output",
        "route": route,
        "span_a": [int(s_s), int(s_e)], "span_b": [int(r_s), int(r_e)],
        "T_src": int(src_h.shape[1]), "T_ref": int(ref_h.shape[1]),
        "T_out": int(final_hidden.shape[1]),
    }

    if selfcheck and inject_layer != len(encoder.layers) - 1:
        # 中間層では両経路が一致する理由が無いので比較しない (誤解を招く数値を残さない)
        meta["selfcheck"] = "skipped_not_final_layer"
    elif selfcheck:
        # 最終層で hook 経路 (pre-norm 置換 → final norm) と
        # 既存経路 (post-norm 置換) が一致するかを実測する。
        fin = build_final_layer_memory(model, text_tokenizer, text_a, text_b,
                                       word_a, word_b, device, add_bos)
        if fin is not None:
            ref_mem = fin[0]
            if ref_mem.shape == final_hidden.shape:
                d = (ref_mem.float() - final_hidden.float()).abs()
                meta["selfcheck_max_abs_diff"] = float(d.max().item())
                meta["selfcheck_mean_abs_diff"] = float(d.mean().item())
                meta["selfcheck_ref_abs_mean"] = float(ref_mem.float().abs().mean().item())
                meta["selfcheck_bitexact"] = bool(torch.equal(ref_mem, final_hidden))
            else:
                meta["selfcheck_shape_mismatch"] = [list(ref_mem.shape), list(final_hidden.shape)]
    return final_hidden, attn, pos_ids, meta


# ----------------------------------------------------------------------------
# 合成 (乱数消費順序は既存スクリプトと1対1)
# ----------------------------------------------------------------------------
def synth_plain(model, text_tokenizer, audio_tok, text, device, codec_audio_sr, codec_sr,
                modified_hs=None, attn=None, pos_ids=None):
    """既存 t5_synth (eval_en_crossculture_t5gemma_fixed.py L110-122) の逐語移植。"""
    seed_everything(SEED)
    if modified_hs is None:
        enc = get_encoder_output(model, text_tokenizer, text, device, lang=LANG)
        modified_hs = enc["last_hidden_state"]
        attn = torch.ones((1, modified_hs.shape[1]), dtype=torch.long, device=device)
        if int(getattr(model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([modified_hs.shape[1]]).to(device)
            pos_ids = model._build_position_ids(x_lens, modified_hs.shape[1], device)
    return generate_with_modified_encoder(
        model, text_tokenizer, audio_tok, modified_hs, attn, pos_ids,
        text, device, codec_audio_sr, codec_sr,
    )


def synth_inj_at_layer(model, text_tokenizer, audio_tok, text_a, text_b, word_a, word_b,
                       device, add_bos, inject_layer, use_hook,
                       codec_audio_sr, codec_sr, selfcheck=False):
    """既存 t5_synth_inj (L124-145) と同じ乱数消費順序。

    seed_everything は (i) 関数先頭 (既存 L125) と
    (ii) 生成直前 = synth_plain 先頭 (既存 L111) の2回。
    use_hook=False (最終層) のときは既存と同一の関数を同一引数で呼ぶ。
    """
    seed_everything(SEED)                                   # 既存 L125 相当
    if use_hook:
        built = build_intermediate_layer_memory(
            model, text_tokenizer, text_a, text_b, word_a, word_b,
            device, add_bos, inject_layer, selfcheck=selfcheck)
    else:
        built = build_final_layer_memory(
            model, text_tokenizer, text_a, text_b, word_a, word_b, device, add_bos)
    if built is None:
        return None, {"error": "entity_span_not_found"}
    modified, attn, pos_ids, meta = built
    # 既存は t5_synth(text_a, modified_hs=...) を呼び、その先頭で再度 seed する。
    audio = synth_plain(model, text_tokenizer, audio_tok, text_a, device,
                        codec_audio_sr, codec_sr,
                        modified_hs=modified, attn=attn, pos_ids=pos_ids)
    return audio, meta


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inject_layer", type=int, required=True,
                   help="encoder のどの層の出力を置換するか (0..n_layers-1, -1 で最終層)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--lang", default="en", choices=["en", "ja", "zh"])
    p.add_argument("--wav_dir", required=True, help="層ごとに分けること")
    p.add_argument("--out_dir", required=True, help="層ごとに分けること")
    p.add_argument("--link_bl_tgt_from", default="eval_results/_cc_wavs_b/t5gemma_en")
    p.add_argument("--final_via_hook", action="store_true",
                   help="最終層でも hook 経路 (pre-norm 置換) を通す。等価性の実測用。")
    p.add_argument("--selfcheck", type=int, default=0,
                   help="先頭 N ペアで hook 経路と既存経路の数値差を計測 (--final_via_hook 時のみ意味あり)")
    p.add_argument("--replicate_bl_tgt", action="store_true",
                   help="既存スクリプトと同じ順序で BL / TGT も生成する (RNG/割当履歴を揃える)")
    p.add_argument("--replicate_preload", action="store_true",
                   help="既存スクリプトと同じ順序で WhisperAligner / 特徴抽出器も先に構築する")
    p.add_argument("--overwrite", action="store_true",
                   help="既存の inj wav があっても再生成する (既定は skip して resume)")
    p.add_argument("--no_link", action="store_true", help="BL/TGT の link を作らない")
    args = p.parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    seed_all(SEED)   # 既存 L69 と同じ位置

    # 日本語・中国語を解禁 (2026-09-18)。層別 ablation を 3 言語に広げるため。
    # 英語と同じ cross-culture 300 対を言語ごとに読む。BL / TGT は層に依存しないので
    # --link_bl_tgt_from で既存の音声を持ち込む (英語は _cc_wavs_b、日中は _cc_wavs_a)。
    if args.lang == "en":
        from en_cross_culture_names import build_300_pairs
    elif args.lang == "ja":
        from ja_cross_culture_names import build_300_pairs
    elif args.lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    else:
        raise SystemExit(f"--lang {args.lang} は対象外 (en / ja / zh)。")
    pairs = build_300_pairs()[: args.n]
    print(f"[Info] {args.lang.upper()} cross-culture pairs: {len(pairs)} (n={args.n})")

    if args.replicate_preload:
        # 既存スクリプト L75-76 と同じ順序・同じ引数で構築する
        from forced_alignment_utils import WhisperAligner
        from eval_table3_acoustic_common import AcousticFeatureExtractor
        _feat = AcousticFeatureExtractor(device)
        _aligner = WhisperAligner(device=device)
        print("[Info] replicate_preload: AcousticFeatureExtractor / WhisperAligner loaded")

    print("[Info] Loading T5Gemma-TTS ...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from data.tokenizer import AudioTokenizer
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        "Aratako/T5Gemma-TTS-2b-2b", trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    t5_model.eval()
    cfg = t5_model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    t5_tok = AutoTokenizer.from_pretrained(tok_name)
    audio_tok = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(t5_model.args, "add_bos_to_text", 0)) else 0

    encoder = t5_model.encoder_module
    n_layers = len(encoder.layers)
    cfg_n = None
    try:
        cfg_n = int(cfg.t5_config_dict["encoder"]["num_hidden_layers"])
    except Exception:
        pass
    print(f"[Info] encoder layers: len(encoder.layers)={n_layers} config={cfg_n}")
    if cfg_n is not None and cfg_n != n_layers:
        raise SystemExit(f"[Fatal] layer count mismatch: module={n_layers} config={cfg_n}")

    inject_layer = (n_layers - 1) if args.inject_layer == -1 else args.inject_layer
    if not (0 <= inject_layer < n_layers):
        raise SystemExit(f"[Fatal] --inject_layer {inject_layer} out of range [0, {n_layers-1}]")
    is_final = (inject_layer == n_layers - 1)
    use_hook = (not is_final) or args.final_via_hook
    print(f"[Info] inject_layer={inject_layer} (final={is_final}) "
          f"path={'hook(intermediate)' if use_hook else 'final_output(existing)'}")
    print(f"[Info] add_bos={add_bos} add_eos={int(getattr(t5_model.args,'add_eos_to_text',0))} "
          f"use_pm_rope={int(getattr(t5_model.args,'use_pm_rope',1))} "
          f"codec_audio_sr={codec_audio_sr} codec_sr={codec_sr}")
    print(f"[Info] torch={torch.__version__} cuda_device_count={torch.cuda.device_count()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    import torchaudio

    def _sv(a, sr, pth):
        """既存 L168-170 と同一の保存条件。"""
        a = a if a.dim() > 1 else a.unsqueeze(0)
        torchaudio.save(pth, a.cpu(), sr, encoding="PCM_S", bits_per_sample=16)

    def t5_synth(text):
        return synth_plain(t5_model, t5_tok, audio_tok, text, device,
                           codec_audio_sr, codec_sr)

    def t5_synth_inj_at_layer(text_a, text_b, word_a, word_b, selfcheck=False):
        return synth_inj_at_layer(
            t5_model, t5_tok, audio_tok, text_a, text_b, word_a, word_b,
            device, add_bos, inject_layer, use_hook,
            codec_audio_sr, codec_sr, selfcheck=selfcheck)

    link_src = args.link_bl_tgt_from
    repl_dir = os.path.join(args.out_dir, "repl_check")
    if args.replicate_bl_tgt:
        os.makedirs(repl_dir, exist_ok=True)

    run_header = {
        "script": os.path.abspath(__file__),
        "inject_layer": inject_layer,
        "n_encoder_layers": n_layers,
        "final_layer_index": n_layers - 1,
        "path": "hook_intermediate" if use_hook else "final_output_existing",
        "final_via_hook": bool(args.final_via_hook),
        "replicate_bl_tgt": bool(args.replicate_bl_tgt),
        "replicate_preload": bool(args.replicate_preload),
        "n": args.n, "lang": args.lang, "seed": SEED,
        "add_bos": add_bos,
        "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
        "wav_dir": os.path.abspath(args.wav_dir),
        "link_bl_tgt_from": os.path.abspath(link_src) if not args.no_link else None,
        "torch": torch.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": os.uname().nodename,
    }
    rows, n_gen, n_skip, n_fail = [], 0, 0, 0
    t0 = time.time()
    meta_path = os.path.join(args.out_dir, "layer_inject_meta.json")

    for i, pair in enumerate(pairs):
        pair_idx = pair.get("pair_idx", i)
        inj_path = os.path.join(args.wav_dir, f"s{pair_idx:04d}_inj.wav")
        if os.path.exists(inj_path) and not args.overwrite:
            n_skip += 1
            rows.append({"pair_idx": pair_idx, "status": "skipped_existing"})
            continue
        try:
            if args.replicate_bl_tgt:
                # 既存ループ L161-163 と同じ順序で BL -> TGT -> INJ
                bl = t5_synth(pair["text_a"])
                tg = t5_synth(pair["text_b"])
                if bl is not None:
                    _sv(bl[0], codec_audio_sr, os.path.join(repl_dir, f"s{pair_idx:04d}_bl.wav"))
                if tg is not None:
                    _sv(tg[0], codec_audio_sr, os.path.join(repl_dir, f"s{pair_idx:04d}_tgt.wav"))
            do_sc = bool(args.selfcheck) and (i < args.selfcheck)
            inj, meta = t5_synth_inj_at_layer(
                pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"],
                selfcheck=do_sc)
            if inj is None:
                n_fail += 1
                rows.append({"pair_idx": pair_idx, "status": "no_audio", **meta})
                continue
            _sv(inj[0], codec_audio_sr, inj_path)
            n_gen += 1
            rows.append({
                "pair_idx": pair_idx, "status": "ok",
                "word_a": pair["word_a"], "word_b": pair["word_b"],
                "dur_sec": round(inj[0].shape[-1] / codec_audio_sr, 3),
                **meta,
            })
        except Exception as e:
            import traceback
            n_fail += 1
            rows.append({"pair_idx": pair_idx, "status": "error", "error": str(e)[:300]})
            print(f"  [{i}] FAIL: {str(e)[:300]}")
            traceback.print_exc()
            continue
        if (i + 1) % 10 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{i+1}/{len(pairs)}] gen={n_gen} skip={n_skip} fail={n_fail} "
                  f"elapsed={el:.1f}min", flush=True)
        if (i + 1) % 25 == 0:
            with open(meta_path, "w") as f:
                json.dump({"run": run_header, "rows": rows}, f, ensure_ascii=False, indent=1)

    # BL / TGT の持ち込み (hard link, 失敗したらコピー)
    n_link, n_link_missing = 0, 0
    if not args.no_link:
        import shutil
        for pair in pairs:
            pidx = pair["pair_idx"]
            if not os.path.exists(os.path.join(args.wav_dir, f"s{pidx:04d}_inj.wav")):
                continue
            for cond in ("bl", "tgt"):
                src = os.path.join(link_src, f"s{pidx:04d}_{cond}.wav")
                dst = os.path.join(args.wav_dir, f"s{pidx:04d}_{cond}.wav")
                if os.path.exists(dst):
                    continue
                if not os.path.exists(src):
                    n_link_missing += 1
                    continue
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                n_link += 1
    run_header.update({
        "n_generated": n_gen, "n_skipped": n_skip, "n_failed": n_fail,
        "n_linked": n_link, "n_link_missing": n_link_missing,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    })
    with open(meta_path, "w") as f:
        json.dump({"run": run_header, "rows": rows}, f, ensure_ascii=False, indent=1)

    sc = [r["selfcheck_max_abs_diff"] for r in rows if "selfcheck_max_abs_diff" in r]
    print("\n=== layer injection done ===")
    for k, v in run_header.items():
        print(f"  {k}: {v}")
    if sc:
        print(f"  selfcheck_max_abs_diff: max={max(sc):.6g} over {len(sc)} pairs "
              f"(bitexact={sum(1 for r in rows if r.get('selfcheck_bitexact'))}/{len(sc)})")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
