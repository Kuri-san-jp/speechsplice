#!/usr/bin/env python3
"""SpeechT5 (英語) の複数 entity 同時 splice: BL / INJ_1 / INJ_2 / INJ_12.

T5Gemma 版 `scripts/analysis/generate_multientity.py` の構成を写した**新規**
スクリプト。既存ファイルは 1 行も変更していない。差し替えの中身は
`scripts/projection/splice_multi.py:build_spliced_hidden_multi` を呼ぶだけで、
区間の一括 concat を自分で書き直してはいない。

--------------------------------------------------------------------------
1. 何を差し替えるか
--------------------------------------------------------------------------
`s5_model.speecht5.encoder`（= `SpeechT5EncoderWithTextPrenet`）の forward
出力の `last_hidden_state` を forward hook で置き換える。既存の単一 entity 版
`scripts/projection/eval_speecht5_crossculture_fixed_v2.py:99-141` と同一。

transformers の `modeling_speecht5.py` では `SpeechT5Encoder.forward` が
入口で `self.layer_norm(hidden_states)` を掛け（1310 行付近）、層ループの後には
LayerNorm を置かず `BaseModelOutput(last_hidden_state=hidden_states)` を返す
（1354-1364 行）。したがって最終層（layer 11）の forward hook 出力は
`last_hidden_state` と同一で、層 11 を捕捉しても encoder 出力を捕捉しても同じ。

--------------------------------------------------------------------------
2. 長さの扱い（--len_policy）
--------------------------------------------------------------------------
既存の SpeechT5 単一 entity 実装は**長さ保存**である。
`eval_speecht5_crossculture_fixed_v2.py:127-133`:

    if src_len == tgt_len:  modified[0, ss:se] = h_b[0, rs:re]
    else:                   F.adaptive_avg_pool1d(ref_span, src_len) で長さを合わせる

SpeechT5 の text tokenizer は文字単位（spm_char）なので語長の違いがそのまま
トークン数の違いになり、`data/en_cross_culture_names.py` の 300 対では
270/300 が src_len != tgt_len、つまり 9 割が pooling 経路を通っている
（CPU 検証済み）。よって md5 等価性を満たす既定値は `pool` である。

  --len_policy pool   （既定）区間ごとに ref 側を src 区間長へ
                      adaptive_avg_pool1d で合わせてから splice する。
                      delta_tok は常に 0、T_out == T_src。
                      k=1 のとき既存実装とビット一致する。
  --len_policy concat T5Gemma 版と同じ可変長 concat。T_out != T_src になる。
                      SpeechT5 では decoder の cross-attention mask が
                      input_ids 長（T_src）で作られるため、そのままだと
                      `SpeechT5Attention` の
                      "Attention mask should be of size ..." で落ちる
                      （modeling_speecht5.py:959-962）。本スクリプトは
                      `wrapped_decoder` の forward pre-hook で
                      encoder_attention_mask を None に落として回避する。
                      pad token が id 列に無いこと（mask が全 1 であること）を
                      毎文 assert してから行う。全 1 の加算 mask は 0.0 なので
                      None と数値的に同一。

どちらでも `s_ranges` は**解決したままの src index** を渡す。2 番目の区間を
1 番目の差し替え分だけ手でずらすことは一切しない（ずれは
`build_spliced_hidden_multi` の一括 concat が構造的に吸収する）。

--------------------------------------------------------------------------
3. 区間の解決
--------------------------------------------------------------------------
`splice_multi.resolve_spans_with_cursor` は `return_offsets_mapping=True` を
使うので **SpeechT5 では使えない**。`SpeechT5Tokenizer` は slow tokenizer
（spm_char、fast 版が transformers に無い）で、offset_mapping を求めると
NotImplementedError になる。

代わりに既存の `s5_find_ent`（トークン id 列の部分列探索）へ
**開始カーソルだけを足した** `find_ent_ids(entity, ids, tkz, start)` を使う。
`start=0` のとき本文は既存 `s5_find_ent`（同ファイル 89-97 行）と同一で、
`data/en_cross_culture_names.py` の 300 対すべてで返り値が一致することを
CPU で確認済み。

解決後に
  * 往復確認: `decode(ids[t_s:t_e]).strip() == 表層形`
  * 文字位置の整合: `len(decode(ids[:t_s]).rstrip()) == len(text[:char_start].rstrip())`
  * 重なり検査: `s_ranges[0][1] > s_ranges[1][0]` なら skip
    （さらに `build_spliced_hidden_multi` 内の `check_span_sets` が assert で落とす）
を行い、落ちた文は丸ごと skip する。「末尾 N トークンを entity とみなす」種類の
fallback は用意しない。

--------------------------------------------------------------------------
4. 乱数
--------------------------------------------------------------------------
SpeechT5 の `SpeechT5SpeechDecoderPrenet._consistent_dropout`
（modeling_speecht5.py:671-686）は `torch.bernoulli` を eval でも必ず引く。
しかも毎 step 伸びる `output_sequence` 全体に掛かるので、引く個数が step 数に
依存する。したがって seed は**生成呼び出しの直前**に毎回戻す。
4 条件すべてで `seed_all(SEED)` -> `generate_speech` の間に乱数を引く処理を
挟まない（hook の登録は乱数を引かない）。encoder 側の dropout は
`training=False` なので `F.dropout` は乱数を消費しない。

--------------------------------------------------------------------------
5. 使い方
--------------------------------------------------------------------------
単一区間の md5 等価性試験（n=5）:

  SLURM_TIME=00:30:00 slurm_run.sh \\
    python3 -u /mnt/scripts/projection/run_multientity_speecht5.py \\
      --mode equiv --n 5 \\
      --output_dir /mnt/eval_results/_s5_multi_equiv_n5 \\
      --compare_dir /mnt/eval_results/_cc_wavs_a/speecht5_en \\
      --repeat_check

複数 entity 本番（英語 300 件・4 条件）:

  SLURM_TIME=06:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/projection/run_multientity_speecht5.py \\
      --mode multi --pairs_module en_multientity_300 \\
      --output_dir /mnt/eval_results/multientity_speecht5_en_300

天井用の目標文だけを後から足す（--tgt_only、2026-09-22 追加）:

  SLURM_TIME=06:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/projection/run_multientity_speecht5.py \\
      --mode multi --tgt_only --pairs_module en_multientity_300 \\
      --output_dir /mnt/eval_results/multientity_speecht5_en_300

  注入をしないので区間の解決も往復確認もしない。既にある wav は作り直さない
  ので、既存の bl/inj1/inj2/inj12 と results.json / checkpoint.json は動かない。
  帳面は checkpoint_tgt.json / results_tgt.json（別ファイル）。
  --dry_run を足すとモデルを読まずに、出すつもりの文とファイル名だけ並べる。

出力:
  <out_dir>/results.json     summary + rows（conds に bl/inj1/inj2/inj12）
  <out_dir>/checkpoint.json  途中再開用
  <out_dir>/s{pair_idx:04d}_{bl,inj1,inj2,inj12}.wav
  <out_dir>/s{pair_idx:04d}_{tgt1,tgt2}.wav   （--tgt_only のとき）
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

# HF_HOME は setdefault では効かない（親から継承された値に負ける）ので代入する。
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["TORCH_HOME"] = os.path.join(ROOT, ".cache", "torch")

# **スレッド上限。torch を読む前に置く必要がある。**
# 2026-09-18、この種のジョブを並べたらログインノードの CPU を埋めて
# 他の利用者に迷惑をかけた（note/2026-09-18_gpuoverrun_debug.md）。
# **上限は 128 スレッドの 1/4 = 32**（2026-09-18 ユーザ指示）。
# GPU を 3 枚まで同時に使うので、1 本あたり 10 にして合計 30 に収める。
# 実測では GPU 律速で 1 コアしか使わないが、上限として置く。
_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

sys.path.insert(0, os.path.join(ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(ROOT, "data"))
# 目標文だけを出す経路（--tgt_only）の共通部品。torch を読まない stdlib だけの
# モジュール。**末尾に足す**ので、既存の import の解決順は 1 つも動かない。
sys.path.append(os.path.join(ROOT, "scripts", "multientity"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402

from splice_multi import build_spliced_hidden_multi, span_deltas  # noqa: E402
from tgt_only import dry_run_report, run_targets  # noqa: E402

SEED = 42
SR = 16000
CONDITIONS = ["bl", "inj1", "inj2", "inj12"]
# 条件 -> 使うスロット index のリスト（0-based）
COND_SLOTS = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}


# --- 乱数（既存 eval_speecht5_crossculture_fixed_v2.py:28-30 と同一） -------

def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# --- 区間探索（既存 s5_find_ent に start だけ足したもの） -------------------

def find_ent_ids(entity, idl, tkz, start=0):
    """トークン id 列 `idl` の `start` 以降で `entity` の部分列を探す.

    `start=0` のとき本文は
    `eval_speecht5_crossculture_fixed_v2.py:89-97` の `s5_find_ent` と同一。
    先に裸の表層形、見つからなければ先頭に空白を付けたものを試す順序も同じ。
    spm_char では語頭の '▁' が表層形の先頭トークンに入るので、区間は
    直前の空白トークンを含む。
    """
    ent_ids = tkz(entity, add_special_tokens=False)["input_ids"]
    for i in range(start, len(idl) - len(ent_ids) + 1):
        if idl[i:i + len(ent_ids)] == ent_ids:
            return i, i + len(ent_ids)
    ent_ids_sp = tkz(" " + entity, add_special_tokens=False)["input_ids"]
    for i in range(start, len(idl) - len(ent_ids_sp) + 1):
        if idl[i:i + len(ent_ids_sp)] == ent_ids_sp:
            return i, i + len(ent_ids_sp)
    return None


def verify_span_roundtrip_ids(tkz, idl, span, surface, char_start=None,
                              text=None):
    """往復確認。offset_mapping が無いので decode 側で見る."""
    t_s, t_e = span
    bad = []
    if not (0 <= t_s < t_e <= len(idl)):
        return False, [f"token range {span} outside [0, {len(idl)}]"]
    cov = tkz.decode(idl[t_s:t_e]).strip()
    if cov != surface:
        bad.append(f"covered {cov!r} != surface {surface!r}")
    if char_start is not None and text is not None:
        pre = tkz.decode(idl[:t_s]).rstrip()
        want = text[:char_start].rstrip()
        if len(pre) != len(want):
            bad.append(f"prefix length {len(pre)} != char_start-derived "
                       f"{len(want)} (pre={pre!r})")
    return (not bad), bad


# --- splice --------------------------------------------------------------

def build_modified(h_src, ref_hs, s_ranges, r_ranges, len_policy):
    """k 区間差し替え後の encoder hidden を返す.

    `s_ranges` は解決したままの src index を渡す。**手でずらさない。**
    長さ変化の吸収は `build_spliced_hidden_multi` の一括 concat が行う。

    返り値: (modified [1,T_out,d], delta_tok, out_ranges, per_span)
      per_span[i] = {"n", "m", "m_eff", "pooled"}
    """
    assert len_policy in ("pool", "concat"), len_policy
    eff_refs, eff_r, per_span = [], [], []
    for (s_s, s_e), (r_s, r_e), hr in zip(s_ranges, r_ranges, ref_hs):
        n = s_e - s_s
        m = r_e - r_s
        if len_policy == "concat" or m == n:
            # 既存実装の `modified[0, ss:se] = h_b[0, rs:re]` と同じ切り出し
            eff_refs.append({"last_hidden_state": hr})
            eff_r.append((int(r_s), int(r_e)))
            per_span.append({"n": n, "m": m, "m_eff": m, "pooled": False})
        else:
            # 既存実装 130-133 行の算術をそのまま写す（同じ op・同じ shape）
            ref_ent = hr[0, r_s:r_e].unsqueeze(0).transpose(1, 2)
            ref_ent_resampled = F.adaptive_avg_pool1d(ref_ent, n).transpose(1, 2)
            eff_refs.append(
                {"last_hidden_state": ref_ent_resampled.to(h_src.dtype)})
            eff_r.append((0, n))
            per_span.append({"n": n, "m": m, "m_eff": n, "pooled": True})

    modified, delta_tok, out_ranges = build_spliced_hidden_multi(
        {"last_hidden_state": h_src}, eff_refs, s_ranges, eff_r)
    if len_policy == "pool":
        assert delta_tok == 0 and modified.shape[1] == h_src.shape[1], (
            f"pool policy must preserve length: delta={delta_tok} "
            f"T_out={modified.shape[1]} T_src={h_src.shape[1]}")
    return modified, delta_tok, out_ranges, per_span


def make_hook(modified):
    """既存 eval_speecht5_crossculture_fixed_v2.py:99-106 と同一."""
    def hook(module, input, output):
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = modified
        elif isinstance(output, tuple) and output:
            output = (modified,) + output[1:]
        return output
    return hook


def _drop_enc_mask_prehook(module, args, kwargs):
    """concat 経路専用: decoder の cross-attention mask を外す.

    `_generate_speech`（modeling_speecht5.py:2310-2352）は
    `encoder_attention_mask` を input_ids 長で作って `wrapped_decoder` に渡す。
    encoder 出力を可変長に差し替えると src_len と合わず
    `SpeechT5Attention`（959-962 行）が ValueError で落ちる。
    pad token が無く mask が全 1 のときに限り、加算される値は 0.0 なので
    None に落として同一。呼び出し側が pad の不在を assert してから使う。
    """
    if "encoder_attention_mask" in kwargs:
        kwargs["encoder_attention_mask"] = None
    return args, kwargs


# --- 合成 ----------------------------------------------------------------

def synth_plain(model, processor, vocoder, spk, text, device):
    """BL。既存 `s5_synth`（108-114 行）と同一."""
    seed_all(SEED)
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        if inputs["input_ids"].shape[1] == 0:
            return None
        audio = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
    return audio


def synth_spliced(model, vocoder, spk, ids_src, modified, drop_mask_handle_fn):
    """INJ。既存 `s5_synth_inj`（134-141 行）と同一の順序.

        seed_all(SEED) -> hook 登録 -> no_grad で generate_speech -> hook 解除

    hook 登録は乱数を引かないので、生成開始時の RNG 状態は
    `seed_all(SEED)` 直後と同一。
    """
    seed_all(SEED)
    handle = model.speecht5.encoder.register_forward_hook(make_hook(modified))
    extra = drop_mask_handle_fn() if drop_mask_handle_fn else None
    try:
        with torch.no_grad():
            audio = model.generate_speech(ids_src, spk, vocoder=vocoder)
    finally:
        handle.remove()
        if extra is not None:
            extra.remove()
    return audio


def save_wav(audio, path):
    """既存 `_sv`（151-155 行）と同一の保存呼び出し（byte 一致のため）."""
    if audio is None:
        return False
    a = audio if audio.dim() >= 2 else audio.unsqueeze(0)
    torchaudio.save(path, a.cpu(), SR, encoding="PCM_S", bits_per_sample=16)
    return True


def md5_of(path):
    """ファイル全体の md5。既存 wav と同じ意味で比べるための値."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_pcm_of(path):
    """int16 サンプル列だけの md5.

    既存 wav のヘッダには ffmpeg の LIST/INFO チャンク
    ("ISFT Lavf58.76.100") が入っているので、ファイル md5 は
    torchaudio のバックエンドと libavformat の版にも依存する。
    音そのものが一致しているかを見るにはサンプル列の md5 を使う。
    """
    wav, sr = torchaudio.load(path, normalize=False)
    return hashlib.md5(wav.numpy().tobytes()).hexdigest(), int(sr)


# --- モデル読み込み（既存 67-79 行と同一） -------------------------------

def load_speecht5(device, xvector_idx):
    from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                              SpeechT5Processor)
    print("[Info] Loading SpeechT5 ...", flush=True)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan").to(device).eval()
    from datasets import load_dataset
    try:
        emb_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        spk = torch.tensor(emb_ds[xvector_idx]["xvector"]).unsqueeze(0).to(device)
        spk_src = f"cmu-arctic-xvectors[{xvector_idx}]"
    except Exception as e:
        spk = torch.randn(1, 512).to(device) * 0.1
        spk_src = f"RANDOM FALLBACK ({type(e).__name__})"
    spk_sum = float(spk.detach().double().sum().item())
    print(f"[Info] speaker embedding: {spk_src} sum={spk_sum:.10f}", flush=True)
    return processor, model, vocoder, spk, spk_src, spk_sum


def get_enc(model, processor, text, device):
    """既存 `s5_get_enc`（82-87 行）と同一."""
    inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        enc = model.speecht5.encoder(
            input_values=inputs["input_ids"],
            attention_mask=inputs["attention_mask"])
    return enc.last_hidden_state, inputs["input_ids"]


# --- 1 文の処理 ----------------------------------------------------------

def resolve_item(processor, model, item, device, is_multi):
    """src 側 k 区間と ref 側 1 区間 x k を解決する.

    返り値 (h_src, ids_src, ref_hs, s_ranges, r_ranges, detail) / 失敗時は
    (None, ..., reason)
    """
    tkz = processor.tokenizer
    h_src, ids_src = get_enc(model, processor, item["text_a"], device)
    idl_src = ids_src[0].cpu().tolist()

    spans = item["spans"]
    surfaces_a = [sp["word_a"] for sp in spans]
    # src: 左から順に、カーソルを持って解決する。k=1 のとき start=0 なので
    # 既存 s5_find_ent と同一。
    cursor = 0
    s_ranges, via = [], []
    for i, surf in enumerate(surfaces_a):
        r = find_ent_ids(surf, idl_src, tkz, cursor)
        if r is None:
            return None, None, None, None, None, \
                f"src span {i}: surface {surf!r} not found at/after token {cursor}"
        s_ranges.append((int(r[0]), int(r[1])))
        via.append("cursor" if cursor else "head")
        cursor = r[1]

    # ref: 区間ごとに別 forward。それぞれ 1 区間だけ解決（start=0）。
    ref_hs, r_ranges = [], []
    for i, sp in enumerate(spans):
        h_ref, ids_ref = get_enc(model, processor, item["text_refs"][i], device)
        idl_ref = ids_ref[0].cpu().tolist()
        r = find_ent_ids(sp["word_b"], idl_ref, tkz, 0)
        if r is None:
            return None, None, None, None, None, \
                f"ref[{i}] span: surface {sp['word_b']!r} not found"
        ok, bad = verify_span_roundtrip_ids(
            tkz, idl_ref, r, sp["word_b"], sp.get("char_start_b"),
            item["text_refs"][i])
        if not ok:
            return None, None, None, None, None, f"ref[{i}] roundtrip: {bad}"
        ref_hs.append(h_ref)
        r_ranges.append((int(r[0]), int(r[1])))

    # src 側の往復確認
    for i, sp in enumerate(spans):
        ok, bad = verify_span_roundtrip_ids(
            tkz, idl_src, s_ranges[i], sp["word_a"], sp.get("char_start_a"),
            item["text_a"])
        if not ok:
            return None, None, None, None, None, f"src[{i}] roundtrip: {bad}"

    # 重なり検査（build_spliced_hidden_multi 内の check_span_sets でも落ちるが、
    # ここで理由を残して skip できるようにする）
    for a in range(len(s_ranges) - 1):
        if s_ranges[a][1] > s_ranges[a + 1][0]:
            return None, None, None, None, None, \
                (f"src spans overlap: {s_ranges[a]} {s_ranges[a + 1]} "
                 f"({surfaces_a})")

    d_per, d_sum = span_deltas(s_ranges, r_ranges)
    detail = {
        "src": [list(x) for x in s_ranges],
        "ref": [list(x) for x in r_ranges],
        "n": [b - a for a, b in s_ranges],
        "m": [b - a for a, b in r_ranges],
        "delta_per_span": d_per,
        "delta_sum_raw": d_sum,
        "src_resolve_via": via,
        "pad_in_src_ids": int(tkz.pad_token_id in idl_src),
    }
    if is_multi:
        detail["gap_tokens"] = s_ranges[1][0] - s_ranges[0][1]
        detail["len_imbalance"] = abs(detail["m"][0] - detail["m"][1])
    return h_src, ids_src, ref_hs, s_ranges, r_ranges, detail


# --- データ --------------------------------------------------------------

def load_multi_items(pairs_module, n):
    mod = __import__(pairs_module)
    items = mod.build_pairs()
    for it in items:
        assert len(it["spans"]) == 2, \
            f"pair {it['pair_idx']}: this script expects exactly 2 spans"
        assert len(it["text_refs"]) == 2
    return items[:n] if n else items


def load_equiv_items(n):
    """単一区間の等価性試験用。1 区間の item 形式に組み替える."""
    from en_cross_culture_names import build_300_pairs
    out = []
    for p in build_300_pairs():
        out.append({
            "pair_idx": p["pair_idx"],
            "template_id": None,
            "lang": "en",
            "text_a": p["text_a"],
            "text_refs": [p["text_b"]],
            "text_both": p["text_b"],
            "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"],
                       "char_start_a": None, "char_start_b": None}],
        })
    return out[:n] if n else out


# --- 本体 ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="multi", choices=["multi", "equiv"])
    ap.add_argument("--pairs_module", default="en_multientity_300")
    ap.add_argument("--lang", default="en", choices=["en"],
                    help="SpeechT5 は英語モデル。JA/ZH は単一 entity で "
                         "3 条件の音声が同一になる退化セルなので対象外")
    ap.add_argument("--len_policy", default="pool", choices=["pool", "concat"])
    ap.add_argument("--n", type=int, default=0, help="0 なら全件")
    ap.add_argument("--s5_xvector_idx", type=int, default=7306)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--compare_dir", default=None,
                    help="equiv: 既存の単一 entity wav があるディレクトリ。"
                         "s{i:04d}_inj.wav と md5 を比べる")
    ap.add_argument("--repeat_check", action="store_true",
                    help="equiv: 同一プロセス内で 2 回生成して md5 を比べ、"
                         "実行ごとのビット再現性そのものを確かめる")
    ap.add_argument("--equiv_full_order", action="store_true",
                    help="equiv: 既存スクリプトと同じ順に bl/tgt も生成して"
                         "捨てる（実行履歴まで揃えたい場合の保険）")
    ap.add_argument("--ckpt_every", type=int, default=10)
    # **後から目標文だけを足すための経路（2026-09-22）。**既定は今までどおり
    # off なので、既存の呼び出しは 1 行も通らない。
    ap.add_argument("--tgt_only", action="store_true",
                    help="枠ごとの目標文 tgt1 / tgt2 だけを合成する（注入なし）。"
                         "既存の bl/inj1/inj2/inj12 と results.json は触らない")
    ap.add_argument("--dry_run", action="store_true",
                    help="モデルを読まずに、出すつもりの文とファイル名を並べる")
    args = ap.parse_args()

    os.chdir(ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(SEED)

    is_multi = args.mode == "multi"
    if is_multi:
        items = load_multi_items(args.pairs_module, args.n)
        conds = CONDITIONS
    else:
        items = load_equiv_items(args.n)
        conds = ["inj"]
    print(f"[Info] mode={args.mode} items={len(items)} "
          f"len_policy={args.len_policy} device={device}", flush=True)

    if args.tgt_only and not is_multi:
        raise SystemExit("--tgt_only は 2 枠のデータ (--mode multi) 専用")
    if args.dry_run:
        # モデルを読む前に返す（CPU で名前と文を確かめるための経路）。
        dry_run_report(items, args.output_dir, tag="speecht5")
        return

    processor, model, vocoder, spk, spk_src, spk_sum = load_speecht5(
        device, args.s5_xvector_idx)
    tkz = processor.tokenizer

    if args.tgt_only:
        # 注入しないので区間の解決も往復確認もしない。合成は bl と同じ
        # synth_plain で、渡す文だけが text_refs[slot] に変わる。
        def _synth_tgt(text, path):
            audio = synth_plain(model, processor, vocoder, spk, text, device)
            if audio is None:
                raise RuntimeError("generate returned None")
            save_wav(audio, path)
            info = torchaudio.info(path)
            return {"dur_sec": round(info.num_frames / info.sample_rate, 4),
                    "md5": md5_of(path)}

        run_targets(items, args.output_dir, _synth_tgt, tag="speecht5",
                    ckpt_every=args.ckpt_every,
                    extra_summary={"method": "speecht5", "lang": args.lang,
                                   "pairs_module": args.pairs_module,
                                   "s5_xvector_idx": args.s5_xvector_idx,
                                   "speaker_source": spk_src,
                                   "speaker_sum": spk_sum})
        return

    drop_mask_fn = None
    if args.len_policy == "concat":
        wd = model.speecht5.decoder.wrapped_decoder

        def drop_mask_fn():  # noqa: F811
            return wd.register_forward_pre_hook(_drop_enc_mask_prehook,
                                                with_kwargs=True)
        print("[Info] len_policy=concat: wrapped_decoder の "
              "encoder_attention_mask を外す pre-hook を使う", flush=True)

    rows, skipped = [], []
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    mode_tag = f"splice_multi_speecht5_{args.len_policy}"
    if os.path.exists(ckpt):
        st = json.load(open(ckpt))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != mode_tag]
        assert not bad, (f"checkpoint mode mismatch in {ckpt}: pairs {bad[:5]} "
                         f"were not produced by this configuration; use a "
                         f"separate --output_dir")
        print(f"[Info] resume: {len(rows)} rows, {len(skipped)} skipped",
              flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}
    t0 = time.time()

    for it in items:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        try:
            (h_src, ids_src, ref_hs, s_ranges, r_ranges,
             detail) = resolve_item(processor, model, it, device, is_multi)
        except Exception as e:
            print(f"  [{pidx}] enc/resolve fail: {str(e)[:160]}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"enc: {e}"})
            done.add(pidx)
            continue
        if h_src is None:
            print(f"  [{pidx}] SKIP {detail}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": detail})
            done.add(pidx)
            continue
        if args.len_policy == "concat":
            assert detail["pad_in_src_ids"] == 0, (
                f"pair {pidx}: pad token present in input_ids; dropping the "
                f"encoder attention mask would change the result")

        row = {
            "pair_idx": pidx, "mode": mode_tag, "method": "speecht5",
            "len_policy": args.len_policy,
            "template_id": it.get("template_id"), "lang": args.lang,
            "text_a": it["text_a"], "text_refs": it["text_refs"],
            "text_both": it.get("text_both"),
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "words_a": [sp["word_a"] for sp in it["spans"]],
            "words_b": [sp["word_b"] for sp in it["spans"]],
            "span_tok": detail,
            "T_src": int(h_src.shape[1]),
            "conds": {},
        }

        if args.equiv_full_order and not is_multi:
            # 既存スクリプトは 1 文につき bl -> tgt -> inj の順に生成する
            # （157-164 行）。どの生成も直前に seed を戻すので inj の結果は
            # 前の 2 本に依らないが、実行履歴まで揃えたい場合の保険として
            # 同じ順に生成して捨てる。
            synth_plain(model, processor, vocoder, spk, it["text_a"], device)
            synth_plain(model, processor, vocoder, spk, it["text_refs"][0],
                        device)

        ok = True
        for cond in conds:
            slots = COND_SLOTS[cond] if is_multi else [0]
            modified, delta_tok, out_ranges, per_span = None, 0, [], []
            if slots:
                try:
                    sub_s = [s_ranges[i] for i in slots]
                    sub_r = [r_ranges[i] for i in slots]
                    sub_h = [ref_hs[i] for i in slots]
                    # s_ranges は解決したまま。2 番目の区間の index を
                    # 1 番目の差し替え分だけずらす処理はどこにも無い。
                    modified, delta_tok, out_ranges, per_span = build_modified(
                        h_src, sub_h, sub_s, sub_r, args.len_policy)
                except Exception as e:
                    print(f"  [{pidx}] {cond} splice fail: {str(e)[:160]}",
                          flush=True)
                    skipped.append({"pair_idx": pidx,
                                    "reason": f"{cond} splice: {e}"})
                    ok = False
                    break

            try:
                if modified is None:
                    audio = synth_plain(model, processor, vocoder, spk,
                                        it["text_a"], device)
                else:
                    audio = synth_spliced(model, vocoder, spk, ids_src,
                                          modified, drop_mask_fn)
                if audio is None:
                    raise RuntimeError("generate returned None")
            except Exception as e:
                print(f"  [{pidx}] {cond} synth fail: {str(e)[:200]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} synth: {e}"})
                ok = False
                break

            wav_name = f"s{pidx:04d}_{cond}.wav"
            wav_path = os.path.join(args.output_dir, wav_name)
            save_wav(audio, wav_path)
            info = torchaudio.info(wav_path)
            cd = {
                "wav": wav_name,
                "delta_tok": int(delta_tok),
                "T_out": int(modified.shape[1]) if modified is not None
                else row["T_src"],
                "out_ranges": [list(x) for x in out_ranges],
                "target_text": it["text_a"],
                "dur_sec": round(info.num_frames / info.sample_rate, 4),
                "slots": slots,
                "per_span": per_span,
                "md5": md5_of(wav_path),
            }
            cd["md5_pcm"], _ = md5_pcm_of(wav_path)
            if args.repeat_check and modified is not None:
                a2 = synth_spliced(model, vocoder, spk, ids_src, modified,
                                   drop_mask_fn)
                p2 = os.path.join(args.output_dir,
                                  f"_repeatcheck_s{pidx:04d}_{cond}.wav")
                save_wav(a2, p2)
                cd["md5_repeat"] = md5_of(p2)
                cd["repeat_bit_identical"] = (cd["md5_repeat"] == cd["md5"])
                os.remove(p2)
            if args.compare_dir:
                ref_p = os.path.join(args.compare_dir, wav_name)
                if os.path.exists(ref_p):
                    cd["md5_existing"] = md5_of(ref_p)
                    cd["md5_match"] = (cd["md5_existing"] == cd["md5"])
                    cd["md5_pcm_existing"], _ = md5_pcm_of(ref_p)
                    cd["md5_pcm_match"] = (cd["md5_pcm_existing"]
                                           == cd["md5_pcm"])
                else:
                    cd["md5_existing"] = None
                    cd["md5_match"] = None
                    cd["md5_pcm_existing"] = None
                    cd["md5_pcm_match"] = None
            row["conds"][cond] = cd

        done.add(pidx)
        if not ok:
            continue
        if is_multi:
            row["total_position_drift"] = row["conds"]["inj12"]["delta_tok"]
        rows.append(row)

        if len(rows) % 5 == 0 or len(rows) <= 3:
            c = row["conds"]
            msg = (f"  [{len(rows)}/{len(items)}] idx={pidx} "
                   f"T_src={row['T_src']} ")
            if is_multi:
                msg += (f"gap_tok={detail.get('gap_tokens')} "
                        f"drift12={row['total_position_drift']} "
                        f"dur(bl/1/2/12)="
                        f"{c['bl']['dur_sec']:.2f}/{c['inj1']['dur_sec']:.2f}/"
                        f"{c['inj2']['dur_sec']:.2f}/{c['inj12']['dur_sec']:.2f}")
            else:
                msg += (f"delta={c['inj']['delta_tok']} "
                        f"md5_match={c['inj'].get('md5_match')} "
                        f"pcm_match={c['inj'].get('md5_pcm_match')} "
                        f"repeat={c['inj'].get('repeat_bit_identical')}")
            print(msg, flush=True)
        if len(rows) % args.ckpt_every == 0:
            with open(ckpt, "w") as f:
                json.dump({"rows": rows, "skipped": skipped}, f,
                          ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f:
        json.dump({"rows": rows, "skipped": skipped}, f, ensure_ascii=False,
                  indent=1)

    summary = {
        "mode": args.mode, "method": "speecht5", "lang": args.lang,
        "len_policy": args.len_policy,
        "pairs_module": args.pairs_module if is_multi
        else "en_cross_culture_names",
        "s5_xvector_idx": args.s5_xvector_idx, "speaker_source": spk_src,
        "speaker_sum": spk_sum,
        "tokenizer": type(tkz).__name__, "tokenizer_is_fast": bool(tkz.is_fast),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "n_items": len(items), "n_generated": len(rows),
        "n_skipped": len(skipped),
        "conds": conds,
        "mean_dur_sec": {c: round(float(np.mean(
            [r["conds"][c]["dur_sec"] for r in rows])), 4) for c in conds}
        if rows else {},
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    if is_multi and rows:
        summary["mean_abs_position_drift"] = round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3)
        summary["n_pooled_spans"] = int(sum(
            1 for r in rows for p in r["conds"]["inj12"]["per_span"]
            if p["pooled"]))
    if args.compare_dir and rows:
        for key in ("md5_match", "md5_pcm_match"):
            vals = [r["conds"][conds[0]].get(key) for r in rows]
            summary[key] = {
                "n": sum(1 for v in vals if v is not None),
                "n_match": sum(1 for v in vals if v is True),
                "n_mismatch": sum(1 for v in vals if v is False),
                "mismatch_pair_idx": [r["pair_idx"] for r in rows
                                      if r["conds"][conds[0]].get(key)
                                      is False][:20],
                "compare_dir": args.compare_dir,
            }
    if args.repeat_check and rows:
        vals = [r["conds"][conds[0]].get("repeat_bit_identical") for r in rows]
        summary["repeat_bit_identical"] = {
            "n": sum(1 for v in vals if v is not None),
            "n_identical": sum(1 for v in vals if v is True),
        }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows, "skipped": skipped}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n=== SpeechT5 multi-entity ({args.mode}) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if skipped:
        print(f"[Warn] {len(skipped)} items skipped; see results.json['skipped']")


if __name__ == "__main__":
    main()
