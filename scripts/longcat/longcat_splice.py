#!/usr/bin/env python3
"""LongCat-AudioDiT-1B 用 SpeechSplice コア実装.

凍結した umt5-base encoder の最終出力 H = LN(h_L) + LN(h_0) を 2 回計算し、
実体語スパンだけを目標読みテキスト側の対応スパンで置換して
    H~ = (H^src_1..s0-1, H^tgt_u0..u1-1, H^src_s1..T)
を作り、公式 forward の text_embedding 引数にそのまま渡す。重み更新なし、hook 不要。
系列長は T -> T - m + n に変わるが、mask は渡したテンソルの実長から導出され
RoPE は毎回組み直されるので公式コードは 1 行も変えなくて通る。

このファイルは third_party/longcat_audiodit_survey/ (公式実装のコピー) を
import するだけで、あちらは 1 行も変更していない。

環境メモ (2026-09-16 実測):
  * 公式 configuration_audiodit.py は transformers v5 の PreTrainedConfig を
    import するが、現環境は 4.57.3 で PretrainedConfig しかない。
    install_v5_compat_shim() で別名を張れば import は通る (v5 依存はこれだけ)。
  * 公式 model.safetensors は v5 形式で書かれており __metadata__ に format キーが
    無い。4.57.3 の from_pretrained はこれを弾く
    ("Incompatible safetensors file. File metadata is not [...] but None")。
    そこで load_state_dict で直接読む。metadata は tied weight の別名表
    {"text_encoder.shared.weight": "text_encoder.encoder.embed_tokens.weight"}
    なので、その別名を張ってから strict=True 相当で読み込む
    (missing/unexpected ともに 0 を実測)。
"""
import contextlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
VENDOR_PARENT = os.path.join(PROJECT_ROOT, "third_party")
VENDOR_PKG = "longcat_audiodit_survey"

DEFAULT_MODEL_ID = "meituan-longcat/LongCat-AudioDiT-1B"
DEFAULT_TOKENIZER_ID = "google/umt5-base"
SEED = 42
DEFAULT_STEPS = 16
DEFAULT_CFG = 4.0
DEFAULT_GUIDANCE = "cfg"


# --------------------------------------------------------------------------
# HF cache: slurm_run.sh は -e HF_HOME=/tmp/.cache/huggingface を渡すが、その実体は
# ${HOME}/.cache/huggingface で空。実際に重みがあるのは
# <PROJECT_ROOT>/.cache/huggingface (コンテナ内 /mnt/.cache/huggingface)。
# setdefault ではコンテナの環境変数に負けるのでここで上書きする。
# scripts/dia/run_dia_eval.py:44-49 と同じ扱い。transformers import 前に済ませる。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


def setup_hf_home(hf_home=None):
    hf_home = hf_home or os.environ.get("LONGCAT_HF_HOME") or DEFAULT_HF_HOME
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_home, "modules")
    return hf_home


setup_hf_home()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def install_v5_compat_shim():
    """公式コードが要求する transformers v5 名を 4.57.3 に生やす.

    v5 依存は configuration_audiodit.py の `from transformers import PreTrainedConfig`
    だけ。config JSON の "dtype" キーと sub_configs は 4.57.3 が既に解釈できる
    (dtype -> torch_dtype の変換つき) ことを実測で確認済み。
    """
    import transformers
    added = []
    if not hasattr(transformers, "PreTrainedConfig"):
        transformers.PreTrainedConfig = transformers.PretrainedConfig
        added.append("PreTrainedConfig")
    return added


def import_audiodit():
    """third_party の公式実装コピーを package として import する."""
    install_v5_compat_shim()
    if VENDOR_PARENT not in sys.path:
        sys.path.insert(0, VENDOR_PARENT)
    import importlib
    return importlib.import_module(VENDOR_PKG)


def vendored_utils():
    """公式 utils.py の normalize_text / approx_duration_from_text をそのまま使う.

    utils.py 自体は transformers を触らないが、package として import すると
    __init__.py 経由で configuration_audiodit.py が読まれるので、
    ここでも shim を先に入れておく (normalize_text だけ使う呼び出し口が
    import_audiodit() を通らずに ImportError になるのを防ぐ)。
    """
    install_v5_compat_shim()
    if VENDOR_PARENT not in sys.path:
        sys.path.insert(0, VENDOR_PARENT)
    import importlib
    return importlib.import_module(f"{VENDOR_PKG}.utils")


def seed_all(seed=SEED):
    """生成の直前に必ず呼ぶ。初期ノイズ y0 = randn(duration, latent_dim) を揃えるため。

    cudnn.benchmark=False は公式 inference.py:29 と同じ。deterministic=True は
    本プロジェクトの既存実装 (speechsplice_release/src/injection.py:44-45) に合わせる。
    """
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def local_snapshot(hub_id):
    """HF_HUB_CACHE 内のスナップショットディレクトリを返す (offline 前提)."""
    if os.path.isdir(hub_id):
        return hub_id
    d = os.path.join(os.environ["HF_HUB_CACHE"],
                     f"models--{hub_id.replace('/', '--')}", "snapshots")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"snapshot dir not found for {hub_id}: {d}")
    snaps = sorted(os.listdir(d))
    return os.path.join(d, snaps[-1])


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
@dataclass
class LongCatBundle:
    model: object
    tokenizer: object
    device: str
    dtype: object
    sampling_rate: int
    latent_hop: int
    max_wav_duration: float
    model_dir: str
    load_report: dict = field(default_factory=dict)


def load_state_dict_into(model, safetensors_path, assign=True):
    """v5 形式 safetensors を 4.57.3 で読む.

    from_pretrained は __metadata__ の format キーを要求して失敗するので
    safetensors を直接読む。metadata は tied weight の別名表なのでそれを展開し、
    missing / unexpected が 0 であることを確認して返す。
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    sd = load_file(safetensors_path)
    with safe_open(safetensors_path, framework="pt") as f:
        meta = f.metadata() or {}
    tied = {}
    for alias, src in meta.items():
        if src in sd and alias not in sd:
            sd[alias] = sd[src]
            tied[alias] = src
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=assign)
    report = {
        "n_tensors": len(sd),
        "tied_aliases": tied,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "safetensors_metadata": meta,
    }
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}")
    return report


def load_longcat(model_id=DEFAULT_MODEL_ID, tokenizer_id=DEFAULT_TOKENIZER_ID,
                 device=None, dtype=torch.float32, vae_half=True, load_mode="meta"):
    """公式の重みを読み、推論用 bundle を返す.

    load_mode="meta": meta device で骨組みを作り assign=True で実体を差し込む
                      (乱数初期化を丸ごと省けるので速い。実測 1 秒未満)
    load_mode="cpu" : 通常どおり CPU 上で構築してから copy する保険用の経路
    vae_half        : 公式 inference.py と同じ (model.vae.to_half())。
                      Wav-VAE は元実装が fp16 で動く前提なので既定で True。
    """
    audiodit = import_audiodit()
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = local_snapshot(model_id)
    cfg = audiodit.AudioDiTConfig(**json.load(open(os.path.join(model_dir, "config.json"))))

    if load_mode == "meta":
        with torch.device("meta"):
            model = audiodit.AudioDiTModel(cfg)
        report = load_state_dict_into(model, os.path.join(model_dir, "model.safetensors"),
                                      assign=True)
    else:
        model = audiodit.AudioDiTModel(cfg)
        report = load_state_dict_into(model, os.path.join(model_dir, "model.safetensors"),
                                      assign=False)

    model = model.to(device=device, dtype=dtype)
    if vae_half:
        model.vae.to_half()
    model.eval()
    model.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(local_snapshot(tokenizer_id))
    if not tok.is_fast:
        raise RuntimeError(
            "fast tokenizer が必要 (return_offsets_mapping を使うため)。"
            f"{tokenizer_id} の tokenizer.json を確認すること。")

    return LongCatBundle(
        model=model, tokenizer=tok, device=device, dtype=dtype,
        sampling_rate=cfg.sampling_rate, latent_hop=cfg.latent_hop,
        max_wav_duration=cfg.max_wav_duration, model_dir=model_dir,
        load_report=report,
    )


# --------------------------------------------------------------------------
# text / span
# --------------------------------------------------------------------------
def normalize_text(text):
    """公式 utils.normalize_text (小文字化・引用符除去・空白正規化) をそのまま使う."""
    return vendored_utils().normalize_text(text)


def duration_frames(bundle, text, prompt_frames=0):
    """公式 inference.py:95-101 と同じ duration 推定 (latent frame 数).

    正規化済みテキストを渡すこと。文字数ベースの決定的な関数なので、
    同じテキストなら常に同じ値になる = BL と INJ で共有できる。
    """
    u = vendored_utils()
    sr = bundle.sampling_rate
    hop = bundle.latent_hop
    max_dur = bundle.max_wav_duration
    prompt_time = prompt_frames * hop / sr
    dur_sec = u.approx_duration_from_text(text, max_duration=max_dur - prompt_time)
    duration = int(dur_sec * sr // hop)
    return min(duration + prompt_frames, int(max_dur * sr // hop)), float(dur_sec)


def entity_char_span(norm_text, word):
    """正規化済みテキスト中の実体語の文字スパン [start, end) を返す.

    語境界つきで検索し、複数ヒットしたらエラーにする (どのスパンを差し替えたか
    曖昧なまま生成しないため)。
    """
    w = normalize_text(word).strip()
    if not w:
        raise ValueError(f"empty entity word: {word!r}")
    hits = list(re.finditer(r"(?<![0-9a-z])" + re.escape(w) + r"(?![0-9a-z])", norm_text))
    if not hits:
        raise ValueError(f"entity {w!r} not found in normalized text {norm_text!r}")
    if len(hits) > 1:
        raise ValueError(f"entity {w!r} occurs {len(hits)} times in {norm_text!r}")
    return hits[0].span(), w


def entity_token_span(tokenizer, text, word, expand=None):
    """文字スパン -> SentencePiece トークンスパンの写像.

    offsets と交差するトークンを全部拾う。SentencePiece の "▁word" は直前の
    空白を含む offset を持つ (例: ' yamada' -> (10,17)) ので、交差判定にして
    strip() で往復一致を確かめる。

    戻り値の tok_start/tok_end は [start, end) の半開区間。

    expand=(pre, post) を渡すと、要求する表層を実体語の左右に pre / post 文字だけ
    広げてからトークンスパンを取り、往復一致もその表層に対して判定する
    (最小対称拡張のときだけ使う。entity_token_span_pair 参照)。
    **既定は None で、そのときは 1 行も挙動が変わらない** (want == w)。
    """
    norm = normalize_text(text)
    (c0, c1), w = entity_char_span(norm, word)
    want = w
    if expand is not None:
        pre, post = int(expand[0]), int(expand[1])
        if pre < 0 or post < 0:
            raise ValueError(f"expand は非負であること: {expand!r}")
        if c0 - pre < 0 or c1 + post > len(norm):
            raise ValueError(f"expand={expand!r} が文の外に出る: {norm!r}")
        c0, c1 = c0 - pre, c1 + post
        want = norm[c0:c1]
    enc = tokenizer(norm, return_offsets_mapping=True, add_special_tokens=True)
    ids = list(enc["input_ids"])
    offs = list(enc["offset_mapping"])
    sel = [i for i, (a, b) in enumerate(offs) if b > a and a < c1 and b > c0]
    if not sel:
        raise ValueError(f"no token overlaps entity span {(c0, c1)} in {norm!r}")
    if sel != list(range(sel[0], sel[-1] + 1)):
        raise ValueError(f"entity token span is not contiguous: {sel}")
    tok_start, tok_end = sel[0], sel[-1] + 1
    covered = norm[offs[tok_start][0]:offs[tok_end - 1][1]]
    roundtrip_ok = covered.strip() == want.strip()
    span = {
        "norm_text": norm,
        "word": w,
        "char_span": [int(c0), int(c1)],
        "tok_start": int(tok_start),
        "tok_end": int(tok_end),
        "n_tokens_entity": int(tok_end - tok_start),
        "n_tokens_total": int(len(ids)),
        "input_ids": ids,
        "offsets": [[int(a), int(b)] for a, b in offs],
        "covered_text": covered,
        "roundtrip_ok": bool(roundtrip_ok),
        "entity_tokens": tokenizer.convert_ids_to_tokens(
            [ids[i] for i in range(tok_start, tok_end)]),
    }
    # 拡張したときだけ鍵を足す (expand=None の記録は従来と同じ形のまま)。
    if expand is not None:
        span["requested_surface"] = want
        span["expand"] = [int(expand[0]), int(expand[1])]
    return span


def verify_entity_span(span, strict=True):
    if not span["roundtrip_ok"]:
        msg = (f"span roundtrip failed: covered={span['covered_text']!r} "
               f"!= word={span['word']!r} in {span['norm_text']!r}")
        if strict:
            raise ValueError(msg)
        return False, msg
    return True, ""


# --------------------------------------------------------------------------
# 最小対称拡張への配線 (2026-09-18 ユーザ承認)
#
# 規則の実装は scripts/projection/span_extend.py の 1 箇所だけ。ここは配線で、
# **語そのままで往復一致するなら従来の経路をそのまま通す**。拡張が起きるのは
# 従来 "skipped_span_roundtrip" で捨てていた対だけ (中国語の「的」など、
# トークン化器が実体語と隣の機能語を 1 トークンに融合する対)。
# --------------------------------------------------------------------------
SPAN_EXTEND_DIR = os.path.join(PROJECT_ROOT, "scripts", "projection")


def _load_minimal_symmetric_surface():
    """規則の実装を遅延 import する (往復一致しない対に当たったときだけ読む)."""
    if SPAN_EXTEND_DIR not in sys.path:
        sys.path.insert(0, SPAN_EXTEND_DIR)
    from span_extend import minimal_symmetric_surface
    return minimal_symmetric_surface


def _find_occurrence_index(text, needle, pos):
    """pos が needle の何番目の出現かを返す (見つからなければ None).

    走査は span_extend.covered_surface と同じ規約 (find を 1 文字ずつ進める、
    重なりを許す) にしてある。occurrence 番号の食い違いを避けるため。
    """
    k, start = 0, 0
    while True:
        p = text.find(needle, start)
        if p < 0:
            return None
        if p == pos:
            return k
        k += 1
        start = p + 1


def entity_token_span_pair(tokenizer, text_a, word_a, text_b, word_b,
                           span_extension=False):
    """元 (text_a) と目標 (text_b) の実体語スパンをまとめて取る.

    戻り値 (s_span, t_span, ext)。ext は実際に使われた拡張
    {"extra", "pre", "post", "surface_a", "surface_b"} か None。

      * 両側とも語そのままで往復一致する
        -> entity_token_span を 2 回呼んだ結果をそのまま返す。
           **従来と bit 単位で同一** (span_extension の値に依らず、
           minimal_symmetric_surface も import しない)。
      * 往復一致しない かつ span_extension=True
        -> 最小対称拡張を試し、得られた表層でスパンを取り直す。
           取り直したスパンが**この経路で**往復一致しなければ拡張は使わず、
           拡張前のスパンを返す (呼び出し側が従来どおり落とす)。
    """
    s = entity_token_span(tokenizer, text_a, word_a)
    t = entity_token_span(tokenizer, text_b, word_b)
    if s["roundtrip_ok"] and t["roundtrip_ok"]:
        return s, t, None                      # 従来の経路そのまま
    if not span_extension:
        return s, t, None
    occ_a = _find_occurrence_index(s["norm_text"], s["word"], s["char_span"][0])
    occ_b = _find_occurrence_index(t["norm_text"], t["word"], t["char_span"][0])
    if occ_a is None or occ_b is None:
        return s, t, None
    r = _load_minimal_symmetric_surface()(
        tokenizer, s["norm_text"], s["word"], t["norm_text"], t["word"],
        occurrence_a=occ_a, occurrence_b=occ_b)
    if r is None or r["plain"]:
        return s, t, None
    try:
        s2 = entity_token_span(tokenizer, text_a, word_a, expand=(r["pre"], r["post"]))
        t2 = entity_token_span(tokenizer, text_b, word_b, expand=(r["pre"], r["post"]))
    except ValueError:
        return s, t, None
    if not (s2["roundtrip_ok"] and t2["roundtrip_ok"]):
        return s, t, None
    if (s2["requested_surface"] != r["surface_a"]
            or t2["requested_surface"] != r["surface_b"]):
        return s, t, None
    ext = {"extra": r["extra"], "pre": int(r["pre"]), "post": int(r["post"]),
           "surface_a": r["surface_a"], "surface_b": r["surface_b"]}
    s2["span_extension"] = dict(ext)
    t2["span_extension"] = dict(ext)
    return s2, t2, ext


# --------------------------------------------------------------------------
# text encoding (encode_text の分解)
# --------------------------------------------------------------------------
def encode_text_parts(model, input_ids, attention_mask):
    """encode_text と同じ H を、LN(h_L) と LN(h_0) に分けて返す.

    公式 encode_text (modeling_audiodit.py:936-968) は
        emb = LN(last_hidden_state);  emb += LN(hidden_states[0]);  return emb.float()
    text_norm_feat / text_add_embed はいずれも config で True。
    ここで組み直した H が encode_text の出力と bit 単位で一致することは
    smoke test で確認している。
    """
    with torch.no_grad():
        out = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
    d = model.text_encoder.config.d_model
    ln_hL = F.layer_norm(out.last_hidden_state, (d,), eps=1e-6)
    ln_h0 = F.layer_norm(out.hidden_states[0], (d,), eps=1e-6)
    if not model.config.text_norm_feat or not model.config.text_add_embed:
        raise RuntimeError("この実装は text_norm_feat=True かつ text_add_embed=True 前提")
    return {
        "H": (ln_hL + ln_h0).float(),
        "ln_hL": ln_hL.float(),
        "ln_h0": ln_h0.float(),
        "hidden_states": out.hidden_states,
        "attention_mask": attention_mask,
    }


def encode_for_text(bundle, text):
    """テキスト 1 本を tokenize して encode_text 相当を計算する."""
    norm = normalize_text(text)
    enc = bundle.tokenizer([norm], padding="longest", return_tensors="pt")
    input_ids = enc["input_ids"].to(bundle.device)
    attn = enc["attention_mask"].to(bundle.device)
    parts = encode_text_parts(bundle.model, input_ids, attn)
    parts["input_ids"] = input_ids
    parts["norm_text"] = norm
    return parts


def _resample_index(m, n, start):
    """長さ m の区間を長さ n に最近傍で貼り直すインデックス列 (絶対 index)."""
    if n == m:
        return [start + j for j in range(n)]
    return [start + min(m - 1, int(j * m / n)) for j in range(n)]


def build_spliced_embedding(src, tgt, s_span, t_span, splice_part="both"):
    """H~ = cat(H^src[:s0], ENT, H^src[s1:]) を作る.

    splice_part="both"   : ENT = H^tgt[u0:u1]  (= LN(h_L^tgt) + LN(h_0^tgt))
    splice_part="hL_only": ENT = LN(h_L^tgt)[u0:u1] + LN(h_0^src)[s0:s1 を n 点に写像]
                           h_0 は文脈非依存の語埋め込みなので、
                           「文脈依存側 (h_L) だけ差し替える」統制条件になる。
                           m != n のときは最近傍 index 写像で長さを合わせる
                           (m == n のペアでは厳密)。
    """
    s0, s1 = s_span["tok_start"], s_span["tok_end"]
    u0, u1 = t_span["tok_start"], t_span["tok_end"]
    m = s1 - s0
    n = u1 - u0
    H_src, H_tgt = src["H"], tgt["H"]

    if splice_part == "both":
        ent = H_tgt[:, u0:u1]
        h0_map = None
    elif splice_part == "hL_only":
        idx = _resample_index(m, n, s0)
        h0_map = [int(i) for i in idx]
        ent = tgt["ln_hL"][:, u0:u1] + src["ln_h0"][:, idx]
    else:
        raise ValueError(f"unknown splice_part={splice_part}")

    H_tilde = torch.cat([H_src[:, :s0], ent, H_src[:, s1:]], dim=1)
    info = {
        "splice_part": splice_part,
        "m_src_entity_tokens": int(m),
        "n_tgt_entity_tokens": int(n),
        "equal_len": bool(m == n),
        "T_src": int(H_src.shape[1]),
        "T_tgt": int(H_tgt.shape[1]),
        "T_spliced": int(H_tilde.shape[1]),
        "delta_tok": int(n - m),
        "h0_source_index_map": h0_map,
    }
    assert info["T_spliced"] == info["T_src"] - m + n, "splice length bookkeeping broken"
    return H_tilde, info


def kappa_by_layer(src, tgt, s_span, t_span):
    """kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}).

    既存の定義 (実体語スパン内 hidden state をトークン方向に単純平均した
    ベクトル同士の cosine) に合わせる。層 0 は embedding 出力。
    加えて、実際に cross-attention の key/value になる H = LN(h_L)+LN(h_0) 上の
    cosine も kappa_H として返す (Table 3 の既存セルとは母数・題材が違うので
    直接比較はできない点に注意)。
    """
    s0, s1 = s_span["tok_start"], s_span["tok_end"]
    u0, u1 = t_span["tok_start"], t_span["tok_end"]

    def cos(a, b):
        return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))

    ks = []
    for hs, ht in zip(src["hidden_states"], tgt["hidden_states"]):
        ks.append(cos(hs[0, s0:s1].mean(0), ht[0, u0:u1].mean(0)))
    kh = cos(src["H"][0, s0:s1].mean(0), tgt["H"][0, u0:u1].mean(0))
    return ks, kh


# --------------------------------------------------------------------------
# AdaLN 側の text 条件を source に固定する統制条件
# --------------------------------------------------------------------------
def _adaln_text_mean(transformer, text, text_len, cond_mask):
    """transformer.forward 内の text_mean を同じ手順で計算する.

    根拠 (modeling_audiodit.py:601-607, 617-620):
        text = self.text_embed(text, cond_mask)
        if self.text_conv: text = self.text_conv_layer(text); text = masked_fill(...)
        text_mean = text.sum(1) / text_len.unsqueeze(1)
    """
    dtype = next(transformer.parameters()).dtype
    x = transformer.text_embed(text.to(dtype), cond_mask)
    if transformer.text_conv:
        x = transformer.text_conv_layer(x)
        x = x.masked_fill(cond_mask.logical_not().unsqueeze(-1), 0.0)
    return x.sum(1) / text_len.unsqueeze(1).to(x.dtype)


class AdaLNSourceOverride:
    """AdaLN に入る text 条件だけを source 由来に固定する context manager.

    transformer.forward は
        norm_cond = t + text_mean
    を作り、これを adaln_global_mlp (全 24 block の scale/shift/gate) と
    norm_out の両方に渡す。text_mean は splice 後の系列から計算されるので、
    splice は cross-attention だけでなく AdaLN 経由の大域条件も動かしてしまう。
    そこで両モジュールの forward_pre_hook で
        delta = text_mean(source) - text_mean(実際に渡された系列)
    を足し、norm_cond を BL と同じ値に戻す。t は打ち消し合うので時刻に依存しない。

    無条件側 (neg_text = zeros) も、系列長が T' から T に変わる分だけ
    text_mean がずれるので、zeros(T') -> zeros(T) の差分で同様に補正する。
    分岐の判定は「渡された text が全ゼロかどうか」で行う。
    """

    def __init__(self, bundle, H_actual, H_source, enabled=True):
        self.bundle = bundle
        self.enabled = enabled
        self.handles = []
        self.transformer = bundle.model.transformer
        self.stats = {}
        if not enabled:
            return
        tr = self.transformer
        dev = H_actual.device
        Ta, Ts = H_actual.shape[1], H_source.shape[1]
        mask_a = torch.ones(1, Ta, dtype=torch.bool, device=dev)
        mask_s = torch.ones(1, Ts, dtype=torch.bool, device=dev)
        len_a = torch.tensor([Ta], device=dev)
        len_s = torch.tensor([Ts], device=dev)
        with torch.no_grad():
            m_act_cond = _adaln_text_mean(tr, H_actual, len_a, mask_a)
            m_src_cond = _adaln_text_mean(tr, H_source, len_s, mask_s)
            z_a = torch.zeros_like(H_actual)
            z_s = torch.zeros_like(H_source)
            m_act_null = _adaln_text_mean(tr, z_a, len_a, mask_a)
            m_src_null = _adaln_text_mean(tr, z_s, len_s, mask_s)
        self.delta_cond = (m_src_cond - m_act_cond).detach()
        self.delta_null = (m_src_null - m_act_null).detach()
        self._is_null = False
        self.stats = {
            "delta_cond_l2": float(self.delta_cond.norm()),
            "delta_null_l2": float(self.delta_null.norm()),
            "text_mean_cond_l2_actual": float(m_act_cond.norm()),
            "text_mean_cond_l2_source": float(m_src_cond.norm()),
        }

    def _branch_hook(self, module, args, kwargs):
        text = kwargs.get("text")
        if text is None and len(args) > 1:
            text = args[1]
        self._is_null = (text is not None) and (not bool(text.any()))
        return None

    def _delta(self):
        return self.delta_null if self._is_null else self.delta_cond

    def _norm_cond_hook(self, module, args):
        if not args:
            return None
        new = list(args)
        # adaln_global_mlp(norm_cond) は args[0]、norm_out(x, emb) は args[1]
        pos = 1 if module is self.transformer.norm_out else 0
        cond = new[pos]
        new[pos] = cond + self._delta().to(cond.dtype)
        return tuple(new)

    def __enter__(self):
        if not self.enabled:
            return self
        tr = self.transformer
        if tr.adaln_type != "global" or not tr.adaln_use_text_cond:
            raise RuntimeError(
                "AdaLN 統制は dit_adaln_type='global' かつ dit_adaln_use_text_cond=True 前提")
        self.handles.append(tr.register_forward_pre_hook(self._branch_hook, with_kwargs=True))
        self.handles.append(tr.adaln_global_mlp.register_forward_pre_hook(self._norm_cond_hook))
        self.handles.append(tr.norm_out.register_forward_pre_hook(self._norm_cond_hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------
def generate_from_embedding(bundle, text_embedding, duration, seed=SEED,
                            steps=DEFAULT_STEPS, cfg_strength=DEFAULT_CFG,
                            guidance_method=DEFAULT_GUIDANCE, adaln_override=None):
    """公式 forward を text_embedding 経路で呼ぶ.

    BL / TGT / INJ すべてこの経路に統一する (input_ids 経路との違いは
    encode_text を内部で呼ぶかどうかだけで、値は一致することを確認済み)。
    seed は呼び出し直前に必ずリセットする。初期ノイズ y0 は
    randn(duration, latent_dim) なので、duration が同じなら BL と INJ の
    ノイズは完全に同一になる = 差は splice だけになる。
    """
    seed_all(seed)
    attn = torch.ones(1, text_embedding.shape[1], dtype=torch.long, device=bundle.device)
    ctx = adaln_override if adaln_override is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        out = bundle.model(
            text_embedding=text_embedding,
            attention_mask=attn,
            prompt_audio=None,
            duration=duration,
            steps=steps,
            cfg_strength=cfg_strength,
            guidance_method=guidance_method,
        )
    wav = out.waveform.squeeze().detach().float().cpu().numpy()
    return wav


def synthesize(bundle, text, duration_text=None, seed=SEED, steps=DEFAULT_STEPS,
               cfg_strength=DEFAULT_CFG, guidance_method=DEFAULT_GUIDANCE):
    """素の TTS (BL = TTS(text_a), TGT = TTS(text_b)).

    duration_text に別のテキストを渡すと duration だけそちら由来に固定できる。
    """
    parts = encode_for_text(bundle, text)
    dtext = normalize_text(duration_text) if duration_text is not None else parts["norm_text"]
    dur, dur_sec = duration_frames(bundle, dtext)
    wav = generate_from_embedding(bundle, parts["H"], dur, seed=seed, steps=steps,
                                  cfg_strength=cfg_strength, guidance_method=guidance_method)
    return {"audio": wav, "duration_frames": dur, "duration_sec_est": dur_sec,
            "n_tokens": int(parts["H"].shape[1]), "norm_text": parts["norm_text"],
            "duration_text": dtext}


def splice_synthesize(bundle, text_a, word_a, text_b, word_b, seed=SEED,
                      steps=DEFAULT_STEPS, cfg_strength=DEFAULT_CFG,
                      guidance_method=DEFAULT_GUIDANCE, splice_part="both",
                      adaln_source=False, collect_kappa=True, strict_span=True,
                      src_parts=None, tgt_parts=None, span_extension=False):
    """SpeechSplice 本体。

    統制条件:
      (a) duration は必ず source テキスト (text_a) 由来 -> BL と完全一致
      (b) splice_part="hL_only" で h_0 を source のまま残す
      (c) adaln_source=True で AdaLN の text 条件を source に固定

    span_extension=True のときだけ、語そのままで往復一致しない対に最小対称拡張を
    試す (既定 False = 従来どおりその対で ValueError)。往復一致する対は
    span_extension の値に関わらず従来と同じ経路を通る。
    """
    src = src_parts if src_parts is not None else encode_for_text(bundle, text_a)
    tgt = tgt_parts if tgt_parts is not None else encode_for_text(bundle, text_b)
    s_span, t_span, span_ext = entity_token_span_pair(
        bundle.tokenizer, text_a, word_a, text_b, word_b,
        span_extension=span_extension)
    verify_entity_span(s_span, strict=strict_span)
    verify_entity_span(t_span, strict=strict_span)

    H_tilde, info = build_spliced_embedding(src, tgt, s_span, t_span, splice_part=splice_part)
    dur, dur_sec = duration_frames(bundle, src["norm_text"])

    override = AdaLNSourceOverride(bundle, H_tilde, src["H"], enabled=adaln_source)
    with override:
        wav = generate_from_embedding(bundle, H_tilde, dur, seed=seed, steps=steps,
                                      cfg_strength=cfg_strength,
                                      guidance_method=guidance_method,
                                      adaln_override=None)
    out = {
        "audio": wav, "duration_frames": dur, "duration_sec_est": dur_sec,
        "s_span": [s_span["tok_start"], s_span["tok_end"]],
        "t_span": [t_span["tok_start"], t_span["tok_end"]],
        "s_tokens": s_span["entity_tokens"], "t_tokens": t_span["entity_tokens"],
        "adaln_source": bool(adaln_source),
        "adaln_stats": override.stats,
    }
    out.update(info)
    if span_ext is not None:
        out["span_extension"] = span_ext
    if collect_kappa:
        ks, kh = kappa_by_layer(src, tgt, s_span, t_span)
        out["kappa_by_layer"] = [round(v, 6) for v in ks]
        out["kappa_last_layer"] = round(ks[-1], 6)
        out["kappa_H"] = round(kh, 6)
    return out
