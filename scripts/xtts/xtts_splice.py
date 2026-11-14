#!/usr/bin/env python3
"""XTTS-v2 (coqui/XTTS-v2) 用 SpeechSplice コア実装.

なぜ XTTS-v2 か
--------------
Llasa-3B も Qwen3-TTS も「事前学習済み LLM に離散 codec を足した」作りなので、
「decoder-only ではなく LLM backbone 特有の現象では」という反論が成り立つ。
XTTS-v2 の trunk は **TTS 用に最初から学習された GPT-2** (30 層 / d=1024 /
16 heads / text vocab 6681) で、事前学習済み LLM ではない。ここでも効かない
なら decoder-only という構造自体の性質だと言える。

構造 (推測ではなく coqui-tts 0.27.5 の実装を読んだ結果)
------------------------------------------------------
* テキストは **prefix として連結**される。cross-attention は無い。
  TTS/tts/layers/xtts/gpt.py: GPT.compute_embeddings()

      text_inputs = F.pad(text_inputs, (0, 1), value=stop_text_token)   # [STOP]=0
      text_inputs = F.pad(text_inputs, (1, 0), value=start_text_token)  # [START]=261
      emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
      emb = torch.cat([cond_latents, emb], dim=1)
      self.gpt_inference.store_prefix_emb(emb)

  したがって GPT2 系列の並びは

      [cond latents x 32][START][ [en] text tokens ... ][STOP][start_audio] mel...
       0            31    32     33 ...                              (pass 1 のみ)

  実体語の行番号 = 32 (cond) + 1 ([START]) + (text token 列中の index)。
  text token 列の index 0 は言語タグ ``[en]`` (id 259) なので、実体語は必ず
  index >= 1 に来る。

* **Perceiver の cross-attention はテキストに対するものではない。**
  gpt_use_perceiver_resampler=True なので GPT.conditioning_perceiver
  (PerceiverResampler, depth=2, num_latents=32) が存在し、その中には
  cross-attention がある。ただし入力は
  ``Xtts.get_gpt_cond_latents`` が作る **参照音声の mel** だけで
  (xtts.py:263-292)、テキストは一切通らない。出力は [1, 32, 1024] の
  cond latents で、これは GPT2 trunk の prefix として前に連結されるだけ。
  本実装の splice は ``model.gpt.gpt.h[l]`` (GPT2Block) の出力にしか触らず、
  cond latents は評価開始時に 1 回だけ計算して 3 条件で bitwise 共有する
  (assert_reference_unchanged() で毎回照合する)。

* **英語は書記素のまま。**VoiceBpeTokenizer.encode は
  multilingual_cleaners (小文字化 / 数詞展開 / 略語展開 / 記号展開 /
  空白整形) を通すだけで phonemizer を通らない (tokenizer.py:645-665)。
  一方 **日本語は cutlet で romaji 化、中国語は pypinyin で拼音化**される
  (japanese_cleaners / chinese_transliterate)。romaji も pinyin も読みを
  直接書き込む入口なので本手法の前提を壊す。**このスクリプトは英語のみ**。

* **推論は GPT2 trunk を 2 回通る。**Xtts.inference (xtts.py:519-551) は
    pass 1: self.gpt.generate(...)          -> mel code を自己回帰生成
    pass 2: self.gpt(..., return_latent=True) -> HiFiGAN に渡す latent を作る
  の 2 段構え。どちらも同じ ``gpt.gpt.h[l]`` を通り、どちらも
  [cond][START][text][STOP] という同じ並びでテキスト行を含む。
  片方だけ差し替えると、もう一方が元テキストのままの latent を作るので
  介入が半分打ち消される。既定では **両方**に同じ置換を掛ける
  (--no_splice_latent_pass で pass 2 を外せる)。
  因果マスクのおかげでテキスト行の hidden state は pass 1 と pass 2 で
  同一 (右側にある start_audio / mel は左のテキスト行に影響しない) なので、
  目標ベクトルは pass 1 形状の prefill から 1 回取れば両方に使える。

長さ不一致の扱い (Qwen3-TTS / Llasa-3B と同じ)
---------------------------------------------
decoder-only で系列長を変えると後続トークンの位置と KV cache がずれる。
したがって **系列長は変えない**。
  m == n : 行対行でそのまま代入
  m != n : 目標スパンをトークン方向に平均した 1 本を m 行に broadcast
XTTS-v2 の text vocab は 6681 と小さく単語が複数トークンに割れるので、
EN cross-culture 300 ペアでは m == n は 40/300 しかない (CPU 実測)。

出所 (この実装が踏襲した既存コード)
----------------------------------
  * prefill 限定 hook / fired flag / 長さ不一致の mean-pool 展開 / kappa
      scripts/llasa/llasa_splice.py (PrefillSplice, kappa_by_layer)
  * スパンの 3 段構え
      speechsplice_release/src/injection.py:401 find_entity_token_range
    トークン境界の検証はしない。窓も広げない。境界に乗ったかどうかだけ
    manifest に aligned で残す。

重みは一切更新しない。推論時のみ。BL と INJ は seed も復号設定も
prompt も系列長も共有し、差は splice だけ。
"""
import os
import re
import sys
import types
from dataclasses import dataclass, field

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

DEFAULT_MODEL_ID = "coqui/XTTS-v2"
SEED = 42

# XTTS-v2 は 30 層。Qwen3-TTS / Llasa-3B (28 層) の格子 0/7/14/21/26 と
# 相対的な深さを合わせた格子。round(l * 30 / 28) (端数は 0.5 で切り上げ):
#   0 -> 0.00 -> 0     7 -> 7.50 -> 8     14 -> 15.00 -> 15
#  21 -> 22.50 -> 23  26 -> 27.86 -> 28
# 深さの比は 0.000 / 0.267 / 0.500 / 0.767 / 0.933 で、28 層側の
# 0.000 / 0.250 / 0.500 / 0.750 / 0.929 に対応する。最深点の 26 = L-2 という
# 位置 (最終層の 1 つ手前) も 28 = L-2 で保存されている。
SWEEP_LAYERS = (0, 8, 15, 23, 28)
EXPECTED_NUM_LAYERS = 30
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_TEXT_VOCAB = 6681
EXPECTED_COND_LEN = 32          # perceiver resampler の num_latents

# coqui/XTTS-v2 の config.json にある公式既定値。BL / TGT / INJ で完全に共有する。
DEFAULT_TEMPERATURE = 0.75
DEFAULT_TOP_P = 0.85
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 5.0
DEFAULT_LENGTH_PENALTY = 1.0
# 参照音声の切り出し (config.json: gpt_cond_len / gpt_cond_chunk_len / max_ref_len)
DEFAULT_GPT_COND_LEN = 30
DEFAULT_GPT_COND_CHUNK_LEN = 4
DEFAULT_MAX_REF_LEN = 30
DEFAULT_SOUND_NORM_REFS = False

OUTPUT_SAMPLE_RATE = 24000      # config.json: model_args.output_sample_rate

# load_checkpoint が読むファイル。dvae.pth (210 MB) は学習用なので落とさない。
XTTS_FILES = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth")

# 参照実装 (pip の coqui-tts 0.27.5 の wheel を展開したもの)。
# coqui-tts が入っていない環境で tokenizer だけ動かすときに使う。
DEFAULT_REF_SRC = os.path.join(PROJECT_ROOT, "scripts", "xtts", "_ref", "src")


# --------------------------------------------------------------------------
# HF cache: slurm_run.sh は -e HF_HOME=/tmp/.cache/huggingface を渡すが、その実体は
# ${HOME}/.cache/huggingface。重みが置いてあるのは <PROJECT_ROOT>/.cache/huggingface
# (コンテナ内 /mnt/.cache/huggingface)。scripts/llasa/llasa_splice.py:80-95 と同じ扱い。
# transformers / huggingface_hub を import する前に済ませる。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


def setup_hf_home(hf_home=None):
    hf_home = hf_home or os.environ.get("XTTS_HF_HOME") or DEFAULT_HF_HOME
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_home, "modules")
    return hf_home


setup_hf_home()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def seed_all(seed=SEED):
    """生成の直前に必ず呼ぶ。BL / TGT / INJ でサンプリングの乱数列を揃えるため。"""
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


# --------------------------------------------------------------------------
# モデルの所在
# --------------------------------------------------------------------------
def local_snapshot(hub_id, required=True):
    """HF_HUB_CACHE 内のスナップショットディレクトリを返す (offline 前提).

    ダウンロードは呼び出し側が明示的に許可したときだけ行う
    (事故で 2 GB を落とさないための既定)。
    """
    if os.path.isdir(hub_id):
        return hub_id
    d = os.path.join(os.environ["HF_HUB_CACHE"],
                     f"models--{hub_id.replace('/', '--')}", "snapshots")
    if not os.path.isdir(d):
        if required:
            raise FileNotFoundError(
                f"snapshot dir not found for {hub_id}: {d}\n"
                f"  ローカルに無い。取得を許可するなら --allow_download を付けること。")
        return None
    snaps = [s for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
    if not snaps:
        if required:
            raise FileNotFoundError(f"no snapshot under {d}")
        return None
    return os.path.join(d, snaps[-1])


def ensure_xtts_dir(model_id=DEFAULT_MODEL_ID, allow_download=False, files=XTTS_FILES):
    """XTTS-v2 の checkpoint ディレクトリを返す (無い分だけ落とす).

    coqui/XTTS-v2 は gated ではないが license: other (Coqui Public Model
    License, 非商用) なので、研究評価以外に使わないこと。
    落とす実体: config.json 4 KB / vocab.json 353 KB /
    speakers_xtts.pth 7.4 MB / model.pth 1.74 GB = 約 1.75 GB。
    dvae.pth (200 MB) は学習専用なので落とさない。
    """
    if os.path.isdir(model_id):
        d = model_id
    else:
        d = local_snapshot(model_id, required=False)
    missing = [] if d is None else [f for f in files if not os.path.exists(os.path.join(d, f))]
    if d is not None and not missing:
        return d
    if not allow_download:
        raise FileNotFoundError(
            f"{model_id} のローカル snapshot が不完全 (dir={d}, missing={missing or 'ALL'})。"
            f" HF から取得するなら --allow_download を付けること (約 1.75 GB)。")
    from huggingface_hub import hf_hub_download
    got = None
    for f in files:
        p = hf_hub_download(model_id, f)
        got = os.path.dirname(p)
    return got


# --------------------------------------------------------------------------
# tokenizer (coqui-tts 本体が無くても動く経路を用意する)
# --------------------------------------------------------------------------
_REF_TOKENIZER_MODULE = None


def _load_ref_tokenizer_module(ref_src=None):
    """coqui-tts 未導入の環境で tokenizer.py だけを読み込む (CPU 検証用).

    TTS パッケージ本体は importlib.metadata.version("coqui-tts") を要求するので
    そのままでは import できない。tokenizer.py が import する 3 つだけを
    差し替える:
      * ko_speech_tools.hangul_romanize          -> 韓国語専用。未使用。
      * TTS.tts.layers.xtts.zh_num2words.TextNorm-> 中国語専用。未使用。
      * TTS.tts.utils.text.cleaners の 2 関数     -> 実体を写したもの:
            lowercase(text)           = text.lower()                (cleaners.py:38-39)
            collapse_whitespace(text) = re.sub(r"\\s+", " ", text).strip()
                                                                    (cleaners.py:42-43)
    差し替えるのは英語経路で 1 度も呼ばれない関数と、上の 2 行だけ。
    multilingual_cleaners / VoiceBpeTokenizer 本体は配布物そのままを使う。
    """
    global _REF_TOKENIZER_MODULE
    if _REF_TOKENIZER_MODULE is not None:
        return _REF_TOKENIZER_MODULE
    import importlib.util

    ref_src = ref_src or os.environ.get("XTTS_REF_SRC") or DEFAULT_REF_SRC
    path = os.path.join(ref_src, "TTS", "tts", "layers", "xtts", "tokenizer.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"参照実装が無い: {path}\n"
            f"  coqui-tts を入れるか、wheel を展開して --xtts_ref_src で指すこと。")

    if "ko_speech_tools" not in sys.modules:
        ko = types.ModuleType("ko_speech_tools")

        def _ko_unused(*a, **k):
            raise NotImplementedError("Korean path is not used (English only).")

        ko.hangul_romanize = _ko_unused
        sys.modules["ko_speech_tools"] = ko

    for name in ("TTS", "TTS.tts", "TTS.tts.layers", "TTS.tts.layers.xtts",
                 "TTS.tts.utils", "TTS.tts.utils.text"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m

    if "TTS.tts.layers.xtts.zh_num2words" not in sys.modules:
        zh = types.ModuleType("TTS.tts.layers.xtts.zh_num2words")

        class TextNorm:  # noqa: D401 - 中国語専用。英語経路では呼ばれない。
            def __init__(self, *a, **k):
                raise NotImplementedError("Chinese path is not used (English only).")

        zh.TextNorm = TextNorm
        sys.modules["TTS.tts.layers.xtts.zh_num2words"] = zh

    if "TTS.tts.utils.text.cleaners" not in sys.modules:
        cl = types.ModuleType("TTS.tts.utils.text.cleaners")
        _ws_re = re.compile(r"\s+")
        cl.lowercase = lambda text: text.lower()
        cl.collapse_whitespace = lambda text: re.sub(_ws_re, " ", text).strip()
        sys.modules["TTS.tts.utils.text.cleaners"] = cl

    spec = importlib.util.spec_from_file_location("_xtts_tokenizer_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_xtts_tokenizer_ref"] = mod
    spec.loader.exec_module(mod)
    _REF_TOKENIZER_MODULE = mod
    return mod


def get_tokenizer_class(allow_reference=True, ref_src=None):
    """VoiceBpeTokenizer クラスと、どこから来たか ("package"/"reference") を返す."""
    try:
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        return VoiceBpeTokenizer, "package"
    except Exception:
        if not allow_reference:
            raise
    mod = _load_ref_tokenizer_module(ref_src)
    return mod.VoiceBpeTokenizer, "reference"


def load_tokenizer(vocab_path=None, model_id=DEFAULT_MODEL_ID, allow_download=False,
                   allow_reference=True, ref_src=None):
    """vocab.json から VoiceBpeTokenizer を作る (モデル重みは読まない).

    CPU でスパンの検証だけをするときに使う。GPU 側では load_xtts が
    Xtts.load_checkpoint 経由で同じものを作る。
    """
    if vocab_path is None:
        d = ensure_xtts_dir(model_id, allow_download=allow_download, files=("vocab.json",))
        vocab_path = os.path.join(d, "vocab.json")
    cls, origin = get_tokenizer_class(allow_reference=allow_reference, ref_src=ref_src)
    tok = cls(vocab_file=vocab_path)
    n = tok.get_number_tokens()
    if n != EXPECTED_TEXT_VOCAB:
        raise RuntimeError(
            f"text vocab が {n} (想定 {EXPECTED_TEXT_VOCAB})。vocab.json が "
            f"coqui/XTTS-v2 のものか確認すること: {vocab_path}")
    return tok, origin, vocab_path


# --------------------------------------------------------------------------
# テキスト -> BPE
# --------------------------------------------------------------------------
def normalize_sentence(text):
    """Xtts.inference が本文に掛ける前処理そのもの (xtts.py:514 `sent.strip().lower()`)."""
    return text.strip().lower()


def bpe_input_string(tokenizer, text, lang="en"):
    """VoiceBpeTokenizer.encode が実際に BPE に渡す文字列を再現する.

    tokenizer.py:658-665
        txt = self.preprocess_text(txt, lang)      # multilingual_cleaners
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids
    ここでは lang tag を付けない版も要るので、文字列を作る所だけ切り出す。
    """
    lang = lang.split("-")[0]
    prep = tokenizer.preprocess_text(text, lang)
    tag = "zh-cn" if lang == "zh" else lang
    return f"[{tag}]" + prep.replace(" ", "[SPACE]")


def bpe_word_string(tokenizer, word, lang="en"):
    """実体語を、本文の中に現れるのと同じ表記に正規化する (lang tag は付けない)."""
    lang = lang.split("-")[0]
    return tokenizer.preprocess_text(word, lang).replace(" ", "[SPACE]")


def raw_bpe_ids(tokenizer, s):
    """lang tag も [SPACE] 置換も済んだ文字列を、そのまま BPE に掛ける."""
    return list(tokenizer.tokenizer.encode(s).ids)


def encode_text(tokenizer, text, lang="en"):
    """本文 -> text token id 列。VoiceBpeTokenizer.encode と一致することを確かめる."""
    s = bpe_input_string(tokenizer, text, lang)
    ids = raw_bpe_ids(tokenizer, s)
    ref = list(tokenizer.encode(text, lang))
    if ids != ref:
        raise RuntimeError(
            "自前の BPE 入力文字列が VoiceBpeTokenizer.encode と一致しない "
            f"({len(ids)} vs {len(ref)} tokens)。スパン特定が使えないので中断する。")
    return ids, s


# --------------------------------------------------------------------------
# スパン特定 (speechsplice_release/src/injection.py:401 と同じ 3 段構え)
# --------------------------------------------------------------------------
def find_entity_token_range(tokenizer, full_text, entity_text, add_bos=0, add_eos=0):
    """実体語が占める token index の半開区間 [start, end) を返す (無ければ None).

    speechsplice_release/src/injection.py:401 find_entity_token_range の移植。
    段の順番も fallback の中身も変えていない。違うのは
    ``text_tokenizer.encode(x, add_special_tokens=False)`` の代わりに
    ``raw_bpe_ids`` (lang tag を付けない生の BPE) を使う点だけ。
    full_text / entity_text は bpe_input_string / bpe_word_string を通した
    「BPE に入る直前の文字列」を渡すこと。

    返す index は **text token 列** の中での位置。GPT2 系列の行番号にするには
    gpt_row_offset(cond_len) を足す。
    """
    offset = 1 if add_bos else 0

    # Primary method: character offsets -> token range via offset mapping.
    # This is position-robust. The legacy subsequence search below fails for
    # mid-sentence entities in space-delimited languages: encode(entity) yields
    # tokens WITHOUT the SentencePiece "\u2581" space marker while the in-sentence
    # entity carries it, and the char-level fallback then mis-computes the
    # prefix length by one token (trailing-space token), silently replacing the
    # word AFTER the entity. (Bug found 2026-07-11.)
    try:
        enc = text_tokenizer(full_text, add_special_tokens=False,
                             return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        pos = full_text.find(entity_text)
        if pos < 0:
            pos = full_text.lower().find(entity_text.lower())
        if pos >= 0 and offsets:
            end_char = pos + len(entity_text)
            idx = [i for i, (s, e) in enumerate(offsets) if s < end_char and e > pos]
            if idx:
                return (min(idx) + offset, max(idx) + 1 + offset)
    except Exception:
        pass  # fast tokenizer が無い場合などは旧ロジックへ

    full_tokens = raw_bpe_ids(tokenizer, full_text)
    entity_tokens = raw_bpe_ids(tokenizer, entity_text)


    # 1. トークン列の部分一致
    for start in range(len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[start:start + len(entity_tokens)] == entity_tokens:
            return (start + offset, start + len(entity_tokens) + offset, "subseq")

    # 2. 文字位置から prefix のトークン数で数える (大小文字を区別)
    entity_pos = full_text.find(entity_text)
    if entity_pos >= 0:
        prefix = full_text[:entity_pos]
        prefix_tokens = raw_bpe_ids(tokenizer, prefix)
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(entity_tokens) + offset, "charpos")

    # 3. 大小文字を無視して探し、実際の表記で取り直す
    low_full = full_text.lower()
    low_ent = entity_text.lower()
    pos = low_full.find(low_ent)
    if pos >= 0:
        actual_surface = full_text[pos:pos + len(entity_text)]
        actual_entity_tokens = raw_bpe_ids(tokenizer, actual_surface)
        for start in range(len(full_tokens) - len(actual_entity_tokens) + 1):
            if full_tokens[start:start + len(actual_entity_tokens)] == actual_entity_tokens:
                return (start + offset,
                        start + len(actual_entity_tokens) + offset, "ci_subseq")
        prefix_tokens = raw_bpe_ids(tokenizer, full_text[:pos])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(actual_entity_tokens) + offset, "ci_charpos")

    return None


def entity_token_span(tokenizer, text, word, lang="en"):
    """本文と実体語から、text token 列でのスパンと検証用の情報をまとめて返す.

    **スパン自体は find_entity_token_range が返したものをそのまま使う。**
    トークン境界に乗っているかどうかは aligned に記録するだけで、
    スパンを広げたり縮めたりしない (ユーザ決定済みの方針)。

      roundtrip_ok : スパンのトークン文字列を連結したら実体語表記に戻るか
      aligned      : tokenizers の offsets から見て、スパンが実体語の
                     文字範囲とちょうど一致するか (境界に乗ったか)
    """
    sent = normalize_sentence(text)
    full_s = bpe_input_string(tokenizer, sent, lang)
    word_s = bpe_word_string(tokenizer, normalize_sentence(word), lang)
    r = find_entity_token_range(tokenizer, full_s, word_s)
    if r is None:
        raise ValueError(f"entity {word!r} not found in {full_s!r}")
    tok_start, tok_end, how = r

    enc = tokenizer.tokenizer.encode(full_s)
    ids = list(enc.ids)
    offs = [(int(a), int(b)) for a, b in enc.offsets]
    if not (0 <= tok_start < tok_end <= len(ids)):
        raise ValueError(
            f"span [{tok_start},{tok_end}) が token 列 (len {len(ids)}) の外: {full_s!r}")

    toks = [tokenizer.tokenizer.id_to_token(i) for i in ids[tok_start:tok_end]]
    covered_tokens = "".join(toks)
    c0, c1 = offs[tok_start][0], offs[tok_end - 1][1]
    covered_chars = full_s[c0:c1]
    return {
        "word": word,
        "word_bpe": word_s,
        "sent": sent,
        "bpe_input": full_s,
        "how": how,
        "tok_start": int(tok_start),
        "tok_end": int(tok_end),
        "n_tokens_entity": int(tok_end - tok_start),
        "n_tokens_text": int(len(ids)),
        "char_span": [int(c0), int(c1)],
        "entity_tokens": toks,
        "covered_tokens": covered_tokens,
        "covered_chars": covered_chars,
        "roundtrip_ok": bool(covered_tokens == word_s),
        "aligned": bool(covered_chars == word_s),
    }


def verify_entity_span(span, strict=True):
    if not span["roundtrip_ok"]:
        msg = (f"span roundtrip failed: covered={span['covered_tokens']!r} "
               f"!= word={span['word_bpe']!r} (sent={span['sent']!r})")
        if strict:
            raise ValueError(msg)
        return False, msg
    return True, ""


def gpt_row_offset(cond_len):
    """text token 列の index -> GPT2 系列の行番号 に足すオフセット.

    [cond latents x cond_len][START][text tokens ...] という並びなので
    cond_len + 1。cond_len を足し忘れると 32 行ずれる。
    """
    return int(cond_len) + 1


# --------------------------------------------------------------------------
# モデル読み込み
# --------------------------------------------------------------------------
@dataclass
class XttsBundle:
    model: object
    tokenizer: object
    device: str
    dtype: object
    num_layers: int
    hidden_size: int
    cond_len: int
    start_text_token: int
    stop_text_token: int
    start_audio_token: int
    stop_audio_token: int
    model_dir: str
    sampling_rate: int = OUTPUT_SAMPLE_RATE
    info: dict = field(default_factory=dict)

    @property
    def gpt(self):
        return self.model.gpt

    @property
    def blocks(self):
        """GPT2 trunk のブロック列 (= splice の対象)."""
        return self.model.gpt.gpt.h


def load_xtts(model_id=DEFAULT_MODEL_ID, device=None, dtype=torch.float32,
              allow_download=False):
    """XTTS-v2 を読む。

    coqui-tts (idiap/coqui-ai-TTS fork, pip の coqui-tts) が要る。
    VRAM は float32 で GPT 約 1.5 GB + HiFiGAN 少々 = 数 GB。

    注意: ``Xtts.eval()`` は ``self.gpt.init_gpt_for_inference()`` を呼び直し、
    しかも **None を返す** (xtts.py:690-693)。``model = model.eval()`` と書くと
    model が None になる。ここでは load_checkpoint(eval=True) が済ませた
    eval 状態をそのまま使い、training フラグだけ確かめる。
    deepspeed は使わない (eval() を挟むと黙って剥がれるため)。
    """
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = ensure_xtts_dir(model_id, allow_download=allow_download)

    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True,
                          use_deepspeed=False)
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    gpt = model.gpt
    if gpt.training:
        # GPT.forward は self.training で latent の切り出し量 (sub) を変える
        # (sub = -1 なら latent が mel code より 4 行多くなり HiFiGAN が壊れる)。
        raise RuntimeError("gpt が train モードのまま。load_checkpoint(eval=True) を確認すること。")
    if model.hifigan_decoder.training:
        raise RuntimeError("hifigan_decoder が train モードのまま。")
    n_layers = len(gpt.gpt.h)
    if n_layers != EXPECTED_NUM_LAYERS:
        print(f"[warn] gpt.gpt.h が {n_layers} 層 (想定 {EXPECTED_NUM_LAYERS})。"
              f"層格子 {SWEEP_LAYERS} を見直すこと。", flush=True)
    if int(gpt.model_dim) != EXPECTED_HIDDEN_SIZE:
        print(f"[warn] model_dim={gpt.model_dim} (想定 {EXPECTED_HIDDEN_SIZE})", flush=True)
    if not gpt.use_perceiver_resampler:
        raise RuntimeError(
            "gpt_use_perceiver_resampler=False。XTTS-v2 では True のはずで、"
            "cond latents の長さ (32) が変わる。checkpoint を確認すること。")
    if not hasattr(gpt, "gpt_inference"):
        raise RuntimeError("init_gpt_for_inference が走っていない (eval=True で読むこと)")

    return XttsBundle(
        model=model, tokenizer=model.tokenizer, device=device, dtype=dtype,
        num_layers=n_layers, hidden_size=int(gpt.model_dim),
        cond_len=EXPECTED_COND_LEN,
        start_text_token=int(gpt.start_text_token),
        stop_text_token=int(gpt.stop_text_token),
        start_audio_token=int(gpt.start_audio_token),
        stop_audio_token=int(gpt.stop_audio_token),
        model_dir=model_dir,
        sampling_rate=int(model.args.output_sample_rate),
        info={
            "layers": n_layers,
            "model_dim": int(gpt.model_dim),
            "heads": int(gpt.heads),
            "number_text_tokens": int(gpt.number_text_tokens),
            "max_text_tokens": int(model.args.gpt_max_text_tokens),
            "max_gen_mel_tokens": int(gpt.max_gen_mel_tokens),
            "code_stride_len": int(gpt.code_stride_len),
            "use_perceiver_resampler": bool(gpt.use_perceiver_resampler),
            "kv_cache": bool(model.args.kv_cache),
            "output_sample_rate": int(model.args.output_sample_rate),
        },
    )


# --------------------------------------------------------------------------
# 参照音声 (話者条件)
# --------------------------------------------------------------------------
def load_reference(bundle, wav_path, gpt_cond_len=DEFAULT_GPT_COND_LEN,
                   gpt_cond_chunk_len=DEFAULT_GPT_COND_CHUNK_LEN,
                   max_ref_length=DEFAULT_MAX_REF_LEN,
                   sound_norm_refs=DEFAULT_SOUND_NORM_REFS):
    """参照音声から cond latents と speaker embedding を **1 回だけ** 作る.

    ここが Perceiver を通る唯一の場所 (Xtts.get_gpt_cond_latents ->
    GPT.get_style_emb -> conditioning_perceiver)。入力は参照音声の mel だけで
    テキストは通らない。出力の cond latents [1, 32, 1024] を BL / TGT / INJ で
    そのまま共有する。splice は GPT2 trunk の中でしか働かないので、
    Perceiver の重みも出力もこの後 1 度も変わらない。
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"参照音声が無い: {wav_path}")
    cond, spk = bundle.model.get_conditioning_latents(
        audio_path=wav_path, max_ref_length=max_ref_length,
        gpt_cond_len=gpt_cond_len, gpt_cond_chunk_len=gpt_cond_chunk_len,
        sound_norm_refs=sound_norm_refs)
    cond = cond.to(bundle.device)
    spk = spk.to(bundle.device)
    if cond.dim() != 3 or cond.shape[0] != 1:
        raise RuntimeError(f"cond latents の形が想定外: {tuple(cond.shape)}")
    if cond.shape[1] != EXPECTED_COND_LEN:
        raise RuntimeError(
            f"cond latents の長さが {cond.shape[1]} (想定 {EXPECTED_COND_LEN})。"
            f"行番号のオフセットが狂うので中断する。")
    ref = {
        "wav_path": wav_path,
        "gpt_cond_latent": cond,
        "speaker_embedding": spk,
        "cond_len": int(cond.shape[1]),
        "cond_checksum": reference_checksum(cond, spk),
        "gpt_cond_len": gpt_cond_len,
        "gpt_cond_chunk_len": gpt_cond_chunk_len,
        "max_ref_length": max_ref_length,
        "sound_norm_refs": bool(sound_norm_refs),
    }
    return ref


def reference_checksum(cond, spk):
    """cond latents / speaker embedding が 3 条件で不変であることを見張るための指紋."""
    # inference tensor (get_conditioning_latents は @torch.inference_mode) を
    # 外から読むので inference_mode で包む。
    with torch.inference_mode():
        a = cond.detach().float().reshape(-1)
        b = spk.detach().float().reshape(-1)
        return {
            "cond_sum": float(a.sum()), "cond_absmax": float(a.abs().max()),
            "cond_shape": list(cond.shape),
            "spk_sum": float(b.sum()), "spk_absmax": float(b.abs().max()),
            "spk_shape": list(spk.shape),
        }


def assert_reference_unchanged(ref):
    """Perceiver 由来の条件が 1 bit も変わっていないことを毎回確かめる."""
    now = reference_checksum(ref["gpt_cond_latent"], ref["speaker_embedding"])
    if now != ref["cond_checksum"]:
        raise RuntimeError(
            "参照音声由来の条件 (Perceiver の出力 / speaker embedding) が変わっている。"
            f"\n  before={ref['cond_checksum']}\n  after ={now}")
    return True


# --------------------------------------------------------------------------
# prefill の hidden state / kappa
# --------------------------------------------------------------------------
def build_prefill_embeddings(bundle, cond_latents, text_tokens):
    """generate の 1 回目 (prefill) に GPT2 が受け取る inputs_embeds を再現する.

    GPT.compute_embeddings (gpt.py:477-497) と
    GPT2InferenceModel.forward の prefill 分岐 (gpt_inference.py:46-59) を
    そのまま並べたもの:

        text_inputs = [START] + text + [STOP]
        emb  = text_embedding(text_inputs) + text_pos_embedding(text_inputs)
        emb  = cat([cond_latents, emb])
        gen  = mel_embedding([start_audio]) + mel_pos_embedding(...)   # 位置 0
        emb  = cat([emb, gen])

    戻り値の長さが prefill_length() と一致することを確かめる。
    """
    gpt = bundle.gpt
    tt = text_tokens
    tt = F.pad(tt, (0, 1), value=bundle.stop_text_token)
    tt = F.pad(tt, (1, 0), value=bundle.start_text_token)
    emb = gpt.text_embedding(tt) + gpt.text_pos_embedding(tt)
    emb = torch.cat([cond_latents.to(emb.dtype), emb], dim=1)
    audio_in = torch.full((emb.shape[0], 1), bundle.start_audio_token,
                          dtype=torch.long, device=emb.device)
    gen_emb = gpt.mel_embedding(audio_in)
    gen_emb = gen_emb + gpt.mel_pos_embedding(gen_emb)
    emb = torch.cat([emb, gen_emb], dim=1)
    return emb


def prefill_length(cond_len, n_text_tokens):
    """pass 1 (generate の prefill) の系列長: cond + [START] + text + [STOP] + start_audio."""
    return int(cond_len) + int(n_text_tokens) + 2 + 1


def verify_prefill_layout(bundle, cond_latents, text_tokens, atol=0.0):
    """自前の prefill 再現が、モデル自身が store_prefix_emb する物と一致するか確かめる.

    ここが 1 行でもずれていると、実体語の行番号が静かにずれたまま完走する。
    CPU 検証ではモデル重みが無いので、この照合だけは GPU ジョブの冒頭で
    1 回だけ走らせて manifest に残す。

    GPT.compute_embeddings は cached_prefix_emb (= cond + [START] + text + [STOP])
    を書き込み、start_audio を足した長さの gpt_inputs を返す
    (gpt.py:477-497)。自前の emb はそれに start_audio の行を 1 つ足したもの。
    """
    gpt = bundle.gpt
    with torch.inference_mode():
        gpt_inputs = gpt.compute_embeddings(cond_latents, text_tokens)
        cached = gpt.gpt_inference.cached_prefix_emb
        mine = build_prefill_embeddings(bundle, cond_latents, text_tokens)
    want = prefill_length(cond_latents.shape[1], text_tokens.shape[1])
    rep = {
        "prefill_length_expected": int(want),
        "gpt_inputs_len": int(gpt_inputs.shape[1]),
        "cached_prefix_len": int(cached.shape[1]),
        "mine_len": int(mine.shape[1]),
        "row_offset": gpt_row_offset(cond_latents.shape[1]),
        "start_text_token": int(gpt.start_text_token),
        "stop_text_token": int(gpt.stop_text_token),
        "start_audio_token": int(gpt.start_audio_token),
    }
    if int(gpt_inputs.shape[1]) != want:
        raise RuntimeError(f"gpt_inputs の長さが {gpt_inputs.shape[1]} (想定 {want}): {rep}")
    if int(mine.shape[1]) != want:
        raise RuntimeError(f"自前 prefill emb の長さが {mine.shape[1]} (想定 {want}): {rep}")
    if int(cached.shape[1]) != want - 1:
        raise RuntimeError(f"cached_prefix_emb の長さが {cached.shape[1]} (想定 {want - 1}): {rep}")
    head = mine[:, : cached.shape[1]]
    d = (head.float() - cached.float()).abs().max().item()
    rep["max_abs_diff_vs_cached"] = float(d)
    rep["bitwise_equal"] = bool(torch.equal(head, cached.to(head.dtype)))
    if d > atol and not rep["bitwise_equal"]:
        raise RuntimeError(
            f"自前 prefill emb がモデルの cached_prefix_emb と一致しない "
            f"(max|diff|={d}): {rep}")
    # gpt_inputs の最後は start_audio、それ以外は 1 で埋められている (gpt.py:488-496)
    ids = gpt_inputs[0].tolist()
    if int(ids[-1]) != int(gpt.start_audio_token) or any(int(v) != 1 for v in ids[:-1]):
        raise RuntimeError(f"gpt_inputs の中身が想定と違う (末尾が start_audio でない): {rep}")
    return rep


def latent_pass_length(cond_len, n_text_tokens, n_mel_codes):
    """pass 2 (return_latent=True の forward) の系列長.

    GPT.forward (gpt.py:...) の組み立て:
      text 側 : [START] + text + [STOP]              -> n_text + 2
      mel 側  : code_lengths = M + 3 まで pad, さらに末尾に 1 個 pad,
                先頭に start_audio                    -> M + 5
    """
    return int(cond_len) + int(n_text_tokens) + 2 + int(n_mel_codes) + 5


class LayerTap:
    """h[l] の出力をそのまま控えておくだけの forward hook (splice はしない)."""

    def __init__(self, module):
        self.module = module
        self.handle = None
        self.outputs = []

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(h):
            self.outputs.append(h.detach())
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


def prefill_hidden_states(bundle, cond_latents, text_tokens):
    """prefill を 1 回だけ回して、全層の hidden state と h[l] の出力を取る.

    戻り値:
      hidden_states : 長さ num_layers+1 のタプル。index 0 = 埋め込み出力、
                      index l+1 = layers[l] の出力。ただし最後の index だけは
                      GPT2Model が ln_f を掛けた後の値 (Llasa/Llama 側の
                      hidden_states[-1] が post-norm なのと同じ)。kappa 用。
      taps          : taps[l] = h[l] の **生の** 出力。splice が書き換える
                      テンソルそのもの。目標ベクトルはこちらから取る。
    """
    emb = build_prefill_embeddings(bundle, cond_latents, text_tokens)
    want = prefill_length(cond_latents.shape[1], text_tokens.shape[1])
    if emb.shape[1] != want:
        raise RuntimeError(f"prefill emb の長さが {emb.shape[1]} (想定 {want})")
    taps = [LayerTap(b) for b in bundle.blocks]
    for t in taps:
        t.__enter__()
    try:
        with torch.inference_mode():
            out = bundle.model.gpt.gpt(inputs_embeds=emb, use_cache=False,
                                       output_hidden_states=True, return_dict=True)
    finally:
        for t in taps:
            t.__exit__()
    got = []
    for l, t in enumerate(taps):
        if len(t.outputs) != 1:
            raise RuntimeError(f"layer {l} の tap が {len(t.outputs)} 回発火した (1 のはず)")
        got.append(t.outputs[0])
    return tuple(h.detach() for h in out.hidden_states), tuple(got)


def hidden_index_for_layer(layer):
    """layers[l] の出力は hidden_states[l+1]."""
    return int(layer) + 1


def span_vector(hidden, row_start, row_end):
    """スパン内 hidden state をトークン方向に単純平均したベクトル."""
    return hidden[0, row_start:row_end].float().mean(dim=0)


def kappa_by_layer(src_hidden_states, tgt_hidden_states, s_rows, t_rows):
    """kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}).

    scripts/llasa/llasa_splice.py:kappa_by_layer と同じ定義
    (スパン内 hidden state をトークン方向に単純平均したベクトル同士の cosine)。
    index 0 = 埋め込み出力、index l+1 = layers[l] の出力。
    """
    s0, s1 = s_rows
    u0, u1 = t_rows
    ks = []
    for hs, ht in zip(src_hidden_states, tgt_hidden_states):
        a = span_vector(hs, s0, s1)
        b = span_vector(ht, u0, u1)
        ks.append(float(F.cosine_similarity(a, b, dim=0)))
    return ks


# --------------------------------------------------------------------------
# splice hook
# --------------------------------------------------------------------------
class RowSplice:
    """gpt.gpt.h[l] の出力の、指定した行だけを差し替える forward hook.

    **いつ発火するか** (ここを間違えると KV cache と食い違う):
      1. h.shape[1] <= 1              -> decode ステップ。何もしない。
         (kv_cache=True の自己回帰生成では 2 回目以降は必ず seq_len == 1)
      2. expect_len を渡してあるとき、h.shape[1] != expect_len なら見送る。
         XTTS は 1 回の合成で GPT2 trunk を 2 回通る (prefill と latent pass)。
         それぞれ長さが違うので、**どちらの pass を狙っているかを長さで名指しする**。
         見送った形状は skipped_shapes に残す。
      3. 既に発火していたら二度と発火しない (fired flag)。
    合成後に assert_fired() を呼び、想定回数でなければ例外にする
    (splice されていない音声を INJ として保存しないため)。

    **何を書くか**:
      m = 元テキストの実体語トークン数、n = 目標テキストのそれ。
      m == n : 行対行でそのまま置換。
      m != n : 目標スパンをトークン方向に平均した 1 本を m 行に broadcast。
      いずれも系列長 T は変えない。
    """

    def __init__(self, layer_module, row_start, row_end, tgt_rows, expect_len=None,
                 name="prefill"):
        self.layer_module = layer_module
        self.r0 = int(row_start)
        self.r1 = int(row_end)
        self.tgt = tgt_rows.detach()
        self.expect_len = int(expect_len) if expect_len is not None else None
        self.name = name
        self.handle = None
        self.fired = 0
        self.decode_calls = 0
        self.skipped_shapes = []
        self.m = self.r1 - self.r0
        self.n = int(self.tgt.shape[0])
        self.equal_len = bool(self.m == self.n)
        if self.m <= 0:
            raise ValueError(f"empty source span: [{self.r0}, {self.r1})")
        if self.n <= 0:
            raise ValueError("empty target hidden span")

    def _replacement(self, dtype, device):
        if self.equal_len:
            rep = self.tgt
        else:
            rep = self.tgt.mean(dim=0, keepdim=True).expand(self.m, -1)
        return rep.to(dtype=dtype, device=device)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            return output
        T = h.shape[1]
        if T <= 1:
            self.decode_calls += 1
            return output
        if self.expect_len is not None and T != self.expect_len:
            self.skipped_shapes.append(int(T))
            return output
        if self.fired:
            self.skipped_shapes.append(int(T))
            return output
        if self.r1 > T:
            raise RuntimeError(f"entity span [{self.r0},{self.r1}) が系列長 {T} を超えている")
        if h.shape[0] != 1:
            raise RuntimeError(f"batch size {h.shape[0]} は未対応 (1 のみ)")
        h_mod = h.clone()
        h_mod[0, self.r0:self.r1] = self._replacement(h.dtype, h.device)
        self.fired += 1
        if isinstance(output, tuple):
            return (h_mod,) + tuple(output[1:])
        return h_mod

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def assert_fired(self, want=1):
        if self.fired != want:
            raise RuntimeError(
                f"splice hook[{self.name}] の発火回数が {self.fired} ({want} であるべき)。"
                f" expect_len={self.expect_len} skipped_shapes={self.skipped_shapes[:8]}")

    def stats(self):
        return {"name": self.name, "fired": self.fired, "decode_calls": self.decode_calls,
                "skipped_shapes": self.skipped_shapes[:16], "expect_len": self.expect_len,
                "rows": [self.r0, self.r1],
                "m_src_entity_tokens": self.m, "n_tgt_entity_tokens": self.n,
                "equal_len": self.equal_len}


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def assert_splice_target(bundle, layer):
    """差し替え先が GPT2 trunk のブロックであり、Perceiver ではないことを確かめる.

    Perceiver (話者条件) の cross-attention は参照音声に対するものであって
    テキストに対するものではない。splice が誤ってそちらに触れていないことを
    ここで機械的に保証する。
    """
    if not (0 <= int(layer) < bundle.num_layers):
        raise ValueError(f"layer {layer} が範囲外 (0..{bundle.num_layers - 1})")
    target = bundle.blocks[int(layer)]
    trunk_ids = {id(m) for m in bundle.model.gpt.gpt.modules()}
    if id(target) not in trunk_ids:
        raise RuntimeError("splice 対象が gpt.gpt (GPT2 trunk) の中に無い")
    perceiver = getattr(bundle.model.gpt, "conditioning_perceiver", None)
    if perceiver is not None:
        per_ids = {id(m) for m in perceiver.modules()}
        if id(target) in per_ids:
            raise RuntimeError("splice 対象が conditioning_perceiver の中にある (禁止)")
        if trunk_ids & per_ids:
            raise RuntimeError("GPT2 trunk と conditioning_perceiver が共有モジュールを持つ")
    cond_enc = getattr(bundle.model.gpt, "conditioning_encoder", None)
    if cond_enc is not None and id(target) in {id(m) for m in cond_enc.modules()}:
        raise RuntimeError("splice 対象が conditioning_encoder の中にある (禁止)")
    return target


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
@dataclass
class GenConfig:
    seed: int = SEED
    do_sample: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    length_penalty: float = DEFAULT_LENGTH_PENALTY
    num_beams: int = 1

    def as_dict(self):
        return {"seed": self.seed, "do_sample": self.do_sample,
                "temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "repetition_penalty": self.repetition_penalty,
                "length_penalty": self.length_penalty, "num_beams": self.num_beams}


def prepare_text(bundle, text, lang="en", word=None, strict_span=True):
    """1 本のテキストについて token 列 / tensor / スパンをまとめて作る."""
    sent = normalize_sentence(text)
    ids, bpe_s = encode_text(bundle.tokenizer, sent, lang)
    max_text = int(bundle.model.args.gpt_max_text_tokens)
    if len(ids) >= max_text:
        raise ValueError(f"text token が {len(ids)} で上限 {max_text} を超える: {sent!r}")
    tt = torch.IntTensor(ids).unsqueeze(0).to(bundle.device)
    out = {"text": text, "sent": sent, "bpe_input": bpe_s, "input_ids": ids,
           "text_tokens": tt, "n_text_tokens": len(ids)}
    if word is not None:
        span = entity_token_span(bundle.tokenizer, text, word, lang)
        verify_entity_span(span, strict=strict_span)
        off = gpt_row_offset(bundle.cond_len)
        span["row_start"] = int(span["tok_start"] + off)
        span["row_end"] = int(span["tok_end"] + off)
        span["row_offset"] = int(off)
        out["span"] = span
    return out


def _generate_codes(bundle, cond, tt, gen, splice=None):
    """pass 1: mel code の自己回帰生成。splice があれば prefill でだけ発火させる."""
    seed_all(gen.seed)
    ctx = splice if splice is not None else _null_ctx()
    with ctx:
        codes = bundle.model.gpt.generate(
            cond_latents=cond,
            text_inputs=tt,
            do_sample=gen.do_sample,
            top_p=gen.top_p,
            top_k=gen.top_k,
            temperature=gen.temperature,
            num_return_sequences=bundle.model.gpt_batch_size,
            num_beams=gen.num_beams,
            length_penalty=gen.length_penalty,
            repetition_penalty=gen.repetition_penalty,
            output_attentions=False,
        )
    if splice is not None:
        splice.assert_fired(1)
    return codes


def _codes_to_wav(bundle, cond, spk, tt, codes, splice=None):
    """pass 2: latent を作って HiFiGAN に流す。splice があればここでも発火させる."""
    expected_output_len = torch.tensor(
        [codes.shape[-1] * bundle.model.gpt.code_stride_len], device=tt.device)
    text_len = torch.tensor([tt.shape[-1]], device=bundle.device)
    ctx = splice if splice is not None else _null_ctx()
    with ctx:
        latents = bundle.model.gpt(
            tt, text_len, codes, expected_output_len,
            cond_latents=cond, return_attentions=False, return_latent=True)
    if splice is not None:
        splice.assert_fired(1)
    wav = bundle.model.hifigan_decoder(latents, g=spk).cpu().squeeze()
    # inference tensor の view を外に出さないよう複製してから返す。
    return wav.detach().float().numpy().copy(), latents


def synthesize(bundle, text, ref, lang="en", gen=GenConfig(), ctx=None):
    """素の TTS。BL = TTS(text_a), TGT = TTS(text_b) はどちらもこれ.

    Xtts.inference (xtts.py:503-556) を splice を挟めるように開いただけで、
    順序も引数も同じ (enable_text_splitting=False / speed=1.0 相当)。
    """
    assert_reference_unchanged(ref)
    c = ctx if ctx is not None else prepare_text(bundle, text, lang)
    cond = ref["gpt_cond_latent"]
    spk = ref["speaker_embedding"]
    with torch.inference_mode():
        codes = _generate_codes(bundle, cond, c["text_tokens"], gen)
        wav, _ = _codes_to_wav(bundle, cond, spk, c["text_tokens"], codes)
    assert_reference_unchanged(ref)
    return {
        "audio": wav, "sampling_rate": bundle.sampling_rate,
        "n_text_tokens": c["n_text_tokens"], "n_mel_codes": int(codes.shape[-1]),
        "dur_sec": float(len(wav) / bundle.sampling_rate),
        "hit_max_mel": bool(codes.shape[-1] >= bundle.model.gpt.max_gen_mel_tokens),
    }


def splice_synthesize(bundle, text_a, word_a, text_b, word_b, layer, ref, lang="en",
                      gen=GenConfig(), collect_kappa=True, strict_span=True,
                      splice_latent_pass=True, src_ctx=None, tgt_ctx=None):
    """SpeechSplice 本体.

    1. 目標文脈 (text_b) の prefill を 1 回回し、h[layer] の出力の
       実体語スパンを取り出す (n 行)。
    2. 元文脈 (text_a) で合成する。GPT2 trunk を通る 2 つの pass
       (prefill と latent pass) の、指定した層の出力の実体語行だけを
       その目標ベクトルで置き換える。系列長は変えない。
    3. HiFiGAN で波形にする。

    重み更新なし。BL とは splice 以外すべて同じ (seed / 復号設定 / prompt /
    cond latents / speaker embedding / 系列長)。
    """
    assert_reference_unchanged(ref)
    cond = ref["gpt_cond_latent"]
    spk = ref["speaker_embedding"]
    target_block = assert_splice_target(bundle, layer)

    src = src_ctx if src_ctx is not None else prepare_text(
        bundle, text_a, lang, word=word_a, strict_span=strict_span)
    tgt = tgt_ctx if tgt_ctx is not None else prepare_text(
        bundle, text_b, lang, word=word_b, strict_span=strict_span)

    with torch.inference_mode():
        # 目標側 hidden state (pass 1 形状の prefill)。因果マスクのおかげで
        # テキスト行の値は pass 2 でも同じなので、これを両方に使える。
        tgt_hidden, tgt_taps = prefill_hidden_states(bundle, cond, tgt["text_tokens"])
        t0, t1 = tgt["span"]["row_start"], tgt["span"]["row_end"]
        tgt_rows = tgt_taps[int(layer)][0, t0:t1].clone()

        src_hidden = None
        if collect_kappa:
            src_hidden, _ = prefill_hidden_states(bundle, cond, src["text_tokens"])

        s0, s1 = src["span"]["row_start"], src["span"]["row_end"]
        p1 = prefill_length(bundle.cond_len, src["n_text_tokens"])
        sp1 = RowSplice(target_block, s0, s1, tgt_rows, expect_len=p1, name="prefill")
        codes = _generate_codes(bundle, cond, src["text_tokens"], gen, splice=sp1)

        n_codes = int(codes.shape[-1])
        sp2 = None
        if splice_latent_pass:
            p2 = latent_pass_length(bundle.cond_len, src["n_text_tokens"], n_codes)
            sp2 = RowSplice(target_block, s0, s1, tgt_rows, expect_len=p2,
                            name="latent_pass")
        wav, _ = _codes_to_wav(bundle, cond, spk, src["text_tokens"], codes, splice=sp2)

    assert_reference_unchanged(ref)

    out = {
        "audio": wav, "sampling_rate": bundle.sampling_rate,
        "layer": int(layer), "hidden_index": hidden_index_for_layer(layer),
        "n_text_tokens_a": src["n_text_tokens"], "n_text_tokens_b": tgt["n_text_tokens"],
        "n_mel_codes": n_codes,
        "dur_sec": float(len(wav) / bundle.sampling_rate),
        "hit_max_mel": bool(n_codes >= bundle.model.gpt.max_gen_mel_tokens),
        "s_span": [src["span"]["tok_start"], src["span"]["tok_end"]],
        "t_span": [tgt["span"]["tok_start"], tgt["span"]["tok_end"]],
        "s_rows": [s0, s1], "t_rows": [t0, t1],
        "s_tokens": src["span"]["entity_tokens"], "t_tokens": tgt["span"]["entity_tokens"],
        "s_aligned": src["span"]["aligned"], "t_aligned": tgt["span"]["aligned"],
        "s_how": src["span"]["how"], "t_how": tgt["span"]["how"],
        "m_src_entity_tokens": sp1.m, "n_tgt_entity_tokens": sp1.n,
        "equal_len": sp1.equal_len,
        "splice_mode": "token_wise" if sp1.equal_len else "mean_pool_broadcast",
        "splice_latent_pass": bool(splice_latent_pass),
        "fired_prefill": sp1.fired,
        "fired_latent_pass": (sp2.fired if sp2 is not None else 0),
        "decode_calls": sp1.decode_calls,
        "skipped_shapes": sp1.skipped_shapes[:16],
        "hook_stats": [sp1.stats()] + ([sp2.stats()] if sp2 is not None else []),
    }
    if collect_kappa and src_hidden is not None:
        ks = kappa_by_layer(src_hidden, tgt_hidden, (s0, s1), (t0, t1))
        out["kappa_by_layer"] = [round(v, 6) for v in ks]
        out["kappa_at_splice"] = round(ks[hidden_index_for_layer(layer)], 6)
        out["kappa_last_layer"] = round(ks[-1], 6)
    return out
