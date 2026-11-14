#!/usr/bin/env python3
"""SpeechSplice for Kiseki-TTS (telecomadm1145/Kiseki-TTS, MIT).

凍結した Kiseki-TTS の text encoder 最終出力に対して、実体語スパンの hidden
state を「目標の読みを書いたテキスト」の 2 回目 forward pass の対応スパンで
差し替える。重み更新は一切しない。

    H~ = (h^src_{1:s0},  h^tgt_{t0:t1},  h^src_{s1:T})
    系列長: T -> T - m + n   （m = s1-s0, n = t1-t0）

Kiseki 固有の事情（すべて HF の modeling_mamba2_s2s.py を読んで確認済み。
ローカル写しは scripts/kiseki/_repo/modeling_mamba2_s2s.py）
------------------------------------------------------------------------
1. splice の入口は ``generate_audio`` の ``encoder_outputs`` 引数
   ``generate_audio``(modeling_mamba2_s2s.py:543-613) は

       if encoder_outputs is None:
           encoder_outputs = self.encoder(input_ids=..., attention_mask=...)
       enc = encoder_outputs[0]
       B, S, _ = enc.shape
       cross_mask = self._build_cross_mask(attention_mask, input_ids, B, S, device)

   という順で走る。``S`` は **splice 後の enc から** 取られるので、
   ``encoder_outputs`` に差し替え済みテンソルを渡し、かつ ``input_ids`` と
   ``attention_mask`` を **両方 None** にすれば、``_build_cross_mask``
   (同:419-425) が ``torch.ones(B, 1, 1, S)`` を返して長さが必ず一致する。
   batch size 1 なら padding が無いので全 1 mask が正解。

   **register_forward_hook で長さを変えてはいけない。** hook で enc だけ
   伸ばすと ``_build_cross_mask`` が元の ``input_ids`` から (B,1,1,T) の mask を
   作る一方 ``_precompute_cross_kv`` は (B,H,T-m+n,Dh) の K/V を作るので、
   SDPA が shape mismatch で落ちる。

2. splice は全層・全フレームに届く
   ``_precompute_cross_kv``(同:427-435) が encoder 最終出力から decoder 6 層ぶんの
   K/V を **一度だけ** 作り、以後すべてのフレームで使い回す(同:557, 577-579)。
   decoder は cross-attention -> Mamba2 SSM の順で、causal self-attention が無い。
   よって encoder 出力を差し替えれば 6 層 × 全フレームに届く。

3. 系列長が変わってよい理由
   ``CrossAttention.forward``(同:157-161) は q_proj / SDPA だけで **RoPE が
   一切入っていない**。位置情報は encoder 内部の RoPE
   (``SelfAttention.forward`` 同:135-144、``pos = arange(S)``) で hidden state に
   焼き込まれている。よって splice 後に位置を振り直す必要がない。
   duration predictor も length regulator も固定長 latent も無く、decoder は
   第 0 codebook が EOS(2048) を出すまで自己回帰するだけ(同:602-607)。

4. 入力列と offset
   ``build_tts_input_ids``(同:672-674) は

       ids = [tts_token_id, lang_token_id] + list(text_ids) + [eos_token_id]

   を組む。encoder に downsampling も conv も無い(``Mamba2Seq2SeqEncoder.forward``
   同:312-331)ので **token index と encoder 出力の時間 index は 1 対 1**。
   実体語スパンは「特殊トークン抜きの座標」で求めてから、この先頭ぶんだけ
   ずらせばよい。**個数はハードコードせず、実行時に
   ``locate_subsequence`` で確定させる**（Irodori で同じ箇所を間違えた）。

5. 実体語スパンの決め方は既存 3 モデルと同一
   tokenizer は PreTrainedTokenizerFast の subword（生テキスト入力。
   本ファイルでも G2P / phonemizer / pyopenjtalk は一切使わない）。
   スパン探索は ``speechsplice_release/src/injection.py:401``
   ``find_entity_token_range`` と **同じ計算**。3 段構え:

     1 段目: 文全体のトークン id 列の中から、実体語を単独でトークナイズした
             id 列を部分列として探す
     2 段目: ``full_text.find(entity_text)`` で文字位置を取り、開始 =
             実体語より前の部分を単独でトークナイズした長さ、
             終了 = 開始 + 実体語単独のトークン数
     3 段目: 大文字小文字を無視した文字検索（日本語では発火しない）

   **トークン境界の検証はしない。窓は広げない。** 境界に乗ったかどうかは
   ``aligned`` フラグとして記録するだけで、挙動は変えない。

6. 乱数
   ``generate_audio`` は ``torch.multinomial``(同:359) を使うのでグローバル RNG
   に依存する。BL / TGT / INJ は生成呼び出しの **直前** に同じ seed を入れて
   初期状態を厳密に揃える。復号設定（max_new_frames / temperature / top_k /
   top_p / repetition_penalty）も 3 条件で共有する。差は splice だけ。

7. 波形化
   出力は Qwen3-TTS-Tokenizer-12Hz の離散 codec token（12.5Hz、Q=16、
   codebook 2048）。波形化は ``qwen_tts.Qwen3TTSTokenizer``。
   ``decode({"audio_codes": codes})`` は (T, Q) を受けて (wavs, sr) を返す。

このモジュールは import 時に torch / transformers を要求しない。純粋な
スパン計算・splice 計算はモデル無しで CPU からテストできるように分離してある
（``smoke_test_kiseki_splice.py`` を参照）。本番生成は ``run_kiseki_eval.py``。
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Optional

SEED = 42
DEFAULT_MODEL_ID = "telecomadm1145/Kiseki-TTS"
DEFAULT_CODEC_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
DEFAULT_LANG_TOKEN = "<|2ja|>"

# モデルカードの既定（README "Sampling parameters"）。
# max_new_frames は 12.5Hz なので 250 frame = 20 秒。
DEFAULT_MAX_NEW_FRAMES = 250
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.0

# qwen_tts / sox はイメージに入っていないので、プロジェクト内の vendor ディレクトリ
# から読む（site-packages には一切触れていない）。README の「依存」を参照。
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")


def ensure_vendor_on_path() -> str:
    """``scripts/kiseki/_vendor`` を sys.path に足す（qwen_tts / sox 用）。"""
    import sys

    if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    return VENDOR_DIR


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def seed_all(seed: int = SEED) -> None:
    """BL / TGT / INJ の初期 RNG 状態を厳密に揃える。

    ``generate_audio`` は ``torch.multinomial`` を使うのでグローバル RNG に
    依存する。生成呼び出しの直前に毎回これを呼ぶこと。
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# span utilities（モデル不要・純粋関数。CPU でテストできる）
# --------------------------------------------------------------------------
def encode_ids(tokenizer, text: str) -> list[int]:
    """``injection.py:404-406`` と同じ ``encode(..., add_special_tokens=False)``。"""
    return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]


def find_entity_token_range(tokenizer, full_text: str, entity_text: str,
                            offset: int = 0) -> Optional[tuple[int, int, str]]:
    """``injection.py:401`` ``find_entity_token_range`` と同一の計算。

    既存 3 モデル（T5Gemma 系）と同じスパンを使うためのもの。戻り値は
    ``(start, end, stage)``。``start``/``end`` は **特殊トークンを含まない
    トークン座標** に ``offset`` を足したもの。見つからなければ ``None``。

    **窓は広げない。トークン境界の検証もしない。** ずれたまま進むのが
    既存 3 モデルの挙動であり、ここだけ変えると比較が揃わない。
    """
    full_tokens = encode_ids(tokenizer, full_text)
    entity_tokens = encode_ids(tokenizer, entity_text)

    # 1 段目: トークン id 列の部分列検索
    for start in range(len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[start:start + len(entity_tokens)] == entity_tokens:
            return (start + offset, start + len(entity_tokens) + offset, "subsequence")

    # フォールバック 1: 文字位置から prefix をトークナイズして開始位置を出す
    entity_pos = full_text.find(entity_text)
    if entity_pos >= 0:
        prefix_tokens = encode_ids(tokenizer, full_text[:entity_pos])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(entity_tokens) + offset,
                "char_prefix")

    # フォールバック 2: 大文字小文字を無視した文字検索（injection.py:422-437）
    pos = full_text.lower().find(entity_text.lower())
    if pos >= 0:
        actual_surface = full_text[pos:pos + len(entity_text)]
        actual_tokens = encode_ids(tokenizer, actual_surface)
        for start in range(len(full_tokens) - len(actual_tokens) + 1):
            if full_tokens[start:start + len(actual_tokens)] == actual_tokens:
                return (start + offset, start + len(actual_tokens) + offset,
                        "subsequence_ci")
        prefix_tokens = encode_ids(tokenizer, full_text[:pos])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(actual_tokens) + offset,
                "char_prefix_ci")

    return None


def nospec_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    """特殊トークン抜きの offset mapping（``describe_entity_span`` 用）。"""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(a), int(b)) for a, b in enc["offset_mapping"]]


def describe_entity_span(tokenizer, full_text: str, entity_text: str,
                         offsets: list[tuple[int, int]], span: tuple[int, int],
                         stage: str) -> dict:
    """スパンが実際に何を覆っているかを記録する（**挙動は変えない**）。

    ``span`` は特殊トークンを含まないトークン座標。``offsets`` は同じ座標系。

    ``aligned`` = そのスパンのトークンが実体語を **ちょうど** 綴っているか。
    ``False`` ならサブワード境界と実体語の境界がずれている。ずれていても進む。
    """
    s, t = span
    out = {
        "stage": stage,
        "span_nospec": [int(s), int(t)],
        "n_tokens": int(t - s),
        "entity_text": entity_text,
        "entity_chars": None,
        "covered_chars": None,
        "covered_text": None,
        "aligned": False,
        "in_range": False,
    }
    pos = full_text.find(entity_text)
    if pos >= 0:
        out["entity_chars"] = [pos, pos + len(entity_text)]
    if not (0 <= s < t <= len(offsets)):
        return out
    out["in_range"] = True
    c0 = int(offsets[s][0])
    c1 = int(offsets[t - 1][1])
    out["covered_chars"] = [c0, c1]
    out["covered_text"] = full_text[c0:c1]
    out["aligned"] = bool(full_text[c0:c1] == entity_text)
    return out


def locate_subsequence(haystack: list[int], needle: list[int]) -> int:
    """haystack の中で needle が現れる唯一の開始位置を返す。

    ``build_tts_input_ids`` が前に足す ``[TTS]`` と言語タグの個数を
    **実行時に確定させる** ために使う。ハードコードしない。
    複数箇所に現れたら曖昧なので ValueError。
    """
    if not needle:
        raise ValueError("empty needle")
    n, m = len(haystack), len(needle)
    hits = [i for i in range(0, n - m + 1) if haystack[i:i + m] == needle]
    if not hits:
        raise ValueError(
            f"text ids are not a subsequence of build_tts_input_ids output "
            f"(needle[:8]={needle[:8]} haystack[:16]={haystack[:16]})")
    if len(hits) > 1:
        raise ValueError(f"ambiguous alignment: needle occurs at {hits}")
    return hits[0]


def spliced_surface_text(text_src: str, word_src: str,
                         text_tgt: str, word_tgt: str) -> str:
    """記録用の表層テキスト（``P + word_tgt + S``）。

    Kiseki には duration predictor が無いので生成には使わない。manifest に
    「文字レベルでは何に相当する splice なのか」を残すためだけのもの。
    ja_cross_culture_names は同一テンプレートの実体語差し替えなので、これは
    ``text_tgt`` に一致する（一致しない入力が来たら ValueError）。
    """
    i_src = text_src.find(word_src)
    i_tgt = text_tgt.find(word_tgt)
    if i_src < 0:
        raise ValueError(f"word {word_src!r} not in {text_src!r}")
    if i_tgt < 0:
        raise ValueError(f"word {word_tgt!r} not in {text_tgt!r}")
    prefix_src = text_src[:i_src]
    suffix_src = text_src[i_src + len(word_src):]
    prefix_tgt = text_tgt[:i_tgt]
    suffix_tgt = text_tgt[i_tgt + len(word_tgt):]
    if prefix_src != prefix_tgt or suffix_src != suffix_tgt:
        raise ValueError(
            f"source/target context differs: {prefix_src!r}|{suffix_src!r} vs "
            f"{prefix_tgt!r}|{suffix_tgt!r}")
    return prefix_src + word_tgt + suffix_src


# --------------------------------------------------------------------------
# splice（純粋テンソル演算。モデル不要）
# --------------------------------------------------------------------------
def splice_encoder_output(src_enc, s_span: tuple[int, int],
                          tgt_enc, t_span: tuple[int, int]):
    """H~ = cat(h^src[:s0], h^tgt[t0:t1], h^src[s1:]) を作る。

    src_enc / tgt_enc: (1, T, D)。batch size 1 固定（padding が無いことを
    前提にした全 1 cross mask を使うため。README の「制約」を参照）。
    戻り値: (spliced (1, T-m+n, D), info dict)
    """
    import torch

    if src_enc.dim() != 3 or src_enc.shape[0] != 1:
        raise ValueError(f"src_enc must be (1, T, D), got {tuple(src_enc.shape)}")
    if tgt_enc.dim() != 3 or tgt_enc.shape[0] != 1:
        raise ValueError(f"tgt_enc must be (1, T, D), got {tuple(tgt_enc.shape)}")
    if src_enc.shape[2] != tgt_enc.shape[2]:
        raise ValueError(
            f"hidden dim mismatch: {src_enc.shape[2]} vs {tgt_enc.shape[2]}")

    t_src = int(src_enc.shape[1])
    t_tgt = int(tgt_enc.shape[1])
    s0, s1 = int(s_span[0]), int(s_span[1])
    t0, t1 = int(t_span[0]), int(t_span[1])
    if not 0 <= s0 < s1 <= t_src:
        raise ValueError(f"s_span={s_span} outside src length {t_src}")
    if not 0 <= t0 < t1 <= t_tgt:
        raise ValueError(f"t_span={t_span} outside tgt length {t_tgt}")

    m = s1 - s0
    n = t1 - t0
    spliced = torch.cat(
        [
            src_enc[:, :s0],
            tgt_enc[:, t0:t1].to(device=src_enc.device, dtype=src_enc.dtype),
            src_enc[:, s1:],
        ],
        dim=1,
    )
    t_out = t_src - m + n
    if int(spliced.shape[1]) != t_out:
        raise AssertionError(f"spliced length {int(spliced.shape[1])} != {t_out}")

    info = {
        "n_tokens_src": t_src,
        "n_tokens_tgt": t_tgt,
        "n_tokens_spliced": t_out,
        "span_len_src": m,
        "span_len_tgt": n,
        "delta_tok": n - m,
        "hidden_dim": int(src_enc.shape[2]),
    }
    return spliced, info


def entity_cosine(src_h, tgt_h, s_span: tuple[int, int], t_span: tuple[int, int]) -> float:
    """kappa = cos( mean_j h^src_{s_j}, mean_k h^tgt_{u_k} )。

    既存実装（eval_layer_convergence_6lang.py:426-431、injection.py:508-512、
    irodori_splice.py:391-403）と同式。スパン内 hidden state をトークン方向に
    単純平均したベクトル同士の cosine。
    """
    import torch

    a = src_h[0, s_span[0]:s_span[1]].float().mean(0)
    b = tgt_h[0, t_span[0]:t_span[1]].float().mean(0)
    return float(
        torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# --------------------------------------------------------------------------
# モデル読み込み
# --------------------------------------------------------------------------
@dataclass
class KisekiBundle:
    model: Any
    tokenizer: Any
    lang_token: str
    lang_token_id: int
    device: str
    dtype: Any
    codec: Any = None
    codec_sr: Optional[int] = None
    meta: dict = None


def load_kiseki(model_id: str = DEFAULT_MODEL_ID,
                device: str = "cuda",
                precision: str = "fp32",
                lang_token: str = DEFAULT_LANG_TOKEN,
                codec_id: Optional[str] = DEFAULT_CODEC_ID,
                codec_device: Optional[str] = None) -> KisekiBundle:
    """Kiseki-TTS 本体と Qwen3-TTS codec を読む。

    ``trust_remote_code=True`` が要る（config.json の auto_map が
    ``modeling_mamba2_s2s`` を指す）。codec_id を None にすると codec を
    読まない（スパン検査だけしたいとき用）。
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast

    dtype = {"fp32": torch.float32,
             "bf16": torch.bfloat16,
             "fp16": torch.float16}[precision]

    tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=dtype)
    model = model.eval().to(device)

    lang_id = tok.convert_tokens_to_ids(lang_token)
    if lang_id is None or lang_id == tok.unk_token_id:
        raise RuntimeError(
            f"language token {lang_token!r} is not in the tokenizer "
            f"(got id={lang_id})")

    codec = None
    codec_sr = None
    if codec_id:
        codec = load_codec(codec_id, codec_device or device)
        codec_sr = int(codec.get_output_sample_rate())

    cfg = model.config
    meta = {
        "model_id": model_id,
        "codec_id": codec_id,
        "lang_token": lang_token,
        "lang_token_id": int(lang_id),
        "tts_token_id": int(cfg.tts_token_id),
        "eos_token_id": int(cfg.eos_token_id),
        "pad_token_id": int(cfg.pad_token_id),
        "d_model": int(cfg.d_model),
        "enc_layers": int(cfg.enc_layers),
        "dec_layers": int(cfg.dec_layers),
        "num_q": int(cfg.num_q),
        "audio_codebook": int(cfg.audio_codebook),
        "audio_eos_token_id": int(cfg.audio_eos_token_id),
        "vocab_size": int(cfg.vocab_size),
        "precision": precision,
        "codec_sample_rate": codec_sr,
    }
    return KisekiBundle(model=model, tokenizer=tok, lang_token=lang_token,
                        lang_token_id=int(lang_id), device=device, dtype=dtype,
                        codec=codec, codec_sr=codec_sr, meta=meta)


def load_codec(codec_id: str = DEFAULT_CODEC_ID, device: str = "cuda"):
    """Qwen3-TTS-Tokenizer-12Hz（Apache-2.0、24kHz mono）を読む。"""
    ensure_vendor_on_path()
    from qwen_tts import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer.from_pretrained(codec_id, device_map=device)


def codes_to_wav(bundle: KisekiBundle, codes):
    """(T, Q) の codec token を波形にする。戻り値 (np.float32 1-D, sr)。"""
    import numpy as np

    if bundle.codec is None:
        raise RuntimeError("codec is not loaded (load_kiseki(codec_id=...))")
    if int(codes.shape[0]) == 0:
        raise RuntimeError("no audio frames were generated (immediate EOS)")
    wavs, sr = bundle.codec.decode({"audio_codes": codes})
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if wav.size == 0:
        raise RuntimeError("codec returned an empty waveform")
    return wav, int(sr)


# --------------------------------------------------------------------------
# text encoding（offset を実行時に確定させる）
# --------------------------------------------------------------------------
def build_input_ids(bundle: KisekiBundle, text: str) -> dict:
    """``build_tts_input_ids`` で入力列を組み、実体語座標の offset を確定させる。

    ``build_tts_input_ids``(modeling_mamba2_s2s.py:672-674) が前に足す
    ``[TTS]`` と言語タグの個数は **ハードコードしない**。素のトークン列を
    組み上がった列の中に部分列として探して実行時に確定させる
    （Irodori でここを間違えてパディング込みの列と突き合わせて落ちた）。

    Returns dict:
      input_ids (1, T) LongTensor / text_ids（README と同じ ``tok.encode``）/
      nospec_ids（``add_special_tokens=False``）/ offsets / entity_offset /
      prefix_len / suffix_len / n_tokens
    """
    import torch

    tok = bundle.tokenizer
    model = bundle.model

    # README の使い方（`m.build_tts_input_ids(tok.encode(text), lang, ...)`）に
    # そのまま合わせる。この tokenizer には post-processor が無いので
    # add_special_tokens の有無で結果は変わらないが、変わったとしても下の
    # locate_subsequence が offset に吸収する。
    text_ids = [int(i) for i in tok.encode(text)]
    nospec_ids = encode_ids(tok, text)

    input_ids = model.build_tts_input_ids(text_ids, bundle.lang_token_id,
                                          device=bundle.device)
    if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
        raise AssertionError(
            f"build_tts_input_ids returned {tuple(input_ids.shape)}, expected (1, T)")
    built = [int(x) for x in input_ids[0].tolist()]

    entity_offset = locate_subsequence(built, nospec_ids)
    prefix_len = entity_offset
    suffix_len = len(built) - entity_offset - len(nospec_ids)
    if suffix_len < 0:
        raise AssertionError("negative suffix length; build_tts_input_ids changed shape")

    # pad_id がテキスト中に現れると encoder が
    # ``(input_ids != pad_id)`` で作る mask にその位置が落ちる
    # (Mamba2Seq2SeqEncoder.forward:319-324)。起きない前提だが黙って進まない。
    pad_id = bundle.meta["pad_token_id"]
    if pad_id in nospec_ids:
        raise RuntimeError(
            f"text tokenizes to a pad id ({pad_id}); encoder would mask it out: {text!r}")

    return {
        "text": text,
        "input_ids": input_ids,
        "built_ids": built,
        "text_ids": text_ids,
        "nospec_ids": nospec_ids,
        "offsets": nospec_offsets(tok, text),
        "entity_offset": int(entity_offset),
        "prefix_len": int(prefix_len),
        "suffix_len": int(suffix_len),
        "n_tokens": len(built),
    }


def encode_text(bundle: KisekiBundle, text: str,
                collect_hidden: bool = False) -> dict:
    """text encoder を 1 回だけ走らせる。

    ``collect_hidden=True`` のとき、各 EncoderLayer の出力を **読み取り専用の
    forward hook** で拾う（長さは変えないので 1. の落とし穴には当たらない）。
    ``hidden_states`` は HF の慣習に合わせて
    ``(embedding 出力, layer1 出力, ..., layer12 出力)`` の 13 本。
    ``enc``（= splice する対象）はさらに最終 RMSNorm を通ったもの。
    """
    import torch

    built = build_input_ids(bundle, text)
    model = bundle.model
    input_ids = built["input_ids"]

    hidden = []
    handles = []
    if collect_hidden:
        def _hook(_mod, _inp, out):
            hidden.append(out.detach())
        for layer in model.encoder.layers:
            handles.append(layer.register_forward_hook(_hook))

    try:
        with torch.inference_mode():
            if collect_hidden:
                embed_out = model.encoder.embed_tokens(input_ids).detach()
            enc = model.encoder(input_ids=input_ids)[0]
    finally:
        for h in handles:
            h.remove()

    hidden_states = None
    if collect_hidden:
        hidden_states = [embed_out] + list(hidden)
        n_expect = bundle.meta["enc_layers"] + 1
        if len(hidden_states) != n_expect:
            raise AssertionError(
                f"collected {len(hidden_states)} hidden states, expected {n_expect}")

    if int(enc.shape[1]) != built["n_tokens"]:
        raise AssertionError(
            f"encoder output length {int(enc.shape[1])} != input length "
            f"{built['n_tokens']}; the 1:1 token/time assumption is broken")

    built["enc"] = enc
    built["hidden_states"] = hidden_states
    return built


# --------------------------------------------------------------------------
# splice の組み立て
# --------------------------------------------------------------------------
def build_splice(bundle: KisekiBundle, text_a: str, word_a: str,
                 text_b: str, word_b: str, collect_kappa: bool = True) -> dict:
    """text_a / text_b を 1 回ずつ encode し、spliced な encoder 出力を作る。

    スパンは ``find_entity_token_range``（= ``injection.py:401`` と同じ計算）で
    src / tgt それぞれ独立に決める。**窓は広げず、トークン境界の検証もしない。**
    境界に乗っていたかどうかは ``span_info`` に記録するだけ。

    生成は行わない（``splice_synthesize`` から呼ばれる）。
    """
    tok = bundle.tokenizer

    src = encode_text(bundle, text_a, collect_hidden=collect_kappa)
    tgt = encode_text(bundle, text_b, collect_hidden=collect_kappa)

    s_found = find_entity_token_range(tok, text_a, word_a)
    t_found = find_entity_token_range(tok, text_b, word_b)
    if s_found is None:
        raise ValueError(f"entity {word_a!r} not found in {text_a!r}")
    if t_found is None:
        raise ValueError(f"entity {word_b!r} not found in {text_b!r}")

    s_report = describe_entity_span(tok, text_a, word_a, src["offsets"],
                                    s_found[:2], s_found[2])
    t_report = describe_entity_span(tok, text_b, word_b, tgt["offsets"],
                                    t_found[:2], t_found[2])

    # 特殊トークン抜きの座標 -> build_tts_input_ids 座標へ移す。
    s_span = (s_found[0] + src["entity_offset"], s_found[1] + src["entity_offset"])
    t_span = (t_found[0] + tgt["entity_offset"], t_found[1] + tgt["entity_offset"])

    # スパンがテキスト本体の範囲に収まっていることだけは確認する
    # （はみ出すと [TTS]/言語タグ/eos を差し替えてしまう。窓の拡大ではなく
    #   単なる範囲チェックなので、既存 3 モデルとの計算の同一性は保たれる）。
    _check_span_in_text(s_span, src, text_a, word_a, "src")
    _check_span_in_text(t_span, tgt, text_b, word_b, "tgt")

    spliced, info = splice_encoder_output(src["enc"], s_span, tgt["enc"], t_span)

    kappa_enc = entity_cosine(src["enc"], tgt["enc"], s_span, t_span)
    kappa_by_layer = None
    if collect_kappa and src["hidden_states"] is not None:
        kappa_by_layer = [
            entity_cosine(sh, th, s_span, t_span)
            for sh, th in zip(src["hidden_states"], tgt["hidden_states"])
        ]

    span_info = {
        "src": s_report,
        "tgt": t_report,
        "aligned": bool(s_report["aligned"] and t_report["aligned"]),
        "src_span_text": tok.decode(src["built_ids"][s_span[0]:s_span[1]]),
        "tgt_span_text": tok.decode(tgt["built_ids"][t_span[0]:t_span[1]]),
        "prefix_len_src": src["prefix_len"],
        "prefix_len_tgt": tgt["prefix_len"],
        "suffix_len_src": src["suffix_len"],
        "suffix_len_tgt": tgt["suffix_len"],
    }

    out = {
        "spliced_enc": spliced,
        "s_span": [int(s_span[0]), int(s_span[1])],
        "t_span": [int(t_span[0]), int(t_span[1])],
        "span_info": span_info,
        "spliced_surface_text": spliced_surface_text(text_a, word_a, text_b, word_b),
        "kappa_encoder_out": kappa_enc,
        "kappa_by_layer": kappa_by_layer,
    }
    out.update(info)
    return out


def _check_span_in_text(span: tuple[int, int], built: dict,
                        text: str, word: str, side: str) -> None:
    lo = built["entity_offset"]
    hi = built["entity_offset"] + len(built["nospec_ids"])
    if not (lo <= span[0] < span[1] <= hi):
        raise ValueError(
            f"{side} span {tuple(span)} leaves the text region [{lo}, {hi}) of "
            f"{text!r} (entity {word!r}); it would overwrite [TTS]/lang/eos")


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _decode_kwargs(max_new_frames: int, temperature: float, top_k: int,
                   top_p: float, repetition_penalty: float) -> dict:
    """BL / TGT / INJ で完全に共有する復号設定。差は splice だけ。"""
    return {
        "max_new_frames": int(max_new_frames),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
    }


def _codes_from_output(out) -> Any:
    """``generate_audio`` の戻りから有効フレームだけの (T, Q) を切り出す。

    README: ``codes = out["audio_codes"][0][out["valid_mask"][0]]``。
    ``valid_mask`` は第 0 codebook が EOS/PAD でないフレーム
    (modeling_mamba2_s2s.py:611)。
    """
    return out["audio_codes"][0][out["valid_mask"][0]]


def synthesize(bundle: KisekiBundle, text: str, seed: int = SEED,
               max_new_frames: int = DEFAULT_MAX_NEW_FRAMES,
               temperature: float = DEFAULT_TEMPERATURE,
               top_k: int = DEFAULT_TOP_K,
               top_p: float = DEFAULT_TOP_P,
               repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
               to_wav: bool = True) -> dict:
    """素の TTS（BL / TGT 用）。"""
    import torch

    built = build_input_ids(bundle, text)
    kw = _decode_kwargs(max_new_frames, temperature, top_k, top_p, repetition_penalty)

    seed_all(seed)
    with torch.inference_mode():
        out = bundle.model.generate_audio(input_ids=built["input_ids"],
                                          attention_mask=None, **kw)
    codes = _codes_from_output(out)

    res = {
        "text": text,
        "codes": codes,
        "n_frames": int(codes.shape[0]),
        "n_tokens": built["n_tokens"],
        "used_seed": int(seed),
        "decode_kwargs": kw,
        "hit_frame_cap": bool(int(out["audio_codes"].shape[1]) >= int(max_new_frames)),
    }
    if to_wav:
        wav, sr = codes_to_wav(bundle, codes)
        res["audio"] = wav
        res["sample_rate"] = sr
    return res


def splice_synthesize(bundle: KisekiBundle, text_a: str, word_a: str,
                      text_b: str, word_b: str, seed: int = SEED,
                      max_new_frames: int = DEFAULT_MAX_NEW_FRAMES,
                      temperature: float = DEFAULT_TEMPERATURE,
                      top_k: int = DEFAULT_TOP_K,
                      top_p: float = DEFAULT_TOP_P,
                      repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
                      collect_kappa: bool = True,
                      to_wav: bool = True) -> dict:
    """SpeechSplice 生成（INJ 用）。

    ``input_ids`` と ``attention_mask`` を **両方 None** にするのが必須。
    両方 None のとき ``_build_cross_mask`` は ``torch.ones(B, 1, 1, S)`` を返し、
    この ``S`` は ``generate_audio`` 内で ``B, S, _ = enc.shape`` として
    **splice 後の enc から** 取られるので長さが必ず一致する。
    """
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    sp = build_splice(bundle, text_a, word_a, text_b, word_b,
                      collect_kappa=collect_kappa)
    kw = _decode_kwargs(max_new_frames, temperature, top_k, top_p, repetition_penalty)

    seed_all(seed)
    with torch.inference_mode():
        out = bundle.model.generate_audio(
            encoder_outputs=BaseModelOutput(last_hidden_state=sp["spliced_enc"]),
            input_ids=None, attention_mask=None, **kw)
    codes = _codes_from_output(out)

    res = dict(sp)
    res.pop("spliced_enc", None)
    res.update({
        "text_a": text_a, "text_b": text_b,
        "word_a": word_a, "word_b": word_b,
        "codes": codes,
        "n_frames": int(codes.shape[0]),
        "used_seed": int(seed),
        "decode_kwargs": kw,
        "hit_frame_cap": bool(int(out["audio_codes"].shape[1]) >= int(max_new_frames)),
    })
    if to_wav:
        wav, sr = codes_to_wav(bundle, codes)
        res["audio"] = wav
        res["sample_rate"] = sr
    return res
