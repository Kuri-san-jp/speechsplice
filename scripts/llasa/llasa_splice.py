#!/usr/bin/env python3
"""Llasa-3B (HKUSTAudio/Llasa-3B) 用 SpeechSplice コア実装.

Llasa-3B は LlamaForCausalLM そのもの (config.json の architectures は
LlamaForCausalLM だけ)。text encoder も cross-attention も無い。テキストは
    <|begin_of_text|> ... user header ... "Convert the text to speech:"
    <|TEXT_UNDERSTANDING_START|> {text} <|TEXT_UNDERSTANDING_END|>
    ... assistant header ... <|SPEECH_GENERATION_START|> {参照音声トークン}
という **1 本の prefix** として prefill され、その続きとして speech token
(<|s_0|>..<|s_65535|>) を自己回帰生成する。self-attention のみ。

したがって差し替え地点は「encoder の最終出力」ではなく
    model.model.layers[l] の **出力** (= output_hidden_states[l+1])
の、prefill 系列の中のテキスト部・実体語トークン行。
Qwen3-TTS talker に対して既に行った注入と同じ作法で、同じ層格子
(0/7/14/21/26, 28 層) を掃く。

方針の出所 (推測ではなく既存コードに合わせてある):
  * prefill 限定の hook          : scripts/projection/eval_qwen3_injection_en.py:660-701
  * 1 回だけ発火させる fired flag: scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py:170-187
  * 長さ不一致時の mean-pool 展開: 同上 (m != n なら目標スパンを平均して m 行に broadcast)
  * kappa の定義                 : scripts/longcat/longcat_splice.py:kappa_by_layer と同じ
                                   (スパン内 hidden state をトークン方向に単純平均した
                                    ベクトル同士の cosine、entity ごと → 層ごとに平均)

重みは一切更新しない。推論時のみ。BL と INJ は seed も復号設定も系列長も共有し、
差は「prefill 中に実体語トークン行を目標文脈の同一層 hidden state で置換したか」だけ。

環境メモ:
  * transformers 4.57.3 の LlamaDecoderLayer.forward は **tuple ではなく Tensor**
    を返す (4.57.3 実測)。hook は両方受けられるように書いてある。
  * xcodec2 0.1.7 は Docker image に導入済み (Dockerfile:56)。
    BigCodecConfig の既定は sample_rate=16000 / hop_length=320 / upsample_factors=[]
    で、CodecDecoderVocos は upsample_factors が空なら upsampler=None のまま
    16000/50 == 320 のチェックを通る (実測)。本家 HKUSTAudio/xcodec2
    (16 kHz, 50 Hz フレーム, FSQ levels [4]*8 = 65536 codes) を読むが、
    **鍵名の改名が要る**。この xcodec2 0.1.7 は Anime-XCodec2 の fork で、
    SnakeBeta の magnitude を act.bias という名前で持つ。本家 checkpoint は
    act.beta。XCodec2Model.from_pretrained はこの食い違いを警告で済ませ、
    CodecEnc の活性化 36 個を乱数初期化したまま推論に入る (2026-09-16 実測:
    3.25 s の参照音声 163 トークンが GPU で全部 0、CPU で全部 INT_MIN)。
    load_codec で .beta -> .bias に改名してから strict に load_state_dict する。
    CPU 検証: taskset -c 0-63 python3 scripts/llasa/verify_xcodec2_cpu.py
  * 日本語は非対応 (英語と中国語のみ)。ライセンスは CC-BY-NC-4.0 (非商用)。
"""
import os
import re
import sys
from dataclasses import dataclass, field

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

DEFAULT_MODEL_ID = "HKUSTAudio/Llasa-3B"
DEFAULT_CODEC_ID = "HKUSTAudio/xcodec2"
SEED = 42

# Qwen3-TTS talker (28 層) で使った掃引格子をそのまま使う。Llasa-3B も 28 層。
SWEEP_LAYERS = (0, 7, 14, 21, 26)
EXPECTED_NUM_LAYERS = 28

# 公式 (model card) の復号設定。BL / TGT / INJ で完全に共有する。
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0          # 0 = 無効化 (公式は top_k を指定していない)
DEFAULT_MAX_NEW_TOKENS = 1024

TEXT_START = "<|TEXT_UNDERSTANDING_START|>"
TEXT_END = "<|TEXT_UNDERSTANDING_END|>"
SPEECH_GEN_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GEN_END = "<|SPEECH_GENERATION_END|>"
USER_INSTRUCTION = "Convert the text to speech:"

SPEECH_TOKEN_RE = re.compile(r"^<\|s_(\d+)\|>$")


# --------------------------------------------------------------------------
# HF cache: slurm_run.sh は -e HF_HOME=/tmp/.cache/huggingface を渡すが、その実体は
# ${HOME}/.cache/huggingface で空。重みが置いてあるのは
# <PROJECT_ROOT>/.cache/huggingface (コンテナ内 /mnt/.cache/huggingface)。
# setdefault ではコンテナの環境変数に負けるのでここで上書きする。
# scripts/longcat/longcat_splice.py:57-65 と同じ扱い。transformers import 前に済ませる。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


def setup_hf_home(hf_home=None):
    hf_home = hf_home or os.environ.get("LLASA_HF_HOME") or DEFAULT_HF_HOME
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


def local_snapshot(hub_id, required=True):
    """HF_HUB_CACHE 内のスナップショットディレクトリを返す (offline 前提).

    見つからなければ required=True で FileNotFoundError。
    ダウンロードは呼び出し側が明示的に許可した場合だけ行う (load_llasa の
    allow_download)。事故で巨大な重みを落とさないための既定。
    """
    if os.path.isdir(hub_id):
        return hub_id
    d = os.path.join(os.environ["HF_HUB_CACHE"],
                     f"models--{hub_id.replace('/', '--')}", "snapshots")
    if not os.path.isdir(d):
        if required:
            raise FileNotFoundError(
                f"snapshot dir not found for {hub_id}: {d}\n"
                f"  ローカルに無い。ダウンロードを許可するなら --allow_download を付けること "
                f"(GPU ジョブ内で HF から取得する)。")
        return None
    snaps = [s for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
    if not snaps:
        if required:
            raise FileNotFoundError(f"no snapshot under {d}")
        return None
    return os.path.join(d, snaps[-1])


def resolve_source(hub_id, allow_download):
    """local snapshot があればそのパス、無ければ (許可されていれば) hub id を返す."""
    p = local_snapshot(hub_id, required=not allow_download)
    return p if p is not None else hub_id


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
@dataclass
class LlasaBundle:
    model: object
    tokenizer: object
    codec: object
    device: str
    dtype: object
    num_layers: int
    hidden_size: int
    speech_start_id: int
    speech_end_id: int
    model_dir: str
    codec_dir: str
    sampling_rate: int = 16000
    info: dict = field(default_factory=dict)


def load_llasa(model_id=DEFAULT_MODEL_ID, codec_id=DEFAULT_CODEC_ID, device=None,
               dtype=torch.bfloat16, allow_download=False, with_codec=True,
               attn_implementation="sdpa"):
    """Llasa-3B 本体と XCodec2 を読む。

    Llasa-3B は素の LlamaForCausalLM なので AutoModelForCausalLM で読める
    (trust_remote_code 不要)。VRAM は bf16 で本体 約7GB + XCodec2 約2GB。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src = resolve_source(model_id, allow_download)

    tok = AutoTokenizer.from_pretrained(src)
    if not tok.is_fast:
        raise RuntimeError(
            "fast tokenizer が必要 (return_offsets_mapping でスパンを取るため)。"
            f"{model_id} の tokenizer.json を確認すること。")

    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=dtype, attn_implementation=attn_implementation)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    if n_layers != EXPECTED_NUM_LAYERS:
        print(f"[warn] num_hidden_layers={n_layers} (想定 {EXPECTED_NUM_LAYERS})。"
              f"層格子 {SWEEP_LAYERS} を見直すこと。", flush=True)

    sp_start = tok.convert_tokens_to_ids(SPEECH_GEN_START)
    sp_end = tok.convert_tokens_to_ids(SPEECH_GEN_END)
    unk = getattr(tok, "unk_token_id", None)
    for name, tid in (("SPEECH_GENERATION_START", sp_start), ("SPEECH_GENERATION_END", sp_end),
                      ("TEXT_UNDERSTANDING_START", tok.convert_tokens_to_ids(TEXT_START)),
                      ("TEXT_UNDERSTANDING_END", tok.convert_tokens_to_ids(TEXT_END))):
        if tid is None or (unk is not None and tid == unk):
            raise RuntimeError(f"特殊トークン {name} が tokenizer に無い ({tid})。model_id を確認すること。")
    # speech token が 1 個 1 トークンであること。ここが崩れていると prompt の
    # トークン数もスパン index も静かにずれる。
    probe = tok("<|s_0|><|s_65535|>", add_special_tokens=False)["input_ids"]
    if len(probe) != 2:
        raise RuntimeError(
            f"speech token が 1 トークンになっていない (<|s_0|><|s_65535|> -> {len(probe)} tokens)。"
            " tokenizer が Llasa 用のものか確認すること。")

    codec = None
    codec_dir = ""
    codec_info = None
    if with_codec:
        codec, codec_dir, codec_info = load_codec(
            codec_id, device=device, allow_download=allow_download)

    return LlasaBundle(
        model=model, tokenizer=tok, codec=codec, device=device, dtype=dtype,
        num_layers=n_layers, hidden_size=int(cfg.hidden_size),
        speech_start_id=int(sp_start), speech_end_id=int(sp_end),
        model_dir=src, codec_dir=codec_dir,
        info={
            "vocab_size": int(cfg.vocab_size),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
            "architectures": list(getattr(cfg, "architectures", []) or []),
            "attn_implementation": attn_implementation,
            "codec": codec_info,
        },
    )


SNAKE_BETA_SUFFIX = ".beta"
EXPECTED_BETA_RENAMES = 36      # 公式 HKUSTAudio/xcodec2 の *.act.beta の個数 (実測)
CODEC_SAMPLE_RATE = 16000
CODEC_HOP_LENGTH = 320          # 16000 / 320 = 50 frames/sec
CODEC_FRAME_RATE = CODEC_SAMPLE_RATE // CODEC_HOP_LENGTH


def _codec_checkpoint_path(src, codec_id, allow_download):
    """XCodec2 の重みファイル (model.safetensors) の実体パスを返す."""
    if os.path.isdir(src):
        p = os.path.join(src, "model.safetensors")
        if not os.path.exists(p):
            raise FileNotFoundError(f"model.safetensors が無い: {src}")
        return p
    if not allow_download:
        raise FileNotFoundError(
            f"codec のローカル snapshot が無い ({codec_id})。--allow_download が要る。")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=codec_id, filename="model.safetensors")


def remap_xcodec2_state_dict(raw):
    """公式 HKUSTAudio/xcodec2 の鍵名を xcodec2 0.1.7 の鍵名に合わせる.

    公式 checkpoint は SnakeBeta の magnitude パラメータを ``...act.beta``
    という名前で持つ (36 個)。一方この環境に入っている xcodec2 0.1.7
    (Anime-XCodec2 fork) の SnakeBeta は同じパラメータを ``self.bias``
    として定義している:
        site-packages/xcodec2/vq/activations.py:97-103  self.bias = Parameter(...)
        site-packages/xcodec2/vq/activations.py:114     beta = self.bias.unsqueeze(0)...

    名前が食い違っていても ``from_pretrained`` は例外を出さず、
      * checkpoint 側の ``act.beta`` を "not used" として捨て
      * model 側の ``act.bias`` を "newly initialized" (torch.ones) にして
    そのまま推論に入る。SnakeBeta は x + (1/beta)·sin²(αx) なので beta が
    全部 1 に潰れると CodecEnc の非線形が別物になり、encode_code が
    全フレーム同じ code を返す (実測: 3.25 s の参照音声 163 トークンが全部 0)。

    改名の作法は既に動いている実装と同じ:
        $T5GEMMA_ROOT/data/tokenizer.py:58-62
        ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
    ここでは末尾一致に限定し、改名で衝突したら例外にしている。
    """
    state, renamed = {}, []
    for k, v in raw.items():
        if k.endswith(SNAKE_BETA_SUFFIX):
            k2 = k[: -len(SNAKE_BETA_SUFFIX)] + ".bias"
            renamed.append(k)
        else:
            k2 = k
        if k2 in state:
            raise RuntimeError(f"state_dict の改名で鍵が衝突した: {k} -> {k2}")
        state[k2] = v
    return state, renamed


def load_codec(codec_id=DEFAULT_CODEC_ID, device=None, allow_download=False,
               strict_state_dict=True):
    """XCodec2 (16 kHz / 50 Hz frame / 65536 codes) を読む。

    ``XCodec2Model.from_pretrained(src)`` を直接使ってはいけない。鍵名の
    食い違い (``act.beta`` vs ``act.bias``) を警告で済ませて黙って壊れた
    エンコーダを返すため。詳細は remap_xcodec2_state_dict の docstring。

    そこで
      1. safetensors を生で読み、``.beta`` -> ``.bias`` に改名し
      2. config から XCodec2Model を素で組み
      3. load_state_dict で **鍵の過不足を例外にして** 流し込む
    という順にする。1 は TTS36/vendor/t5gemma_tts/data/tokenizer.py:58-66
    (本プロジェクトで既に正常動作している読み込み) と同じ作法。

    なお ``BigCodecConfig.model_type == "xcodec"`` に対し公式 config.json は
    ``"model_type": "xcodec2"`` なので
        You are using a model of type xcodec2 to instantiate a model of type xcodec.
    という警告が 1 行出るが、これは無害 (同じ config クラスの別名)。
    実害のあった "not used / newly initialized" の警告は上の手順で消える。

    xcodec2.modeling_xcodec2.XCodec2Model は __init__ で
    facebook/w2v-bert-2.0 を from_pretrained する。これはローカル cache に
    既に存在する (models--facebook--w2v-bert-2.0)。
    """
    from safetensors.torch import load_file
    from xcodec2.configuration_bigcodec import BigCodecConfig
    from xcodec2.modeling_xcodec2 import XCodec2Model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src = resolve_source(codec_id, allow_download)
    ckpt_path = _codec_checkpoint_path(src, codec_id, allow_download)

    raw = load_file(ckpt_path, device="cpu")
    state, renamed = remap_xcodec2_state_dict(raw)
    if strict_state_dict and len(renamed) != EXPECTED_BETA_RENAMES:
        raise RuntimeError(
            f"SnakeBeta の改名数が想定と違う: {len(renamed)} (想定 {EXPECTED_BETA_RENAMES})。"
            f" checkpoint {ckpt_path} が公式 HKUSTAudio/xcodec2 か、"
            f" もしくは xcodec2 パッケージ側の命名が変わっていないか確認すること。")

    config = BigCodecConfig.from_pretrained(src)
    sr = int(getattr(config, "sample_rate", CODEC_SAMPLE_RATE))
    hop = int(getattr(config, "hop_length", CODEC_HOP_LENGTH))
    if (sr, hop) != (CODEC_SAMPLE_RATE, CODEC_HOP_LENGTH):
        raise RuntimeError(
            f"codec の格子が想定と違う: sample_rate={sr} hop_length={hop} "
            f"(想定 {CODEC_SAMPLE_RATE}/{CODEC_HOP_LENGTH} = {CODEC_FRAME_RATE} frames/sec)")

    codec = XCodec2Model(config)
    # strict=True だと差分の中身が見えないので、strict=False で受けてから自分で落とす。
    incompatible = codec.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if strict_state_dict and (missing or unexpected):
        raise RuntimeError(
            "XCodec2 の state_dict が一致しない。"
            f"\n  missing ({len(missing)}): {missing[:8]}"
            f"\n  unexpected ({len(unexpected)}): {unexpected[:8]}")

    codec = codec.to(device)
    codec.eval()
    codec.requires_grad_(False)
    info = {
        "ckpt": ckpt_path,
        "n_keys": len(state),
        "n_beta_renamed": len(renamed),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "sample_rate": sr,
        "hop_length": hop,
        "frame_rate": sr // hop,
        "config_model_type": getattr(config, "model_type", None),
    }
    print(f"[codec] {codec_id} <- {ckpt_path}", flush=True)
    print(f"[codec] keys={len(state)} beta->bias renamed={len(renamed)} "
          f"missing={len(missing)} unexpected={len(unexpected)} "
          f"{sr}Hz/hop{hop}={sr // hop}fps", flush=True)
    return codec, src, info


# --------------------------------------------------------------------------
# speech token <-> id
# --------------------------------------------------------------------------
def ids_to_speech_tokens(speech_ids):
    return [f"<|s_{int(i)}|>" for i in speech_ids]


def speech_tokens_to_string(speech_ids):
    return "".join(ids_to_speech_tokens(speech_ids))


def extract_speech_ids(token_strings, stop_on_foreign=True):
    """<|s_123|> 形式のトークン文字列列を整数 id 列に戻す.

    公式サンプルは batch_decode(skip_special_tokens=True) してから前方一致で
    拾っているが、ここでは convert_ids_to_tokens の結果を直接正規表現で
    判定する (special token 扱いの差で落ちないようにするため)。
    """
    out = []
    n_foreign = 0
    for t in token_strings:
        m = SPEECH_TOKEN_RE.match(t)
        if m:
            out.append(int(m.group(1)))
        else:
            n_foreign += 1
            if stop_on_foreign:
                break
    return out, n_foreign


def load_wav_16k(wav_path, max_sec=None):
    """wav を 16 kHz mono float32 の 1D numpy にして返す (XCodec2 は 16 kHz 前提)."""
    import soundfile as sf
    import scipy.signal

    a, sr = sf.read(wav_path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != CODEC_SAMPLE_RATE:
        a = scipy.signal.resample(
            a, int(len(a) * CODEC_SAMPLE_RATE / sr)).astype(np.float32)
        sr = CODEC_SAMPLE_RATE
    if max_sec is not None:
        a = a[:int(max_sec * sr)]
    return np.ascontiguousarray(a.astype(np.float32)), sr


def encode_reference_wav(bundle, wav_path, max_sec=None):
    """参照音声を XCodec2 で speech token 列にする (voice clone prompt 用)."""
    a, sr = load_wav_16k(wav_path, max_sec=max_sec)
    wav = torch.from_numpy(a).unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        code = bundle.codec.encode_code(input_waveform=wav)
    ids = code.reshape(-1).tolist()
    return {"speech_ids": [int(i) for i in ids], "n_tokens": len(ids),
            "wav_path": wav_path, "max_sec": max_sec,
            "dur_sec": round(len(a) / sr, 3)}


def log_mag_spectrogram(x, n_fft=512, hop=160):
    """位相に依らない比較のための log(1+|STFT|)。返り値 [freq, frames]."""
    t = torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
    if t.numel() < n_fft:
        t = F.pad(t, (0, n_fft - t.numel()))
    S = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=torch.hann_window(n_fft), center=True,
                   return_complex=True)
    return torch.log1p(S.abs())


def spectral_correlation(ref, rec, n_fft=512, hop=160):
    """元波形と再構成波形の log 振幅スペクトログラムの Pearson 相関.

    XCodec2 は 50 Hz / 単一 codebook (65536) の生成型 codec なので位相は
    保存されない。波形の相関は働いていても 0 付近になるため、振幅
    スペクトログラムで見る。壊れた codec (全フレーム同一 code) だと
    スペクトルが時間方向にほぼ一定になるので相関は 0 近くに落ちる。
    """
    n = int(min(len(ref), len(rec)))
    if n < n_fft:
        return float("nan")
    A = log_mag_spectrogram(ref[:n], n_fft=n_fft, hop=hop)
    B = log_mag_spectrogram(rec[:n], n_fft=n_fft, hop=hop)
    m = int(min(A.shape[-1], B.shape[-1]))
    a = A[..., :m].reshape(-1).double()
    b = B[..., :m].reshape(-1).double()
    a = a - a.mean()
    b = b - b.mean()
    den = float(a.norm() * b.norm())
    if den <= 0:
        return 0.0
    return float((a @ b).item() / den)


def codec_roundtrip_check(bundle, ref_enc, min_rms=1e-4, min_unique=8,
                          min_unique_ratio=0.05, min_spec_corr=0.45):
    """XCodec2 が本当に動いているかを、参照音声の encode -> decode で確かめる.

    この環境の xcodec2 0.1.7 は Anime-XCodec2 の fork で、公式
    HKUSTAudio/xcodec2 の checkpoint とは SnakeBeta の鍵名が違う
    (remap_xcodec2_state_dict 参照)。改名を忘れると encode_code が
    全フレーム同じ code を返すが、``from_pretrained`` は警告しか出さず
    評価は最後まで完走してしまう。ここで止める。

    見るもの:
      1. id が全部同じ (id_min == id_max)        -> エンコーダが死んでいる
      2. ユニーク id 数が極端に少ない             -> 同上 (縮退)
      3. 再構成が無音に近い (rms)
      4. 再構成長がフレーム数と合わない
      5. 元音声との log 振幅スペクトログラム相関  -> 別音声になっていないか
    """
    ids = ref_enc["speech_ids"]
    if not ids:
        raise RuntimeError("参照音声から speech token が 1 つも出ていない")
    id_min, id_max = int(min(ids)), int(max(ids))
    if id_max >= 65536 or id_min < 0:
        raise RuntimeError(f"speech id が範囲外 (min={id_min}, max={id_max})")

    n_unique = len(set(ids))
    wav = decode_speech_ids(bundle, ids)
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    sec = wav.size / bundle.sampling_rate
    expect = len(ids) / float(CODEC_FRAME_RATE)

    ref_wav, _ = load_wav_16k(ref_enc["wav_path"], max_sec=ref_enc.get("max_sec"))
    corr = spectral_correlation(ref_wav, wav)

    report = {"n_tokens": len(ids), "n_unique_ids": n_unique,
              "unique_ratio": round(n_unique / max(1, len(ids)), 4),
              "rms": round(rms, 6), "dur_sec": round(sec, 3),
              "expected_sec": round(expect, 3),
              "id_min": id_min, "id_max": id_max,
              "spec_corr": (round(corr, 4) if corr == corr else None)}

    if id_min == id_max:
        raise RuntimeError(
            "XCodec2 のエンコーダが縮退している (全トークンが同一 id)。"
            " state_dict の鍵名 (act.beta -> act.bias) の改名を確認すること: "
            f"{report}")
    need_unique = max(int(min_unique), int(min_unique_ratio * len(ids)))
    if n_unique < need_unique:
        raise RuntimeError(
            f"XCodec2 のトークンが縮退している (unique {n_unique} < {need_unique}): {report}")
    if rms < min_rms:
        raise RuntimeError(f"XCodec2 の再構成が無音に近い: {report}")
    if abs(sec - expect) > max(0.5, 0.2 * expect):
        raise RuntimeError(f"XCodec2 の再構成長がフレーム数と合わない: {report}")
    if corr != corr:
        raise RuntimeError(f"スペクトル相関が計算できない (参照音声が短すぎる): {report}")
    if corr < min_spec_corr:
        raise RuntimeError(
            f"XCodec2 の再構成が元音声から離れすぎ (spec_corr={corr:.4f} "
            f"< {min_spec_corr}): {report}")
    return report


def decode_speech_ids(bundle, speech_ids):
    """speech token id 列 -> 16 kHz 波形 (numpy 1D)."""
    if len(speech_ids) == 0:
        raise RuntimeError("speech_ids が空 (生成が即座に EOS した可能性)")
    code = torch.tensor(speech_ids, dtype=torch.long, device=bundle.device)
    code = code.unsqueeze(0).unsqueeze(0)          # [1, 1, T]
    with torch.no_grad():
        wav = bundle.codec.decode_code(code)
    return wav.squeeze().detach().float().cpu().numpy()


# --------------------------------------------------------------------------
# prompt 構築 (chat template)
# --------------------------------------------------------------------------
def build_chat(input_text, ref_speech_ids=None):
    """公式 model card と同じ chat を組む.

    voice clone するときは
        input_text = ref_text + " " + target_text
    とし、assistant 側に <|SPEECH_GENERATION_START|> + 参照音声トークンを置く。
    生成はその続きになる (= 参照音声の続きを喋らせる)。
    """
    formatted = f"{TEXT_START}{input_text}{TEXT_END}"
    assistant = SPEECH_GEN_START
    if ref_speech_ids:
        assistant = assistant + speech_tokens_to_string(ref_speech_ids)
    return [
        {"role": "user", "content": USER_INSTRUCTION + formatted},
        {"role": "assistant", "content": assistant},
    ]


def chat_template_date(tokenizer):
    """chat template が prompt に日付を差し込むかどうかを返す (2026-09-18 追加).

    Llasa の chat template は Llama-3 系で、`date_string` と "Today Date" と
    `strftime` を含む。`apply_chat_template` に date_string を渡さないと
    **実行日が prompt に入る**ので、後日の再実行で prompt が変わる。
    1 回の実行の中では BL / TGT / INJ で共通なので比較は壊れないが、
    再現のためには使った日付を記録に残す必要がある。
    """
    t = getattr(tokenizer, "chat_template", None) or ""
    return "date_string" in t or "Today Date" in t


def render_prompt(tokenizer, chat, date_string=None):
    """chat -> prompt 文字列。tokenize=True の結果と id が一致することを必ず確かめる.

    offsets が欲しいので文字列経由で tokenize するが、それが公式の
    apply_chat_template(tokenize=True) と 1 token でも違ったら黙って進まない。

    date_string を渡すと chat template の日付をその値に固定する (2026-09-18 追加)。
    **既定は None で、渡さなければ従来どおり実行日が入る。**既存の結果
    (英語 300 対は 2026-09-17 実行) を変えないため既定は変えていない。
    """
    kw = {"continue_final_message": True}
    if date_string is not None:
        kw["date_string"] = date_string
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, **kw)
    if not isinstance(prompt, str):
        raise RuntimeError(f"apply_chat_template(tokenize=False) が str を返さない: {type(prompt)}")
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids_str = list(enc["input_ids"])
    # 照合側にも同じ日付を渡す。渡さないと固定したときに 1 token 違いで止まる。
    ids_tpl = tokenizer.apply_chat_template(chat, tokenize=True, **kw)
    if hasattr(ids_tpl, "tolist"):
        ids_tpl = ids_tpl.tolist()
    if isinstance(ids_tpl, list) and ids_tpl and isinstance(ids_tpl[0], list):
        ids_tpl = ids_tpl[0]
    if list(ids_str) != list(ids_tpl):
        raise RuntimeError(
            "chat template の文字列経由 tokenize が tokenize=True と一致しない "
            f"(len {len(ids_str)} vs {len(ids_tpl)})。offsets 経由のスパン特定が使えないので中断する。")
    return {
        "prompt": prompt,
        "input_ids": ids_str,
        "offsets": [(int(a), int(b)) for a, b in enc["offset_mapping"]],
    }


def locate_text_region(prompt, target_text):
    """prompt 文字列の中で「今回合成させたい本文」が占める文字範囲 [s, e) を返す.

    voice clone では ref_text も同じ <|TEXT_UNDERSTANDING_*|> の中に入るので、
    実体語の検索範囲をこの領域に絞らないと ref_text 側の同綴りを拾う危険がある。
    出現回数が 1 でなければエラーにする (どこを差し替えたか曖昧なまま生成しない)。
    """
    n = prompt.count(target_text)
    if n != 1:
        raise ValueError(f"target text が prompt に {n} 回出現する (1 回であるべき): {target_text!r}")
    s = prompt.index(target_text)
    return s, s + len(target_text)


def entity_char_span(prompt, region, word):
    """本文領域の中の実体語の文字スパン [c0, c1) を返す (語境界つき・一意)."""
    w = word.strip()
    if not w:
        raise ValueError(f"empty entity word: {word!r}")
    s, e = region
    body = prompt[s:e]
    hits = list(re.finditer(r"(?<![0-9A-Za-z])" + re.escape(w) + r"(?![0-9A-Za-z])", body))
    if not hits:
        raise ValueError(f"entity {w!r} not found in body {body!r}")
    if len(hits) > 1:
        raise ValueError(f"entity {w!r} occurs {len(hits)} times in body {body!r}")
    a, b = hits[0].span()
    return s + a, s + b


def entity_token_span(tokenizer, text, word, ref_speech_ids=None, rendered=None,
                      region_text=None, expand=None):
    """chat template を通した prompt の中で、実体語が占める **トークン** スパンを返す.

    半開区間 [tok_start, tok_end)。offsets と交差するトークンを全部拾い、
    連続であることと、カバーした文字列を strip したら word に戻ることを確かめる。
    BPE は直前の空白を語頭トークンに含める (" Yamada") ので strip 判定にしてある。
    chat header や特殊トークンの分の offset は、prompt 全体を tokenize している
    以上そのまま入っている (自前で足し引きしない)。

    text        : <|TEXT_UNDERSTANDING_*|> に入れた文字列全体
                  (voice clone のときは ref_text + " " + 本文)
    region_text : 実体語を探す範囲。既定は text 全体だが、voice clone では
                  本文だけを渡して ref_text 側の同綴りを拾わないようにする。
    expand      : (pre, post) を渡すと、要求する表層を実体語の左右に pre / post
                  文字だけ広げてからトークンスパンを取り、往復一致もその表層に
                  対して判定する (最小対称拡張のときだけ使う)。広げた範囲が本文
                  領域の外に出るときは ValueError。**既定は None で、そのときは
                  1 行も挙動が変わらない** (want == word)。
    """
    r = rendered if rendered is not None else render_prompt(
        tokenizer, build_chat(text, ref_speech_ids))
    prompt = r["prompt"]
    offs = r["offsets"]
    region = locate_text_region(prompt, region_text if region_text is not None else text)
    c0, c1 = entity_char_span(prompt, region, word)
    want = word
    if expand is not None:
        pre, post = int(expand[0]), int(expand[1])
        if pre < 0 or post < 0:
            raise ValueError(f"expand は非負であること: {expand!r}")
        if c0 - pre < region[0] or c1 + post > region[1]:
            raise ValueError(
                f"expand={expand!r} が本文領域 {region} の外に出る "
                f"(char_span={(c0, c1)})")
        c0, c1 = c0 - pre, c1 + post
        want = prompt[c0:c1]
    sel = [i for i, (a, b) in enumerate(offs) if b > a and a < c1 and b > c0]
    if not sel:
        raise ValueError(f"no token overlaps entity char span {(c0, c1)}")
    if sel != list(range(sel[0], sel[-1] + 1)):
        raise ValueError(f"entity token span is not contiguous: {sel}")
    tok_start, tok_end = sel[0], sel[-1] + 1
    covered = prompt[offs[tok_start][0]:offs[tok_end - 1][1]]
    span = {
        "word": word,
        "text": text,
        "char_span": [int(c0), int(c1)],
        "text_region": [int(region[0]), int(region[1])],
        "tok_start": int(tok_start),
        "tok_end": int(tok_end),
        "n_tokens_entity": int(tok_end - tok_start),
        "n_tokens_total": int(len(r["input_ids"])),
        "covered_text": covered,
        "roundtrip_ok": bool(covered.strip() == want.strip()),
        "entity_tokens": tokenizer.convert_ids_to_tokens(
            r["input_ids"][tok_start:tok_end]),
    }
    # 拡張したときだけ鍵を足す (expand=None の記録は従来と同じ形のまま)。
    if expand is not None:
        span["requested_surface"] = want
        span["expand"] = [int(expand[0]), int(expand[1])]
    return span


def verify_entity_span(span, strict=True):
    if not span["roundtrip_ok"]:
        msg = (f"span roundtrip failed: covered={span['covered_text']!r} "
               f"!= word={span['word']!r} (text={span['text']!r})")
        if strict:
            raise ValueError(msg)
        return False, msg
    return True, ""


# --------------------------------------------------------------------------
# 最小対称拡張への配線 (2026-09-18 ユーザ承認)
#
# 規則の実装は scripts/projection/span_extend.py の 1 箇所だけ。ここは配線で、
# **語そのままで往復一致するなら従来の経路をそのまま通す**。拡張が起きるのは
# 従来 ValueError で落ちていた対だけ (中国語の「的」など、トークン化器が実体語と
# 隣の機能語を 1 トークンに融合する対)。prompt / input_ids は変えない。
# 変わるのは差し替えるトークンスパンだけ。
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
    重なりを許す)。voice clone では ref_text も同じ prompt に入るので、
    occurrence 番号を取り違えると別の位置を見てしまう。
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


def _try_span_extension(tokenizer, src, word_a, region_a, tgt, word_b, region_b):
    """語そのままで往復一致しない 1 対に最小対称拡張を試す.

    戻り値は {"extra","pre","post","surface_a","surface_b","span_a","span_b"}
    か None。**prompt も input_ids も一切変えない** (スパンだけ取り直す)。
    取り直したスパンがこの経路で往復一致しなければ None を返す
    (数値を合わせるための後付け補正はしない)。
    """
    r_a = {"prompt": src["prompt"], "input_ids": src["input_ids"],
           "offsets": src["offsets"]}
    r_b = {"prompt": tgt["prompt"], "input_ids": tgt["input_ids"],
           "offsets": tgt["offsets"]}
    wa, wb = word_a.strip(), word_b.strip()
    occ_a = _find_occurrence_index(src["prompt"], wa, src["span"]["char_span"][0])
    occ_b = _find_occurrence_index(tgt["prompt"], wb, tgt["span"]["char_span"][0])
    if occ_a is None or occ_b is None:
        return None
    r = _load_minimal_symmetric_surface()(
        tokenizer, src["prompt"], wa, tgt["prompt"], wb,
        occurrence_a=occ_a, occurrence_b=occ_b)
    if r is None or r["plain"]:
        return None
    try:
        sa = entity_token_span(tokenizer, src["span"]["text"], word_a, rendered=r_a,
                               region_text=region_a, expand=(r["pre"], r["post"]))
        sb = entity_token_span(tokenizer, tgt["span"]["text"], word_b, rendered=r_b,
                               region_text=region_b, expand=(r["pre"], r["post"]))
    except ValueError:
        return None
    if not (sa["roundtrip_ok"] and sb["roundtrip_ok"]):
        return None
    if (sa["requested_surface"] != r["surface_a"]
            or sb["requested_surface"] != r["surface_b"]):
        return None
    ext = {"extra": r["extra"], "pre": int(r["pre"]), "post": int(r["post"]),
           "surface_a": r["surface_a"], "surface_b": r["surface_b"]}
    sa["span_extension"] = dict(ext)
    sb["span_extension"] = dict(ext)
    out = dict(ext)
    out["span_a"], out["span_b"] = sa, sb
    return out


def prepare_context(bundle, text, ref_speech_ids=None, word=None, need_hidden=False,
                    strict_span=True, region_text=None):
    """1 本のテキストについて prompt / ids / span / hidden states をまとめて作る."""
    chat = build_chat(text, ref_speech_ids)
    r = render_prompt(bundle.tokenizer, chat)
    out = {"chat": chat, "prompt": r["prompt"], "input_ids": r["input_ids"],
           "offsets": r["offsets"], "prompt_len": len(r["input_ids"])}
    if word is not None:
        span = entity_token_span(bundle.tokenizer, text, word, rendered=r,
                                 region_text=region_text)
        verify_entity_span(span, strict=strict_span)
        out["span"] = span
    if need_hidden:
        out["hidden_states"] = prefill_hidden_states(bundle, r["input_ids"])
    return out


def prepare_context_pair(bundle, in_a, word_a, in_b, word_b, ref_speech_ids=None,
                         region_text_a=None, region_text_b=None,
                         need_hidden_a=False, need_hidden_b=True,
                         strict_span=True, span_extension=False):
    """元 (in_a) と目標 (in_b) の context をまとめて作る.

    戻り値 (src, tgt, ext)。ext は実際に使われた拡張
    {"extra","pre","post","surface_a","surface_b"} か None。

      * span_extension=False -> prepare_context を 2 回呼ぶだけ。**従来と同一**。
      * 両側とも語そのままで往復一致する -> やはり prepare_context 2 回と同じ
        src / tgt を返す (拡張の規則は import もしない)。
      * 往復一致しない かつ span_extension=True -> 最小対称拡張を試し、
        成立したら span だけ差し替える。救えなければ従来と同じ ValueError
        (strict_span=True のとき)。
    """
    if not span_extension:
        src = prepare_context(bundle, in_a, ref_speech_ids=ref_speech_ids,
                              word=word_a, need_hidden=need_hidden_a,
                              strict_span=strict_span, region_text=region_text_a)
        tgt = prepare_context(bundle, in_b, ref_speech_ids=ref_speech_ids,
                              word=word_b, need_hidden=need_hidden_b,
                              strict_span=strict_span, region_text=region_text_b)
        return src, tgt, None
    # 拡張を試すので往復一致の判定はここで行う (prepare_context には投げさせない)。
    src = prepare_context(bundle, in_a, ref_speech_ids=ref_speech_ids,
                          word=word_a, need_hidden=need_hidden_a,
                          strict_span=False, region_text=region_text_a)
    tgt = prepare_context(bundle, in_b, ref_speech_ids=ref_speech_ids,
                          word=word_b, need_hidden=need_hidden_b,
                          strict_span=False, region_text=region_text_b)
    if src["span"]["roundtrip_ok"] and tgt["span"]["roundtrip_ok"]:
        return src, tgt, None                  # 従来の経路そのまま
    ext = _try_span_extension(
        bundle.tokenizer, src, word_a, region_text_a if region_text_a is not None else in_a,
        tgt, word_b, region_text_b if region_text_b is not None else in_b)
    if ext is None:
        verify_entity_span(src["span"], strict=strict_span)
        verify_entity_span(tgt["span"], strict=strict_span)
        return src, tgt, None
    src["span"] = ext.pop("span_a")
    tgt["span"] = ext.pop("span_b")
    return src, tgt, ext


def build_input_text(target_text, ref_text=None):
    """voice clone のときは公式と同じく「参照文 + 半角空白 + 本文」を 1 本にする."""
    if ref_text:
        return f"{ref_text.strip()} {target_text}"
    return target_text


# --------------------------------------------------------------------------
# hidden states / kappa
# --------------------------------------------------------------------------
def prefill_hidden_states(bundle, input_ids):
    """prefill を 1 回だけ回して全層の hidden state を取る (cache 不使用).

    戻り値は長さ num_layers+1 のタプル。index 0 = embedding 出力、
    index l+1 = layers[l] の出力 (= splice で置き換える対象)。
    """
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=bundle.device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = bundle.model.model(input_ids=ids, attention_mask=attn,
                                 use_cache=False, output_hidden_states=True)
    return tuple(h.detach() for h in out.hidden_states)


def hidden_index_for_layer(layer):
    """layers[l] の出力は hidden_states[l+1]."""
    return int(layer) + 1


def span_vector(hidden, tok_start, tok_end):
    """スパン内 hidden state をトークン方向に単純平均したベクトル."""
    return hidden[0, tok_start:tok_end].float().mean(dim=0)


def kappa_by_layer(src_hidden_states, tgt_hidden_states, s_span, t_span):
    """kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}).

    既存の定義 (スパン内 hidden state をトークン方向に単純平均したベクトル
    同士の cosine、entity ごとに算出) に合わせる。
    index 0 = embedding 出力、index l+1 = layers[l] の出力。
    """
    s0, s1 = s_span["tok_start"], s_span["tok_end"]
    u0, u1 = t_span["tok_start"], t_span["tok_end"]
    ks = []
    for hs, ht in zip(src_hidden_states, tgt_hidden_states):
        a = span_vector(hs, s0, s1)
        b = span_vector(ht, u0, u1)
        ks.append(float(F.cosine_similarity(a, b, dim=0)))
    return ks


# --------------------------------------------------------------------------
# prefill 限定 splice hook
# --------------------------------------------------------------------------
class PrefillSplice:
    """model.model.layers[l] の出力に、prefill のときだけ介入する forward hook.

    **prefill 判定** (ここを間違えると KV cache と不整合になる):
      1. h.shape[1] <= 1                -> decode ステップ。何もしない。
         (KV cache 有効な自己回帰生成では 2 回目以降の forward は必ず seq_len == 1)
      2. prompt_len を渡してあるときは h.shape[1] != prompt_len も見送る。
         chunked prefill や beam 展開で系列長が変わった場合に、意図しない行を
         潰さないための保険。見送った形状は skipped_shapes に記録する。
      3. 既に 1 回発火していたら二度と発火しない (fired flag)。
    生成後に assert_fired() を呼び、1 回も発火していなければ例外にする
    (splice されていない音声を INJ として保存しないため)。

    **置換の中身**:
      m = 元テキストの実体語トークン数、n = 目標テキストのそれ。
      m == n : 行対行でそのまま置換。
      m != n : 目標スパンをトークン方向に平均した 1 本のベクトルを m 行に broadcast。
      いずれも **系列長は変えない** (T のまま)。decoder-only では T を変えると
      後続の位置 (RoPE) と、layers[0..l] が既に書いた KV cache がずれるため。
    """

    def __init__(self, layer_module, s_span, tgt_hidden, prompt_len=None, strict=True):
        self.layer_module = layer_module
        self.s0 = int(s_span["tok_start"])
        self.s1 = int(s_span["tok_end"])
        self.tgt = tgt_hidden.detach()            # (n, hidden)
        self.prompt_len = int(prompt_len) if prompt_len is not None else None
        self.strict = strict
        self.handle = None
        self.fired = 0
        self.skipped_shapes = []
        self.decode_calls = 0
        self.m = self.s1 - self.s0
        self.n = int(self.tgt.shape[0])
        self.equal_len = bool(self.m == self.n)
        if self.m <= 0:
            raise ValueError(f"empty source span: [{self.s0}, {self.s1})")
        if self.n <= 0:
            raise ValueError("empty target hidden span")

    def _replacement(self, dtype):
        if self.equal_len:
            return self.tgt.to(dtype)
        return self.tgt.mean(dim=0, keepdim=True).expand(self.m, -1).to(dtype)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            return output
        T = h.shape[1]
        if T <= 1:
            self.decode_calls += 1
            return output
        if self.prompt_len is not None and T != self.prompt_len:
            self.skipped_shapes.append(int(T))
            return output
        if self.fired:
            self.skipped_shapes.append(int(T))
            return output
        if self.s1 > T:
            raise RuntimeError(f"entity span [{self.s0},{self.s1}) が prefill 長 {T} を超えている")
        if h.shape[0] != 1:
            raise RuntimeError(f"batch size {h.shape[0]} は未対応 (1 のみ)")
        h_mod = h.clone()
        h_mod[0, self.s0:self.s1] = self._replacement(h.dtype).to(h.device)
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

    def assert_fired(self):
        if self.fired != 1:
            raise RuntimeError(
                f"splice hook の発火回数が {self.fired} (1 であるべき)。"
                f"prompt_len={self.prompt_len} skipped_shapes={self.skipped_shapes[:8]}")

    def stats(self):
        return {"fired": self.fired, "decode_calls": self.decode_calls,
                "skipped_shapes": self.skipped_shapes[:16],
                "m_src_entity_tokens": self.m, "n_tgt_entity_tokens": self.n,
                "equal_len": self.equal_len}


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@dataclass
class GenConfig:
    seed: int = SEED
    do_sample: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def as_dict(self):
        return {"seed": self.seed, "do_sample": self.do_sample,
                "temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "max_new_tokens": self.max_new_tokens}


def generate_speech_ids(bundle, input_ids, gen=GenConfig(), splice=None):
    """prefill -> speech token 自己回帰生成。splice があれば prefill 中だけ介入する.

    seed は generate の直前に必ずリセットする。BL / TGT / INJ で
    input_ids の長さも復号設定も同じなので、差は splice だけになる。
    """
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=bundle.device)
    attn = torch.ones_like(ids)
    kw = dict(
        max_new_tokens=gen.max_new_tokens,
        eos_token_id=bundle.speech_end_id,
        pad_token_id=(bundle.tokenizer.pad_token_id
                      if bundle.tokenizer.pad_token_id is not None else bundle.speech_end_id),
        do_sample=gen.do_sample,
        num_beams=1,
    )
    if gen.do_sample:
        kw.update(temperature=gen.temperature, top_p=gen.top_p, top_k=gen.top_k)
    seed_all(gen.seed)
    ctx = splice if splice is not None else _null_ctx()
    with torch.no_grad(), ctx:
        out = bundle.model.generate(input_ids=ids, attention_mask=attn, **kw)
    if splice is not None:
        splice.assert_fired()
    new_ids = out[0][ids.shape[1]:].tolist()
    if new_ids and new_ids[-1] == bundle.speech_end_id:
        new_ids = new_ids[:-1]
    toks = bundle.tokenizer.convert_ids_to_tokens(new_ids)
    speech_ids, n_foreign = extract_speech_ids(toks)
    return {"speech_ids": speech_ids, "n_generated": len(new_ids),
            "n_foreign_tokens": n_foreign,
            "hit_max_new_tokens": bool(len(new_ids) >= gen.max_new_tokens)}


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def synthesize(bundle, text, ref=None, gen=GenConfig()):
    """素の TTS。BL = TTS(text_a), TGT = TTS(text_b) はどちらもこれ。

    ref: {"ref_text": str, "speech_ids": [int]} または None。
    """
    ref_ids = ref["speech_ids"] if ref else None
    input_text = build_input_text(text, ref["ref_text"] if ref else None)
    ctx = prepare_context(bundle, input_text, ref_speech_ids=ref_ids)
    g = generate_speech_ids(bundle, ctx["input_ids"], gen=gen)
    audio = decode_speech_ids(bundle, g["speech_ids"])
    return {"audio": audio, "sampling_rate": bundle.sampling_rate,
            "prompt_len": ctx["prompt_len"], "n_speech_tokens": len(g["speech_ids"]),
            "n_generated": g["n_generated"], "n_foreign_tokens": g["n_foreign_tokens"],
            "hit_max_new_tokens": g["hit_max_new_tokens"]}


def splice_synthesize(bundle, text_a, word_a, text_b, word_b, layer, ref=None,
                      gen=GenConfig(), collect_kappa=True, strict_span=True,
                      src_ctx=None, tgt_ctx=None, span_extension=False):
    """SpeechSplice 本体.

    1. 目標文脈 (text_b) の prefill を 1 回回し、layers[layer] の出力
       (= hidden_states[layer+1]) の実体語スパンを取り出す。
    2. 元文脈 (text_a) で generate する。prefill のときだけ hook が発火し、
       元テキストの実体語トークン行をその目標ベクトルで置き換える。
    3. 生成された speech token を XCodec2 で波形に戻す。

    重み更新なし。系列長も変えない。

    span_extension=True のときだけ、語そのままで往復一致しない対に最小対称拡張を
    試す (既定 False = 従来どおりその対で ValueError)。src_ctx / tgt_ctx を渡した
    ときはそちらのスパンをそのまま使う (呼び出し側が prepare_context_pair で
    作っている場合はそこで拡張済み)。
    """
    ref_ids = ref["speech_ids"] if ref else None
    ref_text = ref["ref_text"] if ref else None
    in_a = build_input_text(text_a, ref_text)
    in_b = build_input_text(text_b, ref_text)

    span_ext = None
    if src_ctx is None and tgt_ctx is None:
        # span_extension=False なら prepare_context 2 回と完全に同じ。
        src, tgt, span_ext = prepare_context_pair(
            bundle, in_a, word_a, in_b, word_b, ref_speech_ids=ref_ids,
            region_text_a=text_a, region_text_b=text_b,
            need_hidden_a=collect_kappa, need_hidden_b=True,
            strict_span=strict_span, span_extension=span_extension)
    else:
        src = src_ctx if src_ctx is not None else prepare_context(
            bundle, in_a, ref_speech_ids=ref_ids, word=word_a, region_text=text_a,
            need_hidden=collect_kappa, strict_span=strict_span)
        tgt = tgt_ctx if tgt_ctx is not None else prepare_context(
            bundle, in_b, ref_speech_ids=ref_ids, word=word_b, region_text=text_b,
            need_hidden=True, strict_span=strict_span)

    li = hidden_index_for_layer(layer)
    if li >= len(tgt["hidden_states"]):
        raise ValueError(f"layer {layer} は範囲外 (層数 {len(tgt['hidden_states']) - 1})")
    t0, t1 = tgt["span"]["tok_start"], tgt["span"]["tok_end"]
    tgt_hidden = tgt["hidden_states"][li][0, t0:t1].clone()

    layer_module = bundle.model.model.layers[layer]
    splice = PrefillSplice(layer_module, src["span"], tgt_hidden,
                           prompt_len=src["prompt_len"])
    g = generate_speech_ids(bundle, src["input_ids"], gen=gen, splice=splice)
    audio = decode_speech_ids(bundle, g["speech_ids"])

    out = {
        "audio": audio, "sampling_rate": bundle.sampling_rate,
        "layer": int(layer), "hidden_index": int(li),
        "prompt_len": src["prompt_len"], "tgt_prompt_len": tgt["prompt_len"],
        "s_span": [src["span"]["tok_start"], src["span"]["tok_end"]],
        "t_span": [tgt["span"]["tok_start"], tgt["span"]["tok_end"]],
        "s_tokens": src["span"]["entity_tokens"], "t_tokens": tgt["span"]["entity_tokens"],
        "n_speech_tokens": len(g["speech_ids"]), "n_generated": g["n_generated"],
        "n_foreign_tokens": g["n_foreign_tokens"],
        "hit_max_new_tokens": g["hit_max_new_tokens"],
        "splice_mode": "token_wise" if splice.equal_len else "mean_pool_broadcast",
    }
    out.update(splice.stats())
    if span_ext is None:
        span_ext = src["span"].get("span_extension")
    if span_ext is not None:
        out["span_extension"] = span_ext
    if collect_kappa and src.get("hidden_states") is not None:
        ks = kappa_by_layer(src["hidden_states"], tgt["hidden_states"],
                            src["span"], tgt["span"])
        out["kappa_by_layer"] = [round(v, 6) for v in ks]
        out["kappa_at_splice"] = round(ks[li], 6)
        out["kappa_last_layer"] = round(ks[-1], 6)
    return out
