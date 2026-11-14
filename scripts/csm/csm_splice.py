#!/usr/bin/env python3
"""CSM-1B (sesame/csm-1b) 用 SpeechSplice コア実装.

CSM-1B は decoder-only の TTS。構造は

    backbone       CsmBackboneModel          16 層 / hidden 2048 (LLaMA 系)
    depth_decoder  CsmDepthDecoderForCausalLM 4 層 / hidden 1024
    codec          MimiModel                  24 kHz / 12.5 Hz / 32 codebook

テキストは `<|begin_of_text|>[話者ID]本文<|end_of_text|>` という 1 本の prefix として
prefill され、その続きとして音声フレームを自己回帰生成する。1 フレームにつき
backbone が codebook 0 を 1 個予測し、depth_decoder が残り 31 個を
backbone の最終 hidden state から予測する。cross-attention も text encoder も無い。

したがって差し替え地点は Qwen3-TTS talker / Llasa-3B / XTTS-v2 と同じ
    model.backbone_model.layers[l] の **出力** の、prefill 系列の実体語トークン行
になる。重みは一切更新しない。推論時のみ。BL と INJ は prompt が 1 トークンも
違わず、seed も復号設定も共有する。差は「prefill 中に実体語行を目標文脈の
同一層 hidden state で置換したか」だけ。

方針の出所 (推測ではなく既存実装に合わせてある):
  * prefill 限定 hook の 3 段ガード   : scripts/llasa/llasa_splice.py:745 PrefillSplice
  * 長さ不一致時の mean-pool broadcast: 同上 (m != n なら目標スパンを平均して m 行に展開)
  * kappa の定義                      : 同上 kappa_by_layer
                                        (スパン内 hidden state をトークン方向に単純平均した
                                         ベクトル同士の cosine、entity ごと)
  * offset_mapping によるスパン特定   : scripts/projection/eval_qwen3_injection.py の
                                        2026-09-17 修正。接頭辞のトークン数を数える旧方式は
                                        英語の文中で start > end の反転を起こす。
  * HF_HOME の代入上書き              : scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py:17-22

層格子 (16 層):
    既存の写像 round(l * N / 28) を half-up で 16 層に写す。
        0 -> 0,  7 -> 4,  14 -> 8,  21 -> 12,  26 -> 14.86
    最深点は既存 3 例の慣例どおり L-2 = 14 を採る (深さ比 0.875)。
    28 層 0/7/14/21/26、30 層 (XTTS) 0/8/15/23/28 と同じ規則。
    layer 15 を使わないのは、直後に self.norm があり hidden_states[-1] が
    すでに norm 適用後で、他の格子点と意味が揃わないため。

CSM 固有の落とし穴 (CPU 実測で確認済み。詳細は scripts/csm/README.md):
  (1) CsmDecoderLayer.forward は **tuple ではなく素の Tensor** を返す
      (modeling_csm.py:336 の戻り値注釈が torch.Tensor)。tuple 前提で out[0] と
      書くと (T, H) を掴み、多くの実装の `if h.dim() < 3: return` に引っかかって
      **例外も警告も出さずに素通し**する。ここでは 3 次元でなければ raise する。
  (2) CsmBackboneModel.forward には @check_model_inputs() が付いており、
      output_hidden_states が真のとき各層の forward を monkey patch する。
      _sample は毎ステップ output_hidden_states=True を強制するので生成中は常に
      掛かっている。forward hook は forward の **後** に走るので書き換えは効くが、
      outputs.hidden_states[l+1] に記録されるのは **書き換え前**の値。
      発火確認に hidden_states を使ってはいけない (fired カウンタで確認する)。
  (3) hidden_states[-1] は final RMSNorm 適用後 (tie_last_hidden_states=True)。
      len(hidden_states) == 17 で、0 = embedding 出力、l+1 = layers[l] の出力
      (l = 0..14)。index 16 だけは layers[15] の生出力ではない。
  (4) 音声プロンプトを付けると processor が <|AUDIO|> をフレーム数ぶんに展開するので、
      会話全体を rendered した文字列の offset_mapping は使えない。
      **最後の BOS を起点に最終メッセージだけを tokenize** してスパンを取る。
  (5) 参照音声はパス文字列で渡さない。load_audio の既定サンプリングレートは 16000
      (CSM は 24000) で、黙って 1.5 倍に引き伸ばされる。自分で 24 kHz に読み直す。
  (6) processor の text_kwargs 既定は padding=True / padding_side="left"。
      batch を 2 以上にすると行番号が全部ずれる。batch = 1 固定。
  (7) StoppingCriteriaList を渡して使い回すと `criterion.max_length -= cur_len` で
      壊れ、2 回目から 1 フレームしか出ない。stopping_criteria は渡さない。
  (8) generation_config に eos_token_id を足さない。足すと codebook_pad_token_id=2050 が
      書き込まれ、Mimi の codebook_size=2048 の範囲外で IndexError になる。
  (9) リポジトリには重みが 2 系統ある。transformers が読むのは config.json の
      "transformers_weights" が指す transformers-*.safetensors (7.13 GB) の方。
      本家形式の model.safetensors / ckpt.pt (計 12.4 GB) を掴むと key が 1 つも
      一致せず、全パラメータ乱数のまま warning だけで推論に入る。
      load_csm は output_loading_info=True で missing_keys を検査する。

ライセンス: sesame/csm-1b は **Apache-2.0** (gated=auto)。codec は Mimi (kyutai/mimi,
CC-BY-4.0) で、その重みは CSM の checkpoint に同梱されている (codec_model.* 350 key)。
言語タグは en のみ。

CPU 検証:
  taskset -c 0-63 python3 scripts/csm/smoke_test_csm_splice.py
"""
import math
import os
import re
from dataclasses import dataclass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

DEFAULT_MODEL_ID = "sesame/csm-1b"
SEED = 42

EXPECTED_NUM_LAYERS = 16
EXPECTED_NUM_CODEBOOKS = 32
EXPECTED_SAMPLING_RATE = 24000
EXPECTED_FRAME_RATE = 12.5
EXPECTED_HIDDEN_SIZE = 2048

# 既存 3 例と同じ深さ比を 16 層に写した格子。最深点は L-2。
SWEEP_LAYERS = (0, 4, 8, 12, 14)
DEFAULT_LAYER = 14

# 公式 generation_config.json の既定。BL / TGT / INJ で完全に共有する。
DEFAULT_TEMPERATURE = 0.9
DEFAULT_DEPTH_TEMPERATURE = 0.9
DEFAULT_DEPTH_TOP_K = 50
# 既定の 125 フレーム = 10.0 秒。EN の 1 文は 3-4 秒なので 256 (20.5 秒) で十分余裕がある。
DEFAULT_MAX_NEW_TOKENS = 256

DEFAULT_SPEAKER = "0"

BOS_TOKEN_ID = 128000          # <|begin_of_text|>
EOS_TOKEN_ID = 128001          # <|end_of_text|>
AUDIO_TOKEN_ID = 128002        # <|AUDIO|>
AUDIO_EOS_TOKEN_ID = 128003    # <|audio_eos|>
CODEBOOK_EOS_TOKEN_ID = 0

# transformers が読むべき重みファイル (config.json の transformers_weights)。
EXPECTED_WEIGHTS_INDEX = "transformers.safetensors.index.json"
# 本家形式 (ckpt.pt / model.safetensors) を落とさないための取得パターン。
DOWNLOAD_ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "README.md",
    "transformers.safetensors.index.json", "transformers-*.safetensors",
]


# --------------------------------------------------------------------------
# HF cache: slurm_run.sh は -e HF_HOME=/tmp/.cache/huggingface を渡すが、その実体は
# 空で書き込めず PermissionError になる。重みが置いてあるのは
# <PROJECT_ROOT>/.cache/huggingface (コンテナ内 /mnt/.cache/huggingface)。
# setdefault ではコンテナ側の環境変数に負けるので **代入で上書き**する。
# scripts/llasa/llasa_splice.py:86-97 と同じ扱い。transformers import 前に済ませる。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


def setup_hf_home(hf_home=None):
    hf_home = hf_home or os.environ.get("CSM_HF_HOME") or DEFAULT_HF_HOME
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_home, "modules")
    for p in ("/tmp/.cache/huggingface/token",
              os.path.join(hf_home, "token"),
              os.path.expanduser("~/.cache/huggingface/token")):
        if os.path.exists(p):
            try:
                with open(p) as f:
                    tok = f.read().strip()
                if tok:
                    os.environ.setdefault("HF_TOKEN", tok)
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
                break
            except Exception:
                pass
    return hf_home


setup_hf_home()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def seed_all(seed=SEED):
    """generate の直前に必ず呼ぶ.

    CSM は backbone のサンプリングと depth decoder のサンプリングが同じ global RNG を
    共有し、既定で do_sample=True / temperature=0.9。seed を固定しないと BL / TGT / INJ
    の差が splice なのかサンプリングの揺れなのか分からなくなる。
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


# --------------------------------------------------------------------------
# 層格子
# --------------------------------------------------------------------------
def _round_half_up(x):
    return int(math.floor(float(x) + 0.5))


def layer_grid(num_layers, base_layers=(0, 7, 14, 21), base_total=28):
    """28 層で使った格子 0/7/14/21/26 を num_layers 層へ同じ深さ比で写す.

    浅い 4 点は round(l * N / 28) (half-up)、最深点は既存 3 例の慣例どおり N-2。
        28 層 -> (0, 7, 14, 21, 26)   Qwen3-TTS talker / Llasa-3B  (実績どおり)
        30 層 -> (0, 8, 15, 23, 28)   XTTS-v2                      (実績どおり)
        16 層 -> (0, 4, 8, 12, 14)    CSM-1B                       (本ファイル)
    """
    n = int(num_layers)
    if n < 4:
        raise ValueError(f"num_layers={n} は小さすぎる")
    pts = [_round_half_up(l * n / float(base_total)) for l in base_layers]
    pts.append(n - 2)
    out = []
    for p in pts:
        p = max(0, min(n - 1, int(p)))
        if p not in out:
            out.append(p)
    return tuple(sorted(out))


def hidden_index_for_layer(layer):
    """layers[l] の出力は output_hidden_states[l+1].

    index 0 = embedding 出力。index 16 (= 最終層の次) だけは final RMSNorm 適用後の
    last_hidden_state に差し替わるので、layers[15] の生出力ではない。
    格子の最深点を 14 に採っているので、本実装の経路ではここに触れない。
    """
    return int(layer) + 1


# --------------------------------------------------------------------------
# snapshot / download
# --------------------------------------------------------------------------
def local_snapshot(hub_id, required=True):
    """HF_HUB_CACHE 内のスナップショットディレクトリを返す (offline 前提).

    ダウンロードは呼び出し側が明示的に許可したときだけ (allow_download)。
    事故で 19.6 GB 落とさないための既定。
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


def snapshot_has_weights(path):
    if path is None or not os.path.isdir(path):
        return False
    return any(f.startswith("transformers-") and f.endswith(".safetensors")
               for f in os.listdir(path))


def ensure_csm_snapshot(model_id=DEFAULT_MODEL_ID, allow_download=False):
    """重みまで揃ったローカル snapshot のパスを返す.

    allow_patterns を必ず付ける。付けないと本家形式の ckpt.pt (6.22 GB) と
    model.safetensors (6.21 GB) まで落ちて計 19.6 GB になる。transformers が
    読むのは transformers-*.safetensors の 7.13 GB だけ。
    """
    p = local_snapshot(model_id, required=False)
    if snapshot_has_weights(p):
        return p
    if not allow_download:
        raise FileNotFoundError(
            f"{model_id} の重みがローカルに無い ({p})。\n"
            f"  GPU ジョブ内で --allow_download を付けて取得すること "
            f"(transformers 経路で 7.13 GB)。")
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=DOWNLOAD_ALLOW_PATTERNS)


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
@dataclass
class CsmBundle:
    model: object
    processor: object
    tokenizer: object
    config: object
    device: str
    dtype: object
    num_layers: int
    sampling_rate: int
    frame_rate: float
    num_codebooks: int
    source: str

    @property
    def layers(self):
        return self.model.backbone_model.layers

    def layer_module(self, layer):
        n = len(self.layers)
        if not (0 <= int(layer) < n):
            raise ValueError(f"layer {layer} は範囲外 (0..{n - 1})")
        return self.layers[int(layer)]


def load_csm(model_id=DEFAULT_MODEL_ID, device=None, dtype=torch.bfloat16,
             allow_download=False, attn_implementation="sdpa",
             strict_weights=True):
    """CSM-1B を読み込む.

    strict_weights=True のとき、
      * config.transformers_weights が transformers.safetensors.index.json であること
      * from_pretrained の missing_keys が空であること
    を確かめる。ここを見ないと、本家形式の重みを掴んで **全パラメータ乱数のまま**
    warning だけで推論に入る事故 (Llasa の act.beta / self.bias と同型) を見逃す。
    """
    from transformers import AutoProcessor, CsmForConditionalGeneration

    src = ensure_csm_snapshot(model_id, allow_download=allow_download)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and dtype in (torch.bfloat16, torch.float16):
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(src)
    model, info = CsmForConditionalGeneration.from_pretrained(
        src, dtype=dtype, attn_implementation=attn_implementation,
        output_loading_info=True)
    model = model.to(device).eval()

    missing = [k for k in info.get("missing_keys", [])]
    if missing:
        msg = (f"from_pretrained の missing_keys が {len(missing)} 個ある。"
               f"乱数初期化のまま推論に入る危険がある: {missing[:8]}")
        if strict_weights:
            raise RuntimeError(msg)
        print(f"[warn] {msg}", flush=True)

    cfg = model.config
    tw = getattr(cfg, "transformers_weights", None)
    if strict_weights and tw != EXPECTED_WEIGHTS_INDEX:
        raise RuntimeError(
            f"config.transformers_weights が {tw!r} (期待 {EXPECTED_WEIGHTS_INDEX!r})。"
            f"本家形式の model.safetensors を掴むと key が 1 つも一致しない。")

    n_layers = len(model.backbone_model.layers)
    if n_layers != EXPECTED_NUM_LAYERS:
        raise RuntimeError(f"backbone の層数が {n_layers} (期待 {EXPECTED_NUM_LAYERS})")
    codec_cfg = cfg.codec_config
    sr = int(getattr(codec_cfg, "sampling_rate", EXPECTED_SAMPLING_RATE))
    fr = float(getattr(codec_cfg, "frame_rate", EXPECTED_FRAME_RATE))
    nc = int(getattr(cfg, "num_codebooks", EXPECTED_NUM_CODEBOOKS))
    if sr != EXPECTED_SAMPLING_RATE:
        raise RuntimeError(f"codec sampling_rate が {sr} (期待 {EXPECTED_SAMPLING_RATE})")

    return CsmBundle(model=model, processor=processor, tokenizer=processor.tokenizer,
                     config=cfg, device=device, dtype=dtype, num_layers=n_layers,
                     sampling_rate=sr, frame_rate=fr, num_codebooks=nc, source=src)


def assert_splice_target(bundle, layer):
    """差し替え対象が backbone の層そのものであることを型と identity で確かめる.

    CsmDecoderLayer は depth_decoder (4 層) でも同じクラスが使われている。
    depth decoder の prefill は seq_len == 2 で、「seq_len > 1 なら prefill」という
    判定はそちらでも成立してしまう。名前検索やクラス名で層を探すと
    depth decoder (と、名前検索なら Mimi の 8 層も) を掴む。必ず
    model.backbone_model.layers[l] を直接指定する。
    """
    from transformers.models.csm.modeling_csm import CsmDecoderLayer
    m = bundle.layer_module(layer)
    if not isinstance(m, CsmDecoderLayer):
        raise RuntimeError(f"layers[{layer}] が CsmDecoderLayer ではない: {type(m)}")
    if m is not bundle.model.backbone_model.layers[int(layer)]:
        raise RuntimeError("差し替え対象が backbone_model.layers[l] と同一オブジェクトでない")
    return m


# --------------------------------------------------------------------------
# 参照音声
# --------------------------------------------------------------------------
def load_ref_audio(wav_path, target_sr=EXPECTED_SAMPLING_RATE, max_sec=None):
    """参照 wav を **24 kHz の np.float32 1-D** として読む.

    processor は渡された配列の実際のサンプリングレートを一度も見ない
    (CsmProcessorKwargs の sampling_rate は feature extractor 自身の値と
    突き合わせるだけ)。44.1 kHz の配列をそのまま渡すと例外も出さずに
    1.84 倍に引き伸ばされた参照音声として扱われる (実測: フレーム 41 -> 75)。
    既存の参照 eval_results/_natural_refs_norm/*.wav は 44.1 kHz なので必ず変換が要る。
    """
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
        x = torchaudio.functional.resample(t, sr, target_sr).numpy()
    if max_sec is not None:
        x = x[: int(max_sec * target_sr)]
    return np.ascontiguousarray(x.astype(np.float32)), int(target_sr), int(sr)


def expected_audio_frames(bundle, n_samples):
    """processor が <|AUDIO|> を何個に展開するかを先に計算する (検算用).

    秒数 x 12.5 では求まらない (畳み込みの段数計算なので)。
    実測: 3.25 s (78000 sample, 24 kHz) -> 41 フレーム。
    """
    from transformers.models.csm.processing_csm import CsmProcessorKwargs
    kw = CsmProcessorKwargs._defaults["audio_kwargs"]["encoded_length_kwargs"]
    return int(bundle.processor._get_encoded_length(int(n_samples), **kw))


@dataclass
class CsmRef:
    """参照音声ターン。BL / TGT / INJ の全対で同じものを使うこと.

    CSM は文脈が無いと話者が毎回変わる (モデルカード自身が "sounds best when
    provided with context" と書いている)。BL / TGT / INJ で声が変われば
    音素誤り率の比較に話者要因が混ざる。さらに採点器 phoneme_per_v3.py は
    BL と TGT の全体音素列の差分で実体語区間を切り出すので、声が違うと
    差分区間が文全体に広がり、除外されないまま意味のない値が出る。
    """
    text: str
    audio: object            # np.float32 1-D @ 24 kHz
    path: str = ""
    orig_sr: int = 0
    n_samples: int = 0
    n_frames: int = 0
    speaker: str = DEFAULT_SPEAKER


def build_ref(bundle, wav_path, ref_text, speaker=DEFAULT_SPEAKER, max_sec=None):
    arr, sr, orig_sr = load_ref_audio(wav_path, target_sr=bundle.sampling_rate, max_sec=max_sec)
    return CsmRef(text=ref_text, audio=arr, path=wav_path, orig_sr=orig_sr,
                  n_samples=int(arr.shape[0]),
                  n_frames=expected_audio_frames(bundle, arr.shape[0]),
                  speaker=str(speaker))


# --------------------------------------------------------------------------
# prompt 構築
# --------------------------------------------------------------------------
def render_message(tokenizer, text, speaker=DEFAULT_SPEAKER):
    """1 メッセージ分の rendered 文字列 (音声トークンを含まない).

    chat_template.jinja は各メッセージについて
        {bos_token}[{role}]{text}{eos_token}   (+ 音声があれば <|AUDIO|><|audio_eos|>)
    を並べるだけ。role は「整数の文字列」でなければ raise_exception される。
    """
    s = str(speaker)
    if not s.isdigit():
        raise ValueError(f"speaker は整数の文字列でなければならない: {speaker!r}")
    return f"{tokenizer.bos_token}[{s}]{text}{tokenizer.eos_token}"


def build_conversation(text, ref=None, speaker=DEFAULT_SPEAKER):
    """chat_template に渡す会話を組む.

    ref があれば参照音声ターンを 1 本前置する。前置しても最終メッセージの
    トークン列は 1 バイトも変わらない (実測: input_ids[last_bos:] がバイト一致)。
    """
    conv = []
    if ref is not None:
        conv.append({"role": str(ref.speaker),
                     "content": [{"type": "text", "text": ref.text},
                                 {"type": "audio", "path": ref.audio}]})
    conv.append({"role": str(speaker), "content": [{"type": "text", "text": text}]})
    return conv


def prepare_inputs(bundle, text, ref=None, speaker=DEFAULT_SPEAKER):
    """processor を通して backbone に渡す入力を作る (batch = 1 固定).

    processor の text_kwargs 既定は padding=True / padding_side="left" なので、
    batch を 2 以上にすると行番号が左詰めで全部ずれる。ここでは 1 件しか渡さないが、
    念のため attention_mask が全部 1 であることを確かめる。
    """
    conv = build_conversation(text, ref=ref, speaker=speaker)
    inputs = bundle.processor.apply_chat_template(conv, tokenize=True, return_dict=True)
    ids = inputs["input_ids"]
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise RuntimeError(f"input_ids の形が {tuple(ids.shape)} (batch 1 のみ対応)")
    am = inputs.get("attention_mask")
    if am is not None and int(am.sum()) != int(ids.shape[1]):
        raise RuntimeError(
            f"attention_mask に padding がある (sum={int(am.sum())} != T={int(ids.shape[1])})。"
            f"左詰めで行番号がずれるので batch=1 で使うこと。")
    if ref is not None:
        n_audio = int((ids[0] == AUDIO_TOKEN_ID).sum())
        if n_audio != ref.n_frames:
            raise RuntimeError(
                f"音声トークン数が {n_audio} で期待値 {ref.n_frames} と違う。"
                f"参照音声のサンプリングレートを疑うこと (24 kHz でなければ黙ってずれる)。")
    return {k: (v.to(bundle.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


# --------------------------------------------------------------------------
# 実体語 token span (offset_mapping 方式)
# --------------------------------------------------------------------------
def _entity_char_span(body, word):
    """本文の中の実体語の文字スパン [c0, c1) を返す (語境界つき・一意)."""
    w = word.strip()
    if not w:
        raise ValueError(f"empty entity word: {word!r}")
    hits = list(re.finditer(r"(?<![0-9A-Za-z])" + re.escape(w) + r"(?![0-9A-Za-z])", body))
    if not hits:
        raise ValueError(f"entity {w!r} not found in body {body!r}")
    if len(hits) > 1:
        raise ValueError(f"entity {w!r} occurs {len(hits)} times in body {body!r}")
    return hits[0].span()


def entity_token_span(bundle, text, word, input_ids=None, ref=None,
                      speaker=DEFAULT_SPEAKER):
    """input_ids の中で実体語が占める **トークン** スパン [tok_start, tok_end) を返す.

    backbone の系列は input_ids と 1 対 1 (_merge_input_ids_with_input_values は
    embed_text_tokens(input_ids) を作って音声位置だけ差し替えるので長さ不変)。
    よって **token index = hidden state の行番号**で、text_body_offset のような
    補正は要らない。

    手順 (音声プロンプトの有無に関わらずこの 1 本でよい):
      1. input_ids の中の **最後の BOS** の位置を起点にする。
      2. 最終メッセージだけを rendered して tokenize し、ids[bos_pos:] と一致することを確かめる。
         (会話全体を rendered した文字列は <|AUDIO|> がフレーム数ぶんに展開されるため使えない。
          実測: rendered 31 トークン vs 実 input_ids 55 トークン)
      3. 本文領域に限って実体語の文字スパンを取り、offset_mapping と交差するトークンを拾う。
      4. 絶対スパン = bos_pos + 局所スパン。decode して実体語に戻ることを毎回確かめる。

    **接頭辞のトークン数を数える旧方式は使わない。** prompt[:start] が末尾の空白で
    終わると空白が単独トークンになり開始側だけ 1 多く数えられる。英語 600 文の実測で
    165 件が start > end の反転、300 件が offset_mapping と不一致だった。
    """
    tok = bundle.tokenizer
    if input_ids is None:
        input_ids = prepare_inputs(bundle, text, ref=ref, speaker=speaker)["input_ids"]
    ids = input_ids[0].tolist() if torch.is_tensor(input_ids) and input_ids.dim() == 2 \
        else list(input_ids)

    bos_positions = [i for i, t in enumerate(ids) if t == BOS_TOKEN_ID]
    if not bos_positions:
        raise RuntimeError("input_ids に BOS (128000) が無い")
    bos_pos = bos_positions[-1]

    rendered = render_message(tok, text, speaker=speaker)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    local_ids = list(enc["input_ids"])
    if local_ids != ids[bos_pos:]:
        raise RuntimeError(
            "最終メッセージの tokenize が input_ids[last_bos:] と一致しない "
            f"(len {len(local_ids)} vs {len(ids) - bos_pos})。スパン特定が使えないので中断する。")

    # 本文領域 (特殊トークン文字列の中の同綴りを拾わないため)
    n_body = rendered.count(text)
    if n_body != 1:
        raise ValueError(f"本文が rendered に {n_body} 回出現する (1 回であるべき): {text!r}")
    b0 = rendered.index(text)
    a, b = _entity_char_span(text, word)
    c0, c1 = b0 + a, b0 + b

    offs = [(int(x), int(y)) for x, y in enc["offset_mapping"]]
    sel = [i for i, (x, y) in enumerate(offs) if y > x and x < c1 and y > c0]
    if not sel:
        raise ValueError(f"no token overlaps entity char span {(c0, c1)}")
    if sel != list(range(sel[0], sel[-1] + 1)):
        raise ValueError(f"entity token span is not contiguous: {sel}")
    l0, l1 = sel[0], sel[-1] + 1
    covered = rendered[offs[l0][0]:offs[l1 - 1][1]]
    tok_start, tok_end = bos_pos + l0, bos_pos + l1
    if tok_end > len(ids):
        raise RuntimeError(f"span [{tok_start},{tok_end}) が系列長 {len(ids)} を超えている")
    decoded = tok.decode(ids[tok_start:tok_end])
    return {
        "word": word,
        "text": text,
        "speaker": str(speaker),
        "bos_pos": int(bos_pos),
        "char_span": [int(c0), int(c1)],
        "local_span": [int(l0), int(l1)],
        "tok_start": int(tok_start),
        "tok_end": int(tok_end),
        "n_tokens_entity": int(tok_end - tok_start),
        "n_tokens_total": int(len(ids)),
        "covered_text": covered,
        "decoded_text": decoded,
        "roundtrip_ok": bool(covered.strip() == word.strip()
                             and decoded.strip() == word.strip()),
        "entity_tokens": tok.convert_ids_to_tokens(ids[tok_start:tok_end]),
    }


def verify_entity_span(span, strict=True):
    if not span["roundtrip_ok"]:
        msg = (f"span roundtrip failed: covered={span['covered_text']!r} "
               f"decoded={span['decoded_text']!r} != word={span['word']!r} "
               f"(text={span['text']!r})")
        if strict:
            raise ValueError(msg)
        return False, msg
    return True, ""


# --------------------------------------------------------------------------
# hidden states / kappa
# --------------------------------------------------------------------------
class LayerTap:
    """layers[l] の出力を読み取るだけの forward hook (書き換えない).

    hidden_states の index 規約 (l+1、最後だけ norm 適用後) に依存せずに
    「その層の生の出力」を取れる。目標ベクトルはこちらで取る。
    """

    def __init__(self, layer_module):
        self.layer_module = layer_module
        self.handle = None
        self.hidden = None
        self.calls = 0

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            raise RuntimeError(
                f"CsmDecoderLayer の出力が 3 次元の Tensor でない: "
                f"type={type(output)} shape={getattr(h, 'shape', None)}")
        self.calls += 1
        if self.hidden is None:
            self.hidden = h.detach().clone()
        return output

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


PREFILL_FORWARD_KEYS = ("input_ids", "attention_mask", "input_values",
                        "input_values_cutoffs")


def prefill_forward(bundle, inputs, layer=None, need_all_hidden=True):
    """prefill を 1 回だけ回す (cache 不使用、labels 無し = depth decoder は走らない).

    CsmForConditionalGeneration.forward を通す。input_ids が 2 次元なら
    _merge_input_ids_with_input_values が呼ばれ、音声プロンプトの有無に関わらず
    同じ 1 本の経路になる (長さは変わらないので token index = 行番号のまま)。
    labels=None なので depth decoder は走らず、Mimi は音声の符号化にだけ使われる。

    戻り値:
      hidden_states : 長さ 17 のタプル。0 = embedding 出力、l+1 = layers[l] の出力
                      (l = 0..14)。index 16 は final RMSNorm 適用後。
                      need_all_hidden=False なら None。
      tap_hidden    : layer を指定したときの layers[layer] の **生の出力** (1, T, H)。
                      hidden_states[layer+1] と一致するはずで、突き合わせて確かめる。
    """
    kw = {k: v for k, v in inputs.items() if k in PREFILL_FORWARD_KEYS}
    kw["use_cache"] = False
    kw["output_hidden_states"] = bool(need_all_hidden)
    kw["return_dict"] = True
    tap = LayerTap(assert_splice_target(bundle, layer)) if layer is not None else None
    ctx = tap if tap is not None else _null_ctx()
    with torch.no_grad(), ctx:
        out = bundle.model(**kw)
    hs = tuple(h.detach() for h in out.hidden_states) if need_all_hidden else None
    th = tap.hidden if tap is not None else None
    if layer is not None and th is None:
        raise RuntimeError(f"layers[{layer}] の tap が発火しなかった")
    return hs, th


def span_vector(hidden, tok_start, tok_end):
    """スパン内 hidden state をトークン方向に単純平均したベクトル."""
    return hidden[0, int(tok_start):int(tok_end)].float().mean(dim=0)


def kappa_by_layer(src_hidden_states, tgt_hidden_states, s_span, t_span):
    """kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}).

    既存の定義 (scripts/llasa/llasa_splice.py:kappa_by_layer) と同じ。
    index 0 = embedding 出力、index l+1 = layers[l] の出力。
    **index 16 だけは final RMSNorm 適用後**なので、他の index と意味が揃わない。
    格子の最深点を 14 に採っているので splice 地点はここに触れない。
    """
    s0, s1 = int(s_span["tok_start"]), int(s_span["tok_end"])
    u0, u1 = int(t_span["tok_start"]), int(t_span["tok_end"])
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
    """backbone_model.layers[l] の出力に、prefill のときだけ介入する forward hook.

    **prefill 判定** (ここを間違えると KV cache と不整合になる):
      1. h.shape[1] <= 1 -> decode ステップ。何もしない。
         CSM の _sample は 2 回目以降 input_ids = next_tokens[:, None, :] と
         **代入で上書き**するので decode の forward は必ず T == 1 (実測:
         音声なし [17,1,1,1,1,1]、音声あり [55,1,1,1])。
      2. prompt_len を渡してあるときは h.shape[1] != prompt_len も見送る。
      3. 既に 1 回発火していたら二度と発火しない (fired flag)。
    生成後に assert_fired() を呼び、ちょうど 1 回でなければ例外にする
    (splice されていない音声を INJ として保存しないため)。

    **置換の中身**:
      m = 元テキストの実体語トークン数、n = 目標テキストのそれ。
      m == n : 行対行でそのまま置換。
      m != n : 目標スパンをトークン方向に平均した 1 本を m 行に broadcast。
      いずれも **系列長は変えない** (T のまま)。decoder-only では T を変えると
      後続の位置 (RoPE) と、layers[0..l] が既に書いた KV cache がずれる。

    **CSM 固有**:
      * CsmDecoderLayer.forward は tuple ではなく素の Tensor を返す。
        3 次元の Tensor でなければ黙って素通しせず raise する
        (素通しすると INJ が BL とビット単位で同一になり、例外も警告も出ない)。
      * hook 対象が backbone_model.layers[l] そのものであることを identity で確かめる。
        depth_decoder も同じ CsmDecoderLayer クラスで、その prefill は seq_len == 2
        なので「seq_len > 1 なら prefill」はそちらでも成立してしまう。
      * 書き換えの確認に outputs.hidden_states[l+1] を使ってはいけない。
        check_model_inputs の monkey patch が hook の **前** の値を記録するため、
        書き換えが効いていても hidden_states[l+1] には映らない。
    """

    def __init__(self, layer_module, s_span, tgt_hidden, prompt_len=None,
                 expect_module=None):
        self.layer_module = layer_module
        self.expect_module = expect_module if expect_module is not None else layer_module
        self.s0 = int(s_span["tok_start"])
        self.s1 = int(s_span["tok_end"])
        self.tgt = tgt_hidden.detach()            # (n, hidden)
        self.prompt_len = int(prompt_len) if prompt_len is not None else None
        self.handle = None
        self.fired = 0
        self.skipped_shapes = []
        self.decode_calls = 0
        self.foreign_calls = 0
        self.m = self.s1 - self.s0
        self.n = int(self.tgt.shape[0])
        self.equal_len = bool(self.m == self.n)
        if self.m <= 0:
            raise ValueError(f"empty source span: [{self.s0}, {self.s1})")
        if self.n <= 0:
            raise ValueError("empty target hidden span")
        if self.tgt.dim() != 2:
            raise ValueError(f"tgt_hidden は (n, hidden) であるべき: {tuple(self.tgt.shape)}")

    def _replacement(self, dtype):
        if self.equal_len:
            return self.tgt.to(dtype)
        return self.tgt.mean(dim=0, keepdim=True).expand(self.m, -1).to(dtype)

    def __call__(self, module, args, output):
        if module is not self.expect_module:
            # 別の層 (depth decoder や Mimi) に掛かっている。絶対に書き換えない。
            self.foreign_calls += 1
            return output
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            raise RuntimeError(
                f"CsmDecoderLayer の出力が 3 次元の Tensor でない: "
                f"type={type(output)} shape={getattr(h, 'shape', None)}。"
                f"素通しすると INJ が BL と同一になるので中断する。")
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
                f"prompt_len={self.prompt_len} skipped_shapes={self.skipped_shapes[:8]} "
                f"decode_calls={self.decode_calls}")

    def stats(self):
        return {"fired": self.fired, "decode_calls": self.decode_calls,
                "skipped_shapes": self.skipped_shapes[:16],
                "foreign_calls": self.foreign_calls,
                "m_src_entity_tokens": self.m, "n_tgt_entity_tokens": self.n,
                "equal_len": self.equal_len}


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@dataclass
class GenConfig:
    seed: int = SEED
    do_sample: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    depth_temperature: float = DEFAULT_DEPTH_TEMPERATURE
    depth_top_k: int = DEFAULT_DEPTH_TOP_K
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    min_sec: float = 0.5
    # 上限到達 (EOS が出ないまま max_new_tokens まで走り切った) を error にするか。
    # 既定は False = 行に記録して警告を出すだけ。True にすると例外にする。
    fail_on_cap: bool = False

    def as_dict(self):
        return {"seed": self.seed, "do_sample": self.do_sample,
                "temperature": self.temperature,
                "depth_temperature": self.depth_temperature,
                "depth_top_k": self.depth_top_k,
                "max_new_tokens": self.max_new_tokens, "min_sec": self.min_sec,
                "fail_on_cap": self.fail_on_cap}

    def generate_kwargs(self):
        """公式 generation_config.json の既定を BL / TGT / INJ で完全に共有する.

        **stopping_criteria は渡さない**。_sample が criterion.max_length -= cur_len と
        その場で書き換えるので、同じリストを使い回すと 2 回目から 1 フレームしか出ない
        (実測: frames 6 -> 1 -> 1、警告 1 行のみ)。長さの制御は max_new_tokens だけで行う。

        **eos_token_id も渡さない**。渡すと has_eos_stopping_criteria が真になり、
        終了後のフレームに codebook_pad_token_id=2050 が書き込まれる。Mimi の
        codebook_size は 2048 なので decode で IndexError になる。
        """
        kw = {"max_new_tokens": int(self.max_new_tokens),
              "do_sample": bool(self.do_sample),
              "depth_decoder_do_sample": bool(self.do_sample)}
        if self.do_sample:
            kw["temperature"] = float(self.temperature)
            kw["depth_decoder_temperature"] = float(self.depth_temperature)
            kw["depth_decoder_top_k"] = int(self.depth_top_k)
        return kw


def codes_fingerprint(codes):
    """生成コード列の短い指紋.

    BL と INJ で同じ値なら splice が生成を 1 ビットも変えていない。
    transformers を上げて hook が効かなくなっても **例外は出ず INJ が BL と同じ音に
    なるだけ**なので、この指紋を毎回記録して黙った失敗を検出できるようにする。
    """
    import hashlib
    a = np.ascontiguousarray(codes.numpy().astype(np.int32))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def _generate(bundle, inputs, gen=GenConfig(), splice=None):
    """generate を 1 回回して波形とフレーム統計を返す.

    seed は generate の直前に必ずリセットする。
    """
    kw = dict(inputs)
    kw.update(gen.generate_kwargs())
    kw["output_audio"] = True
    kw["return_dict_in_generate"] = True
    seed_all(gen.seed)
    ctx = splice if splice is not None else _null_ctx()
    with torch.no_grad(), ctx:
        out = bundle.model.generate(**kw)
    if splice is not None:
        splice.assert_fired()

    codes = out.sequences                     # (1, n_frames, 32)
    if codes.dim() != 3 or codes.shape[0] != 1:
        raise RuntimeError(f"sequences の形が {tuple(codes.shape)}")
    n_frames = int(codes.shape[1])
    eos_idxs = (codes[0] == CODEBOOK_EOS_TOKEN_ID).all(dim=-1).nonzero()
    eos_found = bool(eos_idxs.numel() != 0)
    cutoff = int(eos_idxs.min()) if eos_found else n_frames

    audio_list = out.audio
    if not audio_list:
        raise RuntimeError("generate が音声を返さなかった")
    audio = audio_list[0].detach().float().cpu().numpy()
    n_samples = int(audio.shape[-1])
    sec = n_samples / float(bundle.sampling_rate)

    # ------------------------------------------------------------------
    # 生成長の検査 (2026-09-17 差し替え)
    #
    # 旧版は sec と cutoff/frame_rate を許容幅つきで比べていたが、これは
    # **原理的に発火しない**。generation_csm.py:489-502 が音声を作るときの
    # 打ち切り式が本スクリプトの cutoff とまったく同じ
    #     eos_idxs = (codes == codebook_eos_token_id).all(dim=-1).nonzero()
    #     cutoff   = eos_idxs.min() if eos_idxs.numel() else codes.shape[0]
    # なので、両辺が同じ量から作られてしまい差が出ようがない。
    #
    # 代わりに「1 フレーム = sampling_rate / frame_rate サンプル」という
    # codec 側の不変量を **フレーム単位で** 照合する。こちらは
    #   * transformers が打ち切り規約を変えた (EOS フレームを含める / 含めない)
    #   * Mimi の frame_rate / sampling_rate が変わった
    #   * 誰かが途中で resample を挟んだ
    # のどれが起きても半フレーム以上ずれて発火する。
    # ------------------------------------------------------------------
    spf = float(bundle.sampling_rate) / float(bundle.frame_rate)   # 1920.0
    exp_samples = cutoff * spf
    if abs(n_samples - exp_samples) > 0.5 * spf:
        raise RuntimeError(
            f"生成長が整合しない: 波形 {n_samples} sample だが cutoff {cutoff} フレーム "
            f"x {spf:.1f} sample/frame = {exp_samples:.0f} sample のはず "
            f"(差 {abs(n_samples - exp_samples) / spf:.2f} フレーム)。"
            f"transformers 側の打ち切り規約か codec の frame_rate を疑うこと。")

    # ------------------------------------------------------------------
    # 上限到達の検出
    #
    # CSM の _sample (generation_csm.py:305-311) は codebook 0..30 がすべて
    # codebook_eos_token_id になった時点で止まる。eos_token_id を渡していなくても
    # ここで止まるので、n_frames == max_new_tokens は「止まる前に予算を使い切った」
    # ことを意味する。EOS フレームも列に入るので、
    #   hit_max かつ eos 無し -> 文の途中で切れた音声 (truncated_at_cap)
    # となる。旧版はこれを row に置くだけで警告も集計もしていなかった。
    # ------------------------------------------------------------------
    hit_max = bool(n_frames >= int(gen.max_new_tokens))
    truncated_at_cap = bool(hit_max and not eos_found)
    if eos_found:
        stop_reason = "eos"
    elif hit_max:
        stop_reason = "max_new_tokens"
    else:
        # 31 codebook だけが EOS になって _sample が止まった場合
        # (打ち切り式は 32 codebook すべてを見るので cutoff には現れない)。
        stop_reason = "eos_partial"
    if truncated_at_cap:
        msg = (f"生成が max_new_tokens={int(gen.max_new_tokens)} まで走り切り、"
               f"EOS フレームが出ていない ({n_frames} フレーム / {sec:.2f} s)。"
               f"文の途中で切れた音声になっている。")
        if gen.fail_on_cap:
            raise RuntimeError(msg + " (--fail_on_cap)")
        print(f"[warn] {msg}", flush=True)

    if sec < gen.min_sec:
        raise RuntimeError(
            f"生成音声が短すぎる ({sec:.3f} s < {gen.min_sec} s、cutoff={cutoff} "
            f"n_frames={n_frames})。壊れた wav を保存しない。")

    codes_cpu = codes[0].detach().cpu()
    return {"audio": audio, "sampling_rate": bundle.sampling_rate,
            "codes": codes_cpu, "codes_fp": codes_fingerprint(codes_cpu),
            "n_frames": n_frames, "cutoff_frames": cutoff, "eos_found": eos_found,
            "n_samples": n_samples, "sec": round(sec, 3),
            "max_new_tokens": int(gen.max_new_tokens),
            "hit_max_new_tokens": hit_max,
            "truncated_at_cap": truncated_at_cap,
            "stop_reason": stop_reason}


def synthesize(bundle, text, ref=None, speaker=DEFAULT_SPEAKER, gen=GenConfig()):
    """素の TTS。BL = TTS(text_a)、TGT = TTS(text_b) はどちらもこれ."""
    inputs = prepare_inputs(bundle, text, ref=ref, speaker=speaker)
    g = _generate(bundle, inputs, gen=gen)
    g["prompt_len"] = int(inputs["input_ids"].shape[1])
    g["text"] = text
    return g


def splice_synthesize(bundle, text_a, word_a, text_b, word_b, layer=DEFAULT_LAYER,
                      ref=None, speaker=DEFAULT_SPEAKER, gen=GenConfig(),
                      collect_kappa=True, strict_span=True):
    """SpeechSplice 本体.

    1. 目標文脈 (text_b) の prefill を 1 回回し、layers[layer] の出力の
       実体語スパンを取り出す (LayerTap で層の生出力を読む)。
    2. 元文脈 (text_a) で generate する。prefill のときだけ hook が発火し、
       元テキストの実体語トークン行をその目標ベクトルで置き換える。
    3. Mimi が波形に戻す。

    目標ベクトルは **text_a と同じ ref (参照音声ターン) を付けた状態**で取る。
    ref を変えると目標ベクトルも変わるので、全対で固定すること。
    重み更新なし。系列長も変えない。
    """
    assert_splice_target(bundle, layer)

    in_a = prepare_inputs(bundle, text_a, ref=ref, speaker=speaker)
    in_b = prepare_inputs(bundle, text_b, ref=ref, speaker=speaker)
    s_span = entity_token_span(bundle, text_a, word_a, input_ids=in_a["input_ids"],
                               ref=ref, speaker=speaker)
    t_span = entity_token_span(bundle, text_b, word_b, input_ids=in_b["input_ids"],
                               ref=ref, speaker=speaker)
    verify_entity_span(s_span, strict=strict_span)
    verify_entity_span(t_span, strict=strict_span)

    tgt_hs, tgt_tap = prefill_forward(bundle, in_b, layer=layer,
                                      need_all_hidden=collect_kappa)
    if tgt_tap is None:
        raise RuntimeError("目標文脈の layer tap が取れなかった")
    li = hidden_index_for_layer(layer)
    if collect_kappa and tgt_hs is not None and li < len(tgt_hs):
        # tap と hidden_states[l+1] が一致することを確かめる (index 規約の自己検査)。
        if not torch.allclose(tgt_tap.float(), tgt_hs[li].float(), atol=1e-3, rtol=1e-3):
            raise RuntimeError(
                f"layers[{layer}] の tap と hidden_states[{li}] が一致しない。"
                f"index 規約が崩れているので中断する。")
    t0, t1 = t_span["tok_start"], t_span["tok_end"]
    tgt_hidden = tgt_tap[0, t0:t1].clone()

    src_hs = None
    if collect_kappa:
        src_hs, _ = prefill_forward(bundle, in_a, layer=None, need_all_hidden=True)

    splice = PrefillSplice(bundle.layer_module(layer), s_span, tgt_hidden,
                           prompt_len=int(in_a["input_ids"].shape[1]),
                           expect_module=bundle.layer_module(layer))
    g = _generate(bundle, in_a, gen=gen, splice=splice)

    g.update({
        "layer": int(layer), "hidden_index": int(li),
        "prompt_len": int(in_a["input_ids"].shape[1]),
        "tgt_prompt_len": int(in_b["input_ids"].shape[1]),
        "s_span": [s_span["tok_start"], s_span["tok_end"]],
        "t_span": [t_span["tok_start"], t_span["tok_end"]],
        "s_tokens": s_span["entity_tokens"], "t_tokens": t_span["entity_tokens"],
        "splice_mode": "token_wise" if splice.equal_len else "mean_pool_broadcast",
    })
    g.update(splice.stats())

    if collect_kappa and src_hs is not None and tgt_hs is not None:
        ks = kappa_by_layer(src_hs, tgt_hs, s_span, t_span)
        g["kappa_by_layer"] = [round(v, 6) for v in ks]
        g["kappa_at_splice"] = round(ks[li], 6)
        g["kappa_last_layer"] = round(ks[-1], 6)
    return g


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def save_wav(path, audio, sr):
    """PCM_16 で保存。途中で落ちても壊れた wav が残らないよう tmp -> rename."""
    import soundfile as sf
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        raise ValueError(f"empty audio for {path}")
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    clipped = int(np.sum(np.abs(a) > 1.0))
    a = np.clip(a, -1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    # format を明示する。拡張子が .wav.tmp になるので soundfile は推測できない。
    sf.write(tmp, a, int(sr), subtype="PCM_16", format="WAV")
    os.replace(tmp, path)
    return path, round(a.shape[0] / float(sr), 3), {"peak": round(peak, 4),
                                                    "n_clipped": clipped}


if __name__ == "__main__":
    print(f"csm_splice: model={DEFAULT_MODEL_ID} layers={EXPECTED_NUM_LAYERS} "
          f"grid={SWEEP_LAYERS} default_layer={DEFAULT_LAYER}")
    print(f"  HF_HOME={os.environ['HF_HOME']}")
    print(f"  layer_grid(16)={layer_grid(16)} layer_grid(28)={layer_grid(28)} "
          f"layer_grid(30)={layer_grid(30)}")
