#!/usr/bin/env python3
"""suno/bark 用 SpeechSplice コア実装.

Bark は decoder-only の TTS を 3 段重ねた構成 (すべて GPT 系の独自学習):

    semantic          BarkSemanticModel  24 層 / hidden 1024 / in 129600 / out 10048
    coarse_acoustics  BarkCoarseModel    24 層 / hidden 1024 / vocab 12096
    fine_acoustics    BarkFineModel      24 層 / hidden 1024 / vocab 1056
    codec             EncodecModel       24 kHz / 8 codebook

**テキストを見るのは semantic 段だけ**。coarse は semantic 段が吐いた意味トークン
(vocab 10000 / 49.9 Hz) だけを受け取り、テキストには一切触らない
(modeling_bark.py:1548 BarkCoarseModel.generate(semantic_output, ...))。
したがって実体語の位置に対応する行を持つのは semantic 段だけで、差し替え地点は

    model.semantic.layers[l] の **出力** の、prefill 系列の実体語トークン行

になる。重みは更新しない。推論時のみ。BL と INJ は prompt が 1 トークンも違わず、
seed も復号設定も声色 (voice preset) も共有する。差は「prefill 中に実体語行を
目標文脈の同一層 hidden state で置換したか」だけ。

既存 4 例 (Qwen3-TTS talker / XTTS-v2 / Llasa-3B / CSM-1B) との違いは 1 点だけ:
**semantic 段と coarse 段の間が連続値ではなく離散トークン列**であること。
他の 4 例は decoder の hidden がそのまま音響トークンの予測に流れるが、Bark では
一度 vocab 10000 の離散列に落ちる。したがって「hidden を少し動かす」だけでは
音が変わらない場合がある (サンプリングが同じトークンを引く)。この構造差は
decoder-only 4 例目としての対照の要点であり、意味トークン列の指紋を毎回記録して
「splice が音に出たか」を可視化する (inj_differs_from_bl)。

prompt のレイアウト (modeling_bark.py:595-690 BarkSemanticModel.generate):

    ids  = input_ids + text_encoding_offset(10048)
    ids  = ids.masked_fill(attention_mask == 0, text_pad_token=129595)
    hist = history_prompt["semantic_prompt"][-256:] を semantic_pad_token(10000) で右 pad
    input_embeds = cat([ emb(ids[:, :256]) + emb(hist[:, :256]),   # 行 0..255 (和)
                         emb([[129599]]) ], dim=1)                  # 行 256 = infer token
                 -> (1, 257, 1024)

  * **prompt 長は常に 257**。文の長さにも preset の有無にも依らない定数
    (= max_input_semantic_length + 1)。実測でも [257, 1, 1, ...] になる。
  * tokenizer は BertTokenizerFast、processor が add_special_tokens=False /
    padding="max_length" / max_length=256 / padding_side="right" で呼ぶので
    [CLS]/[SEP] が付かない。**token index = hidden state の行番号** (補正ゼロ)。
    CSM の bos_pos 補正や Qwen3-TTS の text_body_offset に相当するものは要らない。
  * 行 0..255 は「テキストの埋め込み + 話者履歴の埋め込み」の **和**。したがって
    m != n のときの mean-pool は履歴ぶんも一緒に平均することになる (limitation)。

層格子 (24 層):
    既存の写像 round(l * N / 28) を half-up で 24 層に写す。24/28 = 6/7 なので
    7 -> 6 / 14 -> 12 / 21 -> 18 は割り切れ、丸め誤差が入らない。
        28 層 -> (0, 7, 14, 21, 26)   Qwen3-TTS talker / Llasa-3B
        30 層 -> (0, 8, 15, 23, 28)   XTTS-v2
        16 層 -> (0, 4, 8, 12, 14)    CSM-1B
        24 層 -> (0, 6, 12, 18, 22)   Bark  (本ファイル)
    最深点は既存 4 例の慣例どおり L-2 = 22。層 23 を採らないのは、その出力のうち
    lm_head に届くのが行 256 だけで、実体語行がその後どの attention にも入らず
    **効果が厳密に 0 になる**ため (実測 max|Δlogit| = 0.0)。

**Bark 固有の最重要事項: 深い層では splice が下流にほとんど届かない。**
    CPU 実機での機構プローブ (1 文対・1 話者・1 seed、教師強制で 212 ステップの
    次トークン分布 KL。上限の参照 = 入力文ごと目標文に差し替えたときの値):

        層      実体語のみ         上限比    top1 が変わったステップ
        L0      0.130179          88.1%     17/212
        L6      0.000053           0.04%     2/212
        L12     0.000020           0.01%     1/212
        L18     0.000011           0.01%     1/212
        L22     0.000001           0.00%     0/212      <- 既定の層

    サンプリング下で意味トークン列が 1 個でも動いた対の数は L0 3/3、L22 1/3。
    原因は prompt 257 行のうち **244 行がパディング行**で、テキストの情報が
    最初の数ブロックでテキスト行からパディング行へ写し取られてしまうこと。
    L6 でも「実体語行 + パディング行」まで広げれば効きが 30 倍戻ることを確認した。
    既存 4 例にはこの「テキストの後ろに 244 行のパディングがぶら下がる」構造が無い。

    **この実装は慣例を勝手に変えない。** 既定は他 4 例と同じ L-2 = 22 のままにし、
    代わりに
      (a) 意味トークン列の指紋で inj_differs_from_bl を毎対記録する
      (b) 掃引に効きが残る浅い層 DIAG_LAYERS = (0,1,2,3,4) を足せるようにする
      (c) run_bark_eval.py に「最初の数対がすべて BL と同一なら止まる」検査を入れる
    ことで、null を黙って「Bark では効かない」と記録しないようにしてある。
    どの層を動作点にするかはユーザの判断に委ねる (研究の筋に関わるため)。

Bark 固有の落とし穴 (すべて CPU 実機か transformers 4.57.3 のソースで確認済み。
詳細は scripts/bark/README.md と note/2026-09-17_bark_research.md):
  (1) coarse 段の 1 窓目の prefill 長も **257** になり得る (preset 無しのとき)。
      `T == 257` という数値検査だけでは semantic と区別できない。hook は
      model.semantic.layers[l] に **名指しで**張り、identity で確かめる。
      named_modules() をパターンで舐めて張る書き方は禁止。
  (2) BarkCoarseModel.generate は先頭で semantic_output.masked_fill_(...) を
      **呼び出し側のテンソルに in-place で**掛ける (modeling_bark.py:826)。
      上限到達の判定を coarse のあとで行うと必ず「EOS 無し」になる。
      意味トークンは semantic 段が返した瞬間に clone して指紋を取る。
  (3) min_eos_p は generation_config に書いても効かない。BarkModel.generate が
      kwargs_semantic に **キーごと None を入れて**渡すため、受け側の
      kwargs.get("min_eos_p", config.min_eos_p) が常に None を返す
      (modeling_bark.py:1522 と :674)。model.generate(..., min_eos_p=0.2) と
      **kwarg で渡す**こと。既定は None で、その場合 Bark は文の後に鼻歌や
      無関係な発話を続けることがある。
  (4) 前置きの無い kwargs は 3 段すべてに配られる (modeling_bark.py:1526-1543)。
      temperature= を素で渡すと fine 段まで確率的になり、max_new_tokens= を
      素で渡すと coarse が TypeError で落ちる。必ず semantic_ / coarse_ / fine_
      の前置きを付ける (min_eos_p と attention_mask だけは前置き無しが正しい)。
  (5) semantic の max_new_tokens の上限は **768**。block_size 1024 に対し
      prompt 257 なので 257 + 768 の最後の位置が 1023 でちょうど収まる。
      769 を渡すと IndexError: index out of range in self (実測)。
      逆に 768 に到達した対は EOS を打てなかった対 = 壊れた対。
  (6) 3 段すべてが乱数を消費する。semantic do_sample=True / T=0.7 / top_k=50、
      coarse も同じ、fine は do_sample=False だが temperature=0.5 のとき
      multinomial を引く (modeling_bark.py:1300-1309)。**生成の直前に毎回
      seed を固定する**。
  (7) voice_preset を渡さないと semantic_history が全部 pad になり、話者を
      固定するものが何も無くなる。BL / TGT / INJ で声が変わると採点器の前提
      (BL と TGT は実体語だけ違う) が崩れる。preset は 1 つに固定する。
  (8) BarkProcessor(..., voice_preset=..., return_offsets_mapping=True) は
      TypeError。**kwargs が _validate_voice_preset_dict にも流れる
      (processing_bark.py:320)。スパンは tokenizer を直叩きして取る。
  (9) attention 実装は eager のみ (BARK_ATTENTION_CLASSES に sdpa が無い)。
      attn_implementation="sdpa" を渡すと落ちる。
 (10) BarkSemanticModel(config) を config から直接構築すると、_init_weights が
      LayerNorm の bias を無条件に触るので bias=False では AttributeError
      (modeling_bark.py:348)。from_pretrained は no_init_weights 下なので安全。
      ただし **missing_keys が出た瞬間に同じ例外**が出るので、ロード時の
      missing/unexpected を必ず検査する。
 (11) enable_cpu_offload は使わない。fine_acoustics_hook が仕込まれて .device の
      意味が変わる。GPU では素直に .to("cuda")。
 (12) 「attention mask and the pad token id were not set」という警告は正常。
      BarkSemanticModel.generate が attention_mask を super().generate に
      渡さない作りなので、正しく渡していても必ず出る。直そうとしないこと。

ライセンス: suno/bark は **MIT** (gated=False)。重みは pytorch_model.bin (4.18 GiB)
のみ。safetensors は無い。リポジトリ全体は 20.7 GiB あるが、text*.pt / coarse*.pt /
fine*.pt (16.4 GiB) は本家 suno 実装用で transformers は一切読まない。

CPU 検証 (GPU も重みも使わない):
  OMP_NUM_THREADS=8 taskset -c 0-63 python3 scripts/bark/smoke_test_bark_splice.py
"""
import math
import os
import re
from dataclasses import dataclass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

DEFAULT_MODEL_ID = "suno/bark"
SEED = 42

EXPECTED_NUM_LAYERS = 24
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_NUM_HEADS = 16
EXPECTED_SAMPLING_RATE = 24000
EXPECTED_BLOCK_SIZE = 1024

# semantic 段の prompt レイアウト (generation_config.json の semantic_config より)。
MAX_INPUT_SEMANTIC_LENGTH = 256          # テキスト行数 (右 pad 固定)
PROMPT_LEN = MAX_INPUT_SEMANTIC_LENGTH + 1   # = 257。文にも preset にも依らない定数
TEXT_ENCODING_OFFSET = 10048
TEXT_PAD_TOKEN = 129595
SEMANTIC_PAD_TOKEN = 10000               # eos_token_id と同じ値
SEMANTIC_EOS_TOKEN = 10000
SEMANTIC_INFER_TOKEN = 129599
SEMANTIC_VOCAB_SIZE = 10000
SEMANTIC_RATE_HZ = 49.9
# block_size 1024 - prompt 257 = 767 だが、位置 index は 0..1023 なので 768 まで入る。
# 769 で IndexError (実測)。これは上げてはいけない上限そのもの。
SEMANTIC_MAX_NEW_TOKENS_CAP = 768

# 既存 4 例と同じ深さ比を 24 層に写した格子。最深点は L-2。
SWEEP_LAYERS = (0, 6, 12, 18, 22)
DEFAULT_LAYER = 22
# Bark 固有の追加掃引。L5 以降は prompt からの読み出しがほぼ無くなるため、
# 効きが残るのはここだけ。格子 (SWEEP_LAYERS) の代わりではなく **足すもの**。
DIAG_LAYERS = (0, 1, 2, 3, 4)

# 公式 generation_config.json の既定。BL / TGT / INJ で完全に共有する。
DEFAULT_SEMANTIC_TEMPERATURE = 0.7
DEFAULT_SEMANTIC_TOP_K = 50
DEFAULT_COARSE_TEMPERATURE = 0.7
DEFAULT_COARSE_TOP_K = 50
DEFAULT_FINE_TEMPERATURE = 0.5
DEFAULT_SEMANTIC_MAX_NEW_TOKENS = 768
# generation_config.json には min_eos_p のキーが無い (= None)。本家 suno の推奨は 0.2。
# None のままだと文の後に余計な発話が続き、採点器の差分区間が実体語から外れる。
DEFAULT_MIN_EOS_P = 0.2

DEFAULT_VOICE_PRESET = "v2/en_speaker_6"

# transformers が読むのは 4.18 GiB だけ。無指定の snapshot_download は 20.7 GiB。
DOWNLOAD_ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "speaker_embeddings_path.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "pytorch_model.bin",
    "speaker_embeddings/*.npy", "speaker_embeddings/v2/*.npy",
]


# --------------------------------------------------------------------------
# HF cache / numba cache
#
# slurm_run.sh:99 は -e HF_HOME=/tmp/.cache/huggingface を固定で渡すが、その実体は
# 書き込めず PermissionError になる。重みが置いてあるのは
# <PROJECT_ROOT>/.cache/huggingface (コンテナ内 /mnt/.cache/huggingface)。
# **setdefault はコンテナ側の環境変数に負けるので代入で上書きする。**
# scripts/csm/csm_splice.py:131-160、scripts/llasa/llasa_splice.py:86-97 と同じ扱い。
# transformers を import する前に済ませること。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
DEFAULT_NUMBA_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "numba")


def setup_hf_home(hf_home=None):
    hf_home = hf_home or os.environ.get("BARK_HF_HOME") or DEFAULT_HF_HOME
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


def setup_numba_cache(path=None):
    """librosa/numba が site-packages にキャッシュを書こうとして落ちるのを防ぐ.

    XTTS で踏んだ RuntimeError。採点側 (phoneme_per_v3) が librosa 経由で
    numba を呼ぶので、生成側でも同じ扱いにして環境を揃えておく。
    """
    p = path or os.environ.get("BARK_NUMBA_CACHE_DIR") or DEFAULT_NUMBA_CACHE_DIR
    os.environ["NUMBA_CACHE_DIR"] = p
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass
    return p


setup_hf_home()
setup_numba_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def seed_all(seed=SEED):
    """generate の直前に必ず呼ぶ.

    Bark は semantic / coarse / fine の 3 段すべてが同じ global RNG を消費する
    (fine も temperature=0.5 で multinomial を引く)。seed を固定しないと
    BL / TGT / INJ の差が splice なのかサンプリングの揺れなのか分からなくなる。
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

    scripts/csm/csm_splice.py:layer_grid と同一の関数 (house 関数)。
        28 層 -> (0, 7, 14, 21, 26)   Qwen3-TTS talker / Llasa-3B
        30 層 -> (0, 8, 15, 23, 28)   XTTS-v2
        16 層 -> (0, 4, 8, 12, 14)    CSM-1B
        24 層 -> (0, 6, 12, 18, 22)   Bark
    浅い 4 点は round(l * N / 28) (half-up)、最深点は慣例どおり N-2。
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

    BarkCausalModel.forward は for ループの **入口**で hidden_states を積むので
    (modeling_bark.py:541-544)、index 0 = 埋め込み + 位置、index l+1 = layers[l] の
    出力 (l = 0..22)。index 24 (= 層数) だけは layernorm_final 適用後で、
    layers[23] の生出力ではない。格子の最深点を 22 に採っているのでここには触れない。
    """
    return int(layer) + 1


# --------------------------------------------------------------------------
# snapshot / download
# --------------------------------------------------------------------------
def local_snapshot(hub_id, required=True):
    """HF_HUB_CACHE 内のスナップショットディレクトリを返す (offline 前提).

    ダウンロードは呼び出し側が明示的に許可したときだけ (allow_download)。
    事故で 20.7 GiB 落とさないための既定。
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
    return os.path.isfile(os.path.join(path, "pytorch_model.bin"))


def snapshot_has_preset(path, preset=DEFAULT_VOICE_PRESET):
    """speaker_embeddings の .npy 3 本が揃っているか.

    BarkProcessor.from_pretrained は .npy を落とさない。オフラインで preset を
    使うには事前に取得しておく必要がある。
    """
    if path is None or not os.path.isdir(path):
        return False
    base = os.path.join(path, "speaker_embeddings", preset)
    return all(os.path.isfile(f"{base}_{k}_prompt.npy")
               for k in ("semantic", "coarse", "fine"))


def ensure_bark_snapshot(model_id=DEFAULT_MODEL_ID, allow_download=False,
                         preset=DEFAULT_VOICE_PRESET):
    """重みと preset まで揃ったローカル snapshot のパスを返す.

    allow_patterns を必ず付ける。付けないと本家 suno 形式の text*.pt /
    coarse*.pt / fine*.pt (16.4 GiB) まで落ちて計 20.7 GiB になる。
    transformers が読むのは pytorch_model.bin の 4.18 GiB だけ。
    """
    p = local_snapshot(model_id, required=False)
    if snapshot_has_weights(p) and snapshot_has_preset(p, preset):
        return p
    if not allow_download:
        what = []
        if not snapshot_has_weights(p):
            what.append("pytorch_model.bin")
        if not snapshot_has_preset(p, preset):
            what.append(f"speaker_embeddings/{preset}_*.npy")
        raise FileNotFoundError(
            f"{model_id} のローカル snapshot に {what} が無い (path={p})。\n"
            f"  取得を許可するなら --allow_download を付けること "
            f"(allow_patterns 付きで約 4.2 GiB)。")
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=DOWNLOAD_ALLOW_PATTERNS)


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
class BarkBundle:
    """model / processor / generation config / voice preset をまとめて持つ."""

    def __init__(self, model, processor, source, loading_info=None,
                 history_prompt=None, voice_preset=None, preset_sha256=None):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.source = source
        self.loading_info = loading_info or {}
        self.history_prompt = history_prompt        # dict[str, Tensor] or None
        self.voice_preset = voice_preset
        self.preset_sha256 = preset_sha256
        gc = model.generation_config
        self.sampling_rate = int(getattr(gc, "sample_rate", EXPECTED_SAMPLING_RATE))
        self.codebook_size = int(getattr(gc, "codebook_size", 1024))
        self.semantic_config = dict(gc.semantic_config)
        self.coarse_config = dict(gc.coarse_acoustics_config)
        self.fine_config = dict(gc.fine_acoustics_config)
        self.num_layers = int(model.semantic.config.num_layers)
        self.hidden_size = int(model.semantic.config.hidden_size)
        # semantic 段のモジュール id 集合。hook を張る相手がここに入っていることを
        # 毎回確かめる (BarkBlock は coarse / fine でも使われるため)。
        self._semantic_ids = {id(m) for m in model.semantic.modules()}
        self._coarse_ids = {id(m) for m in model.coarse_acoustics.modules()}
        self._fine_ids = {id(m) for m in model.fine_acoustics.modules()}

    @property
    def layers(self):
        return self.model.semantic.layers

    def layer_module(self, layer):
        n = len(self.layers)
        if not (0 <= int(layer) < n):
            raise ValueError(f"layer {layer} が範囲外 (0..{n - 1})")
        return self.layers[int(layer)]

    @property
    def device(self):
        return next(self.model.parameters()).device


def _sha256_arrays(arrays):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = arrays[k]
        a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load_voice_preset(processor, preset=DEFAULT_VOICE_PRESET):
    """preset を **1 回だけ**読んで dict[str, Tensor] を返す.

    processor(text, voice_preset=...) を対ごとに呼ぶと BarkProcessor._load_voice_preset
    が毎回 cached_file を叩く (300 対 x 3 条件 x 3 ファイル = 2700 往復)。
    起動時に 1 回読んで使い回す。戻り値の形も検査する
    (semantic_prompt は 1 次元。batch 次元を付けると generate が 0 軸で切って黙って壊れる)。
    """
    if preset is None:
        return None, None
    enc = processor("preset probe", voice_preset=preset)
    hp = {k: v for k, v in enc["history_prompt"].items()}
    if set(hp) != {"semantic_prompt", "coarse_prompt", "fine_prompt"}:
        raise RuntimeError(f"voice preset の鍵が想定と違う: {sorted(hp)}")
    if hp["semantic_prompt"].dim() != 1:
        raise RuntimeError(
            f"semantic_prompt は 1 次元であるべき: {tuple(hp['semantic_prompt'].shape)}")
    if hp["coarse_prompt"].dim() != 2 or hp["coarse_prompt"].shape[0] != 2:
        raise RuntimeError(f"coarse_prompt の形が変: {tuple(hp['coarse_prompt'].shape)}")
    if hp["fine_prompt"].dim() != 2 or hp["fine_prompt"].shape[0] != 8:
        raise RuntimeError(f"fine_prompt の形が変: {tuple(hp['fine_prompt'].shape)}")
    return hp, _sha256_arrays(hp)


def load_bark(model_id=DEFAULT_MODEL_ID, device=None, dtype=torch.float32,
              allow_download=False, strict_weights=True,
              voice_preset=DEFAULT_VOICE_PRESET, attn_implementation="eager"):
    """BarkModel と BarkProcessor を読む.

    * attn_implementation は **eager のみ**。BARK_ATTENTION_CLASSES に sdpa が
      無いので "sdpa" を渡すと落ちる (modeling_bark.py:253-256)。
    * output_loading_info=True で missing/unexpected を検査する。Bark の
      _init_weights は LayerNorm の bias を無条件に触るので、missing_keys が
      出ると AttributeError になるか、最悪そのまま乱数で推論に入る。
    * enable_cpu_offload は使わない (fine_acoustics_hook が .device の意味を変える)。
    """
    from transformers import AutoProcessor, BarkModel
    if attn_implementation not in ("eager", "flash_attention_2"):
        raise ValueError(
            f"Bark は attn_implementation={attn_implementation!r} を持たない "
            f"(eager / flash_attention_2 のみ)")
    src = ensure_bark_snapshot(model_id, allow_download=allow_download,
                               preset=voice_preset)
    processor = AutoProcessor.from_pretrained(src)
    model, info = BarkModel.from_pretrained(
        src, dtype=dtype, attn_implementation=attn_implementation,
        output_loading_info=True)
    miss = list(info.get("missing_keys") or [])
    unexp = list(info.get("unexpected_keys") or [])
    mism = list(info.get("mismatched_keys") or [])
    if strict_weights and (miss or mism):
        raise RuntimeError(
            f"重みのロードに欠損がある: missing={miss[:8]} mismatched={mism[:8]}。"
            f"乱数のまま推論に入るので中断する (source={src})")
    if unexp:
        print(f"[warn] unexpected_keys {len(unexp)} 個: {unexp[:5]}", flush=True)
    model = model.eval()
    if device is not None:
        model = model.to(device)

    # 構造の検査 (config を読み違えたまま走らせない)
    for name, sub in (("semantic", model.semantic),
                      ("coarse_acoustics", model.coarse_acoustics),
                      ("fine_acoustics", model.fine_acoustics)):
        if int(sub.config.num_layers) != EXPECTED_NUM_LAYERS:
            raise RuntimeError(f"{name} の層数が {sub.config.num_layers} "
                               f"(想定 {EXPECTED_NUM_LAYERS})")
        if int(sub.config.hidden_size) != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(f"{name} の hidden が {sub.config.hidden_size}")
    if int(model.semantic.config.block_size) != EXPECTED_BLOCK_SIZE:
        raise RuntimeError(f"block_size が {model.semantic.config.block_size}")

    hp, hp_sha = load_voice_preset(processor, voice_preset)
    if hp is not None and device is not None:
        hp = {k: v.to(device) for k, v in hp.items()}

    b = BarkBundle(model, processor, src,
                   loading_info={"missing_keys": miss, "unexpected_keys": unexp,
                                 "mismatched_keys": mism},
                   history_prompt=hp, voice_preset=voice_preset,
                   preset_sha256=hp_sha)
    if int(b.sampling_rate) != EXPECTED_SAMPLING_RATE:
        raise RuntimeError(f"sample_rate が {b.sampling_rate}")
    sem = b.semantic_config
    for k, want in (("max_input_semantic_length", MAX_INPUT_SEMANTIC_LENGTH),
                    ("text_encoding_offset", TEXT_ENCODING_OFFSET),
                    ("text_pad_token", TEXT_PAD_TOKEN),
                    ("semantic_pad_token", SEMANTIC_PAD_TOKEN),
                    ("semantic_infer_token", SEMANTIC_INFER_TOKEN),
                    ("eos_token_id", SEMANTIC_EOS_TOKEN)):
        if int(sem[k]) != int(want):
            raise RuntimeError(f"semantic_config.{k} が {sem[k]} (想定 {want})。"
                               f"prompt のレイアウトが変わっている可能性がある。")
    return b


def assert_splice_target(bundle, layer):
    """hook を張る相手が semantic 段の BarkBlock そのものであることを確かめる.

    **数値だけの検査 (T == 257) では coarse 段の 1 窓目と区別できない**
    (preset 無しのとき coarse の 1 窓目も 257)。identity で確かめる。
    """
    from transformers.models.bark.modeling_bark import BarkBlock
    mod = bundle.layer_module(layer)
    if not isinstance(mod, BarkBlock):
        raise RuntimeError(f"layers[{layer}] が BarkBlock でない: {type(mod).__name__}")
    if id(mod) not in bundle._semantic_ids:
        raise RuntimeError(f"layers[{layer}] が semantic 段のモジュールではない")
    if id(mod) in bundle._coarse_ids or id(mod) in bundle._fine_ids:
        raise RuntimeError(f"layers[{layer}] が coarse / fine と共有されている")
    return mod


# --------------------------------------------------------------------------
# 入力 (テキスト行) の作成
# --------------------------------------------------------------------------
def prepare_inputs(bundle, text):
    """processor を通して input_ids (1, 256) と attention_mask (1, 256) を作る.

    **processor を通すこと。** BarkProcessor.__call__ は padding="max_length",
    max_length=256, add_special_tokens=False で呼ぶ。自前で padding=False に
    すると emb(ids[:, :256]) + emb(hist[:, :256]) の形が合わなくなり、長さ 1 の
    ときだけ broadcast して黙って通る。
    """
    enc = bundle.processor(text)
    ids = enc["input_ids"]
    am = enc["attention_mask"]
    if tuple(ids.shape) != (1, MAX_INPUT_SEMANTIC_LENGTH):
        # processor は truncation を有効にしていないので、長文は (1, L>256) のまま
        # 返り、その後 generate が黙って [:, :256] で切る。
        raise RuntimeError(
            f"input_ids の形が {tuple(ids.shape)} (想定 (1, {MAX_INPUT_SEMANTIC_LENGTH}))。"
            f"本文が 256 トークンを超えている可能性がある: {text!r}")
    n_real = int(am.sum())
    if n_real <= 0:
        raise RuntimeError(f"実トークンが 0: {text!r}")
    # 右詰め padding であること (行番号 = token index の前提)
    if int(am[0, :n_real].sum()) != n_real:
        raise RuntimeError("attention_mask の 1 が先頭から連続していない "
                           "(padding_side が right でない)")
    # 模型と同じ機器へ載せる (2026-09-18)。探針はすべて CPU で回していたので
    # ここが CPU のままでも通っていたが、GPU に載せた模型に CPU の input_ids を
    # 渡すと index_select が「cuda:0 と cpu が混ざっている」と言って落ちる。
    dev = bundle.device
    ids = ids.to(dev)
    am = am.to(dev)
    return {"input_ids": ids, "attention_mask": am, "n_real_tokens": n_real,
            "text": text}


def semantic_prompt_embeds(bundle, input_ids, attention_mask, history_prompt=None):
    """BarkSemanticModel.generate が作る prompt 埋め込み (1, 257, H) を再現する.

    modeling_bark.py:614-660 と 1 行ずつ対応させてある。**再現であることを
    smoke test で検査する** (generate に実際に渡される input_embeds と bitwise 一致)。
    生成時にも PrefillSplice の verify で突き合わせるので、transformers 側が
    レイアウトを変えたら黙って通らずに落ちる。
    """
    sem = bundle.model.semantic
    M = MAX_INPUT_SEMANTIC_LENGTH
    dev = input_ids.device
    ids = input_ids + TEXT_ENCODING_OFFSET
    if attention_mask is not None:
        ids = ids.masked_fill((1 - attention_mask).bool(), TEXT_PAD_TOKEN)
    b = int(ids.shape[0])
    if history_prompt is not None:
        hist = history_prompt["semantic_prompt"][-M:]
        hist = F.pad(hist, (0, M - len(hist)), value=SEMANTIC_PAD_TOKEN, mode="constant")
    else:
        hist = torch.tensor([SEMANTIC_PAD_TOKEN] * M, dtype=torch.int, device=dev)
    hist = torch.repeat_interleave(hist[None], b, dim=0).to(dev)
    infer = torch.tensor([[SEMANTIC_INFER_TOKEN]] * b, dtype=torch.int, device=dev)
    E = sem.input_embeds_layer
    out = torch.cat([E(ids[:, :M]) + E(hist[:, :M + 1]), E(infer)], dim=1)
    if tuple(out.shape[:2]) != (b, PROMPT_LEN):
        raise RuntimeError(f"prompt 埋め込みの形が {tuple(out.shape)} "
                           f"(想定 (*, {PROMPT_LEN}, H))")
    return out


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


def entity_token_span(bundle, text, word, input_ids=None):
    """input_ids の中で実体語が占める **トークン** スパン [tok_start, tok_end) を返す.

    prompt 行 i にはテキスト token i がそのまま載る ([CLS]/[SEP] 無し・右詰め
    padding) ので **token index = hidden state の行番号**。補正は要らない。

    **接頭辞のトークン数を数える旧方式は使わない。** BERT WordPiece には
    先行空白を単独トークンにする artifact が無いので英語では偶然壊れないが、
    (a) 既存 4 例と作法を揃える (b) tokenize_chinese_chars=True なので
    JA/ZH に広げた瞬間に旧方式は必ず壊れる、の 2 点から offset_mapping を使う。

    processor 経由では offset_mapping を取れない (voice_preset があると
    _validate_voice_preset_dict に **kwargs が流れて TypeError)。tokenizer を
    直叩きし、processor が返す input_ids と一致することを毎回確かめる。
    """
    tok = bundle.tokenizer
    enc = tok(text, add_special_tokens=False, padding="max_length",
              max_length=MAX_INPUT_SEMANTIC_LENGTH, return_attention_mask=True,
              return_token_type_ids=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    am = list(enc["attention_mask"])
    offs = [(int(x), int(y)) for x, y in enc["offset_mapping"]]

    if input_ids is not None:
        ref = input_ids[0].tolist() if torch.is_tensor(input_ids) and input_ids.dim() == 2 \
            else list(input_ids)
        if ref != ids:
            raise RuntimeError(
                "tokenizer 直叩きの input_ids が processor の結果と一致しない。"
                "スパン特定が使えないので中断する。")
    n_real = sum(int(v) for v in am)
    if n_real <= 0:
        raise ValueError(f"実トークンが 0: {text!r}")

    c0, c1 = _entity_char_span(text, word)
    # **i < n_real に限ること。** padding 行の offset は (0, 0) なので、
    # 制限しないと文頭の実体語 (c0 == 0) に 240 行以上の pad が混ざる。
    sel = [i for i, (x, y) in enumerate(offs)
           if i < n_real and y > x and x < c1 and y > c0]
    if not sel:
        raise ValueError(f"no token overlaps entity char span {(c0, c1)} in {text!r}")
    if sel != list(range(sel[0], sel[-1] + 1)):
        raise ValueError(f"entity token span is not contiguous: {sel}")
    t0, t1 = sel[0], sel[-1] + 1
    if t1 > n_real:
        raise RuntimeError(f"span [{t0},{t1}) が実トークン数 {n_real} を超えている")
    if t1 > MAX_INPUT_SEMANTIC_LENGTH:
        raise RuntimeError(f"span [{t0},{t1}) が prompt のテキスト行 "
                           f"{MAX_INPUT_SEMANTIC_LENGTH} を超えている")
    covered = text[offs[t0][0]:offs[t1 - 1][1]]
    decoded = tok.decode(ids[t0:t1])
    return {
        "word": word,
        "text": text,
        "char_span": [int(c0), int(c1)],
        "tok_start": int(t0),
        "tok_end": int(t1),
        "n_tokens_entity": int(t1 - t0),
        "n_real_tokens": int(n_real),
        "prompt_len": int(PROMPT_LEN),
        "covered_text": covered,
        "decoded_text": decoded,
        "roundtrip_ok": bool(covered.strip() == word.strip()
                             and decoded.replace(" ", "").strip().lower()
                             == word.replace(" ", "").strip().lower()),
        "entity_tokens": tok.convert_ids_to_tokens(ids[t0:t1]),
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
    """layers[l] の出力を読み取るだけの forward hook (書き換えない)."""

    def __init__(self, layer_module):
        self.layer_module = layer_module
        self.handle = None
        self.hidden = None
        self.calls = 0

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            raise RuntimeError(
                f"BarkBlock の出力が 3 次元の Tensor でない: "
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


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def prefill_forward(bundle, inputs, layer=None, need_all_hidden=True,
                    history_prompt="bundle"):
    """semantic 段の prefill を 1 回だけ回す (cache 不使用).

    generate の prefill と **同じ入力・同じ経路**にしてある:
      * input_embeds は semantic_prompt_embeds で再現したもの (1, 257, H)
      * attention_mask は渡さない。BarkSemanticModel.generate は attention_mask を
        input_ids の masked_fill にしか使わず super().generate には渡さないので、
        prefill forward が受け取る attention_mask は None (実測)。
      * position_ids も渡さない (past_length = 0 なので arange(0, 257) になる)。

    戻り値:
      hidden_states : 長さ 25 のタプル。0 = 埋め込み + 位置、l+1 = layers[l] の出力
                      (l = 0..22)。index 24 は layernorm_final 適用後。
      tap_hidden    : layer を指定したときの layers[layer] の **生の出力** (1, 257, H)。
    """
    hp = bundle.history_prompt if history_prompt == "bundle" else history_prompt
    embeds = semantic_prompt_embeds(bundle, inputs["input_ids"],
                                    inputs["attention_mask"], history_prompt=hp)
    tap = LayerTap(assert_splice_target(bundle, layer)) if layer is not None else None
    ctx = tap if tap is not None else _null_ctx()
    with torch.no_grad(), ctx:
        out = bundle.model.semantic(input_embeds=embeds, use_cache=False,
                                    output_hidden_states=bool(need_all_hidden),
                                    return_dict=True)
    hs = tuple(h.detach() for h in out.hidden_states) if need_all_hidden else None
    if hs is not None and len(hs) != bundle.num_layers + 1:
        raise RuntimeError(f"hidden_states が {len(hs)} 要素 "
                           f"(想定 {bundle.num_layers + 1})")
    th = tap.hidden if tap is not None else None
    if layer is not None and th is None:
        raise RuntimeError(f"layers[{layer}] の tap が発火しなかった")
    return hs, th


def span_vector(hidden, tok_start, tok_end):
    """スパン内 hidden state をトークン方向に単純平均したベクトル."""
    return hidden[0, int(tok_start):int(tok_end)].float().mean(dim=0)


def kappa_by_layer(src_hidden_states, tgt_hidden_states, s_span, t_span):
    """kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}).

    既存の定義 (scripts/llasa/llasa_splice.py:kappa_by_layer、
    scripts/csm/csm_splice.py:kappa_by_layer) と同じ。
    index 0 = 埋め込み + 位置、index l+1 = layers[l] の出力。
    **index 24 だけは layernorm_final 適用後**なので他と意味が揃わない
    (格子の最深点が 22 なので splice 地点はここに触れない)。
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
    """semantic.layers[l] の出力に、prefill のときだけ介入する forward hook.

    **prefill 判定** (ここを間違えると KV cache と不整合になる):
      1. h.shape[1] <= 1 -> decode ステップ。何もしない。
         (実測の系列: [257, 1, 1, 1, ...])
      2. h.shape[1] != prompt_len (= 257) は見送って skipped_shapes に記録。
         **Bark の prompt 長はデータに依らない定数**なので、他モデルと違い
         実測ではなく generation config から決まる値で照合できる。
      3. 既に 1 回発火していたら二度と発火しない (fired flag)。
      4. module is expect_module でなければ絶対に書き換えない。
         **これが最後の砦**。coarse 段の 1 窓目の入力長も 257 になり得るので、
         長さの検査だけでは semantic と区別できない。
    生成後に assert_fired() を呼び、ちょうど 1 回でなければ例外にする。

    **置換の中身**:
      m = 元テキストの実体語トークン数、n = 目標テキストのそれ。
      m == n : 行対行でそのまま置換。
      m != n : 目標スパンをトークン方向に平均した 1 本を m 行に broadcast。
      いずれも **系列長は変えない** (T = 257 のまま)。

    **verify_hidden**: prefill で受け取った (書き換え前の) hidden を、別に計算した
    元文脈の prefill hidden と突き合わせる。donor (目標側の hidden) を generate と
    別の経路で計算しているので、その経路が生成時の prefill と同じであることを
    実行のたびに確かめる。ここが黙ってずれると、INJ は「別物の hidden を入れた音」に
    なるのに例外は出ない。
    """

    def __init__(self, layer_module, s_span, tgt_hidden, prompt_len=PROMPT_LEN,
                 expect_module=None, verify_hidden=None, verify_atol=1e-3):
        self.layer_module = layer_module
        self.expect_module = expect_module if expect_module is not None else layer_module
        self.s0 = int(s_span["tok_start"])
        self.s1 = int(s_span["tok_end"])
        self.tgt = tgt_hidden.detach()            # (n, hidden)
        self.prompt_len = int(prompt_len) if prompt_len is not None else None
        self.verify_hidden = None if verify_hidden is None else verify_hidden.detach()
        self.verify_atol = float(verify_atol)
        self.verify_max_abs_diff = None
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
            # 別の段 (coarse / fine) に掛かっている。絶対に書き換えない。
            self.foreign_calls += 1
            return output
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            raise RuntimeError(
                f"BarkBlock の出力が 3 次元の Tensor でない: "
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
        if self.verify_hidden is not None:
            if tuple(self.verify_hidden.shape) != tuple(h.shape):
                raise RuntimeError(
                    f"照合用 hidden の形が合わない: {tuple(self.verify_hidden.shape)} "
                    f"vs {tuple(h.shape)}")
            d = float((self.verify_hidden.float().to(h.device) - h.float()).abs().max())
            self.verify_max_abs_diff = d
            if not (d <= self.verify_atol):
                raise RuntimeError(
                    f"生成時の prefill hidden と、別経路で計算した prefill hidden が "
                    f"一致しない (max|Δ| = {d:.3e} > {self.verify_atol:.1e})。"
                    f"donor も同じ経路で計算しているので、このまま INJ を作ると "
                    f"別物の hidden を入れた音になる。中断する。")
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
                f"decode_calls={self.decode_calls} foreign_calls={self.foreign_calls}")

    def stats(self):
        return {"fired": self.fired, "decode_calls": self.decode_calls,
                "skipped_shapes": self.skipped_shapes[:16],
                "foreign_calls": self.foreign_calls,
                "m_src_entity_tokens": self.m, "n_tgt_entity_tokens": self.n,
                "equal_len": self.equal_len,
                "prefill_hidden_max_abs_diff": (None if self.verify_max_abs_diff is None
                                                else round(self.verify_max_abs_diff, 8))}


# --------------------------------------------------------------------------
# 意味トークンの捕捉
# --------------------------------------------------------------------------
class SemanticTap:
    """model.semantic.generate の戻り値を **clone して** 記録する.

    BarkCoarseModel.generate が semantic_output.masked_fill_ で呼び出し側の
    テンソルを in-place 書き換えする (modeling_bark.py:826-829) ので、
    coarse のあとに読むと EOS が消えていて「上限まで走った」と誤判定する。
    戻ってきた瞬間に clone する。

    ここで取れる指紋が **splice が音に出たかどうかの唯一の直接的な証拠**。
    BL と INJ は semantic 段の乱数消費が同じなので、splice が効いていなければ
    意味トークン列は完全一致し、その先 (coarse / fine / codec) も同一になる。
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.calls = 0
        self.tokens = None
        self._orig = None

    def __enter__(self):
        sem = self.bundle.model.semantic
        self._orig = sem.generate

        def wrapped(*a, **kw):
            out = self._orig(*a, **kw)
            self.calls += 1
            if self.tokens is None and torch.is_tensor(out):
                self.tokens = out.detach().clone().cpu()
            return out

        sem.generate = wrapped
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            sem = self.bundle.model.semantic
            # インスタンス属性を消してクラスの束縛メソッドに戻す。代入で戻すと
            # インスタンス側に束縛メソッドが残り続け、次に包んだときに二重包みに
            # なり得る (calls が 2 になって _generate が落ちる)。
            try:
                del sem.generate
            except AttributeError:
                sem.generate = self._orig
            self._orig = None
        return False


def semantic_fingerprint(tokens):
    """意味トークン列の短い指紋 (BL と INJ の同一性の検出に使う)."""
    import hashlib
    a = np.ascontiguousarray(np.asarray(tokens).astype(np.int64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@dataclass
class GenConfig:
    seed: int = SEED
    do_sample: bool = True
    semantic_temperature: float = DEFAULT_SEMANTIC_TEMPERATURE
    semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K
    coarse_temperature: float = DEFAULT_COARSE_TEMPERATURE
    coarse_top_k: int = DEFAULT_COARSE_TOP_K
    fine_temperature: float = DEFAULT_FINE_TEMPERATURE
    semantic_max_new_tokens: int = DEFAULT_SEMANTIC_MAX_NEW_TOKENS
    min_eos_p: float = DEFAULT_MIN_EOS_P
    min_sec: float = 0.5
    # 上限到達 (EOS が出ないまま走り切った) を error にするか。
    # 既定は False = 行に記録して警告を出すだけ。
    fail_on_cap: bool = False

    def __post_init__(self):
        if int(self.semantic_max_new_tokens) > SEMANTIC_MAX_NEW_TOKENS_CAP:
            raise ValueError(
                f"semantic_max_new_tokens={self.semantic_max_new_tokens} は上限 "
                f"{SEMANTIC_MAX_NEW_TOKENS_CAP} を超える。block_size {EXPECTED_BLOCK_SIZE} に対し "
                f"prompt {PROMPT_LEN} なので位置埋め込みが範囲外になり IndexError で落ちる。")
        if int(self.semantic_max_new_tokens) <= 0:
            raise ValueError("semantic_max_new_tokens は正の整数")

    def as_dict(self):
        return {"seed": self.seed, "do_sample": self.do_sample,
                "semantic_temperature": self.semantic_temperature,
                "semantic_top_k": self.semantic_top_k,
                "coarse_temperature": self.coarse_temperature,
                "coarse_top_k": self.coarse_top_k,
                "fine_temperature": self.fine_temperature,
                "semantic_max_new_tokens": self.semantic_max_new_tokens,
                "min_eos_p": self.min_eos_p, "min_sec": self.min_sec,
                "fail_on_cap": self.fail_on_cap}

    def generate_kwargs(self):
        """**前置きを必ず付ける**。素の kwargs は 3 段すべてに配られる.

        (modeling_bark.py:1526-1543)
          * temperature= を素で渡すと fine 段まで確率的になる
          * max_new_tokens= を素で渡すと coarse が TypeError で落ちる
        min_eos_p と attention_mask は BarkModel.generate が個別に拾うので
        前置きを付けてはいけない (kwargs.pop("min_eos_p", None) / :1522)。
        """
        kw = {
            "semantic_max_new_tokens": int(self.semantic_max_new_tokens),
            "semantic_do_sample": bool(self.do_sample),
            "coarse_do_sample": bool(self.do_sample),
            # fine 段は do_sample を読まない。temperature だけを見る (:1234)。
            "fine_temperature": float(self.fine_temperature),
        }
        if self.do_sample:
            kw["semantic_temperature"] = float(self.semantic_temperature)
            kw["semantic_top_k"] = int(self.semantic_top_k)
            kw["coarse_temperature"] = float(self.coarse_temperature)
            kw["coarse_top_k"] = int(self.coarse_top_k)
        if self.min_eos_p is not None:
            kw["min_eos_p"] = float(self.min_eos_p)
        return kw


def _generate(bundle, inputs, gen=GenConfig(), splice=None):
    """BarkModel.generate を 1 回回して波形と統計を返す.

    seed は generate の直前に必ずリセットする。3 段すべてが同じ global RNG を
    消費するので、BL / TGT / INJ で乱数の消費順序と回数を揃える唯一の方法。
    """
    kw = {"input_ids": inputs["input_ids"],
          "attention_mask": inputs["attention_mask"],
          "history_prompt": bundle.history_prompt}
    kw.update(gen.generate_kwargs())
    seed_all(gen.seed)
    ctx = splice if splice is not None else _null_ctx()
    tap = SemanticTap(bundle)
    with torch.no_grad(), tap, ctx:
        audio_t = bundle.model.generate(**kw)
    if splice is not None:
        splice.assert_fired()
    if tap.calls != 1:
        raise RuntimeError(f"semantic.generate の呼び出し回数が {tap.calls} (1 であるべき)")
    if tap.tokens is None:
        raise RuntimeError("意味トークンを捕捉できなかった")

    sem = tap.tokens
    if sem.dim() != 2 or sem.shape[0] != 1:
        raise RuntimeError(f"意味トークンの形が {tuple(sem.shape)}")
    sem_row = sem[0]
    n_semantic = int(sem_row.numel())
    eos_found = bool((sem_row == SEMANTIC_EOS_TOKEN).any())
    hit_cap = bool(n_semantic >= int(gen.semantic_max_new_tokens))
    truncated_at_cap = bool(hit_cap and not eos_found)
    if eos_found:
        stop_reason = "eos"
    elif hit_cap:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "other"

    if not torch.is_tensor(audio_t):
        raise RuntimeError(f"generate の戻りが Tensor でない: {type(audio_t)}")
    if audio_t.dim() != 2 or audio_t.shape[0] != 1:
        raise RuntimeError(f"波形の形が {tuple(audio_t.shape)} (想定 (1, n_samples))")
    audio = audio_t[0].detach().float().cpu().numpy()
    n_samples = int(audio.shape[-1])
    sec = n_samples / float(bundle.sampling_rate)

    # ------------------------------------------------------------------
    # 長さの整合検査。
    # 意味トークン (49.9 Hz) -> coarse フレーム (75 Hz) -> 波形。
    #   n_coarse_frames = floor(n_semantic * coarse_rate_hz / semantic_rate_hz)
    #   n_samples       = n_coarse_frames * sampling_rate / coarse_rate_hz
    # 端数と EOS の扱いで数フレームずれるので、許容幅を 0.5 秒取ったうえで
    # 「桁が違う」ずれだけを捕まえる。resample が挟まった / codec の周波数が
    # 変わった / 波形が別の条件のものになった、を検出するのが目的。
    # ------------------------------------------------------------------
    coarse_rate = float(bundle.coarse_config.get("coarse_rate_hz", 75.0))
    exp_sec = n_semantic / float(SEMANTIC_RATE_HZ)
    if abs(sec - exp_sec) > 0.5 + 0.1 * exp_sec:
        raise RuntimeError(
            f"生成長が整合しない: 波形 {sec:.2f} s だが意味トークン {n_semantic} 個 / "
            f"{SEMANTIC_RATE_HZ} Hz = {exp_sec:.2f} s のはず "
            f"(coarse_rate_hz={coarse_rate})。codec の周波数か打ち切り規約を疑うこと。")

    if truncated_at_cap:
        msg = (f"生成が semantic_max_new_tokens={int(gen.semantic_max_new_tokens)} まで "
               f"走り切り、EOS (10000) が出ていない ({n_semantic} token / {sec:.2f} s)。"
               f"文の途中で切れたか暴走した音声になっている。")
        if gen.fail_on_cap:
            raise RuntimeError(msg + " (--fail_on_cap)")
        print(f"[warn] {msg}", flush=True)

    if sec < gen.min_sec:
        raise RuntimeError(
            f"生成音声が短すぎる ({sec:.3f} s < {gen.min_sec} s、"
            f"n_semantic={n_semantic})。壊れた wav を保存しない。")

    return {"audio": audio, "sampling_rate": bundle.sampling_rate,
            "semantic_tokens": sem_row.tolist(),
            "sem_fp": semantic_fingerprint(sem_row.numpy()),
            "n_semantic": n_semantic, "eos_found": eos_found,
            "n_samples": n_samples, "sec": round(sec, 3),
            "semantic_max_new_tokens": int(gen.semantic_max_new_tokens),
            "hit_cap": hit_cap, "truncated_at_cap": truncated_at_cap,
            "stop_reason": stop_reason,
            "prompt_len": int(PROMPT_LEN),
            "n_real_tokens": int(inputs["n_real_tokens"])}


def synthesize(bundle, text, gen=GenConfig()):
    """素の TTS。BL = TTS(text_a)、TGT = TTS(text_b) はどちらもこれ."""
    inputs = prepare_inputs(bundle, text)
    g = _generate(bundle, inputs, gen=gen)
    g["text"] = text
    return g


def splice_synthesize(bundle, text_a, word_a, text_b, word_b, layer=DEFAULT_LAYER,
                      gen=GenConfig(), collect_kappa=True, strict_span=True,
                      verify_prefill=True):
    """SpeechSplice 本体.

    1. 目標文脈 (text_b) の prefill を 1 回回し、semantic.layers[layer] の出力の
       実体語スパンを取り出す (LayerTap で層の生出力を読む)。
    2. 元文脈 (text_a) で generate する。prefill のときだけ hook が発火し、
       元テキストの実体語トークン行をその目標ベクトルで置き換える。
    3. coarse / fine / EnCodec が波形に戻す。**後段には掛けない。**
       coarse はテキストを一切見ないので、実体語位置に対応する行を持たない。

    目標ベクトルは text_a と同じ voice preset を付けた状態で取る。preset を
    変えると目標ベクトルも変わる (prompt 行は「テキスト埋め込み + 履歴埋め込み」の
    和) ので、全対で固定すること。重み更新なし。系列長も変えない。

    prefill は generate より **前**に回す。seed_all は _generate の中で
    generate の直前に呼ばれるので、prefill が乱数を消費しても BL / TGT / INJ の
    生成開始時の RNG 状態は揃う。
    """
    assert_splice_target(bundle, layer)

    in_a = prepare_inputs(bundle, text_a)
    in_b = prepare_inputs(bundle, text_b)
    s_span = entity_token_span(bundle, text_a, word_a, input_ids=in_a["input_ids"])
    t_span = entity_token_span(bundle, text_b, word_b, input_ids=in_b["input_ids"])
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
    src_tap = None
    if collect_kappa or verify_prefill:
        src_hs, src_tap = prefill_forward(bundle, in_a, layer=layer,
                                          need_all_hidden=collect_kappa)

    splice = PrefillSplice(bundle.layer_module(layer), s_span, tgt_hidden,
                           prompt_len=PROMPT_LEN,
                           expect_module=bundle.layer_module(layer),
                           verify_hidden=(src_tap if verify_prefill else None))
    g = _generate(bundle, in_a, gen=gen, splice=splice)

    g.update({
        "text": text_a, "tgt_text": text_b,
        "layer": int(layer), "hidden_index": int(li),
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
    """PCM_16 で保存。途中で落ちても壊れた wav が残らないよう tmp -> rename.

    採点器 (scripts/per_fix/phoneme_per_v3.py) は内部で 16 kHz に落とすので、
    Bark の 24 kHz をそのまま保存してよい。
    """
    import soundfile as sf
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        raise ValueError(f"empty audio for {path}")
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    clipped = int(np.sum(np.abs(a) > 1.0))
    a = np.clip(a, -1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    # format を明示する (拡張子が .wav.tmp になるので soundfile は推測できない)。
    sf.write(tmp, a, int(sr), subtype="PCM_16", format="WAV")
    os.replace(tmp, path)
    return path, round(a.shape[0] / float(sr), 3), {"peak": round(peak, 4),
                                                    "n_clipped": clipped}


if __name__ == "__main__":
    print(f"bark_splice: model={DEFAULT_MODEL_ID} layers={EXPECTED_NUM_LAYERS} "
          f"grid={SWEEP_LAYERS} default_layer={DEFAULT_LAYER} diag={DIAG_LAYERS}")
    print(f"  HF_HOME={os.environ['HF_HOME']} NUMBA_CACHE_DIR={os.environ['NUMBA_CACHE_DIR']}")
    print(f"  layer_grid(16)={layer_grid(16)} layer_grid(24)={layer_grid(24)} "
          f"layer_grid(28)={layer_grid(28)} layer_grid(30)={layer_grid(30)}")
    print(f"  prompt_len={PROMPT_LEN} (定数) semantic cap={SEMANTIC_MAX_NEW_TOKENS_CAP}")
