#!/usr/bin/env python3
"""suno/bark で英語 300 対の BL / TGT / INJ を生成し、wav_dir に保存する.

3 条件:
  BL  = TTS(text_a)                      元の文をそのまま合成 (実体語は word_a を読む)
  INJ = SpeechSplice(text_a -> word_b)   文面は word_a だが word_b を読ませたい
  TGT = TTS(text_b)                      文面を word_b に書き換えて合成 (正解の読み)

保存名は採点器に合わせて s{pair_idx:04d}_{bl,inj,tgt}.wav (24 kHz / PCM_16)。
採点は **既存の採点器** scripts/per_fix/phoneme_per_v3.py で行う。ここでは作らない。

**Bark 固有の注意 (最重要)**
  既定層 22 (= L-2、既存 4 例と同じ慣例) では、CPU の機構プローブで
  意味トークン列が BL と完全一致する = INJ が BL とビット単位で同じ音になる、
  という結果が出ている。原因は手法ではなく Bark の prompt 構造
  (テキスト 16 行の後ろに 244 行のパディングが続き、テキストの情報が浅い層で
   パディング行へ写し取られてしまう)。詳細は scripts/bark/README.md。
  そのため本スクリプトは
    * 対ごとに BL と INJ の **意味トークン列の指紋**を突き合わせ、
      inj_differs_from_bl を記録する
    * 最初の --null_check_pairs 対がすべて一致したらそこで **停止する**
      (--allow_null_layer で続行できる)
    * summary に n_inj_identical_to_bl / inj_identical_rate を必ず出す
  ようにしてある。「効果なし」と「そもそも splice が音に出ていない」を
  取り違えないため。

**率の分母は必ず試行総数 (n_attempted)**。生き残った対だけで割ると、生成が
落ちた対がまるごと消えて率が勝手に上がる。n_attempted / n_ok / n_failed と
条件別の失敗数を必ず出す。

**再開の照合**。progress.json に実行設定の指紋 (層 / seed / 復号設定 / preset /
実装の AST) を書き、再開時に食い違ったら停止する。wav_dir にも台帳を置き、
別の設定で作られた wav を「今回の結果」として使い回さないようにする
(2026-09-17 の LongCat 統制条件の事故と同じ型の再発防止)。

実行例 (GPU):
  SLURM_TIME=02:00:00 slurm_run.sh \\
    python3 scripts/bark/run_bark_eval.py --lang en --n 12 --layer 0 \\
      --out_dir eval_results/bark_smoke_L0 --allow_download

  SLURM_TIME=08:00:00 slurm_run.sh \\
    python3 scripts/bark/run_bark_eval.py --lang en --n 300 --layer 22 \\
      --out_dir eval_results/bark_en_n300_L22
"""

# 2026-09-27: CPU スレッド上限（1 本あたり 13）。torch などを読む前に設定する。
import os as _os_cap
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os_cap.environ[_v] = "13"
try:
    import torch as _torch_cap
    _torch_cap.set_num_threads(13)
except Exception:
    pass
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
import uuid

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "bark"))

import bark_splice as C  # noqa: E402  (HF_HOME / NUMBA_CACHE_DIR はこの import で代入上書きされる)

import numpy as np  # noqa: E402
import torch  # noqa: E402

COND = ("bl", "inj", "tgt")
DEFAULT_MAX_DROP_RATE = 0.05
MANIFEST_SCHEMA = 1


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pairs(lang, n):
    """Table 3 (cross-architecture) で既存モデルが使っているデータそのもの.

    JA / ZH を解禁した (2026-09-18)。元は「読みが成立するか、スパンの取り方が
    通るかを先に確かめること」という保留だった。
    `scripts/bark/probe_ja_zh_text_cpu.py` で確かめた結果:

      * 文字を切る器は BertTokenizerFast (語彙 119,547 = 多言語 BERT)。
        **日本語・中国語は書記素のまま通る。**ローマ字化も拼音化もされない
        (XTTS-v2 は cutlet / pypinyin を通すため日中では使えない)。
      * 実体語のスパンは offset_mapping でそのまま取れる。12 例すべてで
        覆った文字列が実体語と一致し、食い違いは 0。
      * 中国語では「的」が独立した切片になるので、LongCat の umt5 で 6 対を
        落とした「田中的」型の融合は起きない。
    """
    if lang == "en":
        from en_cross_culture_names import build_300_pairs
    elif lang == "ja":
        from ja_cross_culture_names import build_300_pairs
    elif lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    else:
        raise SystemExit(f"--lang {lang} は対象外 (en / ja / zh)。")
    pairs = build_300_pairs()[:n]
    for i, p in enumerate(pairs):
        p.setdefault("pair_idx", i)
    return pairs


def wav_paths(wav_dir, pidx):
    return {c: os.path.join(wav_dir, f"s{pidx:04d}_{c}.wav") for c in COND}


_WAV_RE = re.compile(r"^s(\d{4})_(bl|inj|tgt)\.wav$")


def scan_wav_dir(wav_dir):
    out = []
    if not os.path.isdir(wav_dir):
        return out
    for name in sorted(os.listdir(wav_dir)):
        m = _WAV_RE.match(name)
        if m:
            out.append((int(m.group(1)), m.group(2), os.path.join(wav_dir, name)))
    return out


# --------------------------------------------------------------------------
# ASR (source_in_bl / target_in_inj)
# --------------------------------------------------------------------------
class Asr:
    """Whisper で書き起こして実体語が入っているかを判定する.

    eval_qwen3_crossculture_acoustic_fixed.py と同じ WhisperAligner を読むので、
    既存の decoder-only 4 例と同じ ASR モデル・同じ判定になる。
    """

    def __init__(self, device="cuda", model_id="openai/whisper-large-v3"):
        from forced_alignment_utils import WhisperAligner
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.aligner = WhisperAligner(device=device, model_id=model_id, dtype=dtype)
        self.model_id = model_id

    def transcribe(self, audio, sr, lang="en"):
        chunks, _ = self.aligner.transcribe_words(audio, sr, language=lang)
        return " ".join(c.get("word", "") for c in chunks).strip()


def transcribe_cond(asr, cond, audio, sr, lang, pair=None):
    """書き起こしの入口をここ 1 箇所にまとめてある (生成経路と --asr_only 経路で共通)."""
    return asr.transcribe(audio, sr, lang=lang)


def in_text(name, transcript, lang="en"):
    """実体語が書き起こしに入っているか (既存 4 例と同じ判定)."""
    if not transcript:
        return False
    if lang in ("ja", "zh"):
        return name in transcript.replace(" ", "")
    return name.lower() in transcript.lower()


# --------------------------------------------------------------------------
# 集計
#
# **率の分母は必ず「試行した対の総数」(n_attempted)**。
# 率は「行が ok の対」ではなく「その条件の生成が成立した対」から数える
# (INJ が落ちても BL が成立していれば source_in_bl には数える)。
# --------------------------------------------------------------------------
def _rate(rows, key, denom):
    """True の個数 / denom。1 対も測れていないときは 0.0 ではなく None を返す.

    --no_asr や ASR が全滅した実行で source_in_bl_rate=0.0 と書かれると
    「BL が word_a を読めていない」と読めてしまう。実際には測っていないだけ。
    """
    d = int(denom)
    if d <= 0:
        return None
    if not any(r.get(key) is not None for r in rows):
        return None
    return round(sum(1 for r in rows if r.get(key) is True) / float(d), 4)


def _rate_survivors(rows, key):
    """生き残りだけを分母にした率 (参考値。主たる率ではない)."""
    vs = [r for r in rows if r.get(key) is not None]
    if not vs:
        return None
    return round(sum(1 for r in vs if r[key]) / float(len(vs)), 4)


def _mean(vals):
    vs = [float(v) for v in vals if v is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def _dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


COND_JUDGEMENTS = {"bl": ("source_in_bl",), "tgt": ("target_in_tgt",),
                   "inj": ("target_in_inj", "source_in_inj")}


def _cond_ok(r, c):
    v = r.get(f"{c}_status")
    if v is not None:
        return v == "ok"
    if r.get(f"{c}_sem_fp") is not None or r.get(f"{c}_sec") is not None:
        return True
    return r.get("status") == "ok"


def summarize(rows, args, extra=None, attempted=None):
    """attempted = 試行した対の pair_idx のリスト (= 率の分母)."""
    if attempted is None:
        attempted = [int(r["pair_idx"]) for r in rows if "pair_idx" in r]
    attempted = sorted({int(i) for i in attempted})
    n_att = len(attempted)
    att = set(attempted)

    in_scope = [r for r in rows if int(r.get("pair_idx", -1)) in att]
    have_row = {int(r["pair_idx"]) for r in in_scope if "pair_idx" in r}
    ok = [r for r in in_scope if r.get("status") == "ok"]
    n_missing = n_att - len(have_row)
    n_failed = n_att - len(ok)
    drop_rate = round(n_failed / float(n_att), 4) if n_att else None

    n_ok_by_cond = {c: sum(1 for r in in_scope if _cond_ok(r, c)) for c in COND}
    fail_by_cond = {c: n_att - n_ok_by_cond[c] for c in COND}
    n_all_cond_ok = sum(1 for r in in_scope if all(_cond_ok(r, c) for c in COND))

    # 失敗の内訳 (どの段で落ちたか)
    fail_stages = {}
    for r in in_scope:
        if r.get("status") != "ok":
            k = str(r.get("fail_stage") or "unknown")
            fail_stages[k] = fail_stages.get(k, 0) + 1
    for i in sorted(att - have_row):
        fail_stages["not_attempted_yet"] = fail_stages.get("not_attempted_yet", 0) + 1

    rows_by_run = {}
    for r in in_scope:
        k = str(r.get("run_id"))
        rows_by_run[k] = rows_by_run.get(k, 0) + 1

    # ---- splice が音に出たか (Bark で最も重要な列) ----
    cmp_rows = [r for r in in_scope if r.get("inj_differs_from_bl") is not None]
    n_identical = sum(1 for r in cmp_rows if r.get("inj_differs_from_bl") is False)
    n_cap = {c: sum(1 for r in in_scope if r.get(f"{c}_truncated_at_cap") is True)
             for c in COND}
    n_hit_cap = {c: sum(1 for r in in_scope if r.get(f"{c}_hit_cap") is True)
                 for c in COND}

    s = {
        "model": "suno/bark",
        "arch": "decoder-only (GPT 独自学習 3 段: semantic / coarse / fine)",
        "splice_target": f"model.semantic.layers[{int(args.layer)}].output",
        "lang": args.lang,
        "layer": int(args.layer),
        "sweep_layers": list(C.SWEEP_LAYERS),
        "diag_layers": list(C.DIAG_LAYERS),
        "num_layers": C.EXPECTED_NUM_LAYERS,
        "prompt_len": C.PROMPT_LEN,
        "seed": int(args.seed),
        "voice_preset": args.voice_preset,

        # ---- 母数 ----
        "n_attempted": n_att,
        "n_ok": len(ok),
        "n_failed": n_failed,
        "n_missing_rows": n_missing,
        "drop_rate": drop_rate,
        "n_ok_by_cond": n_ok_by_cond,
        "n_failed_by_cond": fail_by_cond,
        "n_all_cond_ok": n_all_cond_ok,
        "fail_stages": fail_stages,
        "rows_by_run_id": rows_by_run,

        # ---- 判定 (分母はすべて n_attempted) ----
        "source_in_bl_rate": _rate(in_scope, "source_in_bl", n_att),
        "target_in_tgt_rate": _rate(in_scope, "target_in_tgt", n_att),
        "target_in_inj_rate": _rate(in_scope, "target_in_inj", n_att),
        "source_in_inj_rate": _rate(in_scope, "source_in_inj", n_att),
        # 参考値 (生き残りだけを分母にしたもの)。主たる率ではない。
        "source_in_bl_rate_survivors": _rate_survivors(in_scope, "source_in_bl"),
        "target_in_inj_rate_survivors": _rate_survivors(in_scope, "target_in_inj"),
        "n_transcribed_by_cond": {
            c: sum(1 for r in in_scope if r.get(f"{c}_transcript") is not None)
            for c in COND},

        # ---- splice が音に出たか ----
        "n_compared_bl_inj": len(cmp_rows),
        "n_inj_identical_to_bl": n_identical,
        "inj_identical_rate": (round(n_identical / float(n_att), 4) if n_att else None),

        # ---- 生成の健全性 ----
        "n_hit_cap_by_cond": n_hit_cap,
        "n_truncated_at_cap_by_cond": n_cap,
        "mean_sec_by_cond": {c: _mean([r.get(f"{c}_sec") for r in in_scope])
                             for c in COND},
        "mean_n_semantic_by_cond": {c: _mean([r.get(f"{c}_n_semantic") for r in in_scope])
                                    for c in COND},
        "n_clipped_total": sum(int(r.get(f"{c}_n_clipped") or 0)
                               for r in in_scope for c in COND),

        # ---- スパン ----
        "n_equal_len": sum(1 for r in in_scope if r.get("equal_len") is True),
        "n_mean_pool": sum(1 for r in in_scope if r.get("equal_len") is False),
        "mean_kappa_at_splice": _mean([r.get("kappa_at_splice") for r in in_scope]),
    }

    warnings = list((extra or {}).get("warnings") or [])
    if drop_rate is not None and drop_rate > float(getattr(args, "max_drop_rate",
                                                           DEFAULT_MAX_DROP_RATE)):
        warnings.append(
            f"生成に失敗した対が {n_failed}/{n_att} ({drop_rate:.1%}) で閾値 "
            f"{float(getattr(args, 'max_drop_rate', DEFAULT_MAX_DROP_RATE)):.1%} を超えている。"
            f"率の分母は試行総数のままだが、この結果を表に載せる前に原因を見ること。")
    if cmp_rows and n_identical:
        warnings.append(
            f"INJ の意味トークン列が BL と完全一致した対が {n_identical}/{len(cmp_rows)} ある。"
            f"その対の INJ 音声は BL とビット単位で同じで、ΔPER は必ず 0 になる。"
            f"層 {args.layer} で splice が下流に届いていない可能性が高い "
            f"(Bark は L5 以降で prompt をほとんど読まない)。")
    if s["source_in_bl_rate"] is not None and s["source_in_bl_rate"] < 0.8:
        warnings.append(
            f"source_in_bl_rate={s['source_in_bl_rate']} が低い。BL が word_a を"
            f"読めていないなら、PER を測っても手法の評価にならない (Dia で BL の 55% が"
            f"目標語を発話しなかった前例がある)。先に BL の明瞭性を確かめること。")
    if sum(n_cap.values()):
        warnings.append(f"EOS が出ないまま上限まで走った対がある: {n_cap}")
    if warnings:
        s["warnings"] = _dedupe(warnings)
    if extra:
        for k, v in extra.items():
            if k != "warnings":
                s[k] = v
    return s


def report(summary):
    print("\n" + "=" * 68)
    print(f"  Bark SpeechSplice  layer={summary['layer']} / {summary['num_layers']} 層  "
          f"seed={summary['seed']}  preset={summary['voice_preset']}")
    print("=" * 68)
    print(f"  試行 {summary['n_attempted']} 対 / 成立 {summary['n_ok']} / "
          f"失敗 {summary['n_failed']} (欠落 {summary['n_missing_rows']})")
    print(f"  条件別の成立 {summary['n_ok_by_cond']}  失敗 {summary['n_failed_by_cond']}")
    if summary.get("fail_stages"):
        print(f"  失敗の内訳 {summary['fail_stages']}")
    print(f"  source_in_bl  {summary['source_in_bl_rate']}   "
          f"target_in_tgt {summary['target_in_tgt_rate']}")
    print(f"  target_in_inj {summary['target_in_inj_rate']}   "
          f"source_in_inj {summary['source_in_inj_rate']}")
    print(f"  (分母はすべて試行総数 {summary['n_attempted']}。"
          f"書き起こせた本数 {summary['n_transcribed_by_cond']})")
    print(f"  INJ が BL と同一 {summary['n_inj_identical_to_bl']}/"
          f"{summary['n_compared_bl_inj']}  (rate {summary['inj_identical_rate']})")
    print(f"  上限到達 {summary['n_hit_cap_by_cond']}  "
          f"うち EOS 無し {summary['n_truncated_at_cap_by_cond']}")
    print(f"  平均長 {summary['mean_sec_by_cond']} s  "
          f"平均意味トークン数 {summary['mean_n_semantic_by_cond']}")
    print(f"  m==n {summary['n_equal_len']} 対 / mean-pool {summary['n_mean_pool']} 対  "
          f"kappa@splice 平均 {summary['mean_kappa_at_splice']}")
    for w in summary.get("warnings", []):
        print(f"  [warn] {w}")
    print("=" * 68 + "\n")


# --------------------------------------------------------------------------
# 指紋 (ハッシュの道具)
# --------------------------------------------------------------------------
def _sha256_file(path, limit_mb=64):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if n >= limit_mb * (1 << 20):
                break
    return h.hexdigest()[:16]


def _sha256_text(s):
    return hashlib.sha256(("" if s is None else str(s)).encode("utf-8")).hexdigest()[:16]


def _sha256_obj(d):
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _tf_version():
    try:
        import transformers
        return str(transformers.__version__)
    except Exception:
        return None


# --------------------------------------------------------------------------
# 実装の指紋 (code fingerprint)
#
# 旗で渡した設定だけを指紋にすると、**コードを書き換えても指紋が動かない**。
# ファイルの書き換え (AST ハッシュ) と、実行時の関数差し替え (呼び出し実体の
# バイトコード) の両方を見る。AST は注釈と docstring を落としてから取るので、
# コメントを直しただけで 900 本の wav が作り直しになることはない。
# --------------------------------------------------------------------------
AUDIO_IMPL_FUNCS = (
    "synthesize", "splice_synthesize", "save_wav", "load_bark", "seed_all",
    "semantic_fingerprint", "assert_splice_target", "prepare_inputs",
    "semantic_prompt_embeds", "entity_token_span", "prefill_forward",
    "load_voice_preset",
)
AUDIO_IMPL_METHODS = (
    ("PrefillSplice", "__init__"), ("PrefillSplice", "__call__"),
    ("SemanticTap", "__enter__"), ("GenConfig", "generate_kwargs"),
)


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _ast_sha256(path):
    if not path or not os.path.isfile(path):
        return f"<missing:{path}>"
    try:
        with open(path, "rb") as f:
            tree = _strip_docstrings(ast.parse(f.read()))
        return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "raw:" + _sha256_file(path)


def _const_repr(c):
    """定数の表現を **プロセス間で安定**にする.

    set / frozenset の repr は要素の hash 順に並ぶので、Python のハッシュ乱択
    (PYTHONHASHSEED) のせいで **プロセスごとに変わる**。素の repr を使うと
      * `x in {"a", "b", "c"}` のような式を 1 つ含む関数
    があるだけで実装の指紋が毎回動き、再開のたびに「別の実装で作られている」と
    警告されて全 wav を作り直すことになる (実測で踏んだ)。並べ替えて潰す。
    """
    if isinstance(c, (frozenset, set)):
        return "set(" + ",".join(sorted(_const_repr(x) for x in c)) + ")"
    if isinstance(c, tuple):
        return "(" + ",".join(_const_repr(x) for x in c) + ")"
    return repr(c)


def _code_bytes(code, depth=0):
    parts = [code.co_code, repr(code.co_names).encode("utf-8"),
             repr(code.co_varnames).encode("utf-8")]
    for c in code.co_consts:
        if hasattr(c, "co_code") and depth < 6:
            parts.append(_code_bytes(c, depth + 1))
        else:
            parts.append(_const_repr(c).encode("utf-8"))
    return b"|".join(parts)


def _callable_id(fn):
    if fn is None:
        return "<none>"
    f = getattr(fn, "__func__", fn)
    name = (f"{getattr(f, '__module__', '?')}."
            f"{getattr(f, '__qualname__', getattr(f, '__name__', type(f).__name__))}")
    code = getattr(f, "__code__", None)
    if code is None:
        return name + "|<nocode>"
    return f"{name}|{hashlib.sha256(_code_bytes(code)).hexdigest()[:12]}"


def code_fingerprint():
    d = {"bark_splice_ast": _ast_sha256(getattr(C, "__file__", None)),
         "pairs_data_ast": None}
    try:
        import en_cross_culture_names as _D
        d["pairs_data_ast"] = _ast_sha256(getattr(_D, "__file__", None))
    except Exception as e:
        d["pairs_data_ast"] = f"<import失敗:{type(e).__name__}>"
    for nm in AUDIO_IMPL_FUNCS:
        d[f"fn:{nm}"] = _callable_id(getattr(C, nm, None))
    for cls, meth in AUDIO_IMPL_METHODS:
        d[f"fn:{cls}.{meth}"] = _callable_id(getattr(getattr(C, cls, None), meth, None))
    return d


def driver_ast():
    return _ast_sha256(os.path.abspath(__file__))


def cond_input_fp(pair, cond, fp):
    """その wav を作った入力そのもののハッシュ (wav 1 本ごとに台帳へ書く)."""
    if cond == "bl":
        payload = ["bl", pair["text_a"]]
    elif cond == "tgt":
        payload = ["tgt", pair["text_b"]]
    else:
        payload = ["inj", pair["text_a"], pair["word_a"],
                   pair["text_b"], pair["word_b"], int(fp.get("layer", -1))]
    return _sha256_obj({"pair_idx": int(pair["pair_idx"]), "cond": cond,
                        "payload": payload,
                        "voice_preset": fp.get("voice_preset"),
                        "preset_sha256": fp.get("preset_sha256")})


def want_input_fps(pairs, fp):
    return {(int(p["pair_idx"]), c): cond_input_fp(p, c, fp)
            for p in pairs for c in COND}


# 音のビットを決める設定。これが変われば前の wav は使えない。
AUDIO_FP_STATIC_KEYS = (
    "lang", "layer", "seed", "do_sample", "semantic_temperature", "semantic_top_k",
    "coarse_temperature", "coarse_top_k", "fine_temperature",
    "semantic_max_new_tokens", "min_eos_p", "min_sec", "model_id", "dtype",
    "attn_implementation", "device_type", "voice_preset", "splice_target",
    "splice_len_policy", "splice_impl_sha256",
)
# モデルを読んだあとでないと分からないもの。
AUDIO_FP_DYNAMIC_KEYS = ("model_revision", "preset_sha256", "num_layers",
                         "sampling_rate")
# 再開時に欠けていたら停止する鍵 (指紋を持たない旧版の progress.json を弾く)。
CORE_FP_KEYS = ("layer", "seed", "model_id", "voice_preset", "splice_target",
                "code_fp_hash", "wav_dir")
# 食い違っても停止しない鍵 (警告だけ)。
SOFT_FP_KEYS = ("transformers_version", "asr_model", "run_asr", "device_type",
                "driver_ast")
# 実装の食い違い。停止しないが、その指紋で作られていない wav は信用しない。
CODE_FP_KEYS = ("code_fp_hash", "code_fp")


def static_fingerprint(args):
    """モデルを読む前に決まる分の指紋。GPU を掴む前に照合できる."""
    code_fp = code_fingerprint()
    return {
        "lang": args.lang,
        "layer": int(args.layer),
        "seed": int(args.seed),
        "do_sample": bool(args.do_sample),
        "semantic_temperature": float(args.semantic_temperature),
        "semantic_top_k": int(args.semantic_top_k),
        "coarse_temperature": float(args.coarse_temperature),
        "coarse_top_k": int(args.coarse_top_k),
        "fine_temperature": float(args.fine_temperature),
        "semantic_max_new_tokens": int(args.semantic_max_new_tokens),
        "min_eos_p": (None if args.min_eos_p is None else float(args.min_eos_p)),
        "min_sec": float(args.min_sec),
        "model_id": str(args.model_id),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        # CPU と GPU では同じ seed でもビットが変わる。
        "device_type": str(args.device).split(":")[0],
        "voice_preset": str(args.voice_preset),
        "splice_target": f"model.semantic.layers[{int(args.layer)}].output",
        "splice_len_policy": "token_wise|mean_pool_broadcast",
        # splice を作っている実体そのものを見る (層番号から組み立て直すだけの
        # splice_target では、実装を書き換えても 1 ビットも動かない)。
        "splice_impl_sha256": _sha256_obj(
            {k: v for k, v in code_fp.items()
             if "splice" in k or k == "bark_splice_ast"}),
        "code_fp_hash": _sha256_obj(code_fp),
        "code_fp": code_fp,
        "driver_ast": driver_ast(),
        "asr_model": str(args.asr_model),
        "run_asr": bool(args.run_asr),
        # 出力先。rows と wav の対応が変わるので照合対象に入れる。
        "wav_dir": os.path.abspath(args.wav_dir) if args.wav_dir else None,
        "transformers_version": _tf_version(),
    }


def audio_fp_view(fp, dynamic=False):
    keys = AUDIO_FP_STATIC_KEYS + (AUDIO_FP_DYNAMIC_KEYS if dynamic else ())
    return {k: fp.get(k) for k in keys}


def audio_fp_hashes(fp):
    st = _sha256_obj(audio_fp_view(fp, dynamic=False))
    full = None
    if all(fp.get(k) is not None for k in AUDIO_FP_DYNAMIC_KEYS):
        full = _sha256_obj(audio_fp_view(fp, dynamic=True))
    return st, full


def compare_fingerprint(old, new):
    """(停止すべき食い違い, 停止しない食い違い, 実装の食い違い)."""
    hard, soft, code = [], [], []
    if not isinstance(old, dict):
        return [("<fingerprint>", "<無し>", "<有り>")], [], []
    for k in CORE_FP_KEYS:
        if k not in old:
            hard.append((k, "<欠落>", new.get(k)))
    for k in sorted(set(old) | set(new)):
        if k in old and k in new and old[k] != new[k]:
            if k in CODE_FP_KEYS:
                code.append((k, old[k], new[k]))
            elif k in SOFT_FP_KEYS:
                soft.append((k, old[k], new[k]))
            else:
                hard.append((k, old[k], new[k]))
    return hard, soft, code


def code_fp_diff(old_fp, new_fp):
    o = (old_fp or {}).get("code_fp") or {}
    n = (new_fp or {}).get("code_fp") or {}
    return [(k, o.get(k, "<欠落>"), n.get(k, "<欠落>"))
            for k in sorted(set(o) | set(n)) if o.get(k, "<欠落>") != n.get(k, "<欠落>")]


def check_resume_fingerprint(prev, fp, n_prev_rows, force):
    """食い違えば SystemExit。force=True なら警告文字列を返して続行."""
    if n_prev_rows == 0:
        return []
    hard, soft, code = compare_fingerprint(prev.get("fingerprint"), fp)
    out = []
    for k, o, n in soft:
        out.append(f"再開時に {k} が変わっている (前回={o!r} 今回={n!r})。停止はしていない。")
        print(f"[resume][warn] {out[-1]}", flush=True)
    if code:
        det = code_fp_diff(prev.get("fingerprint"), fp)
        out.append(
            f"再開しようとしている progress.json は **別の実装**で作られている "
            f"(code_fp_hash 前回={prev.get('fingerprint', {}).get('code_fp_hash')!r} "
            f"今回={fp.get('code_fp_hash')!r})。変わった項目 {[k for k, _o, _n in det]}。"
            f"停止はしないが、前の実装で作った wav は信用せず作り直す。")
        print("[resume][warn] " + out[-1], flush=True)
        for k, o, n in det[:8]:
            print(f"    {k}: 前回={o!r}  今回={n!r}", flush=True)
    if not hard:
        return out
    lines = [f"    {k}: 前回={o!r}  今回={n!r}" for k, o, n in hard]
    body = ("再開しようとしている progress.json の実行設定が今回と食い違う。\n"
            + "\n".join(lines) + "\n"
            f"  前回の行数 = {n_prev_rows}。このまま続けると、前の設定で作った wav を\n"
            "  使い回したまま results.json には今回の設定が書かれる\n"
            "  (2026-09-17 の LongCat 統制実験と同じ事故)。")
    if not force:
        raise SystemExit(
            "[resume] " + body + "\n"
            "  対処: 出力先を分ける (--out_dir と --wav_dir を設定ごとに変える) か、\n"
            "        --overwrite で作り直すか、承知のうえなら\n"
            "        --force_resume_mismatch を付けること。")
    print("[resume][warn] " + body, flush=True)
    out.append(f"再開時に設定が食い違ったまま --force_resume_mismatch で続行した: "
               f"{[(k, o, n) for k, o, n in hard]}")
    return out


# --------------------------------------------------------------------------
# wav_dir の台帳
#
# **採点されるのは results.json ではなく wav_dir の中身。**
# 指紋を out_dir にしか置かないと、--out_dir を新しくして --wav_dir だけ別条件の
# 完成品に向ければ、何の警告も出さずに別条件の音声を採点できてしまう。
#
# **ファイルは消さない / 動かさない**。TTS26 は保護パス配下なので、
# 出所を確かめられない wav は「信用しない (= 作り直す)」とだけ扱い、
# 作り直せないもの (今回の範囲外) は summary と warnings に必ず出す。
# --------------------------------------------------------------------------
def wav_key(pidx, cond):
    return f"s{int(pidx):04d}_{cond}"


def manifest_path(wav_dir):
    return os.path.join(wav_dir, "_wav_manifest.json")


def load_manifest(wav_dir):
    p = manifest_path(wav_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        raise SystemExit(f"[wav_dir] 台帳 {p} を読めなかった: {e}\n"
                         f"  中身を確かめること (壊れた台帳を空として黙って通さない)。")


def new_manifest(wav_dir, fp, st_hash, full_hash):
    return {"schema": MANIFEST_SCHEMA, "wav_dir": os.path.abspath(wav_dir),
            "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
            "audio_fingerprint": audio_fp_view(fp, dynamic=True),
            "code_fp_hash": fp.get("code_fp_hash"), "wavs": {}, "producers": []}


def save_manifest(man, wav_dir):
    if man is None:
        return
    p = manifest_path(wav_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def check_manifest_header(man, fp, st_hash, full_hash, args, n_wavs, phase):
    """台帳の見出しと今回の指紋を突き合わせる。食い違えば SystemExit."""
    w = []
    if man is None:
        if n_wavs:
            msg = (f"wav_dir に {n_wavs} 本の wav があるのに台帳 "
                   f"{manifest_path(args.wav_dir)} が無い。どの設定で作られたか分からない。")
            if not args.adopt_unmanifested_wavs:
                raise SystemExit(
                    f"[wav_dir] {msg}\n"
                    f"  対処: 出力先を分けるか、--overwrite で作り直すか、\n"
                    f"        承知のうえなら --adopt_unmanifested_wavs を付けること。")
            w.append(msg + " --adopt_unmanifested_wavs で今回のものとして扱った。")
        return w
    old_st = man.get("audio_fp_static_hash")
    if old_st is not None and old_st != st_hash:
        diff = [(k, man.get("audio_fingerprint", {}).get(k, "<欠落>"), fp.get(k))
                for k in AUDIO_FP_STATIC_KEYS
                if man.get("audio_fingerprint", {}).get(k, "<欠落>") != fp.get(k)]
        raise SystemExit(
            f"[wav_dir] {args.wav_dir} の台帳は別の設定で作られている "
            f"(設定の指紋 台帳={old_st} 今回={st_hash})。\n"
            + "\n".join(f"    {k}: 台帳={o!r}  今回={n!r}" for k, o, n in diff[:10]) + "\n"
            "  この wav_dir をそのまま使うと、別条件の音声が今回の結果として採点される。\n"
            "  対処: --wav_dir を条件ごとに分けること。")
    old_full = man.get("audio_fp_full_hash")
    if full_hash is not None and old_full is not None and old_full != full_hash:
        raise SystemExit(
            f"[wav_dir] 台帳のモデル側の指紋が今回と違う (台帳={old_full} 今回={full_hash})。"
            f"モデルの版か preset か dtype が変わっている。--wav_dir を分けること。")
    if man.get("code_fp_hash") and man["code_fp_hash"] != fp.get("code_fp_hash"):
        w.append(f"wav_dir の台帳は別の実装で作られている "
                 f"(実装の指紋 台帳={man['code_fp_hash']} 今回={fp.get('code_fp_hash')})。"
                 f"その実装で作られた wav は信用せず作り直す。")
    return w


def audit_wav_dir(wav_dir, man, st_hash, full_hash, scope, args, want_input,
                  code_hash=None):
    """台帳と突き合わせて「信用できる wav」の集合を返す (ファイルは動かさない).

    信用の条件は 4 つすべて:
      1. 台帳に載っている
      2. その行の設定の指紋が今回と一致する
      3. ディスク上のファイルの sha256 が台帳の記録と一致する
      4. 入力の指紋 (文・実体語・層) が今回と一致する
    実装の指紋 (code_hash) が違うものも信用しない (= 作り直す)。
    """
    trusted, untrusted, w = set(), [], []
    wavs = (man or {}).get("wavs") or {}
    on_disk = scan_wav_dir(wav_dir)
    out_of_scope = [(p, c) for p, c, _ in on_disk if p not in scope]
    for pidx, cond, path in on_disk:
        if pidx not in scope:
            continue
        e = wavs.get(wav_key(pidx, cond))
        why = None
        if not isinstance(e, dict):
            why = "台帳に無い"
        elif e.get("audio_fp_static_hash") != st_hash:
            why = "設定の指紋が違う"
        elif (full_hash is not None and e.get("audio_fp_full_hash") is not None
              and e.get("audio_fp_full_hash") != full_hash):
            why = "モデル側の指紋が違う"
        elif code_hash is not None and e.get("code_fp_hash") != code_hash:
            why = "実装の指紋が違う"
        elif want_input.get((pidx, cond)) is not None and \
                e.get("input_fp") != want_input.get((pidx, cond)):
            why = "入力 (文 / 実体語 / 層) の指紋が違う"
        else:
            try:
                if _sha256_file(path) != e.get("sha256"):
                    why = "ファイルの中身が台帳の記録と違う"
            except Exception as ex:
                why = f"読めない ({type(ex).__name__})"
        if why is None:
            trusted.add((pidx, cond))
        else:
            untrusted.append((pidx, cond, why))
    if untrusted:
        kinds = {}
        for _p, _c, why in untrusted:
            kinds[why] = kinds.get(why, 0) + 1
        w.append(f"出所を確かめられない wav が {len(untrusted)} 本ある {kinds}。"
                 f"今回の範囲のものは作り直す (ファイルは消さない)。")
    if out_of_scope:
        w.append(
            f"今回の範囲 (--n {args.n}) の外にある wav が {len(out_of_scope)} 本 "
            f"wav_dir に残っている (例 {[f's{p:04d}_{c}' for p, c in out_of_scope[:5]]})。"
            f"**採点器は --n までしか読まないが、範囲を広げると古い音声が混ざる。**"
            f"消していない。必要なら出力先を分けること。")
    return trusted, untrusted, w


def register_wav(man, pidx, cond, path, st_hash, full_hash, out_dir, run_id, g,
                 code_hash=None, input_fp=None):
    sha = _sha256_file(path)
    man.setdefault("wavs", {})[wav_key(pidx, cond)] = {
        "file": os.path.basename(path), "sha256": sha,
        "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
        "code_fp_hash": code_hash, "input_fp": input_fp,
        "run_id": run_id, "out_dir": out_dir,
        "sec": g.get("sec"), "n_semantic": g.get("n_semantic"),
        "sem_fp": g.get("sem_fp"), "stop_reason": g.get("stop_reason"),
    }
    return sha


# --------------------------------------------------------------------------
# 引数
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="suno/bark で BL / TGT / INJ を生成する (SpeechSplice)")
    p.add_argument("--lang", default="en", choices=["en", "ja", "zh"],
                   help="300 対は data/<lang>_cross_culture_names.py から作る。"
                        "JA / ZH は書記素のまま通ることを確認済み "
                        "(scripts/bark/probe_ja_zh_text_cpu.py)。"
                        "話者は --voice_preset で言語に合わせること "
                        "(v2/ja_speaker_0 / v2/zh_speaker_0 など)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--layer", type=int, default=C.DEFAULT_LAYER,
                   help=f"差し替える層。格子 {C.SWEEP_LAYERS} (既定 {C.DEFAULT_LAYER} = L-2)。"
                        f"Bark は L5 以降で prompt をほとんど読まないので、"
                        f"効きを見るなら {C.DIAG_LAYERS} も掃くこと")
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--voice_preset", default=C.DEFAULT_VOICE_PRESET,
                   help="話者を固定する preset。渡さないと 3 条件で声が変わり "
                        "採点器の前提が崩れる")
    p.add_argument("--no_voice_preset", dest="voice_preset", action="store_const",
                   const=None, help="preset 無し (話者が毎回変わる。推奨しない)")

    p.add_argument("--model_id", default=C.DEFAULT_MODEL_ID)
    p.add_argument("--allow_download", action="store_true",
                   help="ローカルに無いときだけ取得する (allow_patterns 付きで約 4.2 GiB)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn_implementation", default="eager", choices=["eager"],
                   help="Bark は eager のみ (BARK_ATTENTION_CLASSES に sdpa が無い)")
    p.add_argument("--hf_home", default=None)
    p.add_argument("--no_strict_weights", dest="strict_weights", action="store_false",
                   help="missing_keys があっても続ける (既定は止める)")

    p.add_argument("--semantic_temperature", type=float, default=C.DEFAULT_SEMANTIC_TEMPERATURE)
    p.add_argument("--semantic_top_k", type=int, default=C.DEFAULT_SEMANTIC_TOP_K)
    p.add_argument("--coarse_temperature", type=float, default=C.DEFAULT_COARSE_TEMPERATURE)
    p.add_argument("--coarse_top_k", type=int, default=C.DEFAULT_COARSE_TOP_K)
    p.add_argument("--fine_temperature", type=float, default=C.DEFAULT_FINE_TEMPERATURE)
    p.add_argument("--semantic_max_new_tokens", type=int,
                   default=C.DEFAULT_SEMANTIC_MAX_NEW_TOKENS,
                   help=f"上限 {C.SEMANTIC_MAX_NEW_TOKENS_CAP}。超えると位置埋め込みが範囲外")
    p.add_argument("--min_eos_p", type=float, default=C.DEFAULT_MIN_EOS_P,
                   help="本家 suno の推奨 0.2。generation_config には無いので明示的に渡す")
    p.add_argument("--no_min_eos_p", dest="min_eos_p", action="store_const", const=None)
    p.add_argument("--greedy", dest="do_sample", action="store_false",
                   help="semantic / coarse を貪欲復号にする (既定は公式どおり sampling)")
    p.add_argument("--min_sec", type=float, default=0.5)
    p.add_argument("--fail_on_cap", action="store_true",
                   help="EOS が出ないまま上限まで走った対を error にする (既定は警告のみ)")

    p.add_argument("--no_asr", dest="run_asr", action="store_false",
                   help="source_in_bl / target_in_inj を測らない")
    p.add_argument("--asr_model", default="openai/whisper-large-v3")
    p.add_argument("--asr_only", action="store_true",
                   help="生成せず、台帳と照合できた既存 wav を書き起こして率だけ出し直す")
    p.add_argument("--no_kappa", dest="collect_kappa", action="store_false")

    p.add_argument("--overwrite", action="store_true", help="再開せず最初からやり直す")
    p.add_argument("--force_resume_mismatch", action="store_true",
                   help="設定が食い違ったまま再開する (事故のもと。既定は停止)")
    p.add_argument("--adopt_unmanifested_wavs", action="store_true",
                   help="台帳の無い wav を今回のものとして扱う (既定は停止)")
    p.add_argument("--max_drop_rate", type=float, default=DEFAULT_MAX_DROP_RATE)

    p.add_argument("--null_check_pairs", type=int, default=12,
                   help="最初の N 対で INJ が BL と同一なら停止する (0 で無効)")
    p.add_argument("--allow_null_layer", action="store_true",
                   help="INJ が BL と同一でも続行する (層掃引で null を記録したいとき)")

    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    return p


def _record_gen(row, cond, g):
    """1 条件ぶんの生成統計を row に書く (3 条件とも同じ形)."""
    row[f"{cond}_status"] = "ok"
    row[f"{cond}_sec"] = g["sec"]
    row[f"{cond}_n_semantic"] = g["n_semantic"]
    row[f"{cond}_eos_found"] = g["eos_found"]
    row[f"{cond}_hit_cap"] = g["hit_cap"]
    row[f"{cond}_truncated_at_cap"] = g["truncated_at_cap"]
    row[f"{cond}_stop_reason"] = g["stop_reason"]
    row[f"{cond}_sem_fp"] = g["sem_fp"]


def main():
    args = build_argparser().parse_args()
    if args.hf_home:
        C.setup_hf_home(args.hf_home)
    os.chdir(PROJECT_ROOT)

    if args.out_dir is None:
        args.out_dir = os.path.join("eval_results",
                                    f"bark_{args.lang}_n{args.n}_L{args.layer}")
    if args.wav_dir is None:
        args.wav_dir = os.path.join(args.out_dir, "wav")
    args.out_dir = os.path.abspath(args.out_dir)
    args.wav_dir = os.path.abspath(args.wav_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    progress_path = os.path.join(args.out_dir, "progress.json")
    results_path = os.path.join(args.out_dir, "results.json")

    pairs = load_pairs(args.lang, args.n)
    print(f"[info] {args.lang} cross-culture: {len(pairs)} pairs", flush=True)
    print(f"[info] run_id={run_id}", flush=True)
    print(f"[info] out_dir={args.out_dir}", flush=True)
    print(f"[info] wav_dir={args.wav_dir}", flush=True)
    print(f"[info] HF_HOME={os.environ['HF_HOME']} "
          f"NUMBA_CACHE_DIR={os.environ.get('NUMBA_CACHE_DIR')}", flush=True)
    print(f"[info] layer={args.layer} (格子 {C.SWEEP_LAYERS} / 追加掃引 {C.DIAG_LAYERS}) "
          f"preset={args.voice_preset} min_eos_p={args.min_eos_p}", flush=True)
    if args.layer not in C.SWEEP_LAYERS and args.layer not in C.DIAG_LAYERS:
        print(f"[warn] layer={args.layer} は格子にも追加掃引にも入っていない", flush=True)
    if args.voice_preset is None:
        print("[warn] --no_voice_preset。話者を固定するものが無いので BL / TGT / INJ で "
              "声が変わる。採点器の前提 (BL と TGT は実体語だけ違う) が崩れる。", flush=True)

    # ---------------- 実行設定の指紋 ----------------
    # モデルを読む前に照合する。GPU を確保してから止めるのは無駄。
    fp = static_fingerprint(args)
    st_hash, full_hash = audio_fp_hashes(fp)          # full_hash はまだ None
    code_hash = fp["code_fp_hash"]
    print(f"[info] 設定の指紋={st_hash} 実装の指紋={code_hash} driver={fp['driver_ast']}",
          flush=True)
    resume_warnings, prov_warnings = [], []

    # ---------------- 再開 (out_dir 側) ----------------
    rows_by_idx = {}
    if os.path.isfile(progress_path) and not args.overwrite:
        try:
            with open(progress_path) as f:
                prev = json.load(f)
        except Exception as e:
            raise SystemExit(f"[resume] {progress_path} を読めなかった: {e}\n"
                             f"  作り直すなら --overwrite を付けること。")
        prev_rows = prev.get("rows", [])
        resume_warnings = check_resume_fingerprint(
            prev, fp, len(prev_rows), args.force_resume_mismatch)
        for r in prev_rows:
            rows_by_idx[int(r["pair_idx"])] = r
        print(f"[resume] {progress_path} から {len(rows_by_idx)} 行を読んだ", flush=True)
    elif os.path.isfile(progress_path) and args.overwrite:
        print(f"[info] --overwrite。{progress_path} を無視して作り直す", flush=True)

    attempted = [int(p["pair_idx"]) for p in pairs]
    scope = set(attempted)
    want_input = want_input_fps(pairs, fp)

    man = load_manifest(args.wav_dir)
    n_wavs_on_disk = len(scan_wav_dir(args.wav_dir))
    prov_warnings += check_manifest_header(
        man, fp, st_hash, None, args, n_wavs_on_disk,
        phase="final" if args.asr_only else "static")

    def provenance_block(trusted, untrusted):
        return {"wav_provenance": {
            "wav_dir": args.wav_dir, "manifest": manifest_path(args.wav_dir),
            "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
            "code_fp_hash": code_hash, "code_fp_checked": (not args.asr_only),
            "driver_ast": fp.get("driver_ast"), "run_id": run_id,
            "n_wavs_on_disk": len(scan_wav_dir(args.wav_dir)),
            "n_trusted": len(trusted), "n_untrusted": len(untrusted),
            "untrusted": [f"s{p:04d}_{c}:{why}" for p, c, why in untrusted][:50],
        }}

    # ======================================================================
    # --asr_only: 生成しない。台帳と照合できた wav だけを書き起こす。
    # ======================================================================
    if args.asr_only:
        import soundfile as sf
        trusted, untrusted, w = audit_wav_dir(
            args.wav_dir, man, st_hash, None, scope, args, want_input, code_hash=None)
        if args.adopt_unmanifested_wavs and man is None:
            trusted = {(p, c) for p, c, _ in scan_wav_dir(args.wav_dir)}
        prov_warnings += w
        asr = Asr(device=args.device, model_id=args.asr_model)
        # 前の行から引き継がない鍵 (今回の書き起こしで必ず作り直す)。
        judged = ({"status", "fail_stage", "fail_conds", "err", "trace",
                   "asr_fail_conds", "source_in_bl", "target_in_tgt",
                   "target_in_inj", "source_in_inj"}
                  | {f"{c}_transcript" for c in COND}
                  | {f"{c}_asr_err" for c in COND}
                  | {f"{c}_status" for c in COND})
        for i, pair in enumerate(pairs):
            pidx = int(pair["pair_idx"])
            prev_row = rows_by_idx.get(pidx) or {}
            row = {"pair_idx": pidx, "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "text_a": pair["text_a"], "text_b": pair["text_b"],
                   "layer": args.layer, "seed": args.seed, "run_id": run_id}
            # **条件ごとに独立に扱う。** 1 条件の欠落で他の条件を巻き添えにしない。
            bad = [c for c in COND if (pidx, c) not in trusted]
            for k, v in prev_row.items():
                if k not in judged and k.split("_")[0] not in bad:
                    row.setdefault(k, v)
            paths = wav_paths(args.wav_dir, pidx)
            tr, failed = {}, []
            for c in COND:
                if c in bad:
                    row[f"{c}_status"] = "error"
                    row[f"{c}_err"] = "wav の出所を台帳で確かめられない"
                    continue
                row[f"{c}_status"] = "ok"
                try:
                    a, sr = sf.read(paths[c], dtype="float32")
                    tr[c] = transcribe_cond(asr, c, a, sr, args.lang, pair=pair)
                    row[f"{c}_transcript"] = tr[c]
                except Exception as e:
                    failed.append(c)
                    row[f"{c}_asr_err"] = f"{type(e).__name__}: {e}"[:200]
            if failed:
                row["asr_fail_conds"] = failed
            if "bl" in tr:
                row["source_in_bl"] = in_text(pair["word_a"], tr["bl"], args.lang)
            if "tgt" in tr:
                row["target_in_tgt"] = in_text(pair["word_b"], tr["tgt"], args.lang)
            if "inj" in tr:
                row["target_in_inj"] = in_text(pair["word_b"], tr["inj"], args.lang)
                row["source_in_inj"] = in_text(pair["word_a"], tr["inj"], args.lang)
            if bad:
                row["status"] = "error"
                row["fail_stage"] = "untrusted_wav"
                row["fail_conds"] = sorted(bad)
            elif failed:
                row["status"] = "error"
                row["fail_stage"] = "asr"
            else:
                row["status"] = "ok"
            rows_by_idx[pidx] = row
            if (i + 1) % args.log_every == 0:
                print(f"  [asr {i + 1}/{len(pairs)}]", flush=True)

        rows = [rows_by_idx[k] for k in sorted(rows_by_idx)]
        ex = {"mode": "asr_only", "asr_model": args.asr_model, "run_id": run_id,
              "fingerprint": fp}
        ex.update(provenance_block(trusted, untrusted))
        ws = _dedupe(list(resume_warnings) + list(prov_warnings))
        if ws:
            ex["warnings"] = ws
        summary = summarize(rows, args, extra=ex, attempted=attempted)
        with open(results_path + ".tmp", "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
        os.replace(results_path + ".tmp", results_path)
        report(summary)
        print(f"[done] {results_path}")
        return

    # ======================================================================
    # 生成
    # ======================================================================
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[info] loading {args.model_id} (dtype={args.dtype}, device={args.device}) ...",
          flush=True)
    t0 = time.time()
    bundle = C.load_bark(args.model_id, device=args.device, dtype=dtype,
                         allow_download=args.allow_download,
                         strict_weights=args.strict_weights,
                         voice_preset=args.voice_preset,
                         attn_implementation=args.attn_implementation)
    print(f"[info] loaded in {time.time() - t0:.1f}s from {bundle.source}", flush=True)
    print(f"[info] semantic {bundle.num_layers} 層 / hidden {bundle.hidden_size} / "
          f"sr {bundle.sampling_rate} / preset sha {bundle.preset_sha256}", flush=True)
    print(f"[info] loading_info: missing={len(bundle.loading_info['missing_keys'])} "
          f"unexpected={len(bundle.loading_info['unexpected_keys'])}", flush=True)
    C.assert_splice_target(bundle, args.layer)

    # モデルを読んで初めて分かる分を指紋に足し、もう一度照合する。
    fp["model_revision"] = os.path.basename(str(bundle.source).rstrip("/"))
    fp["preset_sha256"] = bundle.preset_sha256
    fp["num_layers"] = int(bundle.num_layers)
    fp["sampling_rate"] = int(bundle.sampling_rate)
    st_hash2, full_hash = audio_fp_hashes(fp)
    assert st_hash2 == st_hash
    if os.path.isfile(progress_path) and not args.overwrite and rows_by_idx:
        try:
            with open(progress_path) as f:
                _prev = json.load(f)
        except Exception:
            _prev = {}
        resume_warnings += check_resume_fingerprint(
            _prev, fp, len(rows_by_idx), args.force_resume_mismatch)
    prov_warnings += check_manifest_header(man, fp, st_hash, full_hash, args,
                                           n_wavs_on_disk, phase="final")

    trusted, untrusted, w = audit_wav_dir(
        args.wav_dir, man, st_hash, full_hash, scope, args, want_input,
        code_hash=code_hash)
    prov_warnings += w
    if man is None:
        man = new_manifest(args.wav_dir, fp, st_hash, full_hash)
        if args.adopt_unmanifested_wavs:
            for pidx, cond, path in scan_wav_dir(args.wav_dir):
                register_wav(man, pidx, cond, path, st_hash, full_hash,
                             args.out_dir, run_id, {}, code_hash=code_hash,
                             input_fp=want_input.get((pidx, cond)))
                trusted.add((pidx, cond))
    man["schema"] = MANIFEST_SCHEMA
    man["wav_dir"] = args.wav_dir
    man["audio_fp_static_hash"] = st_hash
    man["audio_fp_full_hash"] = full_hash
    man["audio_fingerprint"] = audio_fp_view(fp, dynamic=True)
    man["code_fp_hash"] = code_hash
    man.setdefault("wavs", {})
    for k, e in list(man["wavs"].items()):
        fn = e.get("file", k + ".wav") if isinstance(e, dict) else k + ".wav"
        if not os.path.isfile(os.path.join(args.wav_dir, fn)):
            man["wavs"].pop(k, None)     # ディスクに無いものを台帳に残さない
    man.setdefault("producers", []).append(
        {"run_id": run_id, "out_dir": args.out_dir,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
         "code_fp_hash": code_hash, "n": args.n, "layer": args.layer})
    man["producers"] = man["producers"][-20:]
    save_manifest(man, args.wav_dir)
    print(f"[wav_dir] 台帳 {manifest_path(args.wav_dir)} / 設定 {st_hash}/{full_hash} / "
          f"照合済み {len(trusted)} 本 / 出所不明 {len(untrusted)} 本", flush=True)

    asr = None
    if args.run_asr:
        print(f"[info] loading ASR {args.asr_model} ...", flush=True)
        asr = Asr(device=args.device, model_id=args.asr_model)

    gen = C.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      semantic_temperature=args.semantic_temperature,
                      semantic_top_k=args.semantic_top_k,
                      coarse_temperature=args.coarse_temperature,
                      coarse_top_k=args.coarse_top_k,
                      fine_temperature=args.fine_temperature,
                      semantic_max_new_tokens=args.semantic_max_new_tokens,
                      min_eos_p=args.min_eos_p, min_sec=args.min_sec,
                      fail_on_cap=args.fail_on_cap)
    meta = {"run_id": run_id, "gen": gen.as_dict(), "model_source": bundle.source,
            "num_layers": bundle.num_layers, "sweep_layers": list(C.SWEEP_LAYERS),
            "diag_layers": list(C.DIAG_LAYERS),
            "voice_preset": args.voice_preset, "preset_sha256": bundle.preset_sha256,
            "loading_info": {k: len(v) for k, v in bundle.loading_info.items()}}
    if asr is not None:
        meta["asr_model"] = args.asr_model

    def flush(final=False):
        rows = [rows_by_idx[k] for k in sorted(rows_by_idx)]
        tmp = progress_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"rows": rows, "args": vars(args), "fingerprint": fp,
                       "run_id": run_id}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, progress_path)
        save_manifest(man, args.wav_dir)
        if final:
            ex = dict(meta)
            ex["fingerprint"] = fp
            ex.update(provenance_block(trusted, untrusted))
            ws = _dedupe(list(resume_warnings) + list(prov_warnings) + list(run_warnings))
            if ws:
                ex["warnings"] = ws
            summary = summarize(rows, args, extra=ex, attempted=attempted)
            tmp = results_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"summary": summary, "rows": rows}, f,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, results_path)
            return summary
        return None

    # ---------------- 本体 ----------------
    #
    # **3 条件はそれぞれ独立に回し、成功した条件はその場で保存して記録する。**
    # 1 対 = 1 つの try にすると、INJ が落ちただけで生成し終えている BL と TGT の
    # 記録まで消え、source_in_bl_rate が INJ の失敗で押し下げられる。
    run_warnings = []
    n_done = n_skip = n_err = 0
    n_cmp = n_same = 0          # BL と INJ の意味トークン列の比較
    t_start = time.time()
    interrupted = None
    try:
        for i, pair in enumerate(pairs):
            pidx = int(pair["pair_idx"])
            # 再開の判定は「行が ok」+「3 本とも台帳と照合済み」。
            # ファイルの有無だけでは飛ばさない (どの設定で作られたか分からないため)。
            if (not args.overwrite and pidx in rows_by_idx
                    and rows_by_idx[pidx].get("status") == "ok"
                    and all((pidx, c) in trusted for c in COND)):
                n_skip += 1
                prev = rows_by_idx[pidx]
                if prev.get("inj_differs_from_bl") is not None:
                    n_cmp += 1
                    n_same += int(prev["inj_differs_from_bl"] is False)
                continue
            paths = wav_paths(args.wav_dir, pidx)
            row = {"pair_idx": pidx, "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "text_a": pair["text_a"], "text_b": pair["text_b"],
                   "layer": args.layer, "seed": args.seed, "run_id": run_id}
            gens, fail_conds = {}, []

            def _try(cond, fn):
                try:
                    g = fn()
                except Exception as e:
                    import traceback
                    row[f"{cond}_status"] = "error"
                    row[f"{cond}_err"] = f"{type(e).__name__}: {e}"[:300]
                    row[f"{cond}_trace"] = traceback.format_exc()[-400:]
                    fail_conds.append(cond)
                    print(f"  [{pidx}] {cond} error: {row[f'{cond}_err']}", flush=True)
                    if args.stop_on_error:
                        row["status"] = "error"
                        row["fail_conds"] = sorted(set(fail_conds))
                        row["fail_stage"] = cond
                        rows_by_idx[pidx] = row
                        flush()
                        raise
                    return None
                gens[cond] = g
                _record_gen(row, cond, g)
                return g

            _try("bl", lambda: C.synthesize(bundle, pair["text_a"], gen=gen))
            _try("tgt", lambda: C.synthesize(bundle, pair["text_b"], gen=gen))
            _try("inj", lambda: C.splice_synthesize(
                bundle, pair["text_a"], pair["word_a"], pair["text_b"], pair["word_b"],
                layer=args.layer, gen=gen, collect_kappa=args.collect_kappa))

            if "inj" in gens:
                for k in ("fired", "decode_calls", "foreign_calls", "splice_mode",
                          "equal_len", "m_src_entity_tokens", "n_tgt_entity_tokens",
                          "s_span", "t_span", "s_tokens", "t_tokens", "prompt_len",
                          "hidden_index", "prefill_hidden_max_abs_diff",
                          "kappa_at_splice", "kappa_last_layer", "kappa_by_layer"):
                    if k in gens["inj"]:
                        row[k] = gens["inj"][k]

            # ---- splice が音に出たか (Bark で最も重要な検査) ----
            check_failed = None
            if "bl" in gens and "inj" in gens:
                if int(gens["bl"]["prompt_len"]) != int(gens["inj"]["prompt_len"]):
                    check_failed = "prompt_len_check"
                    row["err"] = (f"BL と INJ の prompt 長が違う "
                                  f"({gens['bl']['prompt_len']} vs {gens['inj']['prompt_len']})")
                    print(f"  [{pidx}] {row['err']}", flush=True)
                # **意味トークン列の指紋**で見る。波形の指紋では駄目
                # (coarse / fine の乱数差で「違う」と出てしまい検出にならない)。
                same = (gens["inj"]["sem_fp"] == gens["bl"]["sem_fp"])
                row["inj_differs_from_bl"] = bool(not same)
                n_cmp += 1
                n_same += int(same)

            if check_failed is not None:
                for c in COND:
                    row[f"{c}_status"] = "error"
                    row.setdefault(f"{c}_err", f"{check_failed}: wav を保存していない")

            # ---- 保存。成功した条件だけをその場で書き、台帳にも載せる ----
            if check_failed is None:
                for c in list(gens):
                    try:
                        _, dur, qc = C.save_wav(paths[c], gens[c]["audio"],
                                                bundle.sampling_rate)
                        row[f"{c}_peak"] = qc["peak"]
                        # save_wav は [-1,1] に丸めてから PCM_16 で書く。ASR は丸める前の
                        # 配列を読み、採点器は丸めたあとの wav を読むので、丸めが起きた対は
                        # 判定と採点で聞いている音が違う。数えて出す。
                        row[f"{c}_n_clipped"] = qc["n_clipped"]
                        row[f"{c}_wav"] = os.path.basename(paths[c])
                        row[f"{c}_wav_sha256"] = register_wav(
                            man, pidx, c, paths[c], st_hash, full_hash,
                            args.out_dir, run_id, gens[c], code_hash=code_hash,
                            input_fp=want_input.get((pidx, c)))
                        trusted.add((pidx, c))
                    except Exception as e:
                        row[f"{c}_status"] = "error"
                        row[f"{c}_err"] = f"save_wav: {type(e).__name__}: {e}"[:300]
                        fail_conds.append(c)
                        trusted.discard((pidx, c))
                        gens.pop(c, None)
                        print(f"  [{pidx}] {c} save_wav error: {row[f'{c}_err']}", flush=True)
                        if args.stop_on_error:
                            rows_by_idx[pidx] = row
                            flush()
                            raise

            # ---- ASR。条件ごとに独立。ASR の失敗は生成の母数を汚さない ----
            asr_failed = []
            if asr is not None:
                tr = {}
                for c in COND:
                    # wav として保存できた条件だけ書き起こす。
                    if c not in gens or row.get(f"{c}_status") != "ok":
                        continue
                    try:
                        tr[c] = transcribe_cond(asr, c, gens[c]["audio"],
                                                bundle.sampling_rate, args.lang, pair=pair)
                        row[f"{c}_transcript"] = tr[c]
                    except Exception as e:
                        asr_failed.append(c)
                        row[f"{c}_asr_err"] = f"{type(e).__name__}: {e}"[:200]
                if "bl" in tr:
                    row["source_in_bl"] = in_text(pair["word_a"], tr["bl"], args.lang)
                if "tgt" in tr:
                    row["target_in_tgt"] = in_text(pair["word_b"], tr["tgt"], args.lang)
                if "inj" in tr:
                    row["target_in_inj"] = in_text(pair["word_b"], tr["inj"], args.lang)
                    row["source_in_inj"] = in_text(pair["word_a"], tr["inj"], args.lang)
                if asr_failed:
                    row["asr_fail_conds"] = asr_failed

            if fail_conds or check_failed:
                row["status"] = "error"
                row["fail_stage"] = check_failed or sorted(set(fail_conds))[0]
                row["fail_conds"] = sorted(set(fail_conds)) or list(COND)
                n_err += 1
            elif asr_failed:
                row["status"] = "error"
                row["fail_stage"] = "asr"
                row["fail_conds"] = asr_failed
                n_err += 1
            else:
                row["status"] = "ok"
                n_done += 1
            rows_by_idx[pidx] = row

            if (i + 1) % args.log_every == 0:
                el = time.time() - t_start
                done = n_done + n_err
                rate = el / done if done else 0.0
                print(f"  [{i + 1}/{len(pairs)}] ok={n_done} err={n_err} skip={n_skip} "
                      f"同一={n_same}/{n_cmp} {rate:.1f}s/対 "
                      f"残り {rate * (len(pairs) - i - 1) / 60:.1f} 分", flush=True)
            if (i + 1) % args.save_every == 0:
                flush()

            # ---- null 層の早期検出 ----
            # 慣例の既定層 (L-2 = 22) は Bark では恒等になる。300 対を回してから
            # 「ΔPER = 0」と記録するのは GPU 時間の無駄なので、最初の数対で止める。
            if (args.null_check_pairs and not args.allow_null_layer
                    and n_cmp >= int(args.null_check_pairs) and n_same == n_cmp):
                flush()
                raise SystemExit(
                    f"[null] 最初の {n_cmp} 対すべてで、INJ の意味トークン列が BL と "
                    f"完全に一致した。層 {args.layer} では splice が下流に届いていない。\n"
                    f"  その音声は BL とビット単位で同じで、ΔPER は必ず 0 になる。\n"
                    f"  Bark は prompt 257 行のうち 244 行がパディングで、テキストの情報が\n"
                    f"  浅い層でパディング行へ移るため、L5 以降は prompt をほとんど読まない。\n"
                    f"  対処: --layer を {C.DIAG_LAYERS} から選んで掃くか、\n"
                    f"        この層の null を結果として記録したいなら --allow_null_layer\n"
                    f"        を付けて続けること (そのとき summary の\n"
                    f"        n_inj_identical_to_bl が {n_cmp} 以上になる)。\n"
                    f"  progress.json は書いた。results.json は書いていない "
                    f"(途中の母数を確定した結果として残さないため)。")
    except SystemExit:
        raise
    except KeyboardInterrupt as e:
        interrupted = e
        print("\n[warn] 中断された。ここまでの行を書き出す。", flush=True)
    except Exception as e:
        interrupted = e
        print(f"\n[warn] 例外で中断: {type(e).__name__}: {e}", flush=True)

    if n_cmp and n_same:
        run_warnings.append(
            f"INJ の意味トークン列が BL と一致した対が {n_same}/{n_cmp}。"
            f"その対の ΔPER は必ず 0 になる。層 {args.layer} の効きを疑うこと。")
    if n_skip:
        run_warnings.append(
            f"{n_skip} 対は前回の run が作った wav をそのまま使った "
            f"(台帳で設定・実装・入力・中身を照合済み)。")

    summary = flush(final=True)
    report(summary)
    print(f"[done] {results_path}")
    print(f"[done] wav -> {args.wav_dir}")
    print("[next] 採点は既存の採点器で行う (新しい採点器は作らない):")
    print(f"  SLURM_TIME=02:00:00 slurm_run.sh \\")
    print(f"    python3 scripts/per_fix/phoneme_per_v3.py --wav_dir {args.wav_dir} \\")
    print(f"      --lang {args.lang} --n {args.n} \\")
    print(f"      --output_json {os.path.join(args.out_dir, 'per_v3.json')} \\")
    print(f"      --pairs_json {results_path}")
    if interrupted is not None:
        raise SystemExit(f"中断したまま終了した: {type(interrupted).__name__}: {interrupted}")


if __name__ == "__main__":
    main()
