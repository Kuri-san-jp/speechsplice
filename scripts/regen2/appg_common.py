"""付録 G（tab:pronunciation_baselines、JA N=300）の 5 行を「見本なし（zero-prompt）」の T5Gemma-TTS 経路と
同じ書き起こしにそろえるための共通定義（2026-09-29 作成、G2。同日、確認役の指摘を受けて改訂）.

torch を import しない（CPU の点検スクリプトとコンテナ内の GPU スクリプトの両方から読む）。
pyopenjtalk も import しない。pyc は書かない（data/ や scripts/ の __pycache__ を変えないため）。

5 行の音声（すべて generate_with_modified_encoder = 見本なし・zero-length prompt）:
  baseline      eval_results/_cc_wavs_a/t5gemma_ja/s{i:04d}_bl.wav                         （既存、作り直さない）
  g2p           eval_results/_baselines_cc_n300_unified/g2p_baseline/p{i:03d}_g2p_s42.wav   （既存）
  text_hint     eval_results/_appg_redo2/texthint_zp/p{i:03d}_texthint.wav                 （新規、appg_texthint_zp.py）
  rome          eval_results/_baselines_cc_n300_unified/rome_L258_memit/p{i:03d}_text_a_s42.wav（既存）
  speechsplice  eval_results/_cc_wavs_a/t5gemma_ja/s{i:04d}_inj.wav                        （既存）
書き起こしの鍵の名前は採点器 score_trm_g2pcer_nopyojt.py の cond_keys に合わせる
（bl_transcript / g2p_transcript / b7_transcript / rome_transcript / t5_inj_transcript）。

無音の基準は regen2 共通の scripts/regen2/silence_criterion.py（1 か所だけで定義）を使う:
  (RMS < 0.005 かつ peak < 0.05) または（20 ms 区間のうち区間 RMS > 0.01 の割合）< 0.05。

有効な対（著者の判断 (1)）: 生成に成功し（status == "success"）、wav があり、無音でない対。
  無効 = 生成失敗（status == "error"）か無音。作り直す対は、この基準だけで選ぶ（結果は見ない）。
  Text Hint は splice も整列もしないので「整列失敗」は起こらない。
  「まだ作っていない対」（ジョブが途中で止まった）は無効ではない。同じ seed のまま再開する。
"""
import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import silence_criterion as SC  # noqa: E402  regen2 共通の無音の基準（numpy / soundfile は呼ばれたときに読む）

PROJ = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
N_PAIRS = 300
MAIN_SEED = 42
REGEN_SEEDS = (43, 44)
NEXT_SEED = {42: 43, 43: 44}

# ---- 出力の置き場（すべて新規。既存の結果は読むだけ） ----
OUT_ROOT_REL = "eval_results/_appg_redo2"
TEXTHINT_DIR_REL = f"{OUT_ROOT_REL}/texthint_zp"
TEXTHINT_GEN_JSON = "texthint_gen.json"          # 生成の記録（appg_texthint_zp.py が書く）
TRANSCRIPTS_REL = f"{OUT_ROOT_REL}/transcripts_5rows.json"
SCORE_REL = f"{OUT_ROOT_REL}/final_results_5rows_nopyojt.json"
DRYRUN_REL = "eval_results/_regen2_dryrun/appg"

# 書き込んではいけない既存の置き場（本番の音声・書き起こし・採点）
PROTECTED_REL = [
    "eval_results/_cc_wavs_a",
    "eval_results/_cc_wavs_b",
    "eval_results/_baselines_cc_n300_unified",
    "eval_results/baseline_text_edit_ja_n300",
    "eval_results/t5gemma_cc_wavs_ja_n300",
    "eval_results/_appg_redo_nopyojt",
]

# ---- 5 行（原稿の表の並び） ----
# name -> (wav dir（PROJ 相対）, ファイル名の書式, 採点器が読む書き起こしの鍵)
ROWS = {
    "baseline":     ("eval_results/_cc_wavs_a/t5gemma_ja", "s{i:04d}_bl.wav", "bl_transcript"),
    "g2p":          ("eval_results/_baselines_cc_n300_unified/g2p_baseline", "p{i:03d}_g2p_s42.wav", "g2p_transcript"),
    "text_hint":    (TEXTHINT_DIR_REL, "p{i:03d}_texthint.wav", "b7_transcript"),
    "rome":         ("eval_results/_baselines_cc_n300_unified/rome_L258_memit", "p{i:03d}_text_a_s42.wav", "rome_transcript"),
    "speechsplice": ("eval_results/_cc_wavs_a/t5gemma_ja", "s{i:04d}_inj.wav", "t5_inj_transcript"),
}
ROW_ORDER = ["baseline", "g2p", "text_hint", "rome", "speechsplice"]

# ---- 生成の条件（seed 42 / 43 / 44 の回で同じでなければならない provenance の項目） ----
# 判断 (2): vendor の発話長推定がコンテナの pyopenjtalk を使う経路は足しも外しもしない。
# その経路の中身（encoder_injection_t5gemma.py・vendor の duration_estimator.py / inference_tts_utils.py の md5、
# コンテナの pyopenjtalk の版、経路が有効か）が回ごとに同じことを確かめる。
GEN_CONDITION_KEYS = [
    "procedure", "hint_template", "encoder_lang",
    "encoder_injection_md5", "vendor_duration_estimator_md5", "vendor_inference_tts_utils_md5",
    "vendor_pyopenjtalk_version", "vendor_duration_uses_pyopenjtalk", "allow_len_duration",
    "torch", "transformers", "hf_snapshots",
]

# 生成の経路が読む Hub の repo（"cfg:" は T5Gemma-TTS の config.json から読む。gen_t5gemma_pairs_module.py と同じ考え方）
GEN_REPOS = ["Aratako/T5Gemma-TTS-2b-2b", "cfg:text_tokenizer_name", "cfg:xcodec2_model_name",
             "facebook/w2v-bert-2.0"]
ASR_REPOS = ["openai/whisper-large-v3"]


def hint_ja(t, a, b):
    """Text Hint の文。scripts/projection/eval_baseline_text_edit.py L70 の HINT_TEMPLATES['ja'] と同一:
    lambda t, a, b: t.replace(a, f"{a}（{b}）", 1)"""
    return t.replace(a, f"{a}（{b}）", 1)


def proj_path(rel_or_abs):
    """PROJ 相対のパスを絶対パスにする（絶対パスはそのまま）。"""
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(PROJ, rel_or_abs)


def rel_to_proj(path):
    p = os.path.realpath(path)
    for root in (os.path.realpath(PROJ), os.environ.get("REPO_ROOT", "/mnt"), "/mnt"):
        if p == root or p.startswith(root + "/"):
            return os.path.relpath(p, root)
    return p


def same_dir(a, b):
    return os.path.realpath(a) == os.path.realpath(b)


def assert_not_protected(path):
    """本番の置き場（PROTECTED_REL）とその配下には書かない。"""
    r = rel_to_proj(path)
    for pr in PROTECTED_REL:
        if r == pr or r.startswith(pr + "/"):
            raise SystemExit(f"[guard] 本番の置き場には書かない: {path}（{pr}）")


def wav_name(i):
    return f"p{i:03d}_texthint.wav"


def wav_path(row, i, texthint_dir=None):
    d, fmt, _ = ROWS[row]
    if row == "text_hint" and texthint_dir:
        d = texthint_dir
    return os.path.join(proj_path(d), fmt.format(i=i))


def load_pairs():
    """data/ja_cross_culture_names.build_300_pairs()（付録 G の既存 4 行と同じ 300 対）。pyc は書かない。"""
    dpath = os.path.join(PROJ, "data")
    if dpath not in sys.path:
        sys.path.insert(0, dpath)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from ja_cross_culture_names import build_300_pairs
    finally:
        sys.dont_write_bytecode = prev
    pairs = build_300_pairs()
    assert len(pairs) == N_PAIRS, len(pairs)
    assert all(p["pair_idx"] == i for i, p in enumerate(pairs)), "pair_idx が 0..299 の順でない"
    return pairs


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_or_none(path):
    return md5_file(path) if path and os.path.isfile(path) else None


# ------------------------------------------------------------------ 無音（共通の部品に任せる）
def wav_stats(path):
    """silence_criterion.assess_file の結果に、旧い鍵の名前（dur_sec / silent_reason）を足したもの。"""
    a = SC.assess_file(path)
    a["dur_sec"] = a.get("dur")
    a["silent_reason"] = list(a.get("silent_reasons", []))
    return a


def criterion_doc():
    return {"module": "scripts/regen2/silence_criterion.py", "criterion_id": SC.CRITERION_ID,
            "rule": SC.CRITERION_TEXT, "rms_lt": SC.RMS_TH, "peak_lt": SC.PEAK_TH, "frame_ms": SC.FRAME_MS,
            "frame_rms_gt": SC.FRAME_RMS_TH, "active_frac_lt": SC.ACTIVE_FRAC_TH}


# ------------------------------------------------------------------ JSON の書き出し
def atomic_write_json(path, obj):
    """同じ置き場の一時ファイルに書いてから os.replace（途中で落ちても半端なファイルを残さない）。"""
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_keep_prev(path, obj):
    """前のファイルがあれば <path>.prev_<時刻> に写して残してから書く（点検の記録を上書きで失わない）。"""
    kept = None
    if os.path.exists(path):
        kept = f"{path}.prev_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        shutil.copy2(path, kept)
    atomic_write_json(path, obj)
    return kept


# ------------------------------------------------------------------ 生成の記録の要約と、対が有効かの判定
def rows_by_idx(rec):
    return {r["pair_idx"]: r for r in (rec or {}).get("rows", [])}


def summarize_gen(rec, targets):
    """texthint_gen.json の行から数え直す（古い summary は使わない）。wav_stats は生成時に同じ基準で測ったもの。"""
    rows = rows_by_idx(rec)
    targets = sorted(set(targets))
    ok = [i for i in targets if rows.get(i, {}).get("status") == "success"]
    silent = [i for i in ok if (rows[i].get("wav_stats") or {}).get("silent") is True]
    unknown = [i for i in ok if (rows[i].get("wav_stats") or {}).get("silent") is None]
    failed = [i for i in targets if i in rows and rows[i].get("status") != "success"]
    not_attempted = [i for i in targets if i not in rows]
    langs = {}
    for i in ok:
        k = str(rows[i].get("resolved_lang"))
        langs[k] = langs.get(k, 0) + 1
    return {"n_targets": len(targets), "n_success": len(ok), "n_failed": len(failed), "failed_pairs": failed,
            "n_not_attempted": len(not_attempted), "not_attempted_pairs": not_attempted,
            "n_silent": len(silent), "silent_pairs": silent, "n_stats_unknown": len(unknown),
            "stats_unknown_pairs": unknown,
            "regen_targets": sorted(set(silent) | set(failed)),
            "regen_targets_rule": "status == 'error' or silent (silence_criterion); chosen only by validity",
            "resolved_lang_counts": langs, "silence_criterion": criterion_doc(),
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def assess_pair(gen_dir, rows, i, check_md5=True):
    """1 対を、置き場の今の wav で判定する（記録の wav_stats は信じず、共通の基準で測り直す）。
    戻り値 state:
      valid          生成に成功、wav あり、md5 が記録と一致、無音でない
      invalid        生成失敗（status == 'error'）か無音 → 作り直しの対象（判断 (1)）
      not_attempted  記録に行が無く wav も無い（ジョブが途中で止まった）→ 同じ seed で再開
      inconsistent   記録と実物が食い違う（成功なのに wav が無い / md5 が違う / 記録の無い wav /
                     失敗の記録なのに wav がある）→ 調べる
    """
    r = rows.get(i)
    p = os.path.join(gen_dir, wav_name(i))
    exists = os.path.isfile(p)
    if r is None:
        if exists:
            return {"state": "inconsistent", "why": "wav without record", "pair_idx": i}
        return {"state": "not_attempted", "why": "no record", "pair_idx": i}
    if r.get("status") != "success":
        if exists:
            return {"state": "inconsistent", "why": "error record but wav exists", "pair_idx": i}
        return {"state": "invalid", "why": "generation failed", "error": r.get("error"), "pair_idx": i,
                "seed": r.get("seed")}
    if not exists:
        return {"state": "inconsistent", "why": "success record but wav missing", "pair_idx": i}
    md5 = md5_file(p)
    if check_md5 and md5 != r.get("wav_md5"):
        return {"state": "inconsistent", "why": "wav md5 differs from record", "pair_idx": i, "md5": md5,
                "record_md5": r.get("wav_md5")}
    st = wav_stats(p)
    base = {"pair_idx": i, "md5": md5, "seed": r.get("seed"),
            **{k: st[k] for k in ("rms", "peak", "active_frac", "silent_reasons")}}
    if st["silent"]:
        return {"state": "invalid", "why": "silent", **base}
    return {"state": "valid", **base}


def scan_gen_dir(gen_dir, rec, targets=None):
    """置き場の全対（既定 0..299）を assess_pair で判定し、状態ごとの対の一覧を返す。"""
    rows = rows_by_idx(rec)
    targets = list(range(N_PAIRS)) if targets is None else sorted(set(targets))
    per = {i: assess_pair(gen_dir, rows, i) for i in targets}
    by = {s: [i for i in targets if per[i]["state"] == s]
          for s in ("valid", "invalid", "not_attempted", "inconsistent")}
    return per, by


def load_gen_record(gen_dir):
    p = os.path.join(gen_dir, TEXTHINT_GEN_JSON)
    return json.load(open(p)) if os.path.isfile(p) else None


def gen_conditions(prov):
    prov = prov or {}
    return {k: prov.get(k) for k in GEN_CONDITION_KEYS}


def condition_diff(prov_a, prov_b):
    a, b = gen_conditions(prov_a), gen_conditions(prov_b)
    return {k: [a[k], b[k]] for k in GEN_CONDITION_KEYS if a[k] != b[k]}


def all_provenances(rec):
    """その記録に残る生成の provenance すべて（今の回・前の回・差し替えで入った回）を run_id つきで返す。"""
    out = []
    rec = rec or {}
    if rec.get("provenance"):
        out.append(("current", rec["provenance"]))
    for k, p in enumerate(rec.get("provenance_history", [])):
        out.append((f"history[{k}]", p))
    for seed, lst in (rec.get("regen_provenance") or {}).items():
        for k, p in enumerate(lst if isinstance(lst, list) else [lst]):
            out.append((f"regen_provenance[{seed}][{k}]", p))
    return out


# ------------------------------------------------------------------ 生成の経路のファイルの md5（CPU 側でも同じ値を出せる）
def generation_code_md5s():
    return {
        "encoder_injection_md5": md5_or_none(os.path.join(PROJ, "scripts", "encoder_injection_t5gemma.py")),
        "vendor_duration_estimator_md5": md5_or_none(os.path.join(T5GEMMA_ROOT, "duration_estimator.py")),
        "vendor_inference_tts_utils_md5": md5_or_none(os.path.join(T5GEMMA_ROOT, "inference_tts_utils.py")),
    }


def dist_version(name):
    """配布物の版を metadata から読む（モジュールは import しない）。無ければ None。"""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ HF offline（gen_t5gemma_pairs_module.py と同じ考え方）
def hf_offline():
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")


def local_snapshot(repo_id):
    """キャッシュにある Hub の repo の snapshot の置き場（ネットにはつながない）。無ければ例外。"""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, local_files_only=True)


def offline_preflight(repos):
    """repos（"cfg:<key>" は T5Gemma-TTS の config.json から読む）がすべて手元のキャッシュにあるか。torch は読まない。"""
    out, cfg = {}, {}
    for r in repos:
        if r.startswith("cfg:"):
            key = r[4:]
            rid = cfg.get(key) or (cfg.get("t5gemma_model_name") if key == "text_tokenizer_name" else None)
            if not rid:
                out[r] = {"error": "not in T5Gemma-TTS config"}
                continue
            r = rid
        try:
            snap = local_snapshot(r)
            files = sorted(os.listdir(snap))
            out[r] = {"snapshot": snap, "revision": os.path.basename(snap.rstrip("/")),
                      "has_config": "config.json" in files or "preprocessor_config.json" in files}
            if r == "Aratako/T5Gemma-TTS-2b-2b":
                cfg = json.load(open(os.path.join(snap, "config.json")))
        except Exception as e:  # noqa: BLE001
            out[r] = {"error": repr(e)[:300]}
    ok = all("error" not in v and v.get("has_config") for v in out.values())
    return ok, out
