#!/usr/bin/env python3
"""付録 G（tab:pronunciation_baselines、JA N=300）の 5 行を、Baseline / Text Hint と同じ声まね条件にそろえるための
共通定義（2026-09-29 作成。torch を import しない。pyopenjtalk も import しない）.

著者の決定（2026-09-29）:
  (D1) 5 行すべてを、Baseline と Text Hint が既に使っている声まね条件
       （scripts/projection/eval_baseline_text_edit.py の voice_clone L133-156、参照 REFS L59）にそろえる。
       G2P・ROME/MEMIT・SpeechSplice をその声まねで作り直す。既存の Baseline / Text Hint の wav は、無効でなければ残す。
       5 行とも rome_encoder_t5gemma.transcribe_with_whisper で書き起こし、score_unified_trm_g2pcer_nopyojt.py で採点。
  (D2) 無効な対（silence_criterion.py の無音、または生成失敗）は、その対の「同じ生成の回で作った wav すべて」を
       単位として seed 43、まだ無効なら 44 で作り直す。選ぶ基準は有効かどうかだけ。退避と JSON の記録を残し、
       本番のファイルの差し替えは別の apply 段（著者に見せたあと、リードが流す）でだけ行う。
  (D2b, 2026-09-29 追加) Baseline / Text Hint（textedit）の作り直しは b7 だけ（TEXTEDIT_UNIT = "b7"、既定）。
       無音の Text Hint 13 本（対 27, 46, 47, 70, 155, 160, 162, 186, 188, 267, 275, 277, 296 の s{i:04d}_b7.wav）だけを
       seed 43（まだ無音なら 44）で作り、選び、差し替える。本番の bl・b1 は変えない。
       b7 だけで作っても単位の中で作った b7 と同じになる理由: 5/15 の回（eval_baseline_text_edit.py）は合成のたびに
       voice_clone の最初（L133-134）で seed_everything(SEED) を入れ直し、bl（L233）・b1（L235）・b7（L238）を別々の呼び出しで
       作った。ここで使う voice_clone_synth も最初（eval_utmos_speaker_fixed.py L211）で入れ直す。前の合成の乱数は残らない。
       check42（seed 42 の再現の確かめ）は記録だけで、選ぶことには使わない（選ぶのは seed42 → seed43 → seed44 だけ）。
       以前の 3 本の単位は TEXTEDIT_UNIT = "bl_b1_b7" で残す（自己試験の A で使う）。

声まね条件（eval_baseline_text_edit.py の 2026-05-15 の回と同じ）:
  参照音声  eval_results/_natural_refs_norm/ja_natural_norm.wav（md5 3a924426f00a7206fa8e557fa2cafe28）
  見本の文  '太平洋は世界で一番大きな太陽です'（5/15 の回で whisper-large-v3-turbo が出した文。_run.log の prefix 行。
            L61 の正解文ではない。これを渡さないと再現しない）
  見本の長さ 100 秒（= 参照全体。prompt_end_frame = int(100 * 参照の標本化周波数)）
  言語 'ja'、長さの推定は参照の話速から（estimate_duration(reference_speech=参照, reference_transcript=見本の文)）、
  復号 top_k=30, top_p=0.9, min_p=0, T=0.8, stop_repetition=3、合成の直前に seed_everything(seed)。
  生成の関数は scripts/projection/eval_utmos_speaker_fixed.voice_clone_synth（引数・処理が上の voice_clone と同じ。
  SpeechSplice の注入もこの関数の hook で行う。話者差し替え実験と同じ経路）。

単位（D2 の「同じ生成の回で作った wav すべて」）:
  textedit  s{i:04d}_bl.wav / _b1.wav / _b7.wav （Baseline と Text Hint。5/15 の回は 1 対ごとに 3 本を作った）
  g2p       p{i:03d}_g2p.wav
  rome      p{i:03d}_rome.wav
  splice    s{i:04d}_bl.wav / _inj.wav （SpeechSplice。BL は Baseline の再現の確かめに使う）
  単位の中の wav が 1 本でも無音、または生成に失敗していれば、その単位は無効。
表の 5 行と wav:
  baseline = textedit/bl, g2p = g2p/g2p, text_hint = textedit/b7, rome = rome/rome, speechsplice = splice/inj

置き場（本番の既存の結果は読むだけ）:
  textedit の seed 42 = 本番 eval_results/baseline_text_edit_ja_n300/wavs（作らない。apply 段だけが差し替える）
  textedit の作り直し（b7 だけ）= eval_results/_appg_vc/textedit/b7only/<tag>/（s{i:04d}_b7.wav だけ）
  それ以外 = eval_results/_appg_vc/<method>/<tag>/  tag は seed42 / check42（seed 42 の再現の確かめ）/ seed43 / seed44
    wav と items/pNNN.json（1 単位 1 ファイルの記録。これがあれば「作った」）、gen_config.json（条件の固定）、runs/
  選んだ結果 eval_results/_appg_vc/selection.json、差し替えの記録 eval_results/_appg_vc/apply/
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_common as AC  # noqa: E402  torch を import しない（silence_criterion も読む）

SC = AC.SC
PROJ = AC.PROJ
T5GEMMA_ROOT = AC.T5GEMMA_ROOT
N_PAIRS = 300
MAIN_SEED = 42
REGEN_SEEDS = (43, 44)
TAGS = ("seed42", "check42", "seed43", "seed44")
TAG_SEED = {"seed42": 42, "check42": 42, "seed43": 43, "seed44": 44}

OUT_ROOT_REL = "eval_results/_appg_vc"
SMOKE_ROOT_REL = "eval_results/_appg_vc_smoke"
DRYRUN_REL = "eval_results/_regen2_dryrun/appg_vc"
ALLOWED_ROOTS_REL = (OUT_ROOT_REL, SMOKE_ROOT_REL, DRYRUN_REL)

# ---- 声まね条件（eval_baseline_text_edit.py 2026-05-15 の回） ----
REF_WAV_REL = "eval_results/_natural_refs_norm/ja_natural_norm.wav"
REF_WAV_MD5 = "3a924426f00a7206fa8e557fa2cafe28"
PROMPT_TEXT = "太平洋は世界で一番大きな太陽です"
PROMPT_SEC = 100.0
LANG = "ja"
DECODE_DOC = {"top_k": 30, "top_p": 0.9, "min_p": 0.0, "temperature": 0.8, "stop_repetition": 3,
              "silence_tokens": [], "sample_batch_size": 1,
              "note": "constants inside eval_utmos_speaker_fixed.voice_clone_synth == eval_baseline_text_edit.voice_clone"}
BASELINE_DIR_REL = "eval_results/baseline_text_edit_ja_n300/wavs"
BASELINE_LOG_REL = "eval_results/baseline_text_edit_ja_n300/_run.log"
BASELINE_RESULTS_REL = "eval_results/baseline_text_edit_ja_n300/results.json"
OLD_G2P_JSON_REL = "eval_results/_baselines_cc_n300_unified/g2p_baseline/g2p_transcripts.json"
ROME_LAYERS = (2, 5, 8)
ROME_MODE = "memit"
ROME_EPS = 1.0

# ---- 単位と表の行 ----
METHODS = {
    "textedit": (("bl", "s{i:04d}_bl.wav"), ("b1", "s{i:04d}_b1.wav"), ("b7", "s{i:04d}_b7.wav")),
    "g2p": (("g2p", "p{i:03d}_g2p.wav"),),
    "rome": (("rome", "p{i:03d}_rome.wav"),),
    "splice": (("bl", "s{i:04d}_bl.wav"), ("inj", "s{i:04d}_inj.wav")),
}
METHOD_ORDER = ("textedit", "g2p", "rome", "splice")
# Baseline / Text Hint の作り直しで作る・選ぶ・差し替える wav（D2b。既定は b7 だけ。本番の seed 42 は常に 3 本で判定する）
TEXTEDIT_UNITS = {"b7": ("b7",), "bl_b1_b7": ("bl", "b1", "b7")}
TEXTEDIT_UNIT_DEFAULT = "b7"
TEXTEDIT_B7_SUBDIR = "b7only"
_TEXTEDIT_UNIT = [TEXTEDIT_UNIT_DEFAULT]
NEW_METHODS = ("g2p", "rome", "splice")  # seed 42 を新しく作る 3 行
ROWS = {  # 表の行 -> (単位, 単位の中の wav, 採点器が読む書き起こしの鍵)
    "baseline": ("textedit", "bl", "bl_transcript"),
    "g2p": ("g2p", "g2p", "g2p_transcript"),
    "text_hint": ("textedit", "b7", "b7_transcript"),
    "rome": ("rome", "rome", "rome_transcript"),
    "speechsplice": ("splice", "inj", "t5_inj_transcript"),
}
ROW_ORDER = ["baseline", "g2p", "text_hint", "rome", "speechsplice"]
STAT_KEYS = ("dur", "rms", "peak", "active_frac", "silent", "silent_reasons")

hint_ja = AC.hint_ja  # eval_baseline_text_edit.py L70 の HINT_TEMPLATES['ja'] と同一（本番の text_b7 と照合する）
md5_file = AC.md5_file
atomic_write_json = AC.atomic_write_json
write_json_keep_prev = AC.write_json_keep_prev
proj_path = AC.proj_path
rel_to_proj = AC.rel_to_proj
load_pairs = AC.load_pairs


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def new_run_id(tag=""):
    return f"{time.strftime('%Y%m%d_%H%M%S')}{('_' + tag) if tag else ''}_{os.getpid()}"


def load_json(p):
    return json.load(open(p)) if p and os.path.isfile(p) else None


def parse_pairs(spec):
    """'0-149' / '1,5,7' / '0-9,20' -> 並べた pair_idx の一覧."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if a > b:
                raise SystemExit(f"[pairs] 範囲の向きが逆: {part}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if not 0 <= i < N_PAIRS)
    if bad:
        raise SystemExit(f"[pairs] 0..{N_PAIRS - 1} の外: {bad[:10]}")
    return sorted(out)


# ------------------------------------------------------------------ 置き場
def _under(rel, base):
    return rel == base or rel.startswith(base + "/")


def root_path(root=None):
    r = proj_path(root or OUT_ROOT_REL)
    rel = rel_to_proj(r)
    if not any(_under(rel, b) for b in ALLOWED_ROOTS_REL):
        raise SystemExit(f"[guard] 出力の置き場は {ALLOWED_ROOTS_REL} の下だけ: {r}")
    return r


def baseline_dir():
    """textedit の seed 42 = 本番の置き場。自己試験（APPG_VC_BASELINE_DIR）は空運転の置き場の下だけを許す。"""
    ov = os.environ.get("APPG_VC_BASELINE_DIR")
    if ov:
        p = proj_path(ov)
        if not _under(rel_to_proj(p), DRYRUN_REL):
            raise SystemExit(f"[guard] APPG_VC_BASELINE_DIR は {DRYRUN_REL} の下だけ: {p}")
        return p
    return proj_path(BASELINE_DIR_REL)


def is_production(method, tag):
    return method == "textedit" and tag == "seed42"


def method_keys(method):
    return [k for k, _ in METHODS[method]]


def set_textedit_unit(name):
    """textedit の作り直しの単位を決める（各スクリプトの --textedit_unit から。既定 b7 = 著者の決定 D2b）。"""
    if name not in TEXTEDIT_UNITS:
        raise SystemExit(f"[guard] --textedit_unit は {sorted(TEXTEDIT_UNITS)}: {name}")
    _TEXTEDIT_UNIT[0] = name


def textedit_unit():
    return _TEXTEDIT_UNIT[0]


def regen_keys(method):
    """作り直しの段（check42 / seed43 / seed44）で作る・選ぶ・差し替える wav の鍵。textedit だけ単位の決め方による。"""
    if method == "textedit":
        return list(TEXTEDIT_UNITS[textedit_unit()])
    return method_keys(method)


def stage_keys(method, tag):
    """その置き場にある（あるはずの）wav の鍵。本番（textedit seed 42）は 3 本、作り直しの段は regen_keys。"""
    if is_production(method, tag):
        return method_keys(method)
    return regen_keys(method)


def file_name(method, key, i):
    return dict(METHODS[method])[key].format(i=i)


def method_dir(method, tag, root=None):
    if tag not in TAGS:
        raise SystemExit(f"[guard] tag は {TAGS}: {tag}")
    if is_production(method, tag):
        return baseline_dir()
    if method == "textedit" and textedit_unit() == "b7":
        return os.path.join(root_path(root), method, TEXTEDIT_B7_SUBDIR, tag)
    return os.path.join(root_path(root), method, tag)


def file_path(method, tag, key, i, root=None):
    return os.path.join(method_dir(method, tag, root), file_name(method, key, i))


def item_path(method, tag, i, root=None):
    if is_production(method, tag):
        raise SystemExit("[guard] 本番（textedit seed 42）には items の記録を書かない")
    return os.path.join(method_dir(method, tag, root), "items", f"p{i:03d}.json")


def assert_writable_dir(path, root=None):
    """書いてよいのは出力の置き場（root）の下だけ。本番（PROTECTED）には書かない（apply 段は別に確かめる）。"""
    r = rel_to_proj(path)
    rr = rel_to_proj(root_path(root))
    if not (_under(r, rr) or _under(r, DRYRUN_REL)):
        raise SystemExit(f"[guard] 出力の置き場 {rr}（か {DRYRUN_REL}）の外には書かない: {path}")
    AC.assert_not_protected(path)


def pick_stats(st):
    return {k: st.get(k) for k in STAT_KEYS}


# ------------------------------------------------------------------ 1 単位の記録（生成と自己試験で共通）
def commit_unit(method, tag, i, root, tmp_files, item, per_key=None):
    """一時ファイル（同じ置き場の .tmp_*.wav）を本来の名前に置いてから、items/pNNN.json を書く。
    1) 一時ファイルの md5 と無音の判定 2) os.replace で本来の名前へ 3) 記録（atomic）。記録があれば「作った」。"""
    d = method_dir(method, tag, root)
    assert_writable_dir(d, root)
    keys = stage_keys(method, tag)
    if sorted(tmp_files) != sorted(keys):
        raise SystemExit(f"[guard] 1 単位の wav の鍵が単位の決め方と違う: {sorted(tmp_files)} != {sorted(keys)}")
    files = {}
    for k in keys:
        tp = tmp_files[k]
        st = AC.wav_stats(tp)
        files[k] = {"name": file_name(method, k, i), "md5": md5_file(tp), **pick_stats(st),
                    **((per_key or {}).get(k) or {})}
    for k in keys:
        os.replace(tmp_files[k], os.path.join(d, files[k]["name"]))
    rec = {**item, "pair_idx": i, "method": method, "tag": tag, "status": "success", "files": files,
           "silent_keys_at_generation": [k for k in files if files[k]["silent"]], "committed": now(),
           "silence_criterion_id": SC.CRITERION_ID}
    if method == "textedit":
        rec["textedit_unit"] = textedit_unit()
        rec["unit_keys"] = list(keys)
    os.makedirs(os.path.join(d, "items"), exist_ok=True)
    atomic_write_json(item_path(method, tag, i, root), rec)
    return rec


def record_error(method, tag, i, root, item, err):
    d = method_dir(method, tag, root)
    assert_writable_dir(d, root)
    os.makedirs(os.path.join(d, "items"), exist_ok=True)
    rec = {**item, "pair_idx": i, "method": method, "tag": tag, "status": "error",
           "error": str(err)[:300], "error_type": type(err).__name__, "committed": now()}
    if method == "textedit":
        rec["textedit_unit"] = textedit_unit()
        rec["unit_keys"] = stage_keys(method, tag)
    atomic_write_json(item_path(method, tag, i, root), rec)
    return rec


def set_aside_unrecorded(method, tag, i, root, ts):
    """記録の無い本来の名前の wav（途中で落ちた回の残り）は名前を変えて残す（消さない）。"""
    moved = []
    for k in stage_keys(method, tag):
        p = file_path(method, tag, k, i, root)
        if os.path.exists(p):
            dst = f"{p}.unrecorded_{ts}.bak"
            os.replace(p, dst)
            moved.append(rel_to_proj(dst))
    return moved


# ------------------------------------------------------------------ 差し替え（apply）の記録
def apply_dir(root=None):
    return os.path.join(root_path(root), "apply")


def applied_path(root=None):
    return os.path.join(apply_dir(root), "applied_textedit.json")


def load_applied(root=None):
    d = load_json(applied_path(root)) or {}
    return {int(k): v for k, v in (d.get("pairs") or {}).items()}


# ------------------------------------------------------------------ 1 単位の判定
def _state(out, state, why=None, **kw):
    return {**out, "state": state, "why": why, **kw}


def assess_unit(method, tag, i, root=None, applied=None):
    """1 単位を置き場の今のファイルで判定する（記録の無音の判定は信じず、共通の基準で測り直す）。
    state: valid / invalid（生成失敗か無音）/ not_attempted（記録も wav も無い）/ inconsistent（記録と実物が食い違う）。
    本番（textedit seed 42）は記録が無いので wav だけで判定する（常に bl・b1・b7 の 3 本）。差し替え済みの wav は、
    退避した元の wav で判定する（b7 だけ差し替えた対では b7 だけが退避から、bl・b1 は本番のまま）。
    作り直しの段は stage_keys の wav だけ。記録の wav の鍵が単位の決め方と違えば inconsistent（決め方を混ぜない）。"""
    keys = stage_keys(method, tag)
    out = {"pair_idx": i, "method": method, "tag": tag}
    paths = {k: file_path(method, tag, k, i, root) for k in keys}
    if is_production(method, tag):
        applied = load_applied(root) if applied is None else applied
        a = applied.get(i)
        expect = {}
        if a is not None:
            afiles = a.get("files") or {}
            akeys = [k for k in keys if k in afiles]
            if not akeys or set(afiles) - set(keys):
                return _state(out, "inconsistent", f"apply record keys not understood: {sorted(afiles)}")
            cur = {k: (md5_file(paths[k]) if os.path.isfile(paths[k]) else None) for k in akeys}
            bad = [k for k in akeys if cur[k] != afiles[k]["new_md5"]]
            if bad:
                return _state(out, "inconsistent", f"production differs from apply record: {bad}")
            for k in akeys:
                paths[k] = proj_path(afiles[k]["backup"])
                expect[k] = afiles[k]["old_md5"]
            out["assessed_from_backup"] = True
            out["assessed_from_backup_keys"] = akeys
        missing = [k for k in keys if not os.path.isfile(paths[k])]
        if missing:
            return _state(out, "inconsistent", f"production wav missing: {missing}")
        files = {}
        for k in keys:
            md5 = md5_file(paths[k])
            if k in expect and md5 != expect[k]:
                return _state(out, "inconsistent", f"backup md5 differs from apply record: {k}")
            files[k] = {"path": rel_to_proj(paths[k]), "md5": md5, **pick_stats(AC.wav_stats(paths[k]))}
        silent = [k for k in keys if files[k]["silent"]]
        return _state(out, "invalid" if silent else "valid", "silent" if silent else None,
                      silent_keys=silent, files=files, seed=MAIN_SEED)
    ip = item_path(method, tag, i, root)
    rec = load_json(ip)
    exists = [k for k in keys if os.path.isfile(paths[k])]
    if rec is None:
        if exists:
            return _state(out, "inconsistent", f"wav without item record: {exists}")
        return _state(out, "not_attempted", "no item record")
    if rec.get("status") != "success":
        if exists:
            return _state(out, "inconsistent", f"error record but wav exists: {exists}")
        return _state(out, "invalid", "generation failed", error=rec.get("error"), seed=rec.get("seed"))
    if sorted((rec.get("files") or {}).keys()) != sorted(keys):
        return _state(out, "inconsistent", f"item record keys {sorted((rec.get('files') or {}).keys())} "
                                           f"!= unit keys {sorted(keys)}")
    missing = [k for k in keys if k not in exists]
    if missing:
        return _state(out, "inconsistent", f"success record but wav missing: {missing}")
    files = {}
    for k in keys:
        md5 = md5_file(paths[k])
        if md5 != (rec.get("files") or {}).get(k, {}).get("md5"):
            return _state(out, "inconsistent", f"wav md5 differs from item record: {k}")
        files[k] = {"path": rel_to_proj(paths[k]), "md5": md5, **pick_stats(AC.wav_stats(paths[k]))}
    silent = [k for k in keys if files[k]["silent"]]
    return _state(out, "invalid" if silent else "valid", "silent" if silent else None,
                  silent_keys=silent, files=files, seed=rec.get("seed"), run_id=rec.get("run_id"))


def _group(per, targets):
    by = {s: [] for s in ("valid", "invalid", "not_attempted", "inconsistent")}
    for i in targets:
        by[per[i]["state"]].append(i)
    return by


def extra_items(method, tag, targets, root=None):
    """その置き場に、対象の外の対の記録があるか（seed 43/44 で有効な対を作り直していないか）。"""
    if is_production(method, tag):
        return []
    d = os.path.join(method_dir(method, tag, root), "items")
    if not os.path.isdir(d):
        return []
    have = sorted(int(n[1:4]) for n in os.listdir(d) if n.startswith("p") and n.endswith(".json") and len(n) == 9)
    return sorted(set(have) - set(targets))


def scan_method(method, root=None):
    """1 つの単位の種類について、seed 42 → check42 → seed 43 → seed 44 の状態と、次にすることを返す。"""
    applied = load_applied(root) if method == "textedit" else {}
    allp = list(range(N_PAIRS))
    res = {"method": method, "stages": {}, "problems": []}
    s42 = {i: assess_unit(method, "seed42", i, root, applied) for i in allp}
    by42 = _group(s42, allp)
    complete42 = not by42["not_attempted"] and not by42["inconsistent"]
    res["stages"]["seed42"] = {"dir": rel_to_proj(method_dir(method, "seed42", root)), "complete": complete42,
                               **{f"n_{k}": len(v) for k, v in by42.items()},
                               "invalid": by42["invalid"], "not_attempted_head": by42["not_attempted"][:20],
                               "inconsistent": [s42[i] for i in by42["inconsistent"][:10]],
                               "invalid_detail": {str(i): {k: s42[i].get(k) for k in ("why", "silent_keys", "error")}
                                                  for i in by42["invalid"]}}
    if by42["inconsistent"]:
        res["problems"].append(f"seed42: 記録と wav が食い違う対 {by42['inconsistent'][:10]}")
    rk = regen_keys(method)
    res["unit_keys"] = {"seed42": method_keys(method), "regen": rk}
    if method == "textedit":
        res["textedit_unit"] = textedit_unit()
    outside = [i for i in by42["invalid"] if set(s42[i].get("silent_keys") or []) - set(rk)]
    if outside:  # 例: b7 だけの作り直しなのに本番の bl / b1 が無音（この決め方では直らない）
        res["problems"].append(f"seed42: 作り直す wav {rk} の外が無音の単位がある（この単位の決め方では直らない。著者に聞く）: "
                               f"{ {i: s42[i].get('silent_keys') for i in outside[:10]} }")
    per = {"seed42": s42}
    chosen = {i: ("seed42" if s42[i]["state"] == "valid" else None) for i in allp}
    targets = {"check42": None, "seed43": None, "seed44": None}
    unresolved = None
    if complete42:
        t = list(by42["invalid"])
        targets["check42"] = targets["seed43"] = t
        for tag in ("check42", "seed43"):
            st = {i: assess_unit(method, tag, i, root) for i in t}
            per[tag] = st
            by = _group(st, t)
            ex = extra_items(method, tag, t, root)
            res["stages"][tag] = {"dir": rel_to_proj(method_dir(method, tag, root)), "targets": t,
                                  "complete": not by["not_attempted"] and not by["inconsistent"],
                                  **{f"n_{k}": len(v) for k, v in by.items()}, "invalid": by["invalid"],
                                  "not_attempted": by["not_attempted"],
                                  "inconsistent": [st[i] for i in by["inconsistent"][:10]], "extra_records": ex}
            if ex:
                res["problems"].append(f"{tag}: 対象の外の対の記録がある（有効な対は作り直さない）: {ex[:10]}")
            if by["inconsistent"]:
                res["problems"].append(f"{tag}: 記録と wav が食い違う対 {by['inconsistent'][:10]}")
        # seed 42 の再現の確かめ（結果では何も選ばない。記録だけ。比べるのは作り直しの段の wav = rk だけ）
        rep = {}
        for i in t:
            c = per["check42"][i]
            if c["state"] not in ("valid", "invalid") or "files" not in c or "files" not in s42[i]:
                rep[str(i)] = {"state": c["state"]}
                continue
            same = {k: c["files"][k]["md5"] == s42[i]["files"][k]["md5"] for k in rk}
            rep[str(i)] = {"state": c["state"], "md5_identical": same, "silent_keys": c.get("silent_keys")}
        res["stages"]["check42"]["reproduction"] = {
            "n_all_keys_md5_identical": sum(1 for v in rep.values() if v.get("md5_identical") and all(v["md5_identical"].values())),
            "n_still_invalid": sum(1 for v in rep.values() if v.get("state") == "invalid"), "per_pair": rep}
        for i in t:  # 選ぶのは seed43 / seed44 だけ（check42 は有効でも選ばない）
            if per["seed43"][i]["state"] == "valid":
                chosen[i] = "seed43"
        if res["stages"]["seed43"]["complete"] and res["stages"]["check42"]["complete"]:
            t44 = [i for i in t if per["seed43"][i]["state"] == "invalid"]
            targets["seed44"] = t44
            st = {i: assess_unit(method, "seed44", i, root) for i in t44}
            per["seed44"] = st
            by = _group(st, t44)
            ex = extra_items(method, "seed44", t44, root)
            res["stages"]["seed44"] = {"dir": rel_to_proj(method_dir(method, "seed44", root)), "targets": t44,
                                       "complete": not by["not_attempted"] and not by["inconsistent"],
                                       **{f"n_{k}": len(v) for k, v in by.items()}, "invalid": by["invalid"],
                                       "not_attempted": by["not_attempted"],
                                       "inconsistent": [st[i] for i in by["inconsistent"][:10]], "extra_records": ex}
            if ex:
                res["problems"].append(f"seed44: 対象の外の対の記録がある: {ex[:10]}")
            if by["inconsistent"]:
                res["problems"].append(f"seed44: 記録と wav が食い違う対 {by['inconsistent'][:10]}")
            for i in t44:
                if st[i]["state"] == "valid":
                    chosen[i] = "seed44"
            if res["stages"]["seed44"]["complete"]:
                unresolved = by["invalid"]
    else:
        for tag in ("check42", "seed43", "seed44"):
            ex = extra_items(method, tag, [], root)
            if ex:
                res["problems"].append(f"{tag}: seed 42 が終わる前に作り直しの記録がある: {ex[:10]}")
    res["targets"] = targets
    res["unresolved"] = unresolved
    res["chosen"] = chosen
    res["seed_counts"] = {tag: sum(1 for v in chosen.values() if v == tag) for tag in ("seed42", "seed43", "seed44")}
    res["n_unchosen"] = sum(1 for v in chosen.values() if v is None)
    done = (complete42 and not res["problems"] and unresolved is not None and not unresolved) or \
           (complete42 and not res["problems"] and not by42["invalid"])
    res["done"] = bool(done)
    if res["problems"]:
        nxt = "食い違いを調べる（problems を見る）"
    elif not complete42:
        nxt = ("本番の Baseline / Text Hint の wav が欠けている。調べる" if method == "textedit"
               else f"seed 42 を同じ命令で続ける（未生成 {len(by42['not_attempted'])} 単位）")
    elif not by42["invalid"]:
        nxt = "無効な単位なし（作り直し不要）"
    elif unresolved is None:
        unit_s = f"、作る wav {rk}（--textedit_unit {textedit_unit()}）" if method == "textedit" else ""
        nxt = (f"--regen を流す（無効 {len(by42['invalid'])} 単位{unit_s}: check42 → seed43 → seed44。"
               "途中で止まれば同じ命令で続き）")
    elif unresolved:
        nxt = f"seed 44 でも無効な単位がある: {unresolved}。著者に聞く"
    else:
        nxt = "作り直し完了"
    res["next_action"] = nxt
    res["_per"] = per  # 呼び出し側だけが使う（JSON には書かない）
    return res


def public_scan(s):
    return {k: v for k, v in s.items() if not k.startswith("_")}


# ------------------------------------------------------------------ 選んだ結果（selection）
def build_selection(root=None, scans=None):
    """各単位で使う seed を、有効かどうかだけで決める（seed 42 → 43 → 44 の順で最初の有効なもの。check42 は使わない）。
    textedit で 43/44 を使う対は、本番の置き場へ差し替えたあと（apply）に本番のパスを読む。
    作り直しの段で作らない wav（b7 だけの作り直しでの bl・b1）は、選んだ seed に関わらず本番の seed 42 のまま。"""
    scans = scans or {m: scan_method(m, root) for m in METHOD_ORDER}
    problems = []
    for m in METHOD_ORDER:
        s = scans[m]
        if not s["done"]:
            problems.append(f"{m}: {s['next_action']}")
    sel = {"created": now(), "root": rel_to_proj(root_path(root)), "complete": not problems, "problems": problems,
           "silence_criterion": AC.criterion_doc(),
           "rule": "per unit, the first valid of seed42 -> seed43 -> seed44 (validity only: generation success and "
                   "no silent wav in the unit, silence_criterion); check42 is a record only and is never chosen",
           "textedit_unit": textedit_unit(), "textedit_regen_keys": regen_keys("textedit"),
           "scan": {m: {k: scans[m][k] for k in ("next_action", "seed_counts", "n_unchosen", "unresolved", "targets")}
                    for m in METHOD_ORDER}}
    if problems:
        return sel
    units, needs_apply = {}, []
    for m in METHOD_ORDER:
        per = scans[m]["_per"]
        rk = set(regen_keys(m))
        units[m] = {}
        for i in range(N_PAIRS):
            tag = scans[m]["chosen"][i]
            files = {}
            for k in method_keys(m):
                ktag = tag if (tag == "seed42" or k in rk) else "seed42"  # 作り直さない wav は seed 42（本番）のまま
                a = per[ktag][i]
                src = a["files"][k]["path"]
                md5 = a["files"][k]["md5"]
                if m == "textedit":
                    prod = file_path("textedit", "seed42", k, i, root)
                    prod_md5 = md5_file(prod) if os.path.isfile(prod) else None
                    files[k] = {"final_path": rel_to_proj(prod), "source_path": src, "md5": md5,
                                "production_md5_now": prod_md5, "needs_apply": prod_md5 != md5}
                else:
                    files[k] = {"final_path": src, "source_path": src, "md5": md5, "needs_apply": False}
                files[k].update(tag=ktag, seed=TAG_SEED[ktag])
            unit = {"tag": tag, "seed": TAG_SEED[tag], "regen_keys": sorted(rk), "files": files}
            if m == "textedit" and any(f["needs_apply"] for f in files.values()):
                needs_apply.append(i)
            units[m][str(i)] = unit
    rows = {}
    for row in ROW_ORDER:
        m, k, tkey = ROWS[row]
        rows[row] = {"method": m, "key": k, "transcript_key": tkey,
                     "pairs": [{"pair_idx": i, "path": units[m][str(i)]["files"][k]["final_path"],
                                "md5": units[m][str(i)]["files"][k]["md5"], "seed": units[m][str(i)]["files"][k]["seed"],
                                "needs_apply": units[m][str(i)]["files"][k]["needs_apply"]}
                               for i in range(N_PAIRS)]}
    sel["units"] = units
    sel["rows"] = rows
    sel["textedit_needs_apply"] = sorted(set(needs_apply))
    sel["ready_for_transcription"] = not sel["textedit_needs_apply"]
    sel["seed_counts"] = {m: scans[m]["seed_counts"] for m in METHOD_ORDER}
    sel["changed_rows_vs_production"] = {
        row: [i for i in range(N_PAIRS) if units["textedit"][str(i)]["files"][ROWS[row][1]]["seed"] != MAIN_SEED]
        for row in ("baseline", "text_hint")}
    return sel


# ------------------------------------------------------------------ 生成の条件（置き場ごとに固定し、置き場どうしでもそろえる）
COMMON_COND_KEYS = [
    "ref_wav_rel", "ref_wav_md5", "prompt_text", "prompt_sec", "lang", "decode",
    "voice_clone_synth_source_md5", "eval_utmos_speaker_fixed_md5", "splice_clone_md5", "encoder_injection_md5",
    "vendor_duration_estimator_md5", "vendor_inference_tts_utils_md5",
    "vendor_duration_uses_pyopenjtalk", "vendor_pyopenjtalk_version",
    "torch", "torchaudio", "transformers", "model_snapshot", "tokenizer_snapshot",
]


def guard_dir_config(d, cond, root=None, write=True):
    """置き場の gen_config.json と今回の条件が同じか。無ければ書く。戻り値 (ok, diff)。"""
    p = os.path.join(d, "gen_config.json")
    old = load_json(p)
    if old is None:
        if write:
            assert_writable_dir(d, root)
            os.makedirs(d, exist_ok=True)
            atomic_write_json(p, cond)
        return True, {}
    diff = {k: [old.get(k), cond.get(k)] for k in sorted(set(old) | set(cond))
            if k not in ("written", "run_id") and old.get(k) != cond.get(k)}
    return not diff, diff


def guard_root_common(cond, root=None):
    """同じ出力の置き場の中で、声まねの条件（COMMON_COND_KEYS）が全部の置き場で同じか。戻り値 {置き場: diff}。"""
    r = root_path(root)
    bad = {}
    for m in METHOD_ORDER:
        for tag in TAGS:
            if is_production(m, tag):
                continue
            dirs = [os.path.join(r, m, tag)]
            if m == "textedit":  # 単位の決め方ごとの置き場（b7 だけ = textedit/b7only/<tag>）もすべて見る
                dirs.append(os.path.join(r, m, TEXTEDIT_B7_SUBDIR, tag))
            for dd in dirs:
                p = os.path.join(dd, "gen_config.json")
                old = load_json(p)
                if old is None:
                    continue
                diff = {k: [old.get(k), cond.get(k)] for k in COMMON_COND_KEYS if old.get(k) != cond.get(k)}
                if diff:
                    bad[rel_to_proj(p)] = diff
    return bad


# ------------------------------------------------------------------ 生成中の失敗の線引き（appg_texthint_zp.py と同じ考え方）
ENV_ERROR_WORDS = ("cuda", "cublas", "cudnn", "nccl", "device-side assert")
ENV_ERROR_TYPES = ("OutOfMemoryError", "AcceleratorError", "DurationPathError")


class DurationPathError(RuntimeError):
    """vendor の長さ推定が例外を出し、voice_clone_synth が黙って代わりの長さを使った（条件がずれる）。環境の側の失敗として止める。"""


class PairFailure(RuntimeError):
    """その対そのものの失敗（生成失敗として記録する）。"""


def is_environment_failure(e):
    names = {c.__name__ for c in type(e).__mro__}
    if names & set(ENV_ERROR_TYPES):
        return True
    if isinstance(e, (MemoryError, OSError)):
        return True
    if isinstance(e, RuntimeError) and not isinstance(e, PairFailure) and \
            any(w in str(e).lower() for w in ENV_ERROR_WORDS):
        return True
    return False


# ------------------------------------------------------------------ 5/15 の回のログ（見本の文・合成の時間・生成した長さ）
def parse_baseline_log():
    """_run.log の prefix 行と、[Speed] 行（900 行 = 対 i の bl / b1 / b7 の順）。"""
    import re
    p = proj_path(BASELINE_LOG_REL)
    prefix, speed = None, []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("prefix:") and prefix is None:
                prefix = line.split("prefix:", 1)[1].strip("\n").lstrip(" ")
            m = re.search(r"Generated (\d+) tokens x \d+ samples in ([0-9.]+)s", line)
            if m and line.startswith("[Speed]"):
                speed.append((int(m.group(1)), float(m.group(2))))
    tokens = {}
    if len(speed) == 3 * N_PAIRS:
        for j, (n, _) in enumerate(speed):
            tokens.setdefault(j // 3, {})[("bl", "b1", "b7")[j % 3]] = n
    secs = [s for _, s in speed]
    return {"prefix_line": prefix, "n_speed_lines": len(speed), "tokens": tokens,
            "mean_synth_sec": (sum(secs) / len(secs)) if secs else None}
