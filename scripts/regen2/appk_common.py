#!/usr/bin/env python3
"""G1（付録 K: 6 言語の IPA 変化を N=300 の同じ音声集合で採点し直す）の共通定義（2026-09-29）.

torch を import しない。CPU 側（無音の検査・計画・反映・集計）と GPU 側（作り直し・採点）の両方から使う。

無音の判定は regen2 共通の 1 つの部品だけで行う（scripts/regen2/silence_criterion.py の assess_file。G2・G3 も同じ）:
  無音 = (RMS < 0.005 かつ peak < 0.05) または 20 ms 区間のうち区間 RMS > 0.01 の割合 < 0.05
  （著者決定 2026-09-29。閾値がこの値から変わっていたら import の時点で止める）

作り直しの決まり（著者決定 (1)）:
  * 使えない対（無音・欠け・読めない wav、生成の失敗）だけを、無音の基準で選ぶ。採点の結果では選ばない。
  * 対ごと（BL と注入側の両方）に seed 43 で作り直し、43 でも使えなければ seed 44。44 でもだめなら未解決として残す。
  * 既定は unit="pair"（本番の計画では 39 対 = EN1/FR16/DE10/KO7/ZH5）: 使えない対だけを作り直し、seed 44 に
    進むのも seed 43 で使えなかった対だけ。同じ BL 文を共有する使える対は作り直さない。IPA 変化は同じ対の
    BL と注入側を比べるだけなので、同じ文の BL が対ごとに違う音声になっても数え方は崩れない。
  * unit="bl_group"（--unit bl_group）は、同じ BL 文（text_a）を共有する対をまとめて作り直し、まとめて seed 44 に
    進める（64 対。使える 25 対も差し替わる）。著者がはっきり認めたときだけ使う。
  * N=300 必須。T5Gemma-TTS EN の Table 3 (a) は G1 と無関係（触らない）。

検査用: 環境変数 APPK_TEST_ROOT（eval_results/_regen2_dryrun/ の下に限る）を与えると、本番の音声集合の置き場
（SETS の dir）と作り直しの置き場をすべてその下に読み替える（appk_selftest_regen.py が使う）。
"""
import hashlib
import json
import os
import sys

# G1 のスクリプトが読む他の置き場のモジュール（data/、scripts/nontemplate/、scripts/projection/ など）に
# __pycache__ を書かない（G1 が書いてよいのは scripts/regen2/ と作り直し・予行の置き場だけ）
sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import silence_criterion as SC  # noqa: E402  regen2 共通の無音の基準（1 か所だけで定義。G2/G3 と同じ部品）

# 共通の無音の部品が、著者の決めた基準のままであることを確かめる（名前か値が変わっていたら止める）
for _name in ("assess_file", "silent_reasons", "CRITERION_ID", "CRITERION_TEXT"):
    if not hasattr(SC, _name):
        raise SystemExit(f"silence_criterion.{_name} がない（共通の無音の部品が変わった）。G1 を止める。")
if not (SC.RMS_TH == 0.005 and SC.PEAK_TH == 0.05 and SC.FRAME_MS == 20 and SC.FRAME_RMS_TH == 0.01
        and SC.ACTIVE_FRAC_TH == 0.05):
    raise SystemExit("silence_criterion の閾値が著者決定（0.005 / 0.05 / 20 ms / 0.01 / 0.05）と違う。G1 を止める。")
if SC.silent_reasons({"rms": 0.004, "peak": 0.049, "active_frac": 0.5}) != ["rms_and_peak"] \
        or SC.silent_reasons({"rms": 0.1, "peak": 0.9, "active_frac": 0.049}) != ["active_frac"] \
        or SC.silent_reasons({"rms": 0.1, "peak": 0.9, "active_frac": 0.5}) != []:
    raise SystemExit("silence_criterion.silent_reasons が (a) または (b) の判定になっていない。G1 を止める。")

SILENCE_HELPER = "scripts/regen2/silence_criterion.py (assess_file)"


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SILENCE_CRITERION = {"module": "scripts/regen2/silence_criterion.py", "module_md5": md5_file(SC.__file__),
                     "criterion_id": SC.CRITERION_ID, "rule": SC.CRITERION_TEXT,
                     "rms_lt": SC.RMS_TH, "peak_lt": SC.PEAK_TH, "frame_ms": SC.FRAME_MS,
                     "frame_rms_gt": SC.FRAME_RMS_TH, "active_frac_lt": SC.ACTIVE_FRAC_TH}

PROJ = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
REAL_ER = os.path.join(PROJ, "eval_results")
DRYRUN = os.path.join(REAL_ER, "_regen2_dryrun")
TEST_ROOT = os.environ.get("APPK_TEST_ROOT") or None
if TEST_ROOT:
    TEST_ROOT = os.path.abspath(TEST_ROOT)
    if not (TEST_ROOT + "/").startswith(DRYRUN + "/") or TEST_ROOT == DRYRUN:
        raise SystemExit(f"APPK_TEST_ROOT は eval_results/_regen2_dryrun/ の下のフォルダに限る: {TEST_ROOT}")
ER = TEST_ROOT or REAL_ER

THREADS = 13
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
NICE = 10


def cap_threads():
    """numpy / torch を import する前に呼ぶ（13 スレッド）.

    ホストで動くときは、優先度も 10 下げる（`nice -n 10` の代わり。コマンドの前に env や nice を付けなくてよい）。
    何度呼んでも nice 値は 10 のまま（下げすぎない）。GPU のコンテナの中（PROJ が /mnt）では下げない."""
    for v in THREAD_VARS:
        os.environ[v] = str(THREADS)
    if PROJ != "/mnt":
        try:
            cur = os.nice(0)
            if cur < NICE:
                os.nice(NICE - cur)
        except OSError:
            pass


# --------------------------------------------------------------------------
# 付録 K の音声集合（各 300 対。BL と注入側）
# --------------------------------------------------------------------------
SETS = {
    "en": {"dir": "utmos_speaker_fixed_en_n300_splice", "ckpt": "checkpoint.json", "rows_key": "rows",
           "bl": "bl", "inj": "inj", "gen": "E", "json": ["checkpoint.json", "results.json"]},
    "fr": {"dir": "utmos_speaker_fixed_fr_n300_splice", "ckpt": "checkpoint.json", "rows_key": "rows",
           "bl": "bl", "inj": "inj", "gen": "E", "json": ["checkpoint.json", "results.json"]},
    "de": {"dir": "utmos_speaker_fixed_de_n300_splice", "ckpt": "checkpoint.json", "rows_key": "rows",
           "bl": "bl", "inj": "inj", "gen": "E", "json": ["checkpoint.json", "results.json"]},
    "ko": {"dir": "utmos_speaker_fixed_ko_n300_splice", "ckpt": "checkpoint.json", "rows_key": "rows",
           "bl": "bl", "inj": "inj", "gen": "E", "json": ["checkpoint.json", "results.json"]},
    "ja": {"dir": "cjk_n300_rewav", "ckpt": "checkpoint_ja.json", "rows_key": "results",
           "bl": "bl", "inj": "oeoi", "gen": None, "json": ["checkpoint_ja.json"]},
    "zh": {"dir": "cjk_n300_zh_v2", "ckpt": "checkpoint_zh.json", "rows_key": "results",
           "bl": "bl", "inj": "oeoi", "gen": "Z", "json": ["checkpoint_zh.json"]},
}
N_EXPECTED = 300
LANGS = ("en", "fr", "de", "ko", "ja", "zh")
REGEN_LANGS = ("en", "fr", "de", "ko", "zh")   # JA の作り直しは作っていない（今は無音の対 0。出たら計画に警告、採点は runner が拒否）
SEEDS = (43, 44)
SIDES = ("bl", "inj")


def set_dir(lang):
    return os.path.join(ER, SETS[lang]["dir"])


def wav_name(lang, i, side):
    return f"s{i:04d}_{SETS[lang][side]}.wav"


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_rows(lang, d=None):
    """本番（または d）の checkpoint の行."""
    spec = SETS[lang]
    return read_json(os.path.join(d or set_dir(lang), spec["ckpt"]))[spec["rows_key"]]


def wav_stats(path):
    """共通の無音の部品（silence_criterion.assess_file）の結果を、G1 の鍵の名前でも返す."""
    a = SC.assess_file(path)
    return {"sr": int(a["sr"]), "dur_sec": a["dur"], "rms": a["rms"], "peak": a["peak"],
            "active_frac": a["active_frac"], "silent": bool(a["silent"]),
            "silent_reason": list(a["silent_reasons"]), "criterion_id": a.get("criterion_id")}


def wav_check(path):
    """無い・読めない・無音のどれかなら bad=True（対を使えないとする）."""
    if not os.path.exists(path):
        return {"missing": True, "silent": True, "bad": True, "why": "missing"}
    try:
        st = wav_stats(path)
    except Exception as e:  # noqa: BLE001  壊れた wav も使えない音声として扱う
        return {"unreadable": repr(e)[:200], "silent": True, "bad": True, "why": "unreadable"}
    st["bad"] = st["silent"]
    if st["silent"]:
        st["why"] = "silent:" + "+".join(st["silent_reason"])
    return st


def write_json_atomic(obj, path, indent=1):
    """作り直しの置き場（本番ではない）に書くとき用。tmp に書いてから置き換える."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# 生成の条件（本番 7/28–29 と同じ）
# --------------------------------------------------------------------------
GEN_E = os.path.join(PROJ, "scripts", "projection", "eval_utmos_speaker_fixed.py")   # EN/FR/DE/KO（splice、声まね）
GEN_Z = os.path.join(PROJ, "scripts", "projection", "eval_cjk_n300_zhv2.py")        # ZH（見本なし、oeoi）
# 本番の results.json の summary.reference_wav と同じ文字列（ホストの絶対パス。コンテナにも /data が読み取りで見える）
REF_WAV = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav")
REF_PROMPT_SEC = 100.0   # eval_utmos_speaker_fixed.py の既定値（参照を丸ごと使う）
# 本番で渡した参照の書き起こしはログが残っていない。note/2026-07-29_speakersim_report.md は「GT参照転写」。
# seed 42 で対象外の 1 対を作り直して本番と md5 が一致する方を使う（順に試す。一致しなければ先頭を使い、記録する）。
REF_TEXTS = (
    ("gt", "木曽川は、しばしば日本のライン川と呼ばれている。"),        # JSUT BASIC5000_0011 の正解文（data/speaker_variation_refs.json）
    ("whisper_turbo", "基礎川はしばしば日本のライン川と呼ばれている"),  # Whisper large-v3-turbo の書き起こし（logs/autoqueue/multient_en.log、同じ参照 wav）
)

REGEN_ROOT = os.path.join(ER, "_appk_regen")
PROD_RECORD_NAME = "regen2_appk_pairs.json"   # 反映の記録（本番の置き場に 1 つ。既にあれば反映しない）


def plan_path():
    return os.path.join(REGEN_ROOT, "plan.json")


def seed_dir(seed, lang):
    return os.path.join(REGEN_ROOT, f"seed{seed}", lang)


def check_dir(lang, tag):
    return os.path.join(REGEN_ROOT, "check42", f"{lang}_{tag}")


def selection_path(lang):
    return os.path.join(REGEN_ROOT, f"selection_{lang}.json")


def backup_dir(lang):
    return os.path.join(REGEN_ROOT, "original_seed42_backup", lang)


def apply_record_copy(lang):
    return os.path.join(REGEN_ROOT, f"apply_record_{lang}.json")


def assert_regen_path(p):
    p = os.path.abspath(p)
    if not (p + "/").startswith(REGEN_ROOT + "/"):
        raise SystemExit(f"作り直しの出力が {REGEN_ROOT} の外: {p}")
    return p


def assert_prod_write(p):
    """本番（または検査の置き場）へ書く前の確認。原稿と凍結した T5Gemma EN (a) には書かない。symlink には書かない."""
    p = os.path.abspath(p)
    for s in ("/arr_resubmission", "/paper_draft", "/_cc_wavs_a/t5gemma_en", "_t5gemma_en.json", "/naacl2027/"):
        if s in p:
            raise SystemExit(f"書き込み禁止の場所: {p}")
    if not (p + "/").startswith(ER + "/"):
        raise SystemExit(f"eval_results（検査なら APPK_TEST_ROOT）の外には書かない: {p}")
    if os.path.islink(p):
        raise SystemExit(f"symlink には書かない（先の実体を書き換えてしまう）: {p}")
    return p


# --------------------------------------------------------------------------
# 作り直しの計画（無音の検査の結果だけから決める）
# --------------------------------------------------------------------------
UNIT_DECISION = {
    "pair": ("invalid pairs (silent / missing / unreadable wav) are regenerated as a whole pair "
             "(BL and injected side) with seed 43, then 44 only for the pairs still invalid at seed 43; "
             "chosen only by validity. Valid pairs are not regenerated, even when they share a BL sentence."),
    "bl_group": ("invalid pairs (silent / missing / unreadable wav) are regenerated as a whole pair "
                 "(BL and injected side) with seed 43, then 44 if still invalid; chosen only by validity. "
                 "Pairs that share a BL sentence (text_a) are regenerated together and move to seed 44 together "
                 "(valid pairs of the same sentence are replaced too; used only with explicit author approval)."),
}


def build_plan(scan, scan_path, langs=REGEN_LANGS, unit="pair"):
    assert unit in ("bl_group", "pair"), unit
    import collections
    plan = {"created_by": "scripts/regen2/appk_regen_pairs_seed.py --plan_only",
            "scan_path": scan_path, "scan_date": scan.get("date"), "scan_criteria": scan.get("criteria"),
            "silence_helper": SILENCE_HELPER, "silence_criterion": SILENCE_CRITERION,
            "unit": unit, "seeds": list(SEEDS),
            "decision": UNIT_DECISION[unit],
            "per_wav_md5": {lang: {n: v.get("md5") for n, v in scan["per_wav"][lang].items()}
                            for lang in scan["per_wav"]},
            "langs": {}}
    ja = scan["langs"].get("ja")
    if ja and (ja["n_silent_pairs"] or ja.get("n_pairs_missing_or_unreadable")):
        plan["ja_warning"] = ("JA に使えない対がある。JA の作り直しは作っていない（pyopenjtalk を使う発話長推定の扱いを"
                              "著者に確かめてから別に作る）")
    for lang in langs:
        spec = SETS[lang]
        ls = scan["langs"][lang]
        rows = load_rows(lang)
        by = {r["pair_idx"]: r for r in rows}
        assert sorted(by) == list(range(N_EXPECTED)), f"{lang}: pair_idx が 0..299 でない"
        invalid = {}
        for p in ls["silent_pairs"]:
            invalid[p["pair_idx"]] = {s: v.get("silent_reasons") for s, v in p["sides"].items()}
        for i in ls.get("pairs_missing_or_unreadable", []):
            invalid.setdefault(i, {})["missing_or_unreadable"] = True
        groups = collections.defaultdict(list)
        for i in range(N_EXPECTED):
            groups[by[i]["text_a"]].append(i)
        if unit == "bl_group":
            units = sorted({tuple(groups[by[i]["text_a"]]) for i in invalid})
        else:
            units = sorted((i,) for i in invalid)
        targets = sorted({i for u in units for i in u})
        per_wav = scan["per_wav"][lang]

        def _clean(i):
            return all(wav_name(lang, i, s) in per_wav and not per_wav[wav_name(lang, i, s)].get("silent_reasons")
                       for s in SIDES)
        check = next((i for i in range(N_EXPECTED)
                      if i not in targets and _clean(i) and not set(groups[by[i]["text_a"]]) & set(targets)), None)
        assert check is not None, lang
        need = targets + [check]
        plan["langs"][lang] = {
            "set_dir": set_dir(lang), "generator": spec["gen"],
            "invalid_pairs": sorted(invalid), "invalid_detail": {str(i): invalid[i] for i in sorted(invalid)},
            "units": [list(u) for u in units], "targets": targets,
            "added_as_bl_group_member": sorted(set(targets) - set(invalid)),
            "n_invalid": len(invalid), "n_targets": len(targets),
            "n_targets_if_unit_pair": len(invalid),
            "check_pair": check,
            "prod_rows": {str(i): by[i] for i in need},
            "prod_md5": {wav_name(lang, i, s): (per_wav.get(wav_name(lang, i, s)) or {}).get("md5")
                         for i in need for s in SIDES},
        }
    return plan


# --------------------------------------------------------------------------
# 集計の式（生成スクリプトと同じ）
# --------------------------------------------------------------------------
def utmos_summary(rows, old_summary):
    """eval_utmos_speaker_fixed.py main() 580–599 行と同じ式（reference_wav と elapsed_min は元の値のまま）.
    本番 4 言語の results.json で、この式が summary を 1 桁も違わず再現することを確かめてある."""
    import numpy as np
    from scipy import stats as scipy_stats
    bl = np.array([r["bl_utmos"] for r in rows if r.get("bl_utmos") is not None])
    inj = np.array([r["inj_utmos"] for r in rows if r.get("inj_utmos") is not None])
    n_paired = min(len(bl), len(inj))
    if n_paired > 1:
        t_stat, t_pval = scipy_stats.ttest_rel(bl[:n_paired], inj[:n_paired])
        w_stat, w_pval = scipy_stats.wilcoxon(bl[:n_paired], inj[:n_paired])
    else:
        t_stat = t_pval = w_stat = w_pval = None
    return {
        "lang": old_summary["lang"], "n": len(rows), "n_paired": n_paired,
        "reference_wav": old_summary["reference_wav"],
        "bl_utmos": [round(float(bl.mean()), 4), round(float(bl.std()), 4)] if len(bl) else [None, None],
        "inj_utmos": [round(float(inj.mean()), 4), round(float(inj.std()), 4)] if len(inj) else [None, None],
        "mean_diff_inj_minus_bl": round(float((inj[:n_paired] - bl[:n_paired]).mean()), 4) if n_paired else None,
        "paired_t_stat": round(float(t_stat), 4) if t_stat is not None else None,
        "paired_t_pval": round(float(t_pval), 6) if t_pval is not None else None,
        "wilcoxon_stat": round(float(w_stat), 4) if w_stat is not None else None,
        "wilcoxon_pval": round(float(w_pval), 6) if w_pval is not None else None,
        "elapsed_min": old_summary.get("elapsed_min"),
    }


def zh_cjk_summary(rows):
    """eval_cjk_n300_zhv2.py main() 510–531 行が表示する値（本番のファイルには無い。記録にだけ残す）."""
    ok = [r for r in rows if r.get("status") == "success"]
    n = len(ok)
    if not n:
        return {"n_success": 0}
    q = [r for r in ok if "bl_utmos22" in r]
    cos = [r["cos_sim"] for r in ok if "cos_sim" in r]
    return {"n_success": n,
            "bl_acc_pct": round(sum(1 for r in ok if r.get("bl_correct")) / n * 100, 2),
            "oeoi_acc_pct": round(sum(1 for r in ok if r.get("oeoi_correct")) / n * 100, 2),
            "gated_acc_pct": round(sum(1 for r in ok if r.get("gated_correct")) / n * 100, 2),
            "changed_pct": round(sum(1 for r in ok if r.get("changed")) / n * 100, 2),
            "cos_mean": round(sum(cos) / len(cos), 4) if cos else None,
            "bl_utmos22": round(sum(r["bl_utmos22"] for r in q) / len(q), 4) if q else None,
            "oeoi_utmos22": round(sum(r["oeoi_utmos22"] for r in q) / len(q), 4) if q else None,
            "n_quality": len(q)}


# 反映すると古くなる、本番の音声から作った既存の結果（書き換えない。記録と報告に載せるだけ）
STALE_DERIVED = {
    "en": ["speaker_sim_en_n300_splice_wavlm.json", "f0_bl_inj_n300_splice/results.json"],
    "fr": ["speaker_sim_fr_n300_splice_wavlm.json", "f0_bl_inj_n300_splice/results.json"],
    "de": ["speaker_sim_de_n300_splice_wavlm.json", "f0_bl_inj_n300_splice/results.json"],
    "ko": ["speaker_sim_ko_n300_splice_wavlm.json", "f0_bl_inj_n300_splice/results.json"],
    "zh": ["cjk_n300_zh_v2/ctc_lr_judge.json", "cjk_n300_zh_v2/ctc_lr_v3_partial.json",
           "cjk_n300_zh_v2/llr_shift_summary.json"],
}


def zh_shift_impact(targets):
    """ZH の注入成功率 93.67（llr_shift_summary.json、281/300）への影響の幅（判定はやり直さない）.

    既存の ctc_lr_judge.json の行から、声調ありの ΔLLR>0 の数を数え直し（281 を再現することを確かめる）、
    作り直す対のうち今『成功』に数えられている数 k を出す。判定し直したときの値は
    (281 - k) / 300 から (281 - k + len(targets)) / 300 の間になる。"""
    p = os.path.join(REAL_ER, "cjk_n300_zh_v2", "ctc_lr_judge.json")
    s = os.path.join(REAL_ER, "cjk_n300_zh_v2", "llr_shift_summary.json")
    rows = read_json(p)["rows"]
    summ = read_json(s)
    ok = [e for e in rows if "skip" not in e and not e["degenerate"]
          and all(k in e for k in ("bl_lp_src", "bl_lp_tgt", "inj_lp_src", "inj_lp_tgt"))]
    pos = {e["pair_idx"] for e in ok
           if (e["inj_lp_tgt"] - e["inj_lp_src"]) - (e["bl_lp_tgt"] - e["bl_lp_src"]) > 0}
    n = summ["n"]
    k = len(pos & set(targets))
    base = len(pos)
    return {"source": [p, s], "reproduced_count": base, "summary_count": summ["shift_tone_pos"],
            "reproduces_summary": base == summ["shift_tone_pos"],
            "targets": sorted(targets), "targets_counted_positive_now": sorted(pos & set(targets)),
            "range_after_rejudge_pct": [round(100 * (base - k) / n, 2), round(100 * (base - k + len(targets)) / n, 2)],
            "current_pct": summ["shift_tone_pct"],
            "note": "判定（CTC 尤度）はやり直していない。作り直した対の判定は古いまま。値は著者の判断まで変えない。"}
