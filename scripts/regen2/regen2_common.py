#!/usr/bin/env python3
"""G5a（Table 3 の無音の対を seed 43 で対ごとに作り直す）の共通定義（2026-09-29）.

torch を import しない。GPU 側（regen_*_seed.py）と CPU 側（silence_scan / apply / merge）の両方から使う。

対の選び方（著者判断 2026-09-28、2026-09-29 の G5a で Table 3 に適用）:
  * 使えない対（無音の音声・生成の失敗・スパン合わせの失敗）だけを選ぶ。
    無音の基準は全 G 共通（2026-09-29 統一）:
      (RMS < 0.005 かつ peak < 0.05) または 20 ms 区間のうち区間 RMS > 0.01 の割合 < 0.05。
    判定は regen2 共通の部品 scripts/regen2/silence_criterion.py の assess_file だけで行う
    （G1・G2・G3 と同じ部品。このファイルの wav_stats / wav_check はそれを G5a の鍵の名前に写すだけ）。
  * 結果（PIB・書き起こし）では選ばない。seed 43 でも使えなければ seed 44（同じ基準で判定）。
  * 対の中（BL・TGT・INJ、掃引なら 5 点の INJ）は同じ seed にそろえる。
  * T5Gemma-TTS EN の Table 3 (a)（_cc_wavs_a/t5gemma_en、45.33）は触らない。

ホストの CPU で動かすスクリプトは、numpy / torch を読む前に host_cpu_setup() を呼ぶ
（スレッド 13、CPU 0-12、nice 10、GPU を見せない、HF はオフライン）。コマンドは
`taskset -c 0-12 python3 <script> <args>` だけでよい（env・nice・変数の前置きは要らない）。
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import silence_criterion as SC  # noqa: E402  regen2 共通の無音の基準（numpy / soundfile は呼ばれたときに読む）

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
EV = os.path.join(PROJECT_ROOT, "eval_results")

# 検査用: REGEN2_TEST_ROOT を与えると _regen2_* の置き場と予行の報告（DRYRUN_DIR）をすべてその下に移す
# （mock_pipeline_test.py）。本番の音声・採点は読むだけ。apply / merge は、このとき --apply を拒否する。
TEST_ROOT = os.environ.get("REGEN2_TEST_ROOT") or None
if TEST_ROOT and "/eval_results/_regen2_dryrun/" not in os.path.abspath(TEST_ROOT) + "/":
    raise SystemExit("REGEN2_TEST_ROOT は eval_results/_regen2_dryrun/ の下に置く: %s" % TEST_ROOT)
DRYRUN_DIR = TEST_ROOT or os.path.join(EV, "_regen2_dryrun")

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def cap_threads(n="13"):
    """torch / numpy を読む前に呼ぶ（1 本あたり 13 スレッド）."""
    for v in THREAD_VARS:
        os.environ[v] = str(n)


HOST_CPUS = set(range(13))   # CPU 0-12（13 コア）
HOST_NICE = 10


def host_cpu_setup():
    """ホストの CPU で動かすときの設定を、スクリプトの中で済ませる（numpy / torch を読む前に呼ぶ）.

    コマンドの前に env や nice を付けると許可の照合に当たらず承認を求められるので、ここで行う:
      * スレッド上限 13（cap_threads）
      * GPU を見せない（CUDA_VISIBLE_DEVICES=""）、HF はオフライン（HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE。
        既に値があればそのまま）
      * コンテナの外だけ: CPU を 0-12 に絞り（今の affinity との共通部分。広げはしない）、nice を 10 にする
        （既に 10 以上ならそのまま。何度呼んでも同じ）。子プロセス（subprocess）にも引き継がれる。
    """
    cap_threads()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if os.path.isdir("/mnt/scripts"):
        return
    cur = os.sched_getaffinity(0)
    want = (HOST_CPUS & cur) or None
    if want is None:
        raise SystemExit("CPU 0-12 がこのプロセスに許されていない（affinity %s）" % sorted(cur)[:20])
    if want != cur:
        os.sched_setaffinity(0, want)
    os.nice(max(0, HOST_NICE - os.nice(0)))


def host_cpu_state():
    """記録用: 今のプロセスの affinity / nice / スレッド上限."""
    return {"affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0),
            "threads": {v: os.environ.get(v) for v in THREAD_VARS},
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}


# --------------------------------------------------------------------------
# 無音の基準（2026-09-29 に全 G で統一。regen2 共通の部品 silence_criterion.py だけが判定する）
#   silent = (rms < 0.005 and peak < 0.05) or active_frac < 0.05
#   active_frac = 20 ms の区間（重なりなし、端数は捨てる）のうち区間 RMS > 0.01 の割合
#   （区間が 1 つも取れない 20 ms 未満の wav は、全体の RMS > 0.01 なら active_frac = 1）
# 共通の部品の数値が著者の決めた値と違えば止める（g3_common.SILENCE_PARAMS と同じやり方）。
# --------------------------------------------------------------------------
SILENCE_PARAMS = {"RMS_TH": 0.005, "PEAK_TH": 0.05, "FRAME_MS": 20, "FRAME_RMS_TH": 0.01, "ACTIVE_FRAC_TH": 0.05}
_sc_errs = ["silence_criterion.%s = %r（%r のはず）" % (k, getattr(SC, k, None), v)
            for k, v in SILENCE_PARAMS.items() if getattr(SC, k, None) != v]
if getattr(SC, "CRITERION_ID", None) != "regen2-silence-2026-09-29":
    _sc_errs.append("silence_criterion.CRITERION_ID = %r" % getattr(SC, "CRITERION_ID", None))
if _sc_errs:
    raise SystemExit("共通の無音の基準が著者の決めた値と違う。G5a を止める: %s" % _sc_errs)

RMS_SILENT = SC.RMS_TH
PEAK_SILENT = SC.PEAK_TH
FRAME_SEC = SC.FRAME_MS / 1000.0
ACTIVE_RMS = SC.FRAME_RMS_TH
ACTIVE_FRAC_SILENT = SC.ACTIVE_FRAC_TH
SILENCE_RULE = SC.CRITERION_TEXT
SILENCE_CRITERION = {"module": "scripts/regen2/silence_criterion.py", "criterion_id": SC.CRITERION_ID,
                     "module_md5": None,  # 下で埋める（md5_file の定義の後）
                     "rms_lt": RMS_SILENT, "peak_lt": PEAK_SILENT, "frame_ms": SC.FRAME_MS,
                     "frame_rms_gt": ACTIVE_RMS, "active_frac_lt": ACTIVE_FRAC_SILENT, "rule": SILENCE_RULE}


def silence_reasons(rms, peak, active_frac):
    """丸める前の値で判定する（共通の部品の silent_reasons をそのまま呼ぶ）。空なら無音でない."""
    return SC.silent_reasons({"rms": rms, "peak": peak, "active_frac": active_frac})


def wav_stats(path):
    """共通の部品 silence_criterion.assess_file の結果を、G5a の鍵の名前に写す（判定は assess_file のまま。
    記録の値だけ丸める）。読めないファイルは例外になる（呼び出し側は wav_check を使う）."""
    a = SC.assess_file(path)
    return {"dur_sec": a["dur"], "rms": round(a["rms"], 6), "peak": round(a["peak"], 6),
            "active": round(a["active_frac"], 4), "active_frac": round(a["active_frac"], 4),
            "silent": bool(a["silent"]), "silent_reason": list(a["silent_reasons"]),
            "criterion_id": a["criterion_id"]}


def wav_check(path):
    """使えるかの判定に使う形。無い・読めない・無音のどれも "bad": True（対を使えないとする）."""
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


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 共通の部品は他の G も読むので、md5 は記録だけ（一致は求めない。数値と CRITERION_ID は上で確かめた）
SILENCE_CRITERION["module_md5"] = md5_file(SC.__file__)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path, indent=1):
    """本番のファイルは precedent（apply_regen_pairs.py）と同じく上書きで書く（rename しない）."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_json_atomic_regen(obj, path, indent=1):
    """作り直しの置き場（_regen2_*）の帳面だけに使う: tmp に書いてから置き換える
    （run_llasa_eval.atomic_json_dump と同じ。途中で止まっても半端な JSON を残さない）."""
    assert_regen_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Llasa-3B のセル
# date_string: Llasa の chat template は実行日を prompt に入れる（llasa_splice.chat_template_date）。
#   本番の対の日付は wav の mtime（UTC。コンテナの時計は UTC であることを
#   _nontemplate_regen_seed43/meta_t5gemma_en/pairs_module_provenance.json の started/finished と
#   ファイルの mtime で確認）から決めた。変えるのは seed だけにするため、この日付に固定する。
# check_pair: seed 42 のまま同じ日付で作り直して本番と md5 が一致するかを見る対
#   （対象外で、対象の対と同じ実行回・同じ日付に作られた対。無音でない）。
# --------------------------------------------------------------------------
SWEEP_LAYERS = (0, 7, 14, 21, 26)

LLASA_CELLS = {
    "en_a": {
        "lang": "en", "kind": "single", "layer": 26, "pairs": [197, 247],
        "span_extension": False, "date_string": "17 Sep 2026", "check_pair": 0,
        "prod_wav": "_cc_wavs_a/llasa_en", "prod_out": "llasa_cross_culture_en",
        "prod_manifest": "llasa_en_L26_t0.8_p1_manifest.json",
        "orig_cmd": "run_llasa_eval.py --lang en --n 300 --layer 26 "
                    "--wav_dir /mnt/eval_results/_cc_wavs_a/llasa_en "
                    "--out_dir /mnt/eval_results/llasa_cross_culture_en (2026-09-17 09:22-10:31 UTC)",
        "table3": "(a) Llasa-3B EN: MCD 2.12, PIB 11.33",
        "pib_json": {26: "_v1all_llasa_en.json"},
        "mcd_json": {26: "_caseSPTK_whole_llasa_en.json"},
    },
    "zh_a": {
        "lang": "zh", "kind": "single", "layer": 26, "pairs": [145],
        "span_extension": True, "date_string": "18 Sep 2026", "check_pair": 295,
        "prod_wav": "_cc_wavs_a/llasa_zh", "prod_out": "llasa_cross_culture_zh",
        "prod_manifest": "llasa_zh_L26_t0.8_p1_manifest.json",
        "orig_cmd": "run_llasa_eval.py --lang zh --n 300 --layer 26 --span_extension "
                    "(pairs 15/145/295 were generated by the resume run on 2026-09-18 01:22 UTC; "
                    "the other 297 on 2026-09-17/18 without span_extension)",
        "table3": "(a) Llasa-3B ZH: MCD 2.01, PIB 18.00",
        "pib_json": {26: "_v1all_llasa_zh.json"},
        "mcd_json": {26: "_caseSPTK_whole_llasa_zh.json"},
    },
    "en_sweep": {
        "lang": "en", "kind": "sweep", "layers": list(SWEEP_LAYERS), "pairs": [97, 177, 257],
        "span_extension": False, "date_string": "19 Sep 2026", "check_pair": 0,
        "prod_wav": "_cc_wavs_a/llasa_en_sweep_n300", "prod_out": "llasa_layersweep_en_n300",
        "orig_cmd": "run_llasa_layersweep.py --lang en --n 300 (2026-09-19 18:47-21:33 UTC)",
        "table3": "(b) Llasa-3B EN: PIB 13.67--50.67 over l*=1/8/15/22/27",
        "pib_json": {L: "_v1L_llasa_en_L%02d.json" % L for L in SWEEP_LAYERS},
        "mcd_json": {},   # 掃引の MCD は本番に無い（Table 3 (b) は PIB だけ）
    },
    # 対 284 は統一した基準（active_frac < 0.05）で加わった: L00 の INJ が 12.66 秒のうち
    # 声のある区間 0.95%（RMS 0.00686、peak 0.208 のクリックだけ）。2026-09-19 20:00 UTC の回（span_extension なし）。
    "zh_sweep": {
        "lang": "zh", "kind": "sweep", "layers": list(SWEEP_LAYERS), "pairs": [85, 180, 199, 284],
        "span_extension": False, "date_string": "19 Sep 2026", "check_pair": 0,
        "prod_wav": "_cc_wavs_a/llasa_zh_sweep_n300", "prod_out": "llasa_layersweep_zh_n300",
        "orig_cmd": "run_llasa_layersweep.py --lang zh --n 300 (2026-09-19 18:47-21:35 UTC, span_extension off; "
                    "pairs 15/145/295 were added on 2026-09-22 with --span_extension)",
        "table3": "(b) Llasa-3B ZH: PIB 19.67--48.00 over l*=1/8/15/22/27",
        "pib_json": {L: "_v1L_llasa_zh_L%02d_n300.json" % L for L in SWEEP_LAYERS},
        "mcd_json": {},
    },
}
LLASA_CELL_ORDER = ("en_a", "zh_a", "en_sweep", "zh_sweep")


def llasa_layer_dirs(cell):
    """(layer, 相対 subdir) の並び。単一セルは subdir なし."""
    c = LLASA_CELLS[cell]
    if c["kind"] == "single":
        return [(c["layer"], "")]
    return [(L, "L%02d" % L) for L in c["layers"]]


LLASA_KINDS = ("bl", "inj", "tgt")


def llasa_pair_files(root, cell, pidx):
    """その対の wav。単一セルは BL/INJ/TGT、掃引は BL/TGT（最初の層。他の層は同じ inode）と各層の INJ."""
    subs = llasa_layer_dirs(cell)
    first = subs[0][1]
    out = {}
    for k in ("bl", "tgt"):
        out["%s/%s" % (first or "-", k)] = os.path.join(root, cell, "wav", first, "s%04d_%s.wav" % (pidx, k))
    for layer, sub in subs:
        out["%s/inj" % (sub or "-")] = os.path.join(root, cell, "wav", sub, "s%04d_inj.wav" % pidx)
    return out


def read_rows(path, key="rows"):
    """{pair_idx: row}。ファイルが無ければ {}."""
    if not os.path.exists(path):
        return {}
    return {r["pair_idx"]: r for r in read_json(path).get(key, [])}


def llasa_pair_validity(root, cell, pidx):
    """Llasa の対が使えるか（著者判断 (1)）: 対のすべての wav が無音でなく読めること、かつ
    すべての層の progress.json に status == "ok" の行があること（行が無い・"error"・"resumed_wavs_only" は
    生成の失敗として使えない）。run_llasa_eval.run_generation は INJ の wav を保存した後に
    fired != 1 やスパン拡張の食い違いで status="error" にするので、wav だけ見ると失敗を見逃す."""
    st = {"wav": {}, "rows": {}, "reasons": []}
    for name, p in llasa_pair_files(root, cell, pidx).items():
        w = wav_check(p)
        st["wav"][name] = w
        if w["bad"]:
            st["reasons"].append("%s:%s" % (name, w["why"]))
    for layer, sub in llasa_layer_dirs(cell):
        r = read_rows(os.path.join(root, cell, "meta", sub, "progress.json")).get(pidx)
        status = (r or {}).get("status")
        st["rows"]["L%02d" % layer] = {"status": status, "err": (r or {}).get("err")}
        if status != "ok":
            st["reasons"].append("L%02d:row_%s" % (layer, status or "missing"))
    return (not st["reasons"]), st


def multi_pair_validity(d, pidx):
    """2 実体の対が使えるか（著者判断 (1)）:
      * bl / inj1 / inj2 / inj12 の wav が無音でなく読めること（印字値に効く 4 条件）
      * tgt1 / tgt2 の wav があって読めること（無音は記録だけ。対象の選び方と同じ扱い）
      * results.json に 4 条件そろった行、results_tgt.json に tgt1/tgt2 のそろった行があること
        （generate_multientity は合成・採点・スパン合わせの失敗を skipped に入れ、行を作らない）."""
    st = {"wav": {}, "rows": {}, "reasons": [], "tgt_silent_recorded_only": []}
    for k in MULTI["conds"] + MULTI["tgt_conds"]:
        w = wav_check(os.path.join(d, "s%04d_%s.wav" % (pidx, k)))
        st["wav"][k] = w
        if k in MULTI["conds"]:
            if w["bad"]:
                st["reasons"].append("%s:%s" % (k, w["why"]))
        elif w.get("missing") or w.get("unreadable"):
            st["reasons"].append("%s:%s" % (k, w["why"]))
        elif w["silent"]:
            st["tgt_silent_recorded_only"].append(k)
    for name, conds in (("results.json", MULTI["conds"]), ("results_tgt.json", MULTI["tgt_conds"])):
        p = os.path.join(d, name)
        doc = read_json(p) if os.path.exists(p) else {"rows": [], "skipped": []}
        r = {x["pair_idx"]: x for x in doc.get("rows", [])}.get(pidx)
        sk = [s.get("reason") for s in doc.get("skipped", []) if s.get("pair_idx") == pidx]
        have = sorted((r or {}).get("conds", {}))
        st["rows"][name] = {"row": r is not None, "conds": have, "skipped": [str(x)[:300] for x in sk]}
        if r is None:
            st["reasons"].append("%s:row_missing%s" % (name, "(skipped)" if sk else ""))
        elif not set(conds) <= set(have):
            st["reasons"].append("%s:conds_incomplete" % name)
    return (not st["reasons"]), st


# --------------------------------------------------------------------------
# 生成の失敗の扱い（著者判断 (1): 生成の失敗は seed 44 に回す。環境の失敗は回さない）
#
# 環境の失敗: GPU メモリ不足と通信・ハードウェアの失敗だけ。seed の良し悪しではないので、使えない対として
#   次の seed に回さず、ジョブを止めて同じ seed で出し直す（作り直しの置き場には、その対を残さない）。
# CUDA の実行時の失敗（device-side assert、illegal memory access など。上の環境の失敗を除く）:
#   特定の seed で決まって起きるモデル側の失敗（添字の範囲外など）か、一時的な環境の失敗かは 1 回では
#   分からない。しかも CUDA の文脈が壊れるので、同じプロセスで後に処理した対も巻き添えで失敗する
#   （run_llasa_eval.run_generation も generate_multientity も、対ごとの例外を拾って次の対へ進む）。そこで:
#     * 1 回の試行の中で最初に CUDA の失敗が出た対を「疑わしい対」とし、それより後に処理した対は
#       （ok でも）本体に写さない（次の実行で同じ seed のまま作る）。ジョブは止める。
#     * 疑わしい対が、この置き場（seed・層・段階）で初めて CUDA の失敗を出したときは記録（strike）だけ残す。
#       出し直すと同じ seed で作り直す（一時的な失敗ならここで通る）。
#     * 同じ対が同じ置き場で 2 回目の CUDA の失敗を出したら、決まった失敗とみなし、生成の失敗として
#       本体に書く（→ seed 44）。これで同じ seed のまま止まり続けることはない（出し直しは最大 2 回）。
#   その他の例外（モデル側の RuntimeError など）は、これまでどおり生成の失敗として本体に書く（→ seed 44）。
# --------------------------------------------------------------------------
INFRA_ERR_PATTERNS = ("out of memory", "outofmemoryerror", "cublas_status_alloc_failed",
                      "cudaerrormemoryallocation", "nccl error", "cudnn_status_alloc_failed", "ecc error")
CUDA_FAULT_PATTERNS = ("cuda error", "device-side assert", "illegal memory access", "cudnn_status",
                       "cublas_status", "cuda runtime error", "cudaerror", "acceleratorerror")
CUDA_STRIKES_TO_FAIL = 2


def is_infra_error(text):
    t = str(text or "").lower()
    return any(p in t for p in INFRA_ERR_PATTERNS)


def is_cuda_fault(text):
    t = str(text or "").lower()
    return (not is_infra_error(t)) and any(p in t for p in CUDA_FAULT_PATTERNS)


def read_strikes(path):
    """{pair_idx: [記録, ...]}。ファイルが無ければ {}."""
    if not os.path.exists(path):
        return {}
    return {int(k): v for k, v in read_json(path).get("strikes", {}).items()}


def triage_attempt(order, errs, strikes):
    """1 回の試行の失敗を振り分ける（上の説明の規則）.

    order  : その試行で処理した順の対（run_generation / generate_multientity は渡した順に処理する）
    errs   : {pair_idx: エラー文字列}（失敗した対だけ。ok の対は入れない）
    strikes: read_strikes の値（この置き場での前の試行の CUDA の失敗）
    返り値 : {"hold": {p: 理由}          本体に書かない対（次の実行で同じ seed のまま作る）,
              "write_fail": {p: err}     生成の失敗として本体に書く対（→ seed 44）,
              "new_strikes": {p: 記録}   CUDA の失敗の記録に足すもの,
              "stop": [p, ...]           ジョブを止める理由になった対（空なら続けてよい）,
              "cut_after": p or None     CUDA の失敗が最初に出た対}
    """
    out = {"hold": {}, "write_fail": {}, "new_strikes": {}, "stop": [], "cut_after": None}
    cut = None
    for p in order:
        if cut is not None:
            out["hold"][p] = "processed after the CUDA fault of pair %d in the same process" % cut
            continue
        e = errs.get(p)
        if e is None:
            continue
        e = str(e)
        if is_infra_error(e):
            out["hold"][p] = "infra: " + e[:300]
            out["stop"].append(p)
        elif is_cuda_fault(e):
            cut = p
            out["cut_after"] = p
            out["stop"].append(p)
            n_prev = len(strikes.get(p, []))
            decided = "generation_failure" if n_prev + 1 >= CUDA_STRIKES_TO_FAIL else "retry_same_seed"
            out["new_strikes"][p] = {"err": e[:500], "strike": n_prev + 1, "decided": decided}
            if decided == "generation_failure":
                out["write_fail"][p] = e
            else:
                out["hold"][p] = "first CUDA fault at this seed (retry with the same seed): " + e[:300]
        else:
            out["write_fail"][p] = e
    return out


def record_strikes(path, new_strikes, attempt):
    """CUDA の失敗の記録を足す（作り直しの置き場の中だけ。tmp → 置き換え）."""
    if not new_strikes:
        return
    import time as _time
    doc = read_json(path) if os.path.exists(path) else {"strikes": {}}
    for p, rec in sorted(new_strikes.items()):
        doc["strikes"].setdefault(str(p), []).append(
            dict(rec, attempt=attempt, utc=_time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime())))
    doc["rule"] = ("pair held and retried with the same seed on its first CUDA fault in this place; "
                   "a second CUDA fault of the same pair here is a generation failure (-> fallback seed)")
    write_json_atomic_regen(doc, path)


# TEST_ROOT（REGEN2_TEST_ROOT）の定義はファイル冒頭（DRYRUN_DIR もそれに従う）
_BASE = TEST_ROOT or EV


def refuse_apply_in_test(apply_flag):
    if TEST_ROOT and apply_flag:
        raise SystemExit("REGEN2_TEST_ROOT が設定されているときは --apply しない")


def llasa_regen_root(seed):
    return os.path.join(_BASE, "_regen2_seed%d" % seed, "llasa")


def llasa_final_root():
    return os.path.join(_BASE, "_regen2_final", "llasa")


def llasa_backup_root(seed=43):
    return os.path.join(_BASE, "_regen2_seed%d" % seed, "original_seed42_backup", "llasa")


# --------------------------------------------------------------------------
# T5Gemma-TTS EN の 2 実体（Table 3 (c)）
# --------------------------------------------------------------------------
MULTI = {
    "name": "multientity_en_300", "pairs_module": "en_multientity_300", "lang": "en",
    "pairs": [2, 5, 30, 47, 141, 153, 241, 257, 276],
    "conds": ["bl", "inj1", "inj2", "inj12"], "tgt_conds": ["tgt1", "tgt2"],
    # 本番は --reference_text なしで Whisper large-v3-turbo が書き起こした参照文を使った
    # （logs/autoqueue/multient_en.log 13 行、他 9 本のログでも同じ文字列）。
    # 同じ文字列を明示的に渡し、書き起こしのぶれを入れない。
    "reference_text": "基礎川はしばしば日本のライン川と呼ばれている",
    "reference_wav": os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav"),
    "check_pair": 3,          # 対象外・無音でない・0..2（tgt が別の回）を避けた最初の対
    "table3": "(c) T5Gemma-TTS EN: Baseline 96.17 / One 95.33 / Both 96.33 (% of 600 slots)",
    "asr_json": "multientity_asr_scores.json",
    "stale_derived": ["ceiling_asr.json", "pib_multi.json",
                      "../_v1ME_t5en_s1_one.json", "../_v1ME_t5en_s1_both.json",
                      "../_v1ME_t5en_s2_one.json", "../_v1ME_t5en_s2_both.json"],
    # tgt だけが無音の対（tgt は Table 3 の印字値に使われていない: (c) は bl/inj* の書き起こしだけ）。
    # 対象の選び方でも作り直しの判定でも、tgt1/tgt2 は「無い・読めない」ときだけ使えないとし、無音は記録だけ。
    # 2026-09-29 の統一した基準の走査（silence_scan_g5a.py --stage pre）で確定。対 36 の tgt2 は
    # RMS 0.0058・peak 0.029 だが active_frac 0.0488 < 0.05 なので、統一した基準では無音に入る。
    "tgt_only_silent_excluded": [36, 41, 91, 198, 209],
}


def multi_regen_dir(seed):
    return os.path.join(_BASE, "_regen2_seed%d" % seed, MULTI["name"])


def multi_final_dir():
    return os.path.join(_BASE, "_regen2_final", MULTI["name"])


def multi_backup_dir(seed=43):
    return os.path.join(_BASE, "_regen2_seed%d" % seed, "original_seed42_backup", MULTI["name"])


# --------------------------------------------------------------------------
# 触ってはいけないもの
# --------------------------------------------------------------------------
FROZEN_T5GEMMA_EN = {
    "wav_dir": "_cc_wavs_a/t5gemma_en",
    "files": ["_v1all_t5gemma_en.json", "_caseSPTK_whole_t5gemma_en.json",
              "_caseSPTK_phperPER_t5gemma_en.json"],
}
FORBIDDEN_SUBSTR = ("/_cc_wavs_a/t5gemma_en", "_t5gemma_en.json", "/arr_resubmission", "/paper_draft",
                    "/naacl2027/")


def assert_writable_target(path):
    """本番へ書く前の確認。凍結した T5Gemma EN (a) と原稿には絶対に書かない."""
    p = os.path.abspath(path)
    for s in FORBIDDEN_SUBSTR:
        if s in p:
            raise SystemExit("書き込み禁止の場所: %s（%s）" % (p, s))
    return p


def assert_regen_path(path):
    """作り直しの出力は必ず eval_results/_regen2_* の下に置く."""
    p = os.path.abspath(path)
    if "/eval_results/_regen2_" not in p:
        raise SystemExit("作り直しの出力が _regen2_* の外: %s" % p)
    return p


def frozen_digest():
    """凍結した T5Gemma EN (a) の音声と採点ファイルの md5 をまとめたもの（前後比較用）."""
    h = hashlib.md5()
    wd = os.path.join(EV, FROZEN_T5GEMMA_EN["wav_dir"])
    names = sorted(os.listdir(wd))
    for n in names:
        h.update(n.encode())
        h.update(md5_file(os.path.join(wd, n)).encode())
    for f in FROZEN_T5GEMMA_EN["files"]:
        p = os.path.join(EV, f)
        h.update(f.encode())
        h.update(md5_file(p).encode() if os.path.exists(p) else b"missing")
    return {"n_wav_files": len(names), "digest": h.hexdigest()}


# --------------------------------------------------------------------------
# 採点の集計式（元のスクリプトと同じ式。merge で使う。dry-run で既存ファイルの
# summary を 1 桁も違わず再現することを確かめる）
# --------------------------------------------------------------------------
def pib_summary(rows, old_summary):
    """speechsplice_release/src/phoneme_per.py 219-257 行と同じ式."""
    import math
    import numpy as np
    valid = [r for r in rows if r.get("inj_per_vs_tgt") is not None]
    nv = len(valid)
    if nv > 0:
        bl_per_arr = np.array([r["bl_per_vs_tgt"] for r in valid])
        inj_per_arr = np.array([r["inj_per_vs_tgt"] for r in valid])
        delta = bl_per_arr - inj_per_arr
        ib_n = sum(1 for r in valid if r["inj_better"])
        z = 1.96
        p = ib_n / nv
        ci_half = z * math.sqrt(p * (1 - p) / nv + z * z / (4 * nv * nv)) / (1 + z * z / nv)
        from scipy import stats
        try:
            t_, t_p = stats.ttest_rel(inj_per_arr, bl_per_arr)
            w_, w_p = stats.wilcoxon(inj_per_arr, bl_per_arr, zero_method="zsplit")
        except Exception:
            t_p = w_p = float("nan")
        cd = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else None
    else:
        bl_per_arr = inj_per_arr = delta = np.array([])
        ib_n = 0
        ci_half = 0
        t_p = w_p = float("nan")
        cd = None
    return {
        "wav_dir": old_summary["wav_dir"], "lang": old_summary["lang"],
        "n_target": old_summary["n_target"], "n_valid": nv,
        "bl_per_vs_tgt_mean": round(float(bl_per_arr.mean()), 4) if nv else None,
        "bl_per_vs_tgt_std": round(float(bl_per_arr.std(ddof=1)), 4) if nv > 1 else None,
        "inj_per_vs_tgt_mean": round(float(inj_per_arr.mean()), 4) if nv else None,
        "inj_per_vs_tgt_std": round(float(inj_per_arr.std(ddof=1)), 4) if nv > 1 else None,
        "delta_per_mean": round(float(delta.mean()), 4) if nv else None,
        "delta_per_std": round(float(delta.std(ddof=1)), 4) if nv > 1 else None,
        "inj_better_n": ib_n,
        "inj_better_pct": round(100 * ib_n / nv, 2) if nv else 0,
        "inj_better_ci95_half_pp": round(100 * ci_half, 2) if nv else 0,
        "paired_t_pval": float(t_p),
        "wilcoxon_pval": float(w_p),
        "cohens_d_paired": round(cd, 4) if cd is not None else None,
    }


def mcd_summary(rows, old_summary):
    """speechsplice_release/src/mcd_pysptk.py の集計（whole）と同じ式."""
    import numpy as np
    valid = [r for r in rows if r.get("mcd_inj_vs_bl") is not None and r.get("mcd_inj_vs_tgt") is not None
             and r.get("mcd_bl_vs_tgt") is not None]

    def stat(key):
        arr = np.array([r[key] for r in valid])
        if arr.size == 0:
            return None
        return {"n": int(arr.size), "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else None,
                "median": round(float(np.median(arr)), 4)}

    shift_pval = shift_d = shift_mean = None
    if valid:
        bl_tgt = np.array([r["mcd_bl_vs_tgt"] for r in valid])
        inj_tgt = np.array([r["mcd_inj_vs_tgt"] for r in valid])
        shift = bl_tgt - inj_tgt
        try:
            from scipy import stats
            t_, p_ = stats.ttest_1samp(shift, 0)
            shift_pval = float(p_)
            shift_d = float(shift.mean() / shift.std(ddof=1)) if shift.std(ddof=1) > 0 else None
            shift_mean = float(shift.mean())
        except Exception:
            pass
    return {
        "wav_dir": old_summary["wav_dir"], "lang": old_summary["lang"], "mode": old_summary["mode"],
        "order": old_summary["order"], "n_target": old_summary["n_target"], "n_valid": len(valid),
        "unit": old_summary["unit"],
        "mcd_inj_vs_bl": stat("mcd_inj_vs_bl"), "mcd_inj_vs_tgt": stat("mcd_inj_vs_tgt"),
        "mcd_bl_vs_tgt": stat("mcd_bl_vs_tgt"),
        "shift_bl_tgt_minus_inj_tgt": shift_mean, "shift_pval": shift_pval, "shift_cohens_d": shift_d,
    }


ASR_COND_EXPECT = {"bl": ("a", "a"), "inj1": ("b", "a"), "inj2": ("a", "b"), "inj12": ("b", "b")}


def _wilson(k, n, z=1.96):
    import math
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, h


def asr_summary(out_rows, old_summary):
    """scripts/analysis/score_multientity.py 166-183 行と同じ式（n_error は行の *_err から数え直す）."""
    n_err = sum(1 for r in out_rows for k in r if k.endswith("_err"))
    summary = {"dir": old_summary["dir"], "lang": old_summary["lang"], "asr_model": old_summary["asr_model"],
               "n_attempted": len(out_rows), "n_error": n_err}
    for cond in ASR_COND_EXPECT:
        for key in ("slot1_ok", "slot2_ok", "both_ok"):
            vals = [r.get(f"{cond}_{key}") for r in out_rows]
            vals = [v for v in vals if v is not None]
            k, n = sum(1 for v in vals if v), len(vals)
            if n:
                p, h = _wilson(k, n)
                summary[f"{cond}_{key}"] = {"k": k, "n": n, "rate": round(k / n, 4),
                                            "wilson": round(p, 4), "half": round(h, 4)}
        for key in ("slot1_stale", "slot2_stale"):
            vals = [r.get(f"{cond}_{key}") for r in out_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                k = sum(1 for v in vals if v)
                summary[f"{cond}_{key}"] = {"k": k, "n": len(vals), "rate": round(k / len(vals), 4)}
    return summary


def table3c_cells(out_rows):
    """Table 3 (c): 600 枠のうち意図どおり読まれた割合と Wilson 半幅（n=600）."""
    def pooled(k1, k2):
        k = sum(1 for r in out_rows if r.get(k1)) + sum(1 for r in out_rows if r.get(k2))
        n = 2 * len(out_rows)
        return {"k": k, "n": n, "pct": round(100 * k / n, 2), "wilson_half_pp": round(100 * _wilson(k, n)[1], 2)}
    return {"Baseline": pooled("bl_slot1_ok", "bl_slot2_ok"),
            "One": pooled("inj1_slot1_ok", "inj2_slot2_ok"),
            "Both": pooled("inj12_slot1_ok", "inj12_slot2_ok")}


# ==========================================================================
# 「Thank you.」としか書き起こされない音声の作り直し（著者判断 2026-09-30、--campaign thankyou）
#
# Table 3 (c)（T5Gemma-TTS EN 2 実体、multientity_en_300）で、無音の基準には通るが音が小さく、
# Whisper の書き起こしが「Thank you.」だけの 6 本を生成の失敗とみなし、対ごと（6 本とも）次の seed で作る:
#   対 21 / 43 / 251 / 272（inj1）、111（inj2）→ seed 43（まだ使えなければ 44）
#   対 257（inj2。9/29 の G5a で seed 43 に作り直した対）→ seed 44（まだ使えなければ止めて報告）
# 使えるかの基準（結果を見る前に固定）:
#   音声が使えない ⇔ silence_criterion.py で無音、または本番の 2 実体の採点（score_multientity.py）の
#   Whisper の書き起こしから空白と句読点を除き小文字にしたものが "thankyou" とちょうど一致する。
#   対が使える ⇔ bl / inj1 / inj2 / inj12 がすべて使える（tgt1 / tgt2 はこれまでどおり: 有る・読めることだけ
#   求め、無音は記録だけ。results.json / results_tgt.json の行も求める = multi_pair_validity）。
#   seed はこの基準だけで選ぶ。
# 置き場はすべて新しいフォルダ eval_results/_regen2_thankyou/ の下（G5a の _regen2_seed43 / _regen2_final には
# 書かない）。記録は regen2_thankyou_*.json（G5a の regen2_seed43_*.json は上書きしない）。
# ==========================================================================
import string as _string  # noqa: E402
import unicodedata as _ud  # noqa: E402

THANKYOU = {
    "id": "regen2-thankyou-2026-09-30",
    "pairs": [21, 43, 111, 251, 257, 272],
    # 対ごとに試す seed の順（先頭から。使えた最初の seed を採る。最後まで使えなければ unresolved）
    "seed_plan": {21: [43, 44], 43: [43, 44], 111: [43, 44], 251: [43, 44], 257: [44], 272: [43, 44]},
    # 著者が挙げた本番の音声（この基準で本番の書き起こしを走査したときに当たるはずの 6 本）
    "flagged_prod_clips": {21: ["inj1"], 43: ["inj1"], 111: ["inj2"], 251: ["inj1"], 257: ["inj2"], 272: ["inj1"]},
    "decision": ("Author decision 2026-09-30: in Table 3 (c) (T5Gemma-TTS EN two-entity, multientity_en_300) six "
                 "clips pass the silence rule but are quiet and Whisper transcribes them only as 'Thank you.': "
                 "pair 21 (inj1), 43 (inj1), 111 (inj2), 251 (inj1), 257 (inj2; regenerated with seed 43 on "
                 "2026-09-29 by G5a), 272 (inj1). They are treated as generation failures and each WHOLE pair "
                 "(bl, inj1, inj2, inj12, tgt1, tgt2) is regenerated with the next seed: pairs 21, 43, 111, 251, "
                 "272 -> seed 43 (then 44 if still invalid); pair 257 -> seed 44 (then stop and report if still "
                 "invalid)."),
    "rule": ("A clip is invalid iff it is silent under scripts/regen2/silence_criterion.py OR its Whisper "
             "transcript (scripts/analysis/score_multientity.py, the production two-entity scorer, run unchanged "
             "on the seed folder) stripped of whitespace and punctuation and lower-cased is exactly 'thankyou'. "
             "A pair is valid iff bl, inj1, inj2 and inj12 are all valid (tgt1/tgt2 must exist and be readable; "
             "their silence is recorded only) and its results.json / results_tgt.json rows exist. Seeds are "
             "chosen only by this rule (fixed 2026-09-30 before any result)."),
    "normalization": ("lower-case, then drop every character c with c.isspace(), c in string.punctuation, or "
                      "unicodedata.category(c) starting with 'P'"),
    "frozen_digest_prefix": "0c9b56e5",
    "record_pairs": "regen2_thankyou_pairs.json",
    "record_scores": "regen2_thankyou_scores.json",
    "meta_name": "regen2_thankyou_meta.json",
    "asr_final_name": "multientity_asr_scores_regen.json",
    # 本番の Table 3 (c) の今の値（9/29 の G5a の入れ替え後。regen2_seed43_scores.json の table3c_after）
    "prod_table3c_now_record": "regen2_seed43_scores.json",
}
# G5a の作り直し（読むだけ。REGEN2_TEST_ROOT に依らず本物を指す）: 対 257 の今の本番の音声がこれと同じか確かめる
G5A_MULTI_FINAL_REAL = os.path.join(EV, "_regen2_final", MULTI["name"])


def thankyou_root():
    return os.path.join(_BASE, "_regen2_thankyou")


def thankyou_seed_dir(seed):
    return os.path.join(thankyou_root(), "seed%d" % seed, MULTI["name"])


def thankyou_final_dir():
    return os.path.join(thankyou_root(), "final", MULTI["name"])


def thankyou_backup_dir():
    """差し替え前の本番（対 257 は G5a の seed 43 の音声）の退避先."""
    return os.path.join(thankyou_root(), "backup_before_thankyou", MULTI["name"])


def thankyou_dryrun_dir():
    return os.path.join(thankyou_root(), "dryrun")


def job_env():
    """HF はプロジェクトの .cache/huggingface（コンテナでは /mnt/.cache/huggingface）でオフライン、
    numba の cache もプロジェクトの .cache/numba、スレッドは 13 まで（torch / numpy を読む前に呼ぶ）."""
    hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
    os.environ["HF_HOME"] = hf
    os.environ["HF_HUB_CACHE"] = os.path.join(hf, "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(PROJECT_ROOT, ".cache", "numba")
    cap_threads()
    os.environ["GEN_THREADS"] = "13"


def normalize_transcript(txt):
    """小文字にし、空白と句読点（string.punctuation と Unicode の P* 類）を除く."""
    return "".join(ch for ch in str(txt).lower()
                   if not (ch.isspace() or ch in _string.punctuation or _ud.category(ch).startswith("P")))


def is_thankyou(txt):
    return txt is not None and normalize_transcript(txt) == "thankyou"


def thankyou_scan(asr_rows, pairs=None):
    """{pair: [cond, ...]}: 書き起こしが "thankyou" に当たる bl / inj1 / inj2 / inj12（当たる対だけ）."""
    out = {}
    for r in asr_rows:
        if pairs is not None and r["pair_idx"] not in pairs:
            continue
        hit = [k for k in MULTI["conds"] if is_thankyou(r.get(k + "_transcript"))]
        if hit:
            out[r["pair_idx"]] = hit
    return out


def thankyou_pair_validity(d, pidx, asr_row):
    """対が使えるか（THANKYOU["rule"]）。返り値 (valid, st)。st["undetermined"] は、無音などの理由では使えるのに
    bl / inj1 / inj2 / inj12 のどれかの書き起こしが無い（採点の失敗）ために決められないとき True."""
    ok, st = multi_pair_validity(d, pidx)
    st["asr"], st["thankyou_clips"], st["asr_missing"] = {}, [], []
    for k in MULTI["conds"]:
        t = (asr_row or {}).get(k + "_transcript")
        err = (asr_row or {}).get(k + "_err")
        st["asr"][k] = {"transcript": t, "normalized": None if t is None else normalize_transcript(t), "err": err}
        if not isinstance(t, str) or err is not None:
            st["asr_missing"].append(k)
        elif is_thankyou(t):
            st["thankyou_clips"].append(k)
            st["reasons"].append("%s:transcript_thankyou" % k)
    st["silence_rows_ok"] = ok
    st["undetermined"] = bool(ok and st["asr_missing"])
    valid = bool(ok and not st["thankyou_clips"] and not st["asr_missing"])
    return valid, st


def utmos_means(summary):
    """results.json の summary.utmos の平均（bl / inj1 / inj2 / inj12）。explanation_of_revisions.tex ~216 行は
    bl / inj1 / inj12 を 2 桁で使う."""
    u = (summary or {}).get("utmos") or {}
    return {c: (u.get(c) or [None])[0] for c in MULTI["conds"]}


def before_after_lines(t3_before, t3_after, ut_before, ut_after):
    """Table 3 (c)（k/600）と UTMOS の平均の前後を、人が読む行にする."""
    def f3(c):
        return "n/a" if not c else "%d/%d = %.2f%% (±%.2f)" % (c["k"], c["n"], c["pct"], c["wilson_half_pp"])

    def fu(v):
        return "n/a" if v is None else "%.4f (%.2f)" % (v, v)
    lines = ["[before -> after] Table 3 (c) T5Gemma-TTS EN two-entity (slots read as intended, k/600)"]
    for cell in ("Baseline", "One", "Both"):
        lines.append("  %-8s %-26s -> %s" % (cell, f3((t3_before or {}).get(cell)), f3((t3_after or {}).get(cell))))
    lines.append("[before -> after] UTMOS mean, results.json summary.utmos "
                 "(explanation_of_revisions.tex ~216 uses bl / inj1 / inj12)")
    for c, tag in (("bl", "baseline"), ("inj1", "first entity corrected"), ("inj12", "both corrected"),
                   ("inj2", "second entity corrected; not in the EoR sentence")):
        lines.append("  %-5s %-15s -> %-15s %s" % (c, fu((ut_before or {}).get(c)), fu((ut_after or {}).get(c)), tag))
    return lines
