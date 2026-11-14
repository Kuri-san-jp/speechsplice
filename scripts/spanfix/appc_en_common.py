#!/usr/bin/env python3
"""付録 C（tab:ja_crossattn_phoneme）の英語の行を、直したスパン関数で作り直すための共通の定義（2026-10-01）.

torch を読まない。GPU 側（appc_en_gen.py）と CPU 側（appc_en_span_check.py / appc_en_scan.py /
appc_en_aggregate.py）の全部が使う。

元の実行（5/10, 修正前）:
  SLURM_TIME=06:00:00 slurm_run.sh \
    python scripts/projection/eval_crossattn_entity_per.py --lang en --n 300
  （.perm_cmds.txt 15571-15572 行。出力 eval_results/crossattn_entity_per_en_n300/、
    ログ eval_results/crossattn_entity_per_en_resubmit_run.log、elapsed_min 260.08）
  対は data/en_heteronyms_500.py の build_500_pairs()[:300]（30 語 x 10 文。word_a == word_b）。

作り直しの決まり（著者の決定 2026-10-01）:
  * 本番のスクリプト scripts/projection/eval_crossattn_entity_per.py を改変せずに main() を呼ぶ。
    引数は元と同じ（--lang en --n 300）に --output_dir（新しい置き場）だけを足す。
  * スパン関数は scripts/encoder_injection_t5gemma.py の find_entity_token_range（7/11 修正後の版）。
  * seed 42 で 300 対すべて。使えない対（無音・整列の失敗・生成の失敗）は対ごと seed 43、
    それでも使えなければ seed 44。選ぶ基準は有効性だけで、数値は見ない。45 以上は著者の判断。
  * 新しい置き場 eval_results/_spanfix_appC_en/ にだけ書く。既存の結果は読むだけ。何も消さない。

置き場:
  eval_results/_spanfix_appC_en/
    span_check/span_check_<日時>.json      CPU のスパンの確かめ（appc_en_span_check.py）
    seed42/part00 .. part05/               seed 42 の 6 分割（各 50 対）
    seed43/part00 ..                       seed 42 で使えなかった対（50 対ずつ）
    seed44/part00 ..                       seed 43 で使えなかった対
    seed<S>/retry01 ..                     同じ seed のやり直し（環境の問題のときだけ）
    status/status_<日時>.json               状態の集計（appc_en_scan.py）
    aggregate/aggregate_<日時>.json         新旧の表の値（appc_en_aggregate.py）
  各 batch（part/retry）の中:
    manifest.json        対象の対・seed・引数・コードの md5（最初の起動で作る）
    checkpoint.json      本番 main() の checkpoint（対象以外は placeholder の行）
    results.json         本番 main() の出力（batch の分だけの集計。使わない）
    runs/run<NN>/        起動ごとの記録（launch.json, events.jsonl, wav/, checkpoint_at_launch.json）
"""
import glob
import hashlib
import json
import math
import os
import re
import time

HOST_ROOT = os.environ.get("REPO_ROOT", "/mnt")
CONTAINER_ROOT = "/mnt"
PROJECT_ROOT = CONTAINER_ROOT if os.path.isdir("/mnt/scripts") else HOST_ROOT

# ---------------------------------------------------------------- 本番のファイル（読むだけ）
EVAL_REL = "scripts/projection/eval_crossattn_entity_per.py"
INJ_REL = "scripts/encoder_injection_t5gemma.py"
FA_REL = "scripts/projection/forced_alignment_utils.py"
PAIRS_REL = "data/en_heteronyms_500.py"
SHIM_REL = "scripts/nontemplate/gen_t5gemma_pairs_module.py"
SILENCE_REL = "scripts/regen2/silence_criterion.py"
LEGACY_SPAN_REL = "release_artifacts/scripts/encoder_injection_t5gemma.py.bak_spanfix_20260917_095951"
VENDOR_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
VENDOR_FILES = ("inference_tts_utils.py", "duration_estimator.py", "data/tokenizer.py")

# 本番のファイルの md5（2026-10-01 に確かめた値）。違えば GPU ジョブはモデルを読まずに止まる
PINNED_MD5 = {
    EVAL_REL: "7d53ab5c5379e569c08336581084b38a",   # 5/13 15:59 から変わっていない（5/10 の実行の後に T5GEMMA_ROOT だけ変更）
    INJ_REL: "4f36fa570145cde805072438d8a39bb1",    # 7/11 のスパン修正の後の版（offset_mapping）
    FA_REL: "be2b300a061e1374509bd1aa75f41b5d",     # Whisper の単語の時刻（4/30 から変わっていない）
    PAIRS_REL: "5b12509ac76de20d044a590f1248f4a0",  # 4/29 から変わっていない
    SHIM_REL: "53eb4bfc1008beb361a23c6adfc429ba",   # 9/28 の前例の offline の読み替え（G3 と同じ）
}
LEGACY_SPAN_MD5 = "a7aca46875c8b405a89bc3dbfbabddb4"  # 修正前のスパン関数（比べるためだけに読む）

OLD_DIR_REL = "eval_results/crossattn_entity_per_en_n300"
OLD_MD5 = {"results.json": "889656b9576874fe3f0fb5312eb1bf7e",
           "checkpoint.json": "78ad1112e039ac0c325ff363e585d887"}

# 付録 K（tab:six_language）の英語の対の正本（scripts/regen2/appk_common.py の SETS["en"]、
# appk_enjudge_common.py の PROD_CKPT と同じ）。checkpoint.json の "rows" の 300 行。
# 2026-10-01 に build_500_pairs()[:300] と (pair_idx, text_a, text_b, word_a, word_b) が 300/300 同じ並びと確かめた。
# 注意: eval_results/crossattn_mask_en_n300_filtered は 9/29 の G1 より前の「IPA 変化」の列の出どころで、付録 K の対ではない。
APPK_EN_CKPT_REL = "eval_results/utmos_speaker_fixed_en_n300_splice/checkpoint.json"

T5GEMMA_TTS_ID = "Aratako/T5Gemma-TTS-2b-2b"
TOKENIZER_ID = "google/t5gemma-2b-2b-ul2"
LANG = "en"
N_PAIRS = 300
PART_SIZE = 50
N_PARTS_SEED42 = N_PAIRS // PART_SIZE  # 6
PROD_ARGS = ["--lang", LANG, "--n", str(N_PAIRS)]  # 元の引数。この後に --output_dir だけを足す

CONDS = ("bl", "inj", "bl_mask", "inj_mask", "tgt")      # 生成の順（run_condition x4 → synth_simple）
ALIGN_ORDER = ("tgt", "bl", "inj", "bl_mask", "inj_mask")  # entity_ipa の呼ばれる順
PER_CONDS = ("bl", "inj", "bl_mask", "inj_mask")
# 生成の各呼び出しの「形」: (テキスト, 遮断の有無, 直前のスパン呼び出しの対象テキスト)
SLOT_SIGNATURE = {
    "bl": ("a", False, ("a",)),
    "inj": ("a", False, ("a", "b")),
    "bl_mask": ("a", True, ("a",)),
    "inj_mask": ("a", True, ("a", "b")),
    "tgt": ("b", False, ()),
}

SEEDS = (42, 43, 44)
OUT_ROOT_REL = "eval_results/_spanfix_appC_en"
SPAN_CHECK_DIR_REL = OUT_ROOT_REL + "/span_check"
STATUS_DIR_REL = OUT_ROOT_REL + "/status"
AGG_DIR_REL = OUT_ROOT_REL + "/aggregate"
DRYRUN_DIR_REL = OUT_ROOT_REL + "/_dryrun"

# ---------------------------------------------------------------- 状態
STATUS_VALID = "valid"
PAIR_INVALID = ("unaligned", "silent", "generation_failed")            # 次の seed へ
RETRY_SAME_SEED = ("run_error", "alignment_exception", "capture_error", "no_row_unexplained")  # 直して同じ seed
STATUS_PENDING = "pending"          # まだ生成していない（batch の途中で止まった、または未投入）
STATUS_STOP = "stop_author_decision"  # スパンが CPU の確かめと違う等。流し直しでは解決しない
# 同じ seed のやり直しの上限（付録 B の「Whisper の例外が同じ seed で 2 回続けば止める」と揃える）:
#   * 同じ seed の batch で、環境の問題の同じ種類（RETRY_SAME_SEED のどれか）が 2 回出た対は止める
#   * 種類が毎回違っても、同じ seed の retry batch が MAX_SAME_SEED_RETRIES 個になってなお環境の問題なら止める
#   * 前の batch で止まった対は止まったまま
MAX_SAME_SEED_RETRIES = 2

# ---------------------------------------------------------------- 論文の今の値（naacl2027/main.tex L933-940, 読むだけ）
PAPER_EN_CELLS = {
    "match_bl": r"\phantom{0}1.00 \pm 1.28",
    "match_inj": r"\phantom{0}1.00 \pm 1.28",
    "match_bl_mask": r"\phantom{0}3.33 \pm 2.10",
    "match_inj_mask": r"\phantom{0}3.33 \pm 2.10",
    "per_bl": r"0.885 \pm 0.366",
    "per_inj": r"0.915 \pm 0.411",
    "per_bl_mask": r"0.955 \pm 0.491",
    "per_inj_mask": r"0.955 \pm 0.491",
}
PAPER_SOURCE = "naacl2027/main.tex L932-933 (match), L938 (PER); caption: Wilson 95% half-widths / mean ± SD"
EOR_REL = "naacl2027/explanation_of_revisions.tex"
EOR_LINES = (277, 281)  # L279 は付録 B（EN MCD）。付録 C の英語の数値は無い（確かめて表示する）


def P(rel, root=None):
    return os.path.join(root or PROJECT_ROOT, rel)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json_new(obj, path, indent=1):
    """新しいファイルとしてだけ書く（既にあれば止まる。上書き・削除はしない）."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def pinned_errors(root=None):
    errs = []
    for rel, want in PINNED_MD5.items():
        p = P(rel, root)
        if not os.path.exists(p):
            errs.append(f"{rel} が無い")
        elif md5_file(p) != want:
            errs.append(f"{rel} の md5 が固定の値と違う（今 {md5_file(p)} / 固定 {want}）")
    lp = P(LEGACY_SPAN_REL, root)
    if not os.path.exists(lp) or md5_file(lp) != LEGACY_SPAN_MD5:
        errs.append(f"{LEGACY_SPAN_REL} が無いか md5 が違う")
    return errs


def old_result_errors(root=None):
    return [f"{OLD_DIR_REL}/{n} の md5 が違う（読むだけのはずのファイルが変わった）"
            for n, want in OLD_MD5.items() if md5_file(P(f"{OLD_DIR_REL}/{n}", root)) != want]


def code_md5(root=None, own=()):
    rels = list(PINNED_MD5) + [SILENCE_REL, LEGACY_SPAN_REL] + list(own)
    out = {rel: md5_file(P(rel, root)) for rel in rels if os.path.exists(P(rel, root))}
    for f in VENDOR_FILES:
        vp = os.path.join(VENDOR_ROOT, f)
        out["vendor:" + f] = md5_file(vp) if os.path.exists(vp) else None
    return out


# ---------------------------------------------------------------- 対
def load_pairs(root=None):
    """本番 main() と同じ組み立て（build_500_pairs()[:300]; pair_idx は並びの番号 i）."""
    import importlib.util
    path = P(PAIRS_REL, root)
    spec = importlib.util.spec_from_file_location("_appc_en_heteronyms_500", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = mod.build_500_pairs()[:N_PAIRS]
    pairs = []
    for i, p in enumerate(raw):
        assert "pair_idx" not in p  # 本番は pair.get("pair_idx", i) なので i になる
        pairs.append({"pair_idx": i, "text_a": p["text_a"], "text_b": p["text_b"],
                      "word_a": p["word_a"], "word_b": p["word_b"], "direction": p.get("direction")})
    assert len(pairs) == N_PAIRS
    texts = [p["text_a"] for p in pairs]
    assert len(set(texts)) == N_PAIRS, "text_a が対ごとに一意でない（呼び出しの対の特定に使う）"
    return pairs


def pairs_digest(pairs):
    blob = json.dumps([{k: p[k] for k in ("pair_idx", "text_a", "text_b", "word_a", "word_b")} for p in pairs],
                      ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


# ---------------------------------------------------------------- batch の置き場
def seed_dir_rel(seed):
    return f"{OUT_ROOT_REL}/seed{seed}"


def batch_dir_rel(seed, batch):
    return f"{seed_dir_rel(seed)}/{batch}"


def part_name(k):
    return f"part{k:02d}"


def seed42_part_pairs(k):
    assert 0 <= k < N_PARTS_SEED42
    return list(range(k * PART_SIZE, (k + 1) * PART_SIZE))


def list_batches(seed, root=None):
    """seed の batch を作った順（manifest の created_order）に返す: [(name, manifest)]."""
    out = []
    for m in sorted(glob.glob(os.path.join(P(seed_dir_rel(seed), root), "*", "manifest.json"))):
        man = read_json(m)
        out.append((os.path.basename(os.path.dirname(m)), man))
    out.sort(key=lambda x: (x[1].get("created_order", 0), x[0]))
    return out


def list_runs(batch_abs):
    return sorted(d for d in glob.glob(os.path.join(batch_abs, "runs", "run*")) if os.path.isdir(d))


def seed_fingerprint(seed, root=None):
    """seed の置き場の記録ファイルの md5（状態の集計の後に何も変わっていないことを GPU 側で確かめる）."""
    base = P(seed_dir_rel(seed), root)
    top = P("", root)
    out = {}
    for pat in ("*/manifest.json", "*/checkpoint.json", "*/runs/run*/events.jsonl"):
        for f in sorted(glob.glob(os.path.join(base, pat))):
            out[os.path.relpath(f, top)] = md5_file(f)
    return out


def suggest_slurm_time(n_pairs):
    """1 対あたり約 0.9 分（5/10 の実行 260.08 分 / 300 対）＋モデルの読み込み約 3 分。約 2 倍以上の余裕."""
    if n_pairs <= 20:
        return "00:40:00"
    if n_pairs <= 30:
        return "01:00:00"
    if n_pairs <= 50:
        return "02:00:00"
    raise ValueError("1 ジョブは 50 対まで（分けて流す）")


def read_events(run_abs):
    p = os.path.join(run_abs, "events.jsonl")
    ev = []
    if not os.path.exists(p):
        return ev
    with open(p) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                # 時間切れで最後の行が途中で切れたときだけ起こりうる（記録して読み飛ばす）
                ev.append({"ev": "corrupt_line", "line_no": ln})
    return ev


# ---------------------------------------------------------------- 本番と同じ集計
def production_summary(rows):
    """本番 main() の末尾（L305-333）と同じ式。rows は placeholder を含まない 300 行."""
    import numpy as np
    n = len(rows)
    aligned = [r for r in rows if r.get("aligned")]
    n_align = len(aligned)

    def cnt(key):
        return sum(1 for r in aligned if r.get(key))

    def avg_per(key):
        vs = [r[key] for r in aligned if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    return {
        "lang": LANG, "n_total": n, "n_aligned": n_align, "per_threshold_for_match": 0.3,
        "bl_match_count": cnt("bl_match"), "inj_match_count": cnt("inj_match"),
        "bl_mask_match_count": cnt("bl_mask_match"), "inj_mask_match_count": cnt("inj_mask_match"),
        "bl_match_pct": round(100 * cnt("bl_match") / max(1, n_align), 2),
        "inj_match_pct": round(100 * cnt("inj_match") / max(1, n_align), 2),
        "bl_mask_match_pct": round(100 * cnt("bl_mask_match") / max(1, n_align), 2),
        "inj_mask_match_pct": round(100 * cnt("inj_mask_match") / max(1, n_align), 2),
        "bl_per_mean_std": avg_per("bl_per"), "inj_per_mean_std": avg_per("inj_per"),
        "bl_mask_per_mean_std": avg_per("bl_mask_per"), "inj_mask_per_mean_std": avg_per("inj_mask_per"),
    }


def wilson(k, n, z=1.96):
    """regen2_common._wilson / appk_enjudge_common.wilson と同じ式（中心と半幅）."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, h


def table_cells(summary):
    """表 tab:ja_crossattn_phoneme の EN 行と同じ書式（上: 一致率 % ± Wilson 半幅、下: PER 平均 ± SD）.

    0 件のときは今の表と同じく「0.00 + 上端」。PER は results.json の 4 桁の値を 3 桁に丸める（今の表の作り方）。
    """
    n = summary["n_aligned"]
    cells = {}
    for c in PER_CONDS:
        k = summary[f"{c}_match_count"]
        cen, half = wilson(k, n)
        pct = 100 * k / n
        if k == 0:
            s = f"{pct:.2f}~{{+}}{100 * (cen + half):.2f}"
        else:
            s = f"{pct:.2f} \\pm {100 * half:.2f}"
        cells[f"match_{c}"] = (r"\phantom{0}" if pct < 10 else "") + s
        m, sd = summary[f"{c}_per_mean_std"]
        cells[f"per_{c}"] = f"{m:.3f} \\pm {sd:.3f}"
    return cells


# ---------------------------------------------------------------- 1 対の状態（数値は見ない）
def classify_pair(row, gen_events, align_events, err_events, batch_complete, wav_root, span_ref):
    """1 つの seed での 1 対の状態と根拠を返す（数値は見ない）.

    row:        本番 checkpoint の行（placeholder でない）。無ければ None
    gen_events: その対の生成の記録（その対を最後に扱った起動のもの）
    align_events: 同じ起動の整列の記録
    err_events: 同じ起動で、その対について記録された例外の行
    batch_complete: その batch のどれかの起動で main() が例外なく終わったか（終わっていなければ、
                行の無い対は「まだ」= pending。同じコマンドをもう一度流すと続きから再開する）
    span_ref:   CPU の確かめで得たスパン {"a": [s, e], "b": [s, e]}
    """
    import silence_criterion as SC
    ev = {"row_present": row is not None}
    pair_errs = [e for e in err_events if e.get("kind") == "pair_loop"]
    whisper_errs = [e for e in err_events if e.get("kind") == "whisper"]
    gen_exc = [g for g in gen_events if g.get("exception")]
    ev["pair_loop_errors"] = [e.get("line") for e in pair_errs]
    ev["whisper_errors"] = [e.get("line") for e in whisper_errs]
    ev["generate_exceptions"] = [g.get("exception") for g in gen_exc]

    # スパン: 生成の各呼び出しの直前に使われた範囲が CPU の確かめと一致するか。遮断した列もその範囲か
    span_bad = []
    for g in gen_events:
        for sc in g.get("spans", []):
            side = sc.get("side")
            want = span_ref.get(side) if side in ("a", "b") else None
            got = sc.get("result")
            if want is None or got is None or list(got) != list(want):
                span_bad.append({"label": g.get("label"), "side": side, "got": got, "want": want})
        if g.get("label") in ("bl_mask", "inj_mask") and span_ref.get("a"):
            want_cols = list(range(span_ref["a"][0], span_ref["a"][1]))
            if g.get("masked_cols") != want_cols:
                span_bad.append({"label": g.get("label"), "masked_cols": g.get("masked_cols"), "want": want_cols})
    ev["span_mismatch"] = span_bad
    if span_bad:
        return STATUS_STOP, ev
    if row is None and not batch_complete:
        return STATUS_PENDING, ev
    if pair_errs or gen_exc or any(g.get("device_offload") for g in gen_events):
        return "run_error", ev
    if whisper_errs:
        return "alignment_exception", ev

    labels = [g.get("label") for g in gen_events]
    ev["gen_labels"] = labels
    any_none = any(g.get("returned_none") for g in gen_events)
    if row is None:
        if any_none:
            return "generation_failed", ev
        return "no_row_unexplained", ev

    tgt_ok = row.get("aligned") is True
    cond_none = [c for c in PER_CONDS if row.get(f"{c}_per") is None]
    ev["aligned"] = tgt_ok
    ev["conds_without_entity"] = cond_none
    if not tgt_ok or cond_none:
        return "unaligned", ev

    # 写し取りの確かめ: 5 本がそろい、順と形が合い、ファイルの md5 が記録と同じ
    cap_err = [f"tap: {e.get('line')}" for e in err_events if e.get("kind") == "tap"]
    cap_err += [f"save: {g.get('save_error')}" for g in gen_events if g.get("save_error")]
    if sorted(labels) != sorted(CONDS) or len(labels) != len(CONDS):
        cap_err.append(f"labels={labels}")
    if any(not g.get("signature_ok") for g in gen_events):
        cap_err.append("signature")
    wav_info = {}
    for g in gen_events:
        lab = g.get("label")
        rel = g.get("wav")
        if lab not in CONDS or not rel:
            continue
        path = os.path.join(wav_root, rel)
        if not os.path.exists(path):
            cap_err.append(f"missing {rel}")
            continue
        if md5_file(path) != g.get("md5"):
            cap_err.append(f"md5 {rel}")
            continue
        a = SC.assess_file(path)  # 記録ではなくファイルから計算し直す
        wav_info[lab] = {"path": rel, "silent": a["silent"], "silent_reasons": a["silent_reasons"],
                         "rms": a["rms"], "peak": a["peak"], "active_frac": a["active_frac"], "dur": a["dur"]}
    if len(wav_info) != len(CONDS) and not cap_err:
        cap_err.append(f"wavs={sorted(wav_info)}")
    mem = [g.get("mem_check") for g in gen_events if g.get("label") == "inj_mask"]
    if mem and mem[0] and mem[0].get("ok") is False:
        cap_err.append(f"mem_check={mem[0]}")
    # 整列の記録（参考）: 5 本とも Whisper の単語列の中で実体語が見つかった位置と、完全一致か文字位置の比での代替か
    al_tr = [a for a in align_events if a.get("ev") == "align_tr"]
    al_find = {a.get("label"): a for a in align_events if a.get("ev") == "align_find"}
    ev["align_labels"] = [a.get("label") for a in al_tr]
    ev["align_exact"] = {k: v.get("exact") for k, v in al_find.items()}
    if [a.get("label") for a in al_tr] != list(ALIGN_ORDER):
        cap_err.append(f"align_order={[a.get('label') for a in al_tr]}")
    ev["capture_errors"] = cap_err
    ev["wavs"] = wav_info
    if cap_err:
        return "capture_error", ev
    if any(v["silent"] for v in wav_info.values()):
        return "silent", ev
    return STATUS_VALID, ev


# ---------------------------------------------------------------- 同じ seed のやり直しの上限
def same_seed_retry_verdict(history):
    """同じ seed の中での 1 対の状態の並び [(batch 名, 状態), ...]（作った順）から、止めるべきかを返す.

    止める理由の文字列、または None（止めない）。数値は見ない。
    """
    if not history:
        return None
    if any(s == STATUS_STOP for _, s in history[:-1]):
        return "前の batch で止まった対（止まったまま）"
    last = history[-1][1]
    if last not in RETRY_SAME_SEED:
        return None
    kinds = [s for _, s in history if s in RETRY_SAME_SEED]
    if kinds.count(last) >= 2:
        return f"同じ seed で環境の問題 {last} が 2 回出た（付録 B と同じく止める）"
    n_retry = sum(1 for b, _ in history if str(b).startswith("retry"))
    if n_retry >= MAX_SAME_SEED_RETRIES:
        return f"同じ seed のやり直しが {n_retry} 回でも環境の問題が続く（{kinds}）"
    return None


# ---------------------------------------------------------------- 5/10 との BL・TGT の一致（環境の違いの目安）
def bl_tgt_reproduction(new_rows_by_p, old_rows):
    """BL と TGT はスパンを使わないので、seed 42 なら 5/10 と同じ値が出るはず。一致の件数だけを返す.

    表の値（一致率・PER の平均）は出さない。対を選ぶのには使わない。
    new_rows_by_p: {pair_idx: 本番 checkpoint の行}（seed 42 の行だけを渡す）
    """
    old_by_p = {r["pair_idx"]: r for r in old_rows}
    rep = {"n": 0, "bl_per_same": 0, "bl_ipa_len_same": 0, "tgt_ipa_len_same": 0, "all_three_same": 0,
           "aligned_same": 0}
    for p, r in new_rows_by_p.items():
        o = old_by_p.get(p)
        if o is None or r is None:
            continue
        rep["n"] += 1
        a = o.get("bl_per") == r.get("bl_per")
        b = o.get("bl_ipa_len") == r.get("bl_ipa_len")
        t = o.get("tgt_ipa_len") == r.get("tgt_ipa_len")
        rep["bl_per_same"] += int(a)
        rep["bl_ipa_len_same"] += int(b)
        rep["tgt_ipa_len_same"] += int(t)
        rep["all_three_same"] += int(a and b and t)
        rep["aligned_same"] += int(o.get("aligned") == r.get("aligned"))
    return rep
