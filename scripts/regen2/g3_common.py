#!/usr/bin/env python3
"""G3（付録 B, tab:crossattn_mask の EN 対 237・ZH 対 42）の共通定義と判定（2026-09-29）.

torch を読まない。GPU 側（g3_regen_crossattn_pairs.py）と CPU 側（g3_merge_crossattn.py）の両方が使う。

seed の決まり（著者の決定 (1)）:
  * 本番（5/20）は seed 42 で、対象の 2 対は Whisper が実体語を見つけられなかった（aligned=False）。
    これが「seed 42 で使えない」の根拠。seed 42 の再実行は再現の確認だけに使い、取り込まない
    （著者の決定に無い追加の GPU の条件なので、流すのは著者が承認したときだけ。seed 43 はこれを待たない）。
  * seed 43: 本番の行が aligned=False の対だけ（seed 42 の再実行の結果には左右されない）。
  * seed 44: seed 43 の最終の記録が「対として使えない」（下の PAIR_INVALID）の対だけ。
  * 45 以上は決まっていない（著者の判断が要る）ので、生成も取り込みもしない。
  * 選ぶ基準は有効性だけ。数値（MCD・Mel similarity）は見ない。

状態（regen_row.json の status）:
  valid              有効（aligned、5 本の frames、5 本の wav がそろい、どれも無音でない）
  -- 対として使えない（PAIR_INVALID）: 次の seed に進む --
  unaligned          5 本のどれかで Whisper が実体語を見つけられなかった（alignment failure。Whisper の例外は無い）
  generation_failed  生成が音声を返さなかった（generate が None）ので本番の main() が行を作らなかった
  silent             aligned だが、5 本のどれかが無音（silence_criterion.py の基準）
  -- 対の良し悪しではない（RETRY_SAME_SEED）: 直してから同じ seed を --attempt k+1 でやり直す --
  run_error          main() の例外、対の処理中の例外（本番は握りつぶして行を作らない）、T5Gemma の層が CPU/disk に置かれた
  alignment_exception Whisper の呼び出しが例外で落ちた（forced_alignment_utils.WhisperAligner.transcribe_words は
                     例外を握りつぶして "  [Whisper] transcribe failed: ..." を print し、([], None) を返すので、本番の
                     行は aligned=False になる）。これは「Whisper が実体語を見つけられなかった」（unaligned）ではなく
                     環境の問題（CUDA OOM など）なので、次の seed には進まない。同じ seed で 2 回続いたら、決まった
                     例外がくり返している恐れがあるので、3 回目は流さず著者の判断を仰ぐ（stop_author_decision）。
  capture_error      行は aligned だが、wav の写し取りが確かめに通らない（順序・本数）
  no_row_unexplained 行が無いのに、例外も None も見つからない

同じ seed のやり直し（attempt）:
  <seed root>/<lang>_p<idx:04d>/ が 1 回目、<lang>_p<idx:04d>_a2/ が 2 回目 …。
  前の回の状態が RETRY_SAME_SEED のときだけ次の回を作れる。valid か PAIR_INVALID の回の後に回があれば、
  結果で選んだことになるので、生成も取り込みも止まる。
  regen_row.json を書く前に止まった回（時間切れなど）は、本番 main() の checkpoint.json に対象の行がまだ無ければ
  crashed_no_record（やり直してよい）、行があれば crashed_after_row（著者の判断が要る）とする。

コードの md5（出どころ）:
  * 本番のファイル 4 本（評価・注入・整列・offline の読み替え）は PINNED_MD5 に固定。GPU ジョブは始める前に
    今のファイルがこの md5 であることを確かめ、違えばモデルを読まずに止まる。記録の md5 もこれと一致しなければならない。
  * G3 自身のコード（OWN_CODE: ラッパーと g3_common.py）は、記録の md5 が取り込みの時点のファイルと一致しなければならない。
    有効な記録（または seed 44 の根拠になる「使えない」記録）で違うのがこれだけのときは、stop_author_decision
    （GPU を流し直しても解決しない。valid の後に同じ seed の回を足すと結果で選ぶことになるので、生成側も止まる）。
    著者の判断の例: ファイルを記録の md5 の版に戻して取り込む、または違いが動作に関係しないことを確かめて受け入れる。
  * silence_criterion.py は他の作業も読む共通の部品なので、ファイルの md5 は記録するだけで一致は求めない。
    代わりに CRITERION_ID と基準の数値（SILENCE_PARAMS）が同じこと、wav の判定をファイルから計算し直して記録と
    同じことを確かめる。
"""
import glob
import hashlib
import json
import os
import re

import silence_criterion as SC  # 同じ置き場（scripts/regen2）の共通の無音の基準

HOST_ROOT = os.environ.get("REPO_ROOT", "/mnt")
CONTAINER_ROOT = "/mnt"
PROJECT_ROOT = CONTAINER_ROOT if os.path.isdir("/mnt/scripts") else HOST_ROOT

EVAL_REL = "scripts/projection/eval_crossattn_entity_dtw.py"
INJ_REL = "scripts/encoder_injection_t5gemma.py"
FA_REL = "scripts/projection/forced_alignment_utils.py"
SHIM_REL = "scripts/nontemplate/gen_t5gemma_pairs_module.py"
WRAPPER_REL = "scripts/regen2/g3_regen_crossattn_pairs.py"
COMMON_REL = "scripts/regen2/g3_common.py"
SILENCE_REL = "scripts/regen2/silence_criterion.py"
LEGACY_SPAN_REL = "release_artifacts/scripts/encoder_injection_t5gemma.py.bak_spanfix_20260917_095951"
# 記録するコードの md5（7 本）。一致の求め方はファイルの種類で違う（上の「コードの md5」）
CODE_FILES = (EVAL_REL, INJ_REL, FA_REL, SHIM_REL, WRAPPER_REL, COMMON_REL, SILENCE_REL)
# 本番のファイル（このラッパーは改変しない）。GPU ジョブの前と取り込みの時に、この md5 を求める
PINNED_MD5 = {
    EVAL_REL: "c20083b9142439e6a46f88f3f0b2cfe4",  # 5/20 11:53 から変わっていない（本番の実行は 11:54 から）
    INJ_REL: "4f36fa570145cde805072438d8a39bb1",   # 9/17 の span 修正の後の版（span の事前確認で 5/20 と一致を確かめる）
    FA_REL: "be2b300a061e1374509bd1aa75f41b5d",    # Whisper の整列（"[Whisper] transcribe failed" の print を含む）
    SHIM_REL: "53eb4bfc1008beb361a23c6adfc429ba",  # 9/28 の前例の offline の読み替え
}
EXPECTED_EVAL_MD5 = PINNED_MD5[EVAL_REL]
# G3 自身のコード。記録の md5 が取り込みの時点のファイルと一致しなければならない（違えば著者の判断）
OWN_CODE = (WRAPPER_REL, COMMON_REL)
# 共通の無音の基準の数値（silence_criterion.py の定数がこれと違えば止まる）
SILENCE_PARAMS = {"RMS_TH": 0.005, "PEAK_TH": 0.05, "FRAME_MS": 20, "FRAME_RMS_TH": 0.01, "ACTIVE_FRAC_TH": 0.05}
# 本番の forced_alignment_utils.py が Whisper の例外を握りつぶすときの print（この行が無ければ止まる）
FA_WHISPER_FAIL_PRINT = 'print(f"  [Whisper] transcribe failed: {str(e)[:160]}")'
WHISPER_FAIL_PREFIX = "[Whisper] transcribe failed: "

TOKENIZER_ID = "google/t5gemma-2b-2b-ul2"  # Aratako/T5Gemma-TTS-2b-2b の config.text_tokenizer_name
T5GEMMA_TTS_ID = "Aratako/T5Gemma-TTS-2b-2b"
CONTAINER_HF_HOME = "/mnt/.cache/huggingface"

PROD_DIR = {
    "en": "eval_results/crossattn_entity_dtw_encc_lp_n300",
    "zh": "eval_results/crossattn_entity_dtw_zhcc_lp_n300",
}
PROD_MD5 = {  # 5/20 の本番ファイル（読むだけ。変わっていたら止まる）
    "en": {"results.json": "f32dd293df04abf4a99ca5cf83ade871", "checkpoint.json": "84bdf01ddd81b71a50bb7101f3d77793"},
    "zh": {"results.json": "1ad9b54b49670450c300089b0f87cb93", "checkpoint.json": "d99968b0b9331207fe20ad760522e6ea"},
}
TARGETS = {"en": 237, "zh": 42}  # 本番で aligned=False の行（各言語 1 対だけ）
CONDS = ("bl", "inj", "bl_mask", "inj_mask", "tgt")
N_PAIRS = 300

REPRO_SEED = 42
REGEN_SEEDS = (43, 44)
ALL_SEEDS = (42, 43, 44)
PRODUCER = "g3_regen_crossattn_pairs.py"
MODE_REGEN = "regen"
MODE_REPRO = "repro_check_seed42"
MODE_FAKE = "FAKE_plumbing"

STATUS_VALID = "valid"
STATUS_ALIGN_EXC = "alignment_exception"
PAIR_INVALID = ("unaligned", "generation_failed", "silent")
RETRY_SAME_SEED = ("run_error", STATUS_ALIGN_EXC, "capture_error", "no_row_unexplained")
ALL_STATUSES = (STATUS_VALID,) + PAIR_INVALID + RETRY_SAME_SEED
# regen_row.json が無い回（時間切れなど）。行ができる前に止まったものだけ、同じ seed をやり直してよい
CRASHED_NO_RECORD = "crashed_no_record"
CRASHED_AFTER_ROW = "crashed_after_row"
RETRYABLE = RETRY_SAME_SEED + (CRASHED_NO_RECORD,)

MERGED_ROOT_REL = "eval_results/_regen2_crossattn_merged"
DRYRUN_ROOT_REL = "eval_results/_regen2_dryrun/g3_crossattn"


def prod_args(lang):
    """本番の引数（.perm_cmds.txt 16714 行）。この後に --output_dir <新しい置き場> だけを足す."""
    return ["--lang", lang, "--cross_culture", "--inject_mode", "length_preserve", "--n", str(N_PAIRS)]


def seed_root_rel(seed):
    return f"eval_results/_regen2_crossattn_seed{seed}"


def pair_dir_rel(seed, lang, idx, attempt=1):
    base = f"{seed_root_rel(seed)}/{lang}_p{idx:04d}"
    return base if attempt == 1 else f"{base}_a{attempt}"


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


def current_code_md5(code_root):
    return {rel: md5_file(os.path.join(code_root, rel)) for rel in CODE_FILES}


def pinned_file_errors(code_root):
    """本番のファイル 4 本が PINNED_MD5 のままか（GPU ジョブの前に確かめる）."""
    return [f"{rel} の md5 が固定の値と違う（今 {md5_file(os.path.join(code_root, rel))} / 固定 {want}）"
            for rel, want in PINNED_MD5.items() if md5_file(os.path.join(code_root, rel)) != want]


def silence_param_errors():
    """共通の無音の基準の数値が決まりどおりか."""
    errs = [f"silence_criterion.{k} = {getattr(SC, k, None)!r}（{v} のはず）"
            for k, v in SILENCE_PARAMS.items() if getattr(SC, k, None) != v]
    if SC.CRITERION_ID != "regen2-silence-2026-09-29":
        errs.append(f"silence_criterion.CRITERION_ID = {SC.CRITERION_ID!r}")
    return errs


def frames_of(row):
    return [row.get(f"frames_{c}") for c in CONDS] if row else None


# ---------------------------------------------------------------- 状態の判定
def classify(row, main_error, pair_loop_errors, whisper_errors, any_returned_none, capture_ok, n_wavs, any_silent,
             device_offload):
    """1 回の実行の状態を決める（上の表）。数値（MCD・類似度）は見ない.

    whisper_errors（Whisper の例外の print）は、aligned の判定より先に見る。例外で落ちた Whisper は
    ([], None) を返し、本番の行は aligned=False になるが、これは対の良し悪しではない（PAIR_INVALID にしない）。
    """
    if main_error or pair_loop_errors or device_offload:
        return "run_error"
    if whisper_errors:
        return STATUS_ALIGN_EXC
    if row is None:
        return "generation_failed" if any_returned_none else "no_row_unexplained"
    if row.get("aligned") is not True:
        return "unaligned"
    if any(row.get(f"frames_{c}") is None for c in CONDS):
        return "run_error"  # aligned なのに frames が欠ける: 本番のコードでは起きない
    if not capture_ok or n_wavs != len(CONDS):
        return "capture_error"
    if any_silent:
        return "silent"
    return STATUS_VALID


# ---------------------------------------------------------------- 記録の読み出し
def attempts_of(data_root, seed, lang, idx):
    """(attempt, rel_dir) の並び。1, 2, ... と抜けなく並ぶことも確かめる."""
    base_rel = pair_dir_rel(seed, lang, idx, 1)
    errs = []
    found = {}
    if os.path.isdir(os.path.join(data_root, base_rel)):
        found[1] = base_rel
    for d in glob.glob(os.path.join(data_root, base_rel + "_a*")):
        m = re.fullmatch(r"_a(\d+)", d[len(os.path.join(data_root, base_rel)):])
        if not m or int(m.group(1)) < 2:
            errs.append(f"想定外のフォルダ: {os.path.relpath(d, data_root)}")
            continue
        found[int(m.group(1))] = os.path.relpath(d, data_root)
    ks = sorted(found)
    if ks and ks != list(range(1, len(ks) + 1)):
        errs.append(f"attempt が 1 から抜けなく並んでいない: {ks}")
    return [(k, found[k]) for k in ks], errs


def crashed_status(data_root, rel_dir):
    """regen_row.json が無い回（時間切れ・強制終了など）の扱い.

    本番 main() の checkpoint.json に対象の対の行がまだ無ければ、対の結果は出ていないので
    crashed_no_record（同じ seed をやり直してよい）。行があれば結果が出た後に止まったので
    crashed_after_row（やり直すと結果で選ぶことになりうるので、著者の判断が要る）。
    """
    ck = os.path.join(data_root, rel_dir, "checkpoint.json")
    try:
        rows = read_json(ck).get("rows", []) if os.path.isfile(ck) else []
    except Exception:  # noqa: BLE001  書きかけの checkpoint: 結果は確かめられない
        return CRASHED_AFTER_ROW
    return CRASHED_AFTER_ROW if any(not r.get("placeholder") for r in rows) else CRASHED_NO_RECORD


def final_record(data_root, seed, lang, idx):
    """その seed の最後の回の記録を返す: (rec, rel_path, errs, chain).

    途中の回は RETRYABLE でなければならない（結果の良い回を選ばないため）。
    最後の回に regen_row.json が無ければ rec=None（chain の最後の status が crashed_*）。
    ジョブがまだ動いている可能性もあるので、やり直す前に squeue で確かめる。
    """
    att, errs = attempts_of(data_root, seed, lang, idx)
    chain = []
    rec = rel_path = None
    for n, (k, rel) in enumerate(att):
        p_rel = os.path.join(rel, "regen_row.json")
        p = os.path.join(data_root, p_rel)
        last = n == len(att) - 1
        if not os.path.isfile(p):
            st = crashed_status(data_root, rel)
            chain.append({"attempt": k, "regen_row": None, "status": st})
            if st == CRASHED_AFTER_ROW:
                errs.append(f"{rel}: 対の行ができた後に記録を書く前に止まった（著者の判断が要る）")
            continue
        try:
            r = read_json(p)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{p_rel} が読めない: {type(e).__name__}")
            chain.append({"attempt": k, "regen_row": p_rel, "status": None})
            continue
        st = r.get("status")
        chain.append({"attempt": k, "regen_row": p_rel, "md5": md5_file(p), "status": st})
        if r.get("attempt") != k or r.get("seed") != seed or r.get("lang") != lang or r.get("pair_idx") != idx:
            errs.append(f"{p_rel}: attempt/seed/lang/pair_idx が置き場と合わない "
                        f"({r.get('attempt')}, {r.get('seed')}, {r.get('lang')}, {r.get('pair_idx')})")
        if st not in ALL_STATUSES:
            errs.append(f"{p_rel}: 知らない status {st!r}")
        if not last and st not in RETRYABLE:
            errs.append(f"{p_rel}: status={st} の後に別の回がある（結果で選ぶことになるので使わない）")
        if last:
            rec, rel_path = r, p_rel
    return rec, rel_path, errs, chain


def repeated_alignment_exception(chain):
    """同じ seed の最後の 2 回が続けて alignment_exception か（決まった例外のくり返しの恐れ → 著者の判断）."""
    sts = [c.get("status") for c in chain]
    return len(sts) >= 2 and sts[-1] == STATUS_ALIGN_EXC and sts[-2] == STATUS_ALIGN_EXC


def stray_seed_dirs(data_root, lang, idx):
    """42/43/44 以外の seed の置き場にこの対があれば返す（決まりの外の生成）."""
    out = []
    for d in glob.glob(os.path.join(data_root, "eval_results", "_regen2_crossattn_seed*")):
        m = re.fullmatch(r"_regen2_crossattn_seed(\d+)", os.path.basename(d))
        if m and int(m.group(1)) not in ALL_SEEDS and glob.glob(os.path.join(d, f"{lang}_p{idx:04d}*")):
            out.append(os.path.relpath(d, data_root))
    return out


# ---------------------------------------------------------------- 出どころの確かめ
def identity_errors(rec, rel_path, lang, idx, seed, expect_mode):
    """誰がどこで作った記録か（本物の GPU ジョブ・決まった置き場・本番の引数）だけを確かめる.

    RETRY_SAME_SEED の回（捨てる回）はこれだけを確かめる。例外の有無やコードの md5 はその回の状態の原因そのもの
    （直すためにコードを変えることもある）なので、やり直しを止める理由にしない。
    """
    errs = []
    rel_dir = os.path.dirname(rel_path)
    if rec.get("producer") != PRODUCER:
        errs.append(f"producer が {PRODUCER} でない: {rec.get('producer')!r}")
    if rec.get("mode") != expect_mode:
        errs.append(f"mode が {expect_mode} でない: {rec.get('mode')!r}")
    if rec.get("seed") != seed or rec.get("lang") != lang or rec.get("pair_idx") != idx:
        errs.append("seed/lang/pair_idx が合わない")
    if rec.get("pair_dir") != rel_dir:
        errs.append(f"pair_dir が置き場と違う: {rec.get('pair_dir')!r} != {rel_dir!r}")
    env = rec.get("env") or {}
    if env.get("project_root") != CONTAINER_ROOT:
        errs.append(f"コンテナ（/mnt）の中で作られていない: project_root={env.get('project_root')!r}")
    if not env.get("cuda_device"):
        errs.append("GPU の記録が無い（cuda_device）")
    want_argv = prod_args(lang) + ["--output_dir", f"{CONTAINER_ROOT}/{rel_dir}"]
    if rec.get("main_argv") != want_argv:
        errs.append(f"main_argv が本番の引数と違う: {rec.get('main_argv')}")
    return errs


def code_drift_errors(rec, code_root):
    """G3 自身のコード（OWN_CODE）の、記録の md5 と今のファイルの md5 の違い（あれば著者の判断）."""
    cm = rec.get("code_md5") or {}
    out = []
    for rel in OWN_CODE:
        cur = md5_file(os.path.join(code_root, rel))
        if cm.get(rel) != cur:
            out.append(f"{rel} の md5 が GPU の実行の後に変わった（記録 {cm.get(rel)} / 今 {cur}）")
    return out


def provenance_errors(rec, rel_path, data_root, code_root, lang, idx, seed, expect_mode):
    """判断に使う記録（取り込み・seed 44 の根拠）が、本物の GPU ジョブで例外なく作られたかを確かめる.

    G3 自身のコードの md5 の違い（code_drift_errors）はここに入れない。呼び出し側が別に扱う。
    """
    errs = identity_errors(rec, rel_path, lang, idx, seed, expect_mode)
    rel_dir = os.path.dirname(rel_path)
    sp = rec.get("span_preflight") or {}
    if set(sp) != {f"{lang}:{idx}"} or not all(isinstance(v, dict) and v.get("all_match") is True
                                                for v in sp.values()):
        errs.append(f"span_preflight が本物でないか一致していない: keys={sorted(sp)}")
    env = rec.get("env") or {}
    if env.get("HF_HUB_OFFLINE") != "1" or env.get("HF_HOME") != CONTAINER_HF_HOME:
        errs.append(f"HF の環境が決まりと違う: HF_HOME={env.get('HF_HOME')!r} HF_HUB_OFFLINE={env.get('HF_HUB_OFFLINE')!r}")
    if env.get("offline_preflight_ok") is not True:
        errs.append("offline_preflight が通っていない")
    shim = env.get("offline_tokenizer_shim") or []
    if not any(s.get("repo_id") == TOKENIZER_ID and s.get("snapshot") for s in shim):
        errs.append(f"{TOKENIZER_ID} が手元の snapshot から読まれた記録が無い")
    pt = env.get("print_taps") or {}
    if pt.get("eval_module_tapped") is not True or pt.get("forced_alignment_utils_tapped") is not True:
        errs.append(f"本番 main() と forced_alignment_utils の print を記録した証拠が無い: {pt}")
    if env.get("device_offload"):
        errs.append(f"T5Gemma の層が CPU/disk に置かれた: {env.get('hf_device_map')}")
    if rec.get("main_error") is not None or rec.get("pair_loop_errors"):
        errs.append(f"実行中の例外: main_error={rec.get('main_error')!r} pair_loop_errors={rec.get('pair_loop_errors')}")
    if rec.get("whisper_errors") != []:
        errs.append(f"Whisper の例外がある（または記録が無い）: whisper_errors={rec.get('whisper_errors')!r}")
    cm = rec.get("code_md5") or {}
    for rel, want in PINNED_MD5.items():
        if cm.get(rel) != want:
            errs.append(f"本番のファイル {rel} の md5 が固定の値と違う（記録 {cm.get(rel)} / 固定 {want}）")
    if rec.get("silence_criterion_id") != SC.CRITERION_ID:
        errs.append(f"無音の基準が違う: {rec.get('silence_criterion_id')!r}")
    errs += silence_param_errors()
    # 行は main() が書いた checkpoint.json の行と同じでなければならない
    ck_p = os.path.join(data_root, rel_dir, "checkpoint.json")
    row = rec.get("row")
    if not os.path.isfile(ck_p):
        errs.append(f"{rel_dir}/checkpoint.json が無い")
    else:
        rows = read_json(ck_p).get("rows", [])
        real = [r for r in rows if not r.get("placeholder")]
        ph = sorted(r["pair_idx"] for r in rows if r.get("placeholder"))
        if ph != [i for i in range(N_PAIRS) if i != idx]:
            errs.append("checkpoint.json の placeholder が対象以外の 299 対になっていない")
        if row is None:
            if real:
                errs.append("記録に行が無いのに checkpoint.json に行がある")
        elif len(real) != 1 or real[0] != row:
            errs.append("記録の行が main() の書いた checkpoint.json の行と違う")
    # wav: 置き場・md5・無音の判定を、ファイルから計算し直して確かめる
    for c, w in (rec.get("wavs") or {}).items():
        p = os.path.join(data_root, w.get("path", ""))
        if not w.get("path", "").startswith(rel_dir + "/wav/"):
            errs.append(f"wav {c} が置き場の外: {w.get('path')}")
        elif not os.path.isfile(p):
            errs.append(f"wav {c} が無い: {w.get('path')}")
        elif md5_file(p) != w.get("md5"):
            errs.append(f"wav {c} の md5 が記録と違う")
        else:
            a = SC.assess_file(p)
            if a["silent"] != w.get("silent"):
                errs.append(f"wav {c} の無音の判定が記録と違う（今 {a['silent']} / 記録 {w.get('silent')}）")
    return errs


def seed43_allowed(prod_root, lang, idx):
    """seed 43 を作ってよいか: 本番（seed 42）の行が aligned=False（prod_root は本番の置き場のある根）."""
    p = os.path.join(prod_root, PROD_DIR[lang], "results.json")
    rows = {r["pair_idx"]: r for r in read_json(p)["rows"]}
    r = rows.get(idx)
    ev = {"production_results": os.path.join(PROD_DIR[lang], "results.json"), "production_md5": md5_file(p),
          "production_aligned": r.get("aligned") if r else None, "production_frames": frames_of(r)}
    errs = []
    if r is None or r.get("aligned") is not False:
        errs.append(f"本番（seed 42）の対 {idx} は aligned={r.get('aligned') if r else None}。"
                    "seed 43 は本番で使えなかった対だけ")
    return errs, ev


def seed44_allowed(data_root, code_root, lang, idx):
    """seed 44 を作ってよいか: seed 43 の最後の回が本物で、PAIR_INVALID."""
    rec, rel, errs, chain = final_record(data_root, 43, lang, idx)
    ev = {"seed43_chain": chain, "seed43_regen_row": rel,
          "seed43_md5": md5_file(os.path.join(data_root, rel)) if rel else None,
          "seed43_status": rec.get("status") if rec else None}
    if rec is None:
        errs = errs + (["seed 43 の最後の回に記録が無い（止まった回。squeue で動いていないことを確かめてから "
                        "seed 43 を --attempt でやり直す）"] if chain else ["seed 43 の記録が無い"])
        return errs, ev
    st = rec.get("status")
    if st not in PAIR_INVALID:
        errs = errs + identity_errors(rec, rel, lang, idx, 43, MODE_REGEN)
        errs.append(f"seed 43 の状態は {st}。seed 44 は seed 43 で対として使えなかった対だけ"
                    + ("（同じ seed 43 を --attempt でやり直す）" if st in RETRY_SAME_SEED else ""))
        return errs, ev
    drift = code_drift_errors(rec, code_root)
    ev["seed43_code_drift"] = drift
    errs = errs + provenance_errors(rec, rel, data_root, code_root, lang, idx, 43, MODE_REGEN) + [
        d + "（seed 43 の記録を作ったコードと今のコードが違うので、seed 44 に進むかは著者の判断）" for d in drift]
    return errs, ev


CRASH_NOTE = "最後の回に regen_row.json が無い。squeue でジョブが動いていないことを確かめてから --attempt でやり直す"
DRIFT_NOTE = ("記録はほかの確かめに通るが、G3 のコード（{files}）が GPU の実行の後に変わった。GPU を流し直しても"
              "解決しない（決着した回の後に同じ seed の回を足すと結果で選ぶことになるので、生成側も止まる）。"
              "著者の判断: ファイルを記録の md5 の版に戻してから取り込む、または違いが動作に関係しないことを"
              "確かめて受け入れる")


def _settle(out, action, drift=()):
    """errors があれば stop_error。G3 のコードの md5 の違いだけなら stop_author_decision。どちらも無ければ action."""
    if out["errors"]:
        out["errors"] += list(drift)
        out["action"] = "stop_error"
    elif drift:
        out["code_drift"] = list(drift)
        out["action"] = "stop_author_decision"
        out["note"] = DRIFT_NOTE.format(files=", ".join(sorted({d.split(" の md5")[0] for d in drift})))
    else:
        out["action"] = action
    return out


def _retry(out, rec, rel, chain, seed, lang, idx):
    """最後の回が RETRY_SAME_SEED: 出どころ（誰がどこで）だけを確かめ、同じ seed のやり直しを返す."""
    out["errors"] += identity_errors(rec, rel, lang, idx, seed, MODE_REGEN)
    if out["errors"]:
        out["action"] = "stop_error"
    elif repeated_alignment_exception(chain):
        out["action"] = "stop_author_decision"
        out["note"] = (f"seed {seed} で Whisper の例外（alignment_exception）が 2 回続いた。決まった例外がくり返している"
                       f"恐れがあるので 3 回目は流さない。whisper_errors を見て、著者の判断を仰ぐ（次の seed には進まない）")
    else:
        out["action"] = f"rerun_seed{seed}"
        out["note"] = (f"seed {seed} の最後の回は {rec.get('status')}（対の良し悪しではない）。原因を直してから、同じ seed を "
                       f"--attempt {rec.get('attempt', 0) + 1} で流す")
    return out


def next_action(data_root, code_root, lang, idx):
    """今の記録から、次にすることを 1 つだけ返す（取り込みと --status が使う）.

    戻り値: {"action": ..., "use": (seed, rel_path) or None, "errors": [...], "chain": {...}}
      action: run_seed43 / rerun_seed43 / run_seed44 / rerun_seed44 / merge / stop_author_decision / stop_error
    判断に使う記録（valid、または seed 44 の根拠になる PAIR_INVALID）は provenance_errors をすべて確かめ、
    G3 のコードの md5 の違い（code_drift_errors）だけが残るときは stop_author_decision（"code_drift" に中身）。
    RETRY_SAME_SEED の回は identity_errors だけを確かめる（その回は捨てるので）。
    """
    out = {"lang": lang, "pair_idx": idx, "chain": {}, "errors": [], "use": None}
    stray = stray_seed_dirs(data_root, lang, idx)
    if stray:
        out["errors"].append(f"決まりの外の seed の置き場: {stray}")
    r43, p43, e43, c43 = final_record(data_root, 43, lang, idx)
    r44, p44, e44, c44 = final_record(data_root, 44, lang, idx)
    out["chain"] = {"43": c43, "44": c44}
    if e43:
        out["errors"] += e43
    if r43 is None:
        if c44:
            out["errors"].append("seed 43 が無いのに seed 44 がある")
        if c43:  # 最後の回が記録を書く前に止まった（crashed_no_record ならやり直してよい）
            out["note"] = CRASH_NOTE
            return _settle(out, "rerun_seed43")
        return _settle(out, "run_seed43")
    st = r43.get("status")
    if st in RETRY_SAME_SEED:
        if c44:
            out["errors"].append("seed 43 が決着していないのに seed 44 がある")
        return _retry(out, r43, p43, c43, 43, lang, idx)
    out["errors"] += provenance_errors(r43, p43, data_root, code_root, lang, idx, 43, MODE_REGEN)
    drift43 = code_drift_errors(r43, code_root)
    if st == STATUS_VALID:
        if c44:
            out["errors"].append("seed 43 が有効なのに seed 44 がある")
        out["use"] = (43, p43)
        return _settle(out, "merge", drift43)
    if st not in PAIR_INVALID:  # 知らない status（final_record も errors に入れている）
        out["errors"].append(f"seed 43 の status {st!r} は判断できない")
        return _settle(out, "stop_error")
    # seed 43 で対として使えない
    if drift43:  # seed 43 の判定を作ったコードが今と違う: seed 44 に進むかは著者の判断
        return _settle(out, "run_seed44", drift43)
    if e44:
        out["errors"] += e44
    if r44 is None:
        if c44:
            out["note"] = CRASH_NOTE
            return _settle(out, "rerun_seed44")
        return _settle(out, "run_seed44")
    pol = r44.get("seed_policy") or {}
    if pol.get("seed43_regen_row") != p43 or pol.get("seed43_md5") != md5_file(os.path.join(data_root, p43)):
        out["errors"].append("seed 44 の記録が、今の seed 43 の記録（置き場と md5）を根拠にしていない")
    st44 = r44.get("status")
    if st44 in RETRY_SAME_SEED:
        return _retry(out, r44, p44, c44, 44, lang, idx)
    out["errors"] += provenance_errors(r44, p44, data_root, code_root, lang, idx, 44, MODE_REGEN)
    drift44 = code_drift_errors(r44, code_root)
    if st44 == STATUS_VALID:
        out["use"] = (44, p44)
        return _settle(out, "merge", drift44)
    _settle(out, "stop_author_decision")
    if out["action"] == "stop_author_decision":
        out["note"] = "seed 44 でも対として使えない。45 以上は決まっていないので著者の判断を仰ぐ"
    return out
