#!/usr/bin/env python3
"""付録 C（Table A2, tab:ja_crossattn_phoneme）の日本語の行を、付録 K の日本語 300 対と直したスパン関数で
作り直すための共通の定義（2026-10-02）.

torch を読まない。pyopenjtalk も読まない。GPU 側（appc_ja_gen.py）と CPU 側（appc_ja_span_check.py /
appc_ja_scan.py / appc_ja_aggregate.py / appc_ja_mock_flow.py）の全部が使う。

著者の決定（2026-10-02, 決定 (1)）:
  付録 C の日本語の行を、付録 K（tab:six_language）の日本語 300 対で作り直す。5 つの条件（BL, INJ, BL_mask,
  INJ_mask, TGT）・採点・設定は元の日本語の実行と同じ。違うのは対の集合と、直したスパン関数だけ。英語の行は変えない。

元の日本語の実行（修正前のスパン関数。対は 5/11 の時点の crossattn_mask_n300_filtered_v2）:
  SLURM_TIME=06:00:00 slurm_run.sh \
    python scripts/projection/eval_crossattn_entity_per.py --lang ja --n 300
  * 5/7: 同じコマンド（.perm_cmds.txt 15212 行、eval_results/crossattn_entity_per_ja_run.log）。23:40 に取り消し。
  * 5/10 01:58 頃: 同じコマンド（.perm_cmds.txt 15571 行、eval_results/crossattn_entity_per_ja_resubmit_run.log）。
    6 時間で時間切れ（07:58）。checkpoint は 200 対。
  * 5/11: 同じコマンドを scripts/auto_priority1to3_loop.sh 29 行（ja_heteronym_4cond_resume）が流し、
    対 200-299 を再開して完了（eval_results/ja_heteronym_4cond_resume_run.log、elapsed_min 158.69、18:40）。
  出力 eval_results/crossattn_entity_per_ja_n300/{checkpoint,results}.json。表の JA 8 セルはこの results.json。
  評価スクリプトは 5/13 に T5GEMMA_ROOT の 1 行だけ変わった（英語の作り直しと同じ事情。appc_en_common.py 参照）。

対（付録 K の日本語 300 対）:
  正本 eval_results/cjk_n300_rewav/checkpoint_ja.json の "results"（scripts/regen2/appk_common.py の SETS["ja"]）。
  元の定義は scripts/projection/eval_cjk_n300.py の load_ja_pairs(300)（data/entities_10000.jsonl を
  random.Random(42) で並べ替えた先頭 300、テンプレート「{word}は、日本で広く知られています。」）。
  eval_results/cjk_n300/checkpoint_ja.json も同じ 300 対・同じ並び。appc_ja_span_check.py が 3 つの一致と、
  300 対がすべて異なること（対、text_a、text_b、(word_a, word_b)）を確かめる。

対の読み込み（本番のスクリプトは変えない）:
  本番 main() の JA の分岐は相対パス eval_results/crossattn_mask_n300_filtered_v2/results.json を open し、
  r["text"], r["kanji"], r["kata"], r["idx"] から対を作る（text_b = text_a.replace(kanji, kata)）。
  appc_ja_gen.py は本番の module の open をこの 1 つのパスだけ読み替え、付録 K の 300 対を同じ形
  （text=text_a, kanji=word_a, kata=word_b, idx=pair_idx）にしたファイル
  eval_results/_spanfix_appC_ja/pairs/appk_ja_300_v2format.json を渡す。本番の組み立てで付録 K の
  (pair_idx, text_a, text_b, word_a, word_b) と 300/300 同じ並びになることを span_check が確かめる。

作り直しの決まり:
  * seed 42 で 300 対すべて（50 対ずつ 6 part）。使えない対（無音・整列の失敗・生成の失敗）は対ごと seed 43、
    それでも使えなければ seed 44。選ぶ基準は有効性だけで、数値は見ない。45 以上は著者の判断。
  * 無音は scripts/regen2/silence_criterion.py（9/29 の基準）で、保存した 5 本の wav から判定する。
  * pyopenjtalk は T5Gemma-TTS 同梱の長さ推定器（vendor duration_estimator._phoneme_count_ja）の中でだけ使う
    （5/11 の実行もこの経路。ログの JPCommonLabel の警告）。自分のコードでは塞ぐ。
  * 新しい置き場 eval_results/_spanfix_appC_ja/ にだけ書く。既存の結果は読むだけ。何も消さない。

置き場:
  eval_results/_spanfix_appC_ja/
    pairs/appk_ja_300_v2format.json      本番 main() に渡す対のファイル（span_check が作る）
    span_check/span_check_<日時>.json    CPU のスパンの確かめ（appc_ja_span_check.py）
    seed42/part00 .. part05/             seed 42 の 6 分割（各 50 対）
    seed43/part00 .., seed44/part00 ..   使えなかった対
    seed<S>/retry01 ..                   同じ seed のやり直し（環境の問題のときだけ）
    status/status_<日時>.json             状態の集計（appc_ja_scan.py）
    aggregate/aggregate_<日時>.json       新旧の表の値と本文の主張の確かめ（appc_ja_aggregate.py）
    _dryrun/                             試運転・selftest・mock の置き場
"""
import ast
import glob
import hashlib
import json
import math
import os
import re
import time

HOST_ROOT = os.environ.get("REPO_ROOT", "/mnt")
CONTAINER_ROOT = "/mnt"
IN_CONTAINER = os.path.isdir("/mnt/scripts")
PROJECT_ROOT = CONTAINER_ROOT if IN_CONTAINER else HOST_ROOT

# ---------------------------------------------------------------- 本番のファイル（読むだけ）
EVAL_REL = "scripts/projection/eval_crossattn_entity_per.py"
INJ_REL = "scripts/encoder_injection_t5gemma.py"
FA_REL = "scripts/projection/forced_alignment_utils.py"
SHIM_REL = "scripts/nontemplate/gen_t5gemma_pairs_module.py"
SILENCE_REL = "scripts/regen2/silence_criterion.py"
LEGACY_SPAN_REL = "release_artifacts/scripts/encoder_injection_t5gemma.py.bak_spanfix_20260917_095951"
VENDOR_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
VENDOR_DE_FILE = os.path.join(VENDOR_ROOT, "duration_estimator.py")
VENDOR_FILES = ("inference_tts_utils.py", "duration_estimator.py", "data/tokenizer.py")

# 対の出どころ
APPK_JA_CKPT_REL = "eval_results/cjk_n300_rewav/checkpoint_ja.json"   # 付録 K の採点が読む正本
CJK_N300_CKPT_REL = "eval_results/cjk_n300/checkpoint_ja.json"         # 7/23 の作り直しの元（同じ対）
CJK_GEN_REL = "scripts/projection/eval_cjk_n300.py"                    # load_ja_pairs の定義
ENTITIES_REL = "data/entities_10000.jsonl"
APPK_COMMON_REL = "scripts/regen2/appk_common.py"                      # SETS["ja"] の確かめ（読むだけ）
APPK_REWAV_DIR_REL = "eval_results/cjk_n300_rewav"                     # 7/23 の BL（環境の目安に使う）
OLD_PAIR_SRC_REL = "eval_results/crossattn_mask_n300_filtered_v2/results.json"  # main() が開く相対パス（5/11 の対）

# 本番のファイルの md5（2026-10-02 に確かめた値）。違えば GPU ジョブはモデルを読まずに止まる
PINNED_MD5 = {
    EVAL_REL: "7d53ab5c5379e569c08336581084b38a",   # 5/13 から変わっていない（英語の作り直しと同じ版）
    INJ_REL: "4f36fa570145cde805072438d8a39bb1",    # 7/11 のスパン修正の後の版（offset_mapping）
    FA_REL: "be2b300a061e1374509bd1aa75f41b5d",     # Whisper の単語の時刻（4/30 から変わっていない）
    SHIM_REL: "53eb4bfc1008beb361a23c6adfc429ba",   # offline の読み替え（英語の作り直しと同じ）
    APPK_JA_CKPT_REL: "c7817205438bb86a2f46846a75c79a45",  # 7/23 21:21 から変わっていない
}
VENDOR_PINNED_MD5 = {  # 1/15 から変わっていない（5/11 の実行・英語の作り直しと同じ）
    "inference_tts_utils.py": "d6bbe3ac110d56fd3208cbb9c1c4e712",
    "duration_estimator.py": "75bf3506c0aba1e9c543daafe6f651d5",
    "data/tokenizer.py": "2be3114969be6ce59b90f98be021ee04",
}
LEGACY_SPAN_MD5 = "a7aca46875c8b405a89bc3dbfbabddb4"  # 修正前のスパン関数（比べるためだけに読む）
# 対の定義の確かめ（CPU の span_check だけが読む。GPU では見ない）
PAIR_SOURCE_MD5 = {
    CJK_N300_CKPT_REL: "346e2c5a3e97ce82bf6d5b525c761903",
    CJK_GEN_REL: "bccbeb8b8effaf2ad271c057b211c5a2",
    ENTITIES_REL: "eb37d85fc571f662c49dfcd54fa2778e",
}
SILENCE_CRITERION_ID = "regen2-silence-2026-09-29"

OLD_DIR_REL = "eval_results/crossattn_entity_per_ja_n300"
OLD_MD5 = {"results.json": "11a3751c4d0071eb2407c220313530d8",
           "checkpoint.json": "2be8a7b760384c05553a36215e3f7dd0"}

T5GEMMA_TTS_ID = "Aratako/T5Gemma-TTS-2b-2b"
TOKENIZER_ID = "google/t5gemma-2b-2b-ul2"
LANG = "ja"
N_PAIRS = 300
PART_SIZE = 50
N_PARTS_SEED42 = N_PAIRS // PART_SIZE  # 6
PROD_ARGS = ["--lang", LANG, "--n", str(N_PAIRS)]  # 元の引数。この後に --output_dir だけを足す

CONDS = ("bl", "inj", "bl_mask", "inj_mask", "tgt")      # 生成の順（run_condition x4 → synth_simple）
ALIGN_ORDER = ("tgt", "bl", "inj", "bl_mask", "inj_mask")  # entity_ipa の呼ばれる順
PER_CONDS = ("bl", "inj", "bl_mask", "inj_mask")
# 生成の各呼び出しの「形」: (テキスト, 遮断の有無, 直前のスパン呼び出しの対象)
SLOT_SIGNATURE = {
    "bl": ("a", False, ("a",)),
    "inj": ("a", False, ("a", "b")),
    "bl_mask": ("a", True, ("a",)),
    "inj_mask": ("a", True, ("a", "b")),
    "tgt": ("b", False, ()),
}

SEEDS = (42, 43, 44)
OUT_ROOT_REL = "eval_results/_spanfix_appC_ja"
PAIR_FILE_REL = OUT_ROOT_REL + "/pairs/appk_ja_300_v2format.json"
SPAN_CHECK_DIR_REL = OUT_ROOT_REL + "/span_check"
STATUS_DIR_REL = OUT_ROOT_REL + "/status"
AGG_DIR_REL = OUT_ROOT_REL + "/aggregate"
DRYRUN_DIR_REL = OUT_ROOT_REL + "/_dryrun"

# ---------------------------------------------------------------- 状態（英語の作り直しと同じ決まり）
STATUS_VALID = "valid"
PAIR_INVALID = ("unaligned", "silent", "generation_failed")            # 次の seed へ
RETRY_SAME_SEED = ("run_error", "alignment_exception", "capture_error", "no_row_unexplained")  # 直して同じ seed
STATUS_PENDING = "pending"
STATUS_STOP = "stop_author_decision"
MAX_SAME_SEED_RETRIES = 2

# ---------------------------------------------------------------- GPU コマンドの形（決まった形だけを出す）
ALLOWED_SLURM = ("00:20:00", "00:40:00", "01:00:00", "02:00:00", "04:00:00")
SLURM_RUN = "slurm_run.sh"
GUARD_PREFIX = "python3 -u /mnt/scripts/spanfix/gpu_guard.py --min_free_gb 30 --"
GEN_CONTAINER = "/mnt/scripts/spanfix/appc_ja_gen.py"
# 時間の見積もり: 5/11 の JA の再開は 100 対で 158.69 分（1.59 分/対、モデルの読み込みは別）。付録 K の文は
# 文字数が 0.78 倍なので約 1.23 分/対。英語は 10 月の方が 5 月より約 0.85 倍速かった。
MIN_PER_PAIR_EST = 1.23
MIN_PER_PAIR_UPPER = 1.60   # SLURM_TIME はこの上限で決める（5/11 の長い文の速さ）
LOAD_MIN = 5.0
MARGIN = 1.3

# ---------------------------------------------------------------- 論文の今の値（naacl2027/main.tex, 読むだけ）
MAIN_TEX_REL = "naacl2027/main.tex"
EOR_REL = "naacl2027/explanation_of_revisions.tex"
PAPER_JA_CELLS = {  # L956-957（一致率）、L962（PER）。INJ は太字
    "match_bl": r"\phantom{0}5.33 \pm 2.59",
    "match_inj": r"\mathbf{12.67 \pm 3.77}",
    "match_bl_mask": r"\phantom{0}0.00~{+}1.26",
    "match_inj_mask": r"\phantom{0}0.00~{+}1.26",
    "per_bl": r"0.651 \pm 0.294",
    "per_inj": r"\mathbf{0.574 \pm 0.305}",
    "per_bl_mask": r"0.888 \pm 0.166",
    "per_inj_mask": r"0.881 \pm 0.188",
}
PAPER_EN_CELLS_NOW = {  # L958-959, L963（10/02 に英語の作り直しを反映した値。この作業では変えない。表示のためだけ）
    "match_bl": r"\phantom{0}1.00 \pm 1.28",
    "match_inj": r"\phantom{0}2.00 \pm 1.69",
    "match_bl_mask": r"\phantom{0}0.00~{+}1.26",
    "match_inj_mask": r"\phantom{0}0.00~{+}1.26",
    "per_bl": r"0.898 \pm 0.380",
    "per_inj": r"0.917 \pm 0.416",
    "per_bl_mask": r"1.222 \pm 0.480",
    "per_inj_mask": r"1.222 \pm 0.480",
}
PAPER_JA_LINES = (956, 957, 962)
# 日本語についての文（読むだけ。主張の確かめの対象）: 付録 C の Results（L970-977）・Interpretation（L979-985）・
# 表の説明（L966）・対の説明（L938）、説明文書の L281-282
JA_TEXT_RANGES = {MAIN_TEX_REL: [(934, 943), (966, 966), (970, 985)], EOR_REL: [(276, 283)]}


def P(rel, root=None):
    return os.path.join(root or PROJECT_ROOT, rel)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json_new(obj, path, indent=1):
    """新しいファイルとしてだけ書く（既にあれば止まる。上書き・削除はしない）."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
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
    for f, want in VENDOR_PINNED_MD5.items():
        vp = os.path.join(VENDOR_ROOT, f)
        if not os.path.exists(vp) or md5_file(vp) != want:
            errs.append(f"vendor {f} が無いか md5 が違う")
    lp = P(LEGACY_SPAN_REL, root)
    if not os.path.exists(lp) or md5_file(lp) != LEGACY_SPAN_MD5:
        errs.append(f"{LEGACY_SPAN_REL} が無いか md5 が違う")
    errs += silence_criterion_errors(root)
    return errs


def silence_criterion_errors(root=None):
    """無音の基準が 9/29 のもの（ID としきい値）であることを、import せずに確かめる."""
    src = open(P(SILENCE_REL, root), encoding="utf-8").read()
    want = {"RMS_TH": 0.005, "PEAK_TH": 0.05, "FRAME_MS": 20, "FRAME_RMS_TH": 0.01, "ACTIVE_FRAC_TH": 0.05,
            "CRITERION_ID": SILENCE_CRITERION_ID}
    got = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", None) in want:
            try:
                got[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return [f"{SILENCE_REL} の {k} が {got.get(k)!r}（9/29 の基準は {v!r}）" for k, v in want.items() if got.get(k) != v]


def old_result_errors(root=None):
    return [f"{OLD_DIR_REL}/{n} の md5 が違う（読むだけのはずのファイルが変わった）"
            for n, want in OLD_MD5.items() if md5_file(P(f"{OLD_DIR_REL}/{n}", root)) != want]


def code_md5(root=None, own=()):
    rels = list(PINNED_MD5) + [SILENCE_REL, LEGACY_SPAN_REL, PAIR_FILE_REL] + list(own)
    out = {rel: md5_file(P(rel, root)) for rel in rels if os.path.exists(P(rel, root))}
    for f in VENDOR_FILES:
        vp = os.path.join(VENDOR_ROOT, f)
        out["vendor:" + f] = md5_file(vp) if os.path.exists(vp) else None
    return out


# ---------------------------------------------------------------- 対
PAIR_KEYS = ("pair_idx", "text_a", "text_b", "word_a", "word_b")


def load_pairs(root=None):
    """付録 K の日本語 300 対（正本の並び）。pair_idx は並びの番号と同じ（確かめる）."""
    rows = read_json(P(APPK_JA_CKPT_REL, root))["results"]
    pairs = []
    for i, r in enumerate(rows):
        assert r["pair_idx"] == i, ("pair_idx が並びの番号と違う", i, r["pair_idx"])
        assert r.get("lang") == "ja", r.get("lang")
        pairs.append({k: r[k] for k in PAIR_KEYS})
    assert len(pairs) == N_PAIRS, len(pairs)
    for p in pairs:
        # 本番 main() は text_b = text_a.replace(kanji, kata) で作る。付録 K の text_b と同じになること
        assert p["text_a"].replace(p["word_a"], p["word_b"]) == p["text_b"], p["pair_idx"]
        assert p["text_a"].count(p["word_a"]) == 1 and p["text_b"].count(p["word_b"]) == 1, p["pair_idx"]
    d = distinctness(pairs)
    assert d["all_distinct"], d
    return pairs


def distinctness(pairs):
    keys = {
        "pair_tuple": [tuple(p[k] for k in PAIR_KEYS[1:]) for p in pairs],
        "text_a": [p["text_a"] for p in pairs],
        "text_b": [p["text_b"] for p in pairs],
        "word_pair": [(p["word_a"], p["word_b"]) for p in pairs],
        "word_a": [p["word_a"] for p in pairs],
        "word_b": [p["word_b"] for p in pairs],
        "pair_idx": [p["pair_idx"] for p in pairs],
    }
    out = {k: len(set(v)) for k, v in keys.items()}
    out["n"] = len(pairs)
    out["all_distinct"] = all(v == len(pairs) for k, v in out.items() if k != "n")
    return out


def pairs_digest(pairs):
    blob = json.dumps([{k: p[k] for k in PAIR_KEYS} for p in pairs], ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def pairs_v2format(pairs):
    """本番 main() の JA の分岐が読む形（"results" の各行の text / kanji / kata / idx）."""
    return {
        "provenance": {
            "what": "Appendix K JA 300 pairs in the field layout that eval_crossattn_entity_per.py main() reads "
                    "for --lang ja (text=text_a, kanji=word_a, kata=word_b, idx=pair_idx)",
            "source": APPK_JA_CKPT_REL, "source_md5": PINNED_MD5[APPK_JA_CKPT_REL],
            "pairs_digest": pairs_digest(pairs), "producer": "scripts/spanfix/appc_ja_span_check.py",
        },
        "results": [{"idx": p["pair_idx"], "text": p["text_a"], "kanji": p["word_a"], "kata": p["word_b"]}
                    for p in pairs],
    }


def production_ja_branch_source(root=None):
    """本番 main() の `elif args.lang == "ja":` の本体（ast）と、その元の文字列."""
    src = open(P(EVAL_REL, root), encoding="utf-8").read()
    tree = ast.parse(src)
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    found = []
    for n in ast.walk(main):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and len(n.test.comparators) == 1:
            c = n.test.comparators[0]
            left = n.test.left
            if (isinstance(c, ast.Constant) and c.value == "ja" and isinstance(left, ast.Attribute)
                    and left.attr == "lang" and getattr(left.value, "id", None) == "args"):
                found.append(n)
    assert len(found) == 1, f"JA の分岐が 1 つでない（{len(found)}）"
    return found[0].body, "\n".join(ast.get_source_segment(src, b) for b in found[0].body)


def run_production_ja_loader(open_fn, n=N_PAIRS, root=None):
    """本番 main() の JA の分岐を、そのままの文で実行して pairs を返す（open だけ与えたものを使う）."""
    import argparse
    body, _ = production_ja_branch_source(root)
    ns = {"json": json, "open": open_fn, "args": argparse.Namespace(n=n, lang=LANG)}
    exec(compile(ast.Module(body=body, type_ignores=[]), P(EVAL_REL, root), "exec"), ns)
    return ns["pairs"]


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
    out = []
    for m in sorted(glob.glob(os.path.join(P(seed_dir_rel(seed), root), "*", "manifest.json"))):
        man = read_json(m)
        out.append((os.path.basename(os.path.dirname(m)), man))
    out.sort(key=lambda x: (x[1].get("created_order", 0), x[0]))
    return out


def list_runs(batch_abs):
    return sorted(d for d in glob.glob(os.path.join(batch_abs, "runs", "run*")) if os.path.isdir(d))


def seed_fingerprint(seed, root=None):
    base = P(seed_dir_rel(seed), root)
    top = P("", root)
    out = {}
    for pat in ("*/manifest.json", "*/checkpoint.json", "*/runs/run*/events.jsonl"):
        for f in sorted(glob.glob(os.path.join(base, pat))):
            out[os.path.relpath(f, top)] = md5_file(f)
    return out


def suggest_slurm_time(n_pairs):
    """(読み込み 5 分 + 1.6 分/対) x 1.3 以上の、許された最小の SLURM_TIME（1 ジョブは 50 対まで）."""
    if n_pairs > PART_SIZE:
        raise ValueError("1 ジョブは 50 対まで（分けて流す）")
    need = (LOAD_MIN + MIN_PER_PAIR_UPPER * n_pairs) * MARGIN
    for t in ALLOWED_SLURM:
        h, m, s = (int(x) for x in t.split(":"))
        if h * 60 + m + s / 60 >= need:
            return t
    return ALLOWED_SLURM[-1]


def expected_minutes(n_pairs):
    return round(LOAD_MIN + MIN_PER_PAIR_EST * n_pairs, 1)


def format_gpu_command(slurm_time, args_str):
    assert slurm_time in ALLOWED_SLURM, slurm_time
    return f"SLURM_TIME={slurm_time} {SLURM_RUN} {GUARD_PREFIX} {GEN_CONTAINER} {args_str}".rstrip()


def read_events(run_abs):
    p = os.path.join(run_abs, "events.jsonl")
    ev = []
    if not os.path.exists(p):
        return ev
    with open(p, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                ev.append({"ev": "corrupt_line", "line_no": ln})
    return ev


# ---------------------------------------------------------------- 本番と同じ集計・今の表と同じ書式
def production_summary(rows):
    """本番 main() の末尾（L305-333）と同じ式。rows は placeholder を含まない行."""
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
    """regen2_common._wilson / appc_en_common.wilson と同じ式（中心と半幅）."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, h


def strip_bold(s):
    m = re.fullmatch(r"\\mathbf\{(.*)\}", s)
    return m.group(1) if m else s


def per_values(rows, cond):
    return [r[f"{cond}_per"] for r in rows if r.get("aligned") and r.get(f"{cond}_per") is not None]


def table_cells(rows):
    """表の JA 行と同じ書式（太字なし）.

    上: 一致率 % ± Wilson 95% 半幅（0 件は「0.00 + 上端」）。下: PER の平均 ± 標準偏差（np.std, n で割る）。
    PER は生の値から直接 3 桁にする（今の表の JA の 8 セルはこの作り方で一字一句再現する。results.json の
    4 桁の値を経由すると INJ の SD が 0.304 になり、表の 0.305 と合わない）。英語の旧い 8 セルもこの作り方で同じ。
    """
    import numpy as np
    s = production_summary(rows)
    n = s["n_aligned"]
    cells = {}
    for c in PER_CONDS:
        k = s[f"{c}_match_count"]
        cen, half = wilson(k, n)
        pct = 100 * k / n
        if k == 0:
            v = f"{pct:.2f}~{{+}}{100 * (cen + half):.2f}"
        else:
            v = f"{pct:.2f} \\pm {100 * half:.2f}"
        cells[f"match_{c}"] = (r"\phantom{0}" if pct < 10 else "") + v
        vs = per_values(rows, c)
        cells[f"per_{c}"] = f"{float(np.mean(vs)):.3f} \\pm {float(np.std(vs)):.3f}"
    return cells


def table_cells_via4(rows):
    """参考: results.json の 4 桁の値を経由した 3 桁（appc_en_common.table_cells と同じ作り方）."""
    s = production_summary(rows)
    out = {}
    for c in PER_CONDS:
        m, sd = s[f"{c}_per_mean_std"]
        out[f"per_{c}"] = f"{m:.3f} \\pm {sd:.3f}"
    return out


def bold_ok(rows):
    """太字（JA の INJ）の根拠: INJ が 4 条件で一致率が最も高く（同点なし）、PER が最も低い（同点なし）か."""
    import numpy as np
    s = production_summary(rows)
    m = {c: s[f"{c}_match_count"] for c in PER_CONDS}
    per = {c: float(np.mean(per_values(rows, c))) for c in PER_CONDS}
    return {"match_counts": m, "per_means": per,
            "inj_highest_match_strict": all(m["inj"] > m[c] for c in PER_CONDS if c != "inj"),
            "inj_lowest_per_strict": all(per["inj"] < per[c] for c in PER_CONDS if c != "inj")}


# ---------------------------------------------------------------- 1 対の状態（数値は見ない）
def expected_masked_cols(span_ref, label):
    """遮断した列の期待値。BL_mask は a 側の範囲、INJ_mask は継ぎ合わせの後の目標の範囲.

    本番 run_condition: 長さが同じなら置換で [s_s, s_e)、違えば継ぎ合わせで [s_s, s_s + (r_e - r_s))。
    """
    a0, a1 = span_ref["a"]
    if label == "bl_mask":
        return list(range(a0, a1))
    b0, b1 = span_ref["b"]
    return list(range(a0, a0 + (b1 - b0)))


def classify_pair(row, gen_events, align_events, err_events, batch_complete, wav_root, span_ref):
    """1 つの seed での 1 対の状態と根拠を返す（数値は見ない。appc_en_common.classify_pair の JA 版）.

    英語との違いは 2 つ: 遮断した列の期待値（INJ_mask は継ぎ合わせの後の長さ）と、記憶の確かめ
    （appc_ja_gen.Ctx.mem_check: 長さ・前後の文脈・遮断の有無で記憶が同じか）。
    """
    import silence_criterion as SC
    ev = {"row_present": row is not None}
    pair_errs = [e for e in err_events if e.get("kind") == "pair_loop"]
    whisper_errs = [e for e in err_events if e.get("kind") == "whisper"]
    gen_exc = [g for g in gen_events if g.get("exception")]
    ev["pair_loop_errors"] = [e.get("line") for e in pair_errs]
    ev["whisper_errors"] = [e.get("line") for e in whisper_errs]
    ev["generate_exceptions"] = [g.get("exception") for g in gen_exc]

    span_bad = []
    for g in gen_events:
        for sc in g.get("spans", []):
            side = sc.get("side")
            want = span_ref.get(side) if side in ("a", "b") else None
            got = sc.get("result")
            if want is None or got is None or list(got) != list(want):
                span_bad.append({"label": g.get("label"), "side": side, "got": got, "want": want})
        if g.get("label") in ("bl_mask", "inj_mask") and span_ref.get("a") and span_ref.get("b"):
            want_cols = expected_masked_cols(span_ref, g.get("label"))
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
    if not mem or not mem[0] or mem[0].get("ok") is not True:
        cap_err.append(f"mem_check={mem[0] if mem else None}")
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


def same_seed_retry_verdict(history):
    """appc_en_common.same_seed_retry_verdict と同じ決まり."""
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


# ---------------------------------------------------------------- 7/23 の付録 K の BL との一致（環境の目安）
def bl_env_compare(new_wav_abs, pair_idx, root=None):
    """seed 42 の BL は 7/23 の付録 K の BL（eval_cjk_n300.py、同じ文・同じ経路・seed 42）と同じ音になるはず.

    7/23 の wav は torchaudio.save の 16 bit PCM（44.1 kHz）、こちらは float32 で保存しているので、
    「同じ」= 長さと標本化周波数が同じで、すべての標本の差が 16 bit の 1 段（1/32768）以内（振り切れた標本は除く）。
    違う音なら差は桁違いに大きい。標本の一致だけを返す（表の値は見ない。対の選択には使わない）。
    """
    import numpy as np
    import soundfile as sf
    old = P(f"{APPK_REWAV_DIR_REL}/s{pair_idx:04d}_bl.wav", root)
    if not os.path.exists(old) or not os.path.exists(new_wav_abs):
        return {"compared": False}
    x, sx = sf.read(new_wav_abs, dtype="float32", always_2d=False)
    y, sy = sf.read(old, dtype="float32", always_2d=False)
    sub = sf.info(old).subtype
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    out = {"compared": True, "same_sr": sx == sy, "len_new": int(x.size), "len_0723": int(y.size),
           "subtype_0723": sub}
    out["same_len"] = out["len_new"] == out["len_0723"]
    if out["same_len"] and out["same_sr"]:
        d = np.abs(x - y)
        out["max_abs_diff"] = float(d.max()) if d.size else 0.0
        if sub == "PCM_16":
            lsb = 1.0 / 32768
            clipped = np.abs(x) >= 1.0 - lsb
            out["n_clipped"] = int(clipped.sum())
            out["max_abs_diff_unclipped"] = float(d[~clipped].max()) if (~clipped).any() else 0.0
            out["identical"] = bool(out["max_abs_diff_unclipped"] <= lsb * 1.0001)
            out["rule"] = "within one 16-bit step (unclipped samples)"
        else:
            out["identical"] = out["max_abs_diff"] == 0.0
            out["rule"] = "exact"
    else:
        out["identical"] = False
    return out
