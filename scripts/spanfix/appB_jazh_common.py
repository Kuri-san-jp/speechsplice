#!/usr/bin/env python3
"""spanfix (2a): 付録 B（Table A1, tab:crossattn_mask）の JA 4 対・ZH 4 対の作り直しの共通部品（2026-10-02）.

標準ライブラリだけを読む（numpy・torch・transformers は読まない）。英語の作り直しの共通部品 spanfix_common の
道具を読むだけで、英語のファイル（spanfix_common.py, spanfix_appB_en_*.py）は 1 バイトも変えない
（英語の GPU の記録が自分のコードの md5 を確かめているので、書き換えると英語の記録の確かめが止まる）。

著者の決定（2026-10-02, 2a。note/2026-10-02_gpuneeds2_report.md の GPU 表 5 = B4）:
  * 5/19-20 の本番（scripts/projection/eval_crossattn_entity_dtw.py --lang {ja,zh} --cross_culture
    --inject_mode length_preserve --n 300。.perm_cmds.txt 16714 行）は 7/11 に直す前の範囲の関数で作られた。
    範囲が誤っていたのは JA 59, 99, 164, 179 と ZH 15, 145, 246, 295 の 8 対
    （eval_results/_audit_spanB_20261002/cc_pairs_spans.json。164・179 は名前ではなく次の語「さんが・さんに」）。
  * この 8 対だけを、英語（10/02）と同じ手順で作り直す: 同じスクリプトを 1 バイトも変えずに呼び、同じ引数、seed 42。
    違いは範囲の関数（scripts/encoder_injection_t5gemma.py の find_entity_token_range、7/11 修正）だけ。
  * 使えない対（無音・生成の失敗・位置合わせの失敗）は対ごと seed 43、次に 44。選ぶ基準は有効性だけ（数値は見ない）。
  * ほかの対は 5/19-20 の本番の行のまま。ZH は 9/29 に対 42 を seed 43 で入れ替えたファイル
    （eval_results/_regen2_crossattn_merged/zhcc_lp_n300/）から始める。
  * 新しい N=300 のファイルは eval_results/_spanfix_appB_jazh/merged_<日時>/<lang>cc_lp_n300/ に作る。
    既存の結果・原稿（naacl2027/ ほか）には書かない。何も消さない。

修正後の関数の範囲（§3.2 の定義: 実体語の文字と重なるトークンをすべて取る）について:
  JA/ZH では名前と隣の文字が 1 つのトークンになることがある（例: 「を渡|辺」「田|中的」「和小|川」「は何」「を高」）。
  そのときは隣の文字（を・的・和・は）も範囲に入る。これは今の関数の定義どおりの振る舞いなので止めない（記録する）。
  確かめるのは、(1) 名前の文字をすべて覆う、(2) 範囲の外のトークンが名前に掛からない、(3) 復号した文字列が名前を含む、
  (4) offset mapping を使わない別の求め方（前から 1 トークンずつ復号した文字の位置）と範囲が一致する、の 4 つ。

pyopenjtalk（著者の決まり）: T5Gemma-TTS 同梱（vendor）の発話長推定（duration_estimator.py）の中でだけ使う。
  PyopenjtalkGate が、import を vendor の duration_estimator.py からだけ通し、関数の呼び出しも vendor の中
  （と pyopenjtalk 自身の中）からだけ通す。ほかから import・呼び出し・参照の保持があれば記録し、止める。
  ホスト（試運転・集計）では sys.modules["pyopenjtalk"] = None で一切読まない。
"""
import ast
import glob
import importlib
import json
import os
import re
import subprocess
import sys
import types

sys.dont_write_bytecode = True  # scripts/ や data/ に .pyc を作らない
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import spanfix_common as C  # noqa: E402  標準ライブラリだけを読む（英語の共通部品。読むだけ）

# ---------------------------------------------------------------- 置き場
HOST_ROOT = C.HOST_ROOT
CONTAINER_ROOT = C.CONTAINER_ROOT
IN_CONTAINER = C.IN_CONTAINER
PROJECT_ROOT = C.PROJECT_ROOT
P = C.P
md5_file = C.md5_file
read_json = C.read_json
write_json_new = C.write_json_new
now = C.now

LANGS = ("ja", "zh")
TARGETS = {"ja": (59, 99, 164, 179), "zh": (15, 145, 246, 295)}
N_PAIRS = C.N_PAIRS
CONDS = C.CONDS
SEEDS = C.SEEDS
PRODUCER = "appB_jazh_gen.py"
MODE_GPU = C.MODE_GPU
MODE_FAKE = C.MODE_FAKE
THREADS = C.THREADS

GEN_REL = "scripts/spanfix/appB_jazh_gen.py"
COMMON_REL = "scripts/spanfix/appB_jazh_common.py"
COLLECT_REL = "scripts/spanfix/appB_jazh_collect.py"
EN_GEN_REL = C.GEN_REL        # 写し取り・範囲の記録・print の記録の部品を借りる（読むだけ）
EN_COMMON_REL = C.COMMON_REL  # 状態の判定（load_run / pair_status）と道具を借りる（読むだけ）
# 借りる英語のファイルの md5（10/02 の英語の作り直しの後の値）。変わっていれば GPU を始めない
DEP_PINNED_MD5 = {EN_GEN_REL: "f24016927dbf31dcbf4eefda978682d5",
                  EN_COMMON_REL: "4aa829a95684c47281838c9aae8732c8"}
OWN_CODE = (GEN_REL, COMMON_REL, EN_GEN_REL, EN_COMMON_REL)  # GPU の後に変われば著者の判断
CODE_FILES = tuple(C.PINNED_MD5) + (C.SILENCE_REL, C.LEGACY_SPAN_REL) + OWN_CODE

# T5Gemma-TTS 同梱のコード（TTS36。コンテナにも同じ置き場で読み取り専用で見える）。10/02 の英語の GPU の記録の値と同じ
VENDOR_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
VENDOR_PINNED = {
    "data.tokenizer": (VENDOR_ROOT + "/data/tokenizer.py", "2be3114969be6ce59b90f98be021ee04"),
    "duration_estimator": (VENDOR_ROOT + "/duration_estimator.py", "75bf3506c0aba1e9c543daafe6f651d5"),
    "inference_tts_utils": (VENDOR_ROOT + "/inference_tts_utils.py", "d6bbe3ac110d56fd3208cbb9c1c4e712"),
}
VENDOR_DE = VENDOR_PINNED["duration_estimator"][0]
VENDOR_ITU = VENDOR_PINNED["inference_tts_utils"][0]

DATA_REL = C.DATA_REL
PAIRS_MODULE = {"ja": "ja_cross_culture_names", "zh": "zh_cross_culture_names"}
PAIRS_MODULE_MD5 = {"ja": "a74383c4f910c8388031deb2d68662a7", "zh": "1260a6110cdff2591aa5bb0e44836ae2"}

# 5/19-20 の本番（読むだけ）
PROD_DIR_REL = {"ja": "eval_results/crossattn_entity_dtw_jacc_lp_n300",
                "zh": "eval_results/crossattn_entity_dtw_zhcc_lp_n300"}
PROD_MD5 = {"ja": {"results.json": "70f7f3f9f09188eb23ca07262bb3bca2",
                   "checkpoint.json": "ae04825845096da473fd028fc438e71e"},
            "zh": {"results.json": "1ad9b54b49670450c300089b0f87cb93",
                   "checkpoint.json": "d99968b0b9331207fe20ad760522e6ea"}}
# 今の Table A1 の出どころ = 取り込みの元（読むだけ）。ZH は 9/29 に対 42 を入れ替えたもの
BASE_DIR_REL = {"ja": PROD_DIR_REL["ja"], "zh": "eval_results/_regen2_crossattn_merged/zhcc_lp_n300"}
BASE_MD5 = {"ja": PROD_MD5["ja"],
            "zh": {"results.json": "4010f2b9f03ed82fe80a0133194c583d",
                   "checkpoint.json": "6b9315a38f159d523e6fc5d04eea4d62"}}
BASE_NON_PROD_ROWS = {"ja": (), "zh": (42,)}  # 取り込みの元で、本番の行でない行（9/29 の seed 43）
G3_ZH_MERGE_RECORD_REL = "eval_results/_regen2_crossattn_merged/zhcc_lp_n300/merge_record.json"
G3_ZH_MERGE_RECORD_MD5 = "b95e07163df4e320a21032cb2741fe49"
AUDIT_SPANS_REL = "eval_results/_audit_spanB_20261002/cc_pairs_spans.json"

OUT_REL = "eval_results/_spanfix_appB_jazh"
DRYRUN_REL = OUT_REL + "/dryrun"
LOGS_REL = OUT_REL + "/logs"
SPAN_CHECK_REL = OUT_REL + "/span_check"
MERGED_PREFIX = "merged_"  # OUT_REL/merged_<日時>/<lang>cc_lp_n300/
EN_FINAL_TV_REL = "eval_results/_spanfix_appB_en/final_20261002_081631/table_values.json"

# 今の原稿の値（読むだけ。書かない）
MAIN_TEX_REL = "naacl2027/main.tex"
EOR_REL = "naacl2027/explanation_of_revisions.tex"
CURRENT_TABLE = {
    "ja": {"MCD (dB)": ("31.70 \\pm 0.77", "0.00 \\pm 0.00"), "MEL Similarity": ("0.94 \\pm 0.01", "1.00 \\pm 0.00")},
    "zh": {"MCD (dB)": ("35.35 \\pm 0.39", "0.00 \\pm 0.00"), "MEL Similarity": ("0.86 \\pm 0.02", "1.00 \\pm 0.00")},
}
MAIN_TEX_ROWS = {"ja": (919, 920), "zh": (921, 922)}  # naacl2027/main.tex（1 始まり）
CURRENT_EOR = {"en": "35.31\\pm0.27", "ja": "31.70\\pm0.77", "zh": "35.35\\pm0.39"}
EOR_LINES = (279, 280)
TABLE_KEYS = C.TABLE_KEYS
CI_METHOD = C.CI_METHOD


def prod_args(lang):
    """本番の引数（.perm_cmds.txt 16714 行の for L in en ja zh）。この後に --output_dir だけを足す."""
    assert lang in LANGS, lang
    return ["--lang", lang, "--cross_culture", "--inject_mode", "length_preserve", "--n", str(N_PAIRS)]


# ---------------------------------------------------------------- 環境変数（numpy・torch・HF を読む前に呼ぶ）
EXTRA_ENV = {"RAYON_NUM_THREADS": str(THREADS), "TOKENIZERS_PARALLELISM": "false"}


def setup_env():
    """英語と同じ設定（スレッド 13、HF_HOME、HF_HUB_OFFLINE=1、NUMBA_CACHE_DIR、ホストでは GPU を見せない）に、
    tokenizer の Rust のスレッドの上限を足す。起動したときの値を返す."""
    extra_launch = {k: os.environ.get(k) for k in EXTRA_ENV}
    launch = C.setup_env()
    launch.update(extra_launch)
    os.environ.update(EXTRA_ENV)
    sys.dont_write_bytecode = True
    return launch


# ---------------------------------------------------------------- 時間の見積もりと GPU のコマンド
ALLOWED_SLURM = ("00:20:00", "00:40:00", "01:00:00", "02:00:00", "04:00:00")
MIN_PER_PAIR = {"ja": 393.77 / 300, "zh": 277.58 / 300}  # 5/19-20 の本番の summary.elapsed_min
LOAD_MIN = 5.0
MARGIN = 1.5
CHUNK = 40  # 1 つのジョブに入れる対の上限（JA でも約 58 分。75 分以内）
GUARD_PREFIX = "python3 -u /mnt/scripts/spanfix/gpu_guard.py --min_free_gb 30 --"
GEN_IN_CONTAINER = "/mnt/" + GEN_REL


def slurm_time_for(lang, n_pairs):
    est = LOAD_MIN + MIN_PER_PAIR[lang] * n_pairs
    for t in ALLOWED_SLURM:
        h, m, _ = (int(x) for x in t.split(":"))
        if h * 60 + m >= est * MARGIN:
            return t, round(est, 1)
    return None, round(est, 1)


def gpu_command(lang, seed, n_pairs, extra=""):
    st, est = slurm_time_for(lang, n_pairs)
    cmd = (f"SLURM_TIME={st} slurm_run.sh {GUARD_PREFIX} {GEN_IN_CONTAINER} "
           f"--lang {lang} --seed {seed}" + (f" {extra}" if extra else ""))
    return {"lang": lang, "seed": seed, "n_pairs": n_pairs, "slurm_time": st, "expected_min": est,
            "command": cmd, "cwd": HOST_ROOT}


# ---------------------------------------------------------------- 照合（md5）
def current_code_md5(root=None):
    return {rel: (md5_file(P(rel, root)) if os.path.isfile(P(rel, root)) else None) for rel in CODE_FILES}


def vendor_md5_now():
    return {name: {"file": path, "md5": md5_file(path) if os.path.isfile(path) else None}
            for name, (path, _) in VENDOR_PINNED.items()}


def pinned_errors(lang, root=None):
    """GPU の前・試運転で: 本番のファイル、借りる英語のファイル、vendor、対のデータ、取り込みの元が決まりの md5 か."""
    errs = list(C.pinned_file_errors(root))
    for rel, want in DEP_PINNED_MD5.items():
        got = md5_file(P(rel, root))
        if got != want:
            errs.append(f"{rel} の md5 が固定の値と違う（今 {got} / 固定 {want}）")
    for name, (path, want) in VENDOR_PINNED.items():
        got = md5_file(path) if os.path.isfile(path) else None
        if got != want:
            errs.append(f"vendor {name} ({path}) の md5 が固定の値と違う（今 {got} / 固定 {want}）")
    got = md5_file(P(f"{DATA_REL}/{PAIRS_MODULE[lang]}.py", root))
    if got != PAIRS_MODULE_MD5[lang]:
        errs.append(f"{PAIRS_MODULE[lang]}.py の md5 が違う（今 {got} / 固定 {PAIRS_MODULE_MD5[lang]}）")
    for kind, d, pins in (("本番", PROD_DIR_REL[lang], PROD_MD5[lang]), ("取り込みの元", BASE_DIR_REL[lang], BASE_MD5[lang])):
        for name, want in pins.items():
            got = md5_file(P(f"{d}/{name}", root))
            if got != want:
                errs.append(f"{kind} {d}/{name} の md5 が違う（今 {got} / 固定 {want}）")
    return errs


# ---------------------------------------------------------------- 文と対
_NORM_NAMES = ("_REPLACE_MAP", "_FULLWIDTH_ALPHA_TO_HALFWIDTH", "_HALFWIDTH_KATAKANA_CHARS",
               "_FULLWIDTH_KATAKANA_CHARS", "_HALFWIDTH_KATAKANA_TO_FULLWIDTH", "_FULLWIDTH_DIGITS_TO_HALFWIDTH")


def vendor_normalizer(path=VENDOR_ITU):
    """vendor の normalize_text_with_lang(text, lang) と同じ文字の置き換え（lang を明示したとき）.

    本番の get_encoder_output は lang=args.lang で正規化した文に範囲の関数を当てる。vendor のモジュールは
    import しない（duration_estimator 経由で pyopenjtalk・nltk を読むので）。ファイルから置き換えの表と
    _normalize_japanese_text だけを取り出して exec する（中身は vendor のまま。md5 は VENDOR_PINNED で固定）。
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    nodes, found = [], set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            names = {getattr(t, "id", None) for t in n.targets}
            if names & set(_NORM_NAMES):
                nodes.append(n)
                found |= names & set(_NORM_NAMES)
        elif isinstance(n, ast.FunctionDef) and n.name == "_normalize_japanese_text":
            nodes.append(n)
            found.add(n.name)
    assert found == set(_NORM_NAMES) | {"_normalize_japanese_text"}, found
    ns = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    fn = ns["_normalize_japanese_text"]

    def norm(text, lang):
        if text and lang.lower().startswith("ja"):
            return fn(text)
        return text
    return norm


def load_pairs(lang, root=None):
    """eval_crossattn_entity_dtw.py の main() と同じ組み立て（cross_culture, [:300]）. data/ は import の経路に足さない."""
    import importlib.util
    path = P(f"{DATA_REL}/{PAIRS_MODULE[lang]}.py", root)
    got = md5_file(path)
    assert got == PAIRS_MODULE_MD5[lang], f"{path} の md5 が違う: {got}"
    spec = importlib.util.spec_from_file_location(f"_appB_jazh_pairs_{lang}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cc = mod.build_300_pairs()[:N_PAIRS]
    pairs = [{k: q[k] for k in ("text_a", "text_b", "word_a", "word_b", "pair_idx")} for q in cc]
    assert len(pairs) == N_PAIRS and [q["pair_idx"] for q in pairs] == list(range(N_PAIRS))
    # 写し取りは target_text で呼び出しを対と文（a/b）に結びつけるので、文は対ごとに一意で a と b は重ならない
    ta, tb = [q["text_a"] for q in pairs], [q["text_b"] for q in pairs]
    assert len(set(ta)) == N_PAIRS and len(set(tb)) == N_PAIRS and not (set(ta) & set(tb))
    for i in TARGETS[lang]:
        q = pairs[i]
        assert q["text_a"].count(q["word_a"]) == 1 and q["text_b"].count(q["word_b"]) == 1, q
    return pairs


# ---------------------------------------------------------------- 範囲の確かめ（tokenizer だけ）
def decode_char_spans(tok, ids, text):
    """offset mapping を使わない別の求め方: 前から 1 トークンずつ復号し、各トークンの文字の位置を求める."""
    out, prev = [], 0
    for k in range(len(ids)):
        s = tok.decode(ids[:k + 1])
        if "�" in s:
            return None, f"token {k} で文字が途中で切れる"
        if len(s) < prev or text[:len(s)] != s:
            return None, f"token {k} までの復号が文の先頭と一致しない"
        out.append((prev, len(s)))
        prev = len(s)
    if prev != len(text):
        return None, "復号した文が元の文と同じ長さでない"
    return out, None


def independent_overlap_span(tok, text, word):
    ids = tok.encode(text, add_special_tokens=False)
    spans, why = decode_char_spans(tok, ids, text)
    if spans is None:
        return None, why
    pos = text.find(word)
    if pos < 0:
        return None, "実体語が文に無い"
    end = pos + len(word)
    idx = [i for i, (a, b) in enumerate(spans) if b > a and a < end and b > pos]
    if not idx:
        return None, "実体語に重なるトークンが無い"
    return [min(idx), max(idx) + 1], None


def cover_check(tok, text, word, span):
    """範囲 [s, e) のトークンが実体語の文字をすべて覆い、範囲の外のトークンが実体語に掛からないか.

    範囲の中の、実体語の外の文字（助詞など。名前と同じトークンに入ったもの）は extra_chars に記録する（止めない）。
    """
    out = {"ok": False, "exact": False}
    if span is None:
        out["why"] = "範囲が None"
        return out
    s, e = span
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = text.find(word)
    end = pos + len(word)
    if not (0 <= s < e <= len(ids)) or pos < 0:
        out["why"] = f"範囲が文の外 ({s},{e}) / {len(ids)} token"
        return out
    covered, extra = set(), []
    for i in range(s, e):
        a, b = offs[i]
        for c in range(a, b):
            if pos <= c < end:
                covered.add(c)
            elif not text[c].isspace():
                extra.append(c)
    neighbors = [i for i in range(len(ids)) if not (s <= i < e) and offs[i][0] < end and offs[i][1] > pos]
    decoded = tok.decode(ids[s:e])
    out.update({"covers_all_chars": len(covered) == len(word), "extra_chars": "".join(text[c] for c in extra),
                "neighbor_tokens_overlapping_name": neighbors, "decoded": decoded,
                "pieces": tok.convert_ids_to_tokens(ids[s:e])})
    out["ok"] = bool(out["covers_all_chars"] and not neighbors and word in decoded)
    out["exact"] = bool(out["ok"] and not extra)
    if not out["ok"]:
        out["why"] = "実体語を覆っていない、または範囲の外のトークンが実体語に掛かる"
    return out


def span_table(tok, pairs, lang, fixed_fn, legacy_fn, norm):
    """対ごとの範囲（修正後・5/19-20 の時点・別の求め方）。本番と同じく正規化した文に当てる（add_bos=0）."""
    rows = {}
    for p in pairs:
        r = {"pair_idx": p["pair_idx"]}
        for side, text, word in (("a", p["text_a"], p["word_a"]), ("b", p["text_b"], p["word_b"])):
            tn = norm(text, lang)
            fx = fixed_fn(tok, tn, word, add_bos=0)
            fx = list(fx) if fx else None
            lg = legacy_fn(tok, tn, word, add_bos=0)
            lg = list(lg) if lg else None
            ind, why = independent_overlap_span(tok, tn, word)
            r[side] = {"text": text, "text_norm": tn, "normalization_changed_text": tn != text, "word": word,
                       "fixed": fx, "legacy_0520": lg, "independent_overlap": ind, "independent_why": why,
                       "fixed_equals_independent": fx is not None and fx == ind,
                       "changed_vs_0520": lg != fx,
                       "fixed_check": cover_check(tok, tn, word, fx), "legacy_check": cover_check(tok, tn, word, lg)}
        r["fixed_ok"] = bool(r["a"]["fixed_check"]["ok"] and r["b"]["fixed_check"]["ok"]
                             and r["a"]["fixed_equals_independent"] and r["b"]["fixed_equals_independent"])
        r["fixed_exact"] = bool(r["a"]["fixed_check"]["exact"] and r["b"]["fixed_check"]["exact"])
        r["changed_vs_0520"] = bool(r["a"]["changed_vs_0520"] or r["b"]["changed_vs_0520"])
        r["src_span_changed_vs_0520"] = bool(r["a"]["changed_vs_0520"])
        r["legacy_pointed_at_name"] = bool(r["a"]["legacy_check"]["ok"] and r["b"]["legacy_check"]["ok"])
        rows[p["pair_idx"]] = r
    return rows


def span_summary(table, lang):
    ch = sorted(i for i, r in table.items() if r["changed_vs_0520"])
    return {"n_pairs": len(table), "pairs_whose_span_changes_vs_0520": ch,
            "equals_targets": ch == sorted(TARGETS[lang]),
            "src_side_changed": sorted(i for i in ch if table[i]["src_span_changed_vs_0520"]),
            "fixed_ok": sum(1 for r in table.values() if r["fixed_ok"]),
            "fixed_not_ok": sorted(i for i, r in table.items() if not r["fixed_ok"]),
            "fixed_exact": sum(1 for r in table.values() if r["fixed_exact"]),
            "fixed_with_extra_chars": {str(i): {s: r[s]["fixed_check"].get("extra_chars") for s in ("a", "b")
                                                if r[s]["fixed_check"].get("extra_chars")}
                                       for i, r in sorted(table.items()) if not r["fixed_exact"]},
            "targets": {str(i): {s: {"text_norm": table[i][s]["text_norm"], "word": table[i][s]["word"],
                                     "legacy_0520": table[i][s]["legacy_0520"],
                                     "legacy_pieces": table[i][s]["legacy_check"].get("pieces"),
                                     "fixed": table[i][s]["fixed"],
                                     "fixed_pieces": table[i][s]["fixed_check"].get("pieces")}
                                 for s in ("a", "b")} for i in TARGETS[lang] if i in table}}


def audit_agreement(table, lang, root=None):
    """10/02 の点検（scripts/_audit/spanB/span_lib.py）の範囲と、すべての対で一致するか（読むだけ）."""
    d = read_json(P(AUDIT_SPANS_REL, root))[lang]
    rows = {r["idx"]: r for r in d["rows"]}
    bad = []
    for i, r in table.items():
        a = rows.get(i)
        if a is None:
            bad.append({"pair_idx": i, "why": "点検に無い"})
            continue
        for side, key in (("a", "src"), ("b", "ref")):
            if r[side]["fixed"] != list(a[key]["fixed"]) or r[side]["legacy_0520"] != list(a[key]["legacy"]):
                bad.append({"pair_idx": i, "side": side, "here": [r[side]["fixed"], r[side]["legacy_0520"]],
                            "audit": [a[key]["fixed"], a[key]["legacy"]]})
    return {"n_compared": len(table), "n_disagree": len(bad), "disagreements": bad[:10],
            "audit_legacy_differs_idx": d["summary"]["legacy_differs_idx"]}


# ---------------------------------------------------------------- 状態（英語の判定を言語つきで使う）
STATUS_VALID = C.STATUS_VALID
STATUS_ALIGN_EXC = C.STATUS_ALIGN_EXC
CRASHED_NO_RECORD = C.CRASHED_NO_RECORD
STATUS_PYOJT = "pyopenjtalk_violation"
PAIR_INVALID = C.PAIR_INVALID
RETRY_SAME_SEED = C.RETRY_SAME_SEED
STOP_STATUSES = C.STOP_STATUSES + (STATUS_PYOJT,)


def run_dir_rel(lang, seed, tag, stamp, uniq):
    return f"{OUT_REL}/{lang}/seed{seed}/{tag}_{stamp}_{uniq}"


def list_runs(data_root, lang, seed):
    base = os.path.join(data_root, OUT_REL, lang, f"seed{seed}")
    runs, warns = [], []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        rr = os.path.join(d, "run_record.json")
        if not os.path.isfile(rr):
            warns.append(f"{os.path.relpath(d, data_root)}: run_record.json が無い（main() の前に止まった置き場。対は数えない）")
            continue
        try:
            rec = read_json(rr)
        except Exception as e:  # noqa: BLE001
            warns.append(f"{os.path.relpath(rr, data_root)} が読めない: {type(e).__name__}")
            continue
        runs.append((rec.get("started", ""), os.path.relpath(d, data_root), rec))
    runs.sort(key=lambda x: (x[0], x[1]))
    return [(rel, rec) for _, rel, rec in runs], warns


def all_runs(data_root, lang):
    runs, warns = {}, []
    for s in SEEDS:
        lst, w = list_runs(data_root, lang, s)
        warns += w
        runs[s] = [C.load_run(data_root, rel, rec) for rel, rec in lst]
    return runs, warns


def pair_status(run, idx):
    """英語と同じ判定（spanfix_common.pair_status）に、pyopenjtalk の決まりの違反（止まる）を足す."""
    st, info = C.pair_status(run, idx)
    if st is None:
        return st, info
    end = run["end"]
    viol = ((end or {}).get("pyopenjtalk_audit") or {}).get("violations")
    if viol:
        info["pyopenjtalk_violations"] = viol[:3]
        return STATUS_PYOJT, info
    return st, info


def run_identity_errors(run, lang, seed):
    rec = run["record"]
    errs = []
    if rec.get("producer") != PRODUCER:
        errs.append(f"producer が {PRODUCER} でない: {rec.get('producer')!r}")
    if rec.get("lang") != lang:
        errs.append(f"lang が置き場（{lang}）と違う: {rec.get('lang')!r}")
    if rec.get("seed") != seed:
        errs.append(f"seed が置き場（seed{seed}）と違う: {rec.get('seed')}")
    if rec.get("run_dir") != run["rel"]:
        errs.append(f"run_dir が置き場と違う: {rec.get('run_dir')!r} != {run['rel']!r}")
    bad = sorted(set(run["todo"]) - set(TARGETS[lang]))
    if bad and rec.get("mode") == MODE_GPU:
        errs.append(f"todo に作り直しの対象でない対がある: {bad}")
    return errs


def run_provenance_errors(run, lang, seed, fake_ok=False):
    """取り込み・次の seed の根拠に使う実行が、本物の GPU の実行で決まりどおりか（英語の確かめ + vendor + pyopenjtalk）."""
    rec = run["record"]
    errs = run_identity_errors(run, lang, seed) + list(run["errors"])
    env = rec.get("env") or {}
    if not fake_ok:
        if rec.get("mode") != MODE_GPU:
            errs.append(f"mode が {MODE_GPU} でない: {rec.get('mode')!r}")
        if env.get("project_root") != CONTAINER_ROOT:
            errs.append(f"コンテナ（/mnt）の中で作られていない: {env.get('project_root')!r}")
        if not env.get("cuda_device"):
            errs.append("GPU の記録が無い（cuda_device）")
        if env.get("HF_HUB_OFFLINE") != "1" or env.get("HF_HOME") != C.CONTAINER_HF_HOME:
            errs.append(f"HF の環境が決まりと違う: HF_HOME={env.get('HF_HOME')!r} HF_HUB_OFFLINE={env.get('HF_HUB_OFFLINE')!r}")
        if env.get("offline_preflight_ok") is not True:
            errs.append("offline_preflight が通っていない")
        if not any(s.get("repo_id") == C.TOKENIZER_ID and s.get("snapshot") for s in env.get("offline_tokenizer_shim") or []):
            errs.append(f"{C.TOKENIZER_ID} を手元の snapshot から読んだ記録が無い")
        want_argv = prod_args(lang) + ["--output_dir", f"{CONTAINER_ROOT}/{run['rel']}"]
        if rec.get("main_argv") != want_argv:
            errs.append(f"main_argv が本番の引数と違う: {rec.get('main_argv')}")
        vm = env.get("vendor_md5") or {}
        for name, (path, want) in VENDOR_PINNED.items():
            got = (vm.get(name) or {}).get("md5")
            if got != want:
                errs.append(f"vendor {name} の md5 が固定の値と違う（記録 {got} / 固定 {want}）")
        pj = rec.get("pyopenjtalk") or {}
        if pj.get("installed") is not True or pj.get("finalized") is not True:
            errs.append("pyopenjtalk の見張りが入っていない")
        if lang == "ja" and pj.get("vendor_duration_uses_pyopenjtalk") is not True:
            errs.append("JA で vendor の発話長推定が pyopenjtalk を使っていない（5/19-20 と違う推定になる）")
        if pj.get("violations"):
            errs.append(f"pyopenjtalk の決まりの違反が記録されている: {pj.get('violations')[:2]}")
    pt = rec.get("print_taps") or {}
    if pt.get("eval_module_tapped") is not True or pt.get("forced_alignment_utils_tapped") is not True:
        errs.append("本番 main() と forced_alignment_utils の print を記録した証拠が無い")
    if rec.get("span_tap_installed") is not True:
        errs.append("実行時の範囲を記録する差し替えが入っていない")
    sp = rec.get("span_preflight") or {}
    if sorted(int(k) for k in sp) != sorted(run["todo"]) or not all(v.get("fixed_ok") for v in sp.values()):
        errs.append("span_preflight が todo のすべての対で「実体語を覆う」になっていない")
    cm = rec.get("code_md5") or {}
    for rel, want in list(C.PINNED_MD5.items()) + list(DEP_PINNED_MD5.items()):
        if cm.get(rel) != want:
            errs.append(f"固定のファイル {rel} の md5 が固定の値と違う（記録 {cm.get(rel)} / 固定 {want}）")
    end = run["end"]
    if end is not None and end.get("device_offload"):
        errs.append(f"T5Gemma の層が CPU/disk に置かれた: {end.get('hf_device_map')}")
    return errs


def own_code_drift(run, code_root=None):
    cm = run["record"].get("code_md5") or {}
    out = []
    for rel in OWN_CODE:
        cur = md5_file(P(rel, code_root))
        if cm.get(rel) != cur:
            out.append(f"{rel} の md5 が GPU の実行の後に変わった（記録 {cm.get(rel)} / 今 {cur}）")
    return out


def chain(runs_s, idx):
    out = []
    for run in runs_s:
        st, info = pair_status(run, idx)
        if st is not None:
            out.append((run, st, info))
    return out


def next_action(runs, lang, idx, code_root=None, check_provenance=True):
    """対 idx について、今の記録から次にすることを 1 つ返す（英語の spanfix_common.next_action と同じ決まり）.

    action: run_seed{S} / rerun_seed{S} / confirm_stopped_seed{S} / use_seed{S} / stop_author_decision / stop_error
    """
    out = {"pair_idx": idx, "errors": [], "use": None, "chain": {}}
    chains = {s: chain(runs[s], idx) for s in SEEDS}
    for s in SEEDS:
        out["chain"][str(s)] = [{"run": r["rel"], "status": st,
                                 **{k: v for k, v in info.items() if k in ("aligned", "frames", "any_silent",
                                                                           "pair_errors", "whisper_errors",
                                                                           "pyopenjtalk_violations")}}
                                for r, st, info in chains[s]]
    decided_at = SEEDS[-1]
    for s in SEEDS:
        ch = chains[s]
        decided_at = s
        if not ch:
            out["action"] = f"run_seed{s}"
            break
        for k, (run, st, _) in enumerate(ch[:-1]):
            if st not in RETRY_SAME_SEED:
                out["errors"].append(f"seed {s}: {run['rel']} の状態 {st} の後に別の実行がある（結果で選ぶことになるので使わない）")
        run, st, info = ch[-1]
        if st in RETRY_SAME_SEED:
            out["errors"] += run_identity_errors(run, lang, s)
            sts = [x for _, x, _ in ch]
            if len(sts) >= 2 and sts[-1] == STATUS_ALIGN_EXC and sts[-2] == STATUS_ALIGN_EXC:
                out["action"] = "stop_author_decision"
                out["note"] = f"seed {s} で Whisper の例外が 2 回続いた。3 回目は流さず著者の判断"
            elif st == CRASHED_NO_RECORD:
                out["action"] = f"confirm_stopped_seed{s}"
                out["note"] = (f"{run['rel']} は最後まで終わっていない（run_end.json が無い）。squeue でジョブが動いて"
                               f"いないことを確かめてから、--confirm_stopped {run['rel']} を付けて seed {s} をもう一度流す")
                out["unfinished_run"] = run["rel"]
            else:
                out["action"] = f"rerun_seed{s}"
            break
        if st in STOP_STATUSES:
            out["action"] = "stop_author_decision"
            out["note"] = f"seed {s}: {st}（{run['rel']}）"
            break
        if check_provenance:
            out["errors"] += run_provenance_errors(run, lang, s)
            drift = own_code_drift(run, code_root)
            if drift:
                out["code_drift"] = drift
        if st == STATUS_VALID:
            out["action"] = f"use_seed{s}"
            out["use"] = {"seed": s, "run": run["rel"]}
            break
        assert st in PAIR_INVALID, st
        if s == SEEDS[-1]:
            out["action"] = "stop_author_decision"
            out["note"] = "seed 44 でも対として使えない。45 以上は決まっていないので著者の判断"
            break
    later = [s for s in SEEDS if s > decided_at and chains[s]]
    if later:
        out["errors"].append(f"seed {later} の実行があるが、seed {decided_at} で対として使えないと決まっていない")
    if out["errors"]:
        out["action"] = "stop_error"
    elif out.get("code_drift"):
        out["action"] = "stop_author_decision"
        out["note"] = "このラッパー・共通部品・借りた英語のファイルの md5 が GPU の実行の後に変わった（著者の判断）"
    return out


def allowed_todo(runs, lang, seed, candidates, confirm_stopped=(), code_root=None, check_provenance=True):
    ok, why = [], {}
    for idx in candidates:
        na = next_action(runs, lang, idx, code_root, check_provenance)
        a = na["action"]
        if a in (f"run_seed{seed}", f"rerun_seed{seed}"):
            ok.append(idx)
        elif a == f"confirm_stopped_seed{seed}" and na.get("unfinished_run") in confirm_stopped:
            ok.append(idx)
        else:
            why[idx] = {"action": a, "errors": na["errors"][:3], "note": na.get("note")}
    return ok, why


# ---------------------------------------------------------------- pyopenjtalk の見張り
class PyopenjtalkGate:
    """pyopenjtalk を、vendor の duration_estimator.py の中でだけ使わせる（GPU の実行の中で使う）.

    1. install(): sys.meta_path の先頭に入る。pyopenjtalk の最初の import は、import した側が vendor の
       duration_estimator.py のときだけ通す。ほかからの import は ImportError にして記録する（違反）。
    2. finalize(DE): vendor が import した pyopenjtalk の公開の関数を、呼んだ側を確かめる関数に置き換える。
       vendor の duration_estimator.py（と pyopenjtalk 自身の中）からの呼び出しだけ通し、数える。
       ほかからの呼び出しは RuntimeError にして記録する（違反）。ほかのモジュールが pyopenjtalk かその関数を
       持っていれば記録する（違反）。pyopenjtalk は sys.modules に残す（中の遅い import を壊さない）。
    違反は run_end.json の pyopenjtalk_audit.violations に残り、その実行の対は止まって著者の判断になる。
    """

    def __init__(self, allowed_file):
        self.allowed = os.path.realpath(allowed_file)
        self.importlib_dir = os.path.realpath(os.path.dirname(importlib.__file__))
        self.self_file = os.path.realpath(os.path.abspath(__file__))
        self.import_attempts = []
        self.violations = []
        self.calls_from_vendor = {}
        self.calls_internal = 0
        self.patched = []
        self.unpatched_public = []
        self.holders = []
        self.installed = False
        self.finalized = False
        self.vendor_uses = None
        self.real_file = None
        self.pkg_dir = None
        self.version = None

    def _caller_file(self, f):
        while f is not None:
            fn = f.f_code.co_filename
            if fn.startswith("<frozen importlib") or fn.startswith("<frozen zipimport"):
                f = f.f_back
                continue
            if fn.startswith("<"):
                return fn
            rp = os.path.realpath(fn)
            if rp == self.self_file or os.path.dirname(rp) == self.importlib_dir:
                f = f.f_back
                continue
            return rp
        return None

    def _is_real_import(self, f):
        """import（finder と呼んだ側の間に _find_and_load がある）か、importlib.util.find_spec の問い合わせだけか.

        呼んだ側より外の _find_and_load（呼んだ側のモジュール自身が import されている途中など）は見ない。
        """
        while f is not None:
            fn = f.f_code.co_filename
            internal = fn.startswith("<frozen importlib") or fn.startswith("<frozen zipimport") or (
                not fn.startswith("<") and (os.path.realpath(fn) == self.self_file
                                            or os.path.dirname(os.path.realpath(fn)) == self.importlib_dir))
            if not internal:
                return False
            if f.f_code.co_name == "_find_and_load" and fn.startswith("<frozen importlib._bootstrap"):
                return True
            f = f.f_back
        return False

    def find_spec(self, fullname, path=None, target=None):  # sys.meta_path の finder
        if fullname != "pyopenjtalk":
            return None
        who = self._caller_file(sys._getframe(1))
        ok = who == self.allowed
        if not self._is_real_import(sys._getframe(1)):
            # 在るかどうかの問い合わせだけ（モジュールは実行されない）。記録して通常の finder に任せる。
            # その後に本当に import すれば、その import はここで止まる
            self.import_attempts.append({"importer": who, "allowed": ok, "probe_only": True, "t": now()})
            return None
        self.import_attempts.append({"importer": who, "allowed": ok, "t": now()})
        if not ok:
            self.violations.append({"kind": "import_outside_vendor", "importer": who, "t": now()})
            raise ImportError(f"[appB_jazh] pyopenjtalk は {self.allowed} からだけ import できる（import した側: {who}）")
        return None  # 通常の finder に任せる

    def install(self):
        if "pyopenjtalk" in sys.modules:
            raise SystemExit("[appB_jazh] pyopenjtalk が見張りの前に読まれている（決まりの外）")
        sys.meta_path.insert(0, self)
        self.installed = True

    def _guard(self, name, fn):
        gate = self

        def guarded(*a, **k):
            who = gate._caller_file(sys._getframe(1))
            if who == gate.allowed:
                gate.calls_from_vendor[name] = gate.calls_from_vendor.get(name, 0) + 1
            elif gate.pkg_dir and isinstance(who, str) and (who == gate.pkg_dir or who.startswith(gate.pkg_dir + os.sep)):
                gate.calls_internal += 1
            else:
                gate.violations.append({"kind": "call_outside_vendor", "function": name, "caller": who, "t": now()})
                raise RuntimeError(f"[appB_jazh] pyopenjtalk.{name} は vendor の発話長推定の中からだけ呼べる（呼んだ側: {who}）")
            return fn(*a, **k)
        guarded.__wrapped__ = fn
        guarded.__name__ = getattr(fn, "__name__", name)
        guarded.__doc__ = getattr(fn, "__doc__", None)
        return guarded

    def finalize(self, de_module):
        de_file = os.path.realpath(getattr(de_module, "__file__", "") or "")
        if de_file != self.allowed:
            raise SystemExit(f"[appB_jazh] duration_estimator が vendor のファイルでない: {de_file}")
        real = getattr(de_module, "pyopenjtalk", None)
        self.vendor_uses = real is not None
        if real is not None:
            self.real_file = getattr(real, "__file__", None)
            self.version = getattr(real, "__version__", None)
            if self.real_file:
                d = os.path.realpath(os.path.dirname(self.real_file))
                self.pkg_dir = d if os.path.basename(self.real_file).startswith("__init__") else None
            originals = {}
            for name in sorted(dir(real)):
                if name.startswith("_"):
                    continue
                obj = getattr(real, name)
                own = (getattr(obj, "__module__", None) or "").split(".")[0] == "pyopenjtalk"
                if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)) and own:
                    originals[name] = obj
                    setattr(real, name, self._guard(name, obj))
                    self.patched.append(name)
                elif callable(obj) and not isinstance(obj, types.ModuleType) and own:
                    self.unpatched_public.append(name)  # クラスなど（置き換えると isinstance が壊れる）
            watch = {id(real): ("module", real)}
            watch.update({id(f): ("function:" + n, f) for n, f in originals.items()})
            for mname, mod in list(sys.modules.items()):
                if mod is None or mod is de_module or mname == "pyopenjtalk" or mname.startswith("pyopenjtalk."):
                    continue
                try:
                    items = list(vars(mod).items())
                except TypeError:
                    continue
                for k, v in items:
                    hit = watch.get(id(v))
                    if hit is not None and hit[1] is v:
                        self.holders.append({"module": mname, "attr": k, "what": hit[0]})
            if self.holders:
                self.violations.append({"kind": "other_module_holds_pyopenjtalk", "holders": self.holders[:10],
                                        "t": now()})
        self.finalized = True  # finder は残す（vendor が None のとき、あとから誰も import できないように）
        return self.report()

    def report(self):
        return {"allowed_importer": self.allowed, "installed": self.installed, "finalized": self.finalized,
                "vendor_duration_uses_pyopenjtalk": self.vendor_uses, "pyopenjtalk_file": self.real_file,
                "pyopenjtalk_version": self.version, "import_attempts": list(self.import_attempts),
                "patched_functions": list(self.patched), "unpatched_public_callables": list(self.unpatched_public),
                "holders": list(self.holders), "calls_from_vendor": dict(self.calls_from_vendor),
                "calls_internal": self.calls_internal, "violations": list(self.violations)}


def duration_preflight(DE, lang, sample_text):
    """vendor の発話長推定が 5/19-20 と同じ経路か（JA: pyopenjtalk、ZH: pypinyin）を、モデルを読む前に確かめる."""
    info = {"module_file": getattr(DE, "__file__", None),
            "pyopenjtalk_present": getattr(DE, "pyopenjtalk", None) is not None,
            "lazy_pinyin_present": getattr(DE, "lazy_pinyin", None) is not None,
            "langdetect_present": getattr(DE, "detect", None) is not None,
            "g2p_en_present": getattr(DE, "G2p", None) is not None}
    if lang == "ja" and not info["pyopenjtalk_present"]:
        raise SystemExit(f"[appB_jazh] vendor の発話長推定に pyopenjtalk が無い（文字数で推定され 5/19-20 と変わる）: {info}")
    if lang == "zh" and not info["lazy_pinyin_present"]:
        raise SystemExit(f"[appB_jazh] vendor の発話長推定に pypinyin が無い（文字数で推定され 5/19-20 と変わる）: {info}")
    try:
        info["sample"] = {"text": sample_text, "lang": lang,
                          "estimate_sec": DE.estimate_duration(target_text=sample_text, target_lang=lang),
                          "detected_lang": DE.detect_language(sample_text)}
    except Exception as ex:  # noqa: BLE001
        raise SystemExit(f"[appB_jazh] vendor の発話長推定が動かない: {type(ex).__name__}: {ex}")
    return info


# ---------------------------------------------------------------- 見張りの試運転（偽の pyopenjtalk、別のプロセス）
_FAKE_PYOJT = '''__version__ = "fake-selftest"
def run_frontend(text):
    return [text]
def g2p(text, kana=False, join=True):
    run_frontend(text)
    return "a i u"
class OpenJTalk:
    pass
'''
_FAKE_PYOJT_ABSENT = 'raise ImportError("pyopenjtalk is not installed (selftest)")\n'
_FAKE_VENDOR_DE = '''try:
    import pyopenjtalk
except ImportError:
    pyopenjtalk = None
def count(text):
    if pyopenjtalk is None:
        return len(text)
    return len(pyopenjtalk.g2p(text).split(" "))
'''
_FAKE_OTHERS = {
    "early_user.py": "import pyopenjtalk\n",
    "late_caller.py": "def use():\n    import pyopenjtalk\n    return pyopenjtalk.g2p('x')\n",
    "late_holder.py": "import pyopenjtalk as P\n",
    "late_fn_holder.py": "from pyopenjtalk import g2p as G2P\n",
    "prober.py": "import importlib.util\nFOUND = importlib.util.find_spec('pyopenjtalk') is not None\n",
}
GATE_SCENARIOS = ("clean", "early_import", "late_call", "module_holder", "function_holder", "absent", "preimported",
                  "probe")


def _gate_scenario(name, base):
    site = os.path.join(base, "site_absent" if name == "absent" else "site")
    for d in (os.path.join(base, "other"), os.path.join(base, "vendor"), site):
        sys.path.insert(0, d)
    out = {"scenario": name}
    gate = PyopenjtalkGate(os.path.join(base, "vendor", "duration_estimator.py"))
    if name == "preimported":
        import pyopenjtalk  # noqa: F401  偽物（試運転の置き場の site/）
        try:
            gate.install()
            out["refused"] = False
        except SystemExit as e:
            out["refused"] = True
            out["why"] = str(e)
        print(json.dumps(out, ensure_ascii=False))
        return
    gate.install()
    if name == "probe":
        import prober
        out["probe_found"] = prober.FOUND
        out["cached_after_probe"] = "pyopenjtalk" in sys.modules
    if name == "early_import":
        try:
            import early_user  # noqa: F401
            out["early_import_raised"] = False
        except ImportError:
            out["early_import_raised"] = True
    import duration_estimator as DE
    if name == "module_holder":
        import late_holder  # noqa: F401
    if name == "function_holder":
        import late_fn_holder  # noqa: F401
    gate.finalize(DE)
    out["count"] = [DE.count("xyz"), DE.count("xy")]
    if name == "late_call":
        import late_caller
        try:
            late_caller.use()
            out["late_call_raised"] = False
        except RuntimeError:
            out["late_call_raised"] = True
    out["report"] = gate.report()
    print(json.dumps(out, ensure_ascii=False))


def gate_selftest(base):
    """偽の vendor・偽の pyopenjtalk で、見張りが通す・止めるを場面ごとに新しいプロセスで確かめる.

    子のプロセスは python3 -S -B（site-packages を読まない。本物の pyopenjtalk には触れない）。
    """
    os.makedirs(base, exist_ok=False)
    for d, files in (("site/pyopenjtalk", {"__init__.py": _FAKE_PYOJT}),
                     ("site_absent/pyopenjtalk", {"__init__.py": _FAKE_PYOJT_ABSENT}),
                     ("vendor", {"duration_estimator.py": _FAKE_VENDOR_DE}),
                     ("other", _FAKE_OTHERS)):
        os.makedirs(os.path.join(base, d), exist_ok=False)
        for fn, src in files.items():
            with open(os.path.join(base, d, fn), "x", encoding="utf-8") as f:
                f.write(src)
    res = {}
    for name in GATE_SCENARIOS:
        p = subprocess.run([sys.executable, "-S", "-B", os.path.abspath(__file__), "--gate_scenario", name, base],
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise AssertionError(f"gate scenario {name} failed rc={p.returncode}: {p.stderr[-800:]}")
        res[name] = json.loads(p.stdout.strip().splitlines()[-1])
    vend = os.path.realpath(os.path.join(base, "vendor", "duration_estimator.py"))
    r = res["clean"]["report"]
    assert res["clean"]["count"] == [3, 3] and r["violations"] == [] and r["calls_from_vendor"] == {"g2p": 2} \
        and r["calls_internal"] == 2 and r["vendor_duration_uses_pyopenjtalk"] is True \
        and set(r["patched_functions"]) == {"g2p", "run_frontend"} and r["unpatched_public_callables"] == ["OpenJTalk"] \
        and r["import_attempts"] == [dict(r["import_attempts"][0], importer=vend, allowed=True)], res["clean"]
    e = res["early_import"]
    assert e["early_import_raised"] is True and e["report"]["violations"][0]["kind"] == "import_outside_vendor" \
        and e["report"]["violations"][0]["importer"].endswith("early_user.py") and e["count"] == [3, 3], e
    lc = res["late_call"]
    assert lc["late_call_raised"] is True and lc["report"]["violations"][-1]["kind"] == "call_outside_vendor" \
        and lc["report"]["violations"][-1]["caller"].endswith("late_caller.py"), lc
    mh = res["module_holder"]["report"]
    assert mh["holders"] == [{"module": "late_holder", "attr": "P", "what": "module"}] \
        and mh["violations"][0]["kind"] == "other_module_holds_pyopenjtalk", mh
    fh = res["function_holder"]["report"]
    assert fh["holders"] == [{"module": "late_fn_holder", "attr": "G2P", "what": "function:g2p"}], fh
    ab = res["absent"]
    assert ab["report"]["vendor_duration_uses_pyopenjtalk"] is False and ab["count"] == [3, 2] \
        and ab["report"]["violations"] == [], ab
    assert res["preimported"]["refused"] is True, res["preimported"]
    pr = res["probe"]
    att = pr["report"]["import_attempts"]
    assert pr["probe_found"] is True and pr["cached_after_probe"] is False and pr["report"]["violations"] == [] \
        and att[0].get("probe_only") is True and att[0]["importer"].endswith("prober.py") \
        and att[1]["importer"] == vend and att[1]["allowed"] is True and "probe_only" not in att[1] \
        and pr["count"] == [3, 3], pr
    return {"scenarios": {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in res.items()},
            "violations_by_scenario": {k: [x["kind"] for x in (v.get("report") or {}).get("violations", [])]
                                       for k, v in res.items()},
            "ok": True}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--gate_scenario":
        _gate_scenario(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: appB_jazh_common.py --gate_scenario <name> <dir>（gate_selftest が使う）")
