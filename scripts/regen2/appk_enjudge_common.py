#!/usr/bin/env python3
"""付録 K 英語の注入成功率: 中国語の行と同じ判定（CTC 対数尤度の BL からの移り方）を英語で行うための共通定義（2026-09-29）.

著者の決定（2026-09-29、note/2026-09-29_zhjudge_plan.md 11 章の候補 C）:
  * 中国語の行（93.67 → 判定し直して 94.00）と同じ判定を英語に使う。
      d = (ℓ^INJ_tgt − ℓ^INJ_src) − (ℓ^BL_tgt − ℓ^BL_src)、成功 ⇔ d > 0、k = #{d > 0}、値 = k/300
    ℓ は文全体の期待音素列（対象の語だけ src の読み / tgt の読みにしたもの）の CTC 対数尤度。
    CTC は connectionist temporal classification。
  * 認識器は facebook/wav2vec2-xlsr-53-espeak-cv-ft（付録 K の IPA 変化の列と同じもの）。
  * 音声は eval_results/utmos_speaker_fixed_en_n300_splice の BL（s####_bl.wav）と注入（s####_inj.wav）の 300 対
    （対 23 は seed 44 に差し替え済み。反映の記録 regen2_appk_pairs.json を照合する）。
  * 2 つの読みの音素列が同じになる対は失敗に数える（N=300 のまま。d = 0 なので d > 0 にならない）。一覧を出す。
  * 主に出す値は BL からの移り方（中国語の shift_tone_pct と同じ）。INJ 単体で tgt を選んだ率も副として出す。
  * 「BL が src を選ぶ率」は前もって決めた確かめとして報告するだけ。対は選り分けない。

読みの表は 2 版（どちらにするかは著者が決める）: v1 = 最初の置き換え、v2 = 認識器が同じ 600 本に実際に出す記号に合わせた
置き換え（ER → ɚ、AO+R → oː、AA+R → ɑː、ほかの母音+R → 母音 + ɹ。22 時に結果を見る前に固定）。文の中のほかの語も表と同じ版。
表と仮説の記号は、認識器の書き起こし（_appk_redo/ipa_en_n300.json）での記号ごとの回数と照らす（symbol_emission_check）。

CTC の前向き計算は、中国語の判定器 scripts/analysis/zh_ctc_lr_judge.py（md5 7e823d42…、変えない）の
関数 ctc_logprob_batch をそのまま（ast で取り出して）使う。torch の ctc_loss でも同じ値になることを毎回確かめる。

このモジュールは torch も numpy も import しない（CPU 側の段取り・集計と、GPU 側の両方から読む）。
書いてよい場所は eval_results/_appk_enjudge/ と eval_results/_regen2_dryrun/appk_enjudge_selftest/ だけ。
本番の utmos_speaker_fixed_en_n300_splice/ と原稿には書かない。
"""
import ast
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_common as K  # noqa: E402  （torch / numpy を import しない）

if K.TEST_ROOT:
    raise SystemExit("APPK_TEST_ROOT が設定されている。appk_enjudge_* は本番の置き場だけを読む（検査は appk_enjudge_selftest.py）。")

PROJ = K.PROJ
HOST_PROJ = os.environ.get("REPO_ROOT", "/mnt")
ER = K.REAL_ER
N = 300
THREADS = 13
N_WORDS = 30
PAIRS_PER_WORD = 10

# ---------------------------------------------------------------------------
# 入力（読むだけ。md5 は 2026-09-29 にホストで計算）
# ---------------------------------------------------------------------------
PROD_DIR = os.path.join(ER, "utmos_speaker_fixed_en_n300_splice")
PROD_CKPT = os.path.join(PROD_DIR, "checkpoint.json")
APPLY_RECORD = os.path.join(PROD_DIR, "regen2_appk_pairs.json")
POST_APPLY_SCAN = os.path.join(ER, "_appk_redo", "silent_scan_n300.json")
POST_APPLY_CHECK = os.path.join(ER, "_appk_redo", "silent_scan_n300.json.post_apply_check.json")
DATA_FILE = os.path.join(PROJ, "data", "en_heteronyms_500.py")
ZH_JUDGE = os.path.join(PROJ, "scripts", "analysis", "zh_ctc_lr_judge.py")
ZH_STORED_JUDGE = os.path.join(ER, "cjk_n300_zh_v2", "ctc_lr_judge.json")
# 同じ認識器が同じ 600 本を貪欲に書き起こした結果（付録 K の IPA 変化の列、19:04 にコンテナで作成）。
# 記号ごとの出た回数（ファイル全体の合計）だけを使う。どの対で何が出たかは使わない。
IPA_EN_TRANSCRIPTS = os.path.join(ER, "_appk_redo", "ipa_en_n300.json")
IPA_RUN_META = os.path.join(ER, "_appk_redo", "ipa_run_meta_20260929_100358.json")

EXPECTED_MD5 = {
    "data_file": "5b12509ac76de20d044a590f1248f4a0",
    "checkpoint": "6c0db277e19ea3ee12ac444e82720b4c",
    "apply_record": "5c8e5da13898c646e41662d79865ed58",
    "post_apply_scan": "0d386e82fece56155a19ebde723158ce",
    "zh_judge": "7e823d4216967042148cd26690e50738",
    "zh_stored_judge": "cb2dc8beeb9499bd9fc5d12e24a5c8a7",
    "ipa_en_transcripts": "90e07144d2af1a704a954ecb5251a34b",
    "ipa_run_meta": "65f19bf5de121449a6fce75dd7e77bd9",
}
# 反映の記録（regen2_appk_pairs.json）に書かれているはずのこと
APPLY_EXPECT = {"pairs_replaced": [23], "seeds": {"23": 44}, "unresolved_pairs_not_replaced": [],
                "new_md5": {"s0023_bl.wav": "3983d5981994da092036fefa93950cc2",
                            "s0023_inj.wav": "9d9a748f04eca96ecf19b2eca6fa7cd7"}}
REPLACED_PAIRS = [23]

# 認識器（付録 K の IPA 変化の列と同じ。保存済みの snapshot だけを読む）
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_ROOT = os.path.join(PROJ, ".cache", "huggingface", "hub", "models--facebook--wav2vec2-xlsr-53-espeak-cv-ft")
SNAP_MAIN = "2c733782da5604684829819a5eb744c193fe9398"   # refs/main（config・vocab・pytorch_model.bin）
SNAP_PR7 = "3e836924cfd3b858a0bbdbc1f7ef412105d00446"    # refs/pr/7（model.safetensors。transformers が自動で使う）
SNAP_MD5 = {
    SNAP_MAIN: {".gitattributes": "74de041f502d2a429fa15aec78aa4e73", "README.md": "e03b8996dddd925ba452979e07ca7e1d",
                "config.json": "edf65f1ea040a970e9e16dd92ffb8ed7",
                "preprocessor_config.json": "792ac3fb96bad242d4ebdfd08edc54c7",
                "pytorch_model.bin": "ef634f563ca29089eacfd36699ef0598",
                "special_tokens_map.json": "1ac61986e1bc63da965f06d7f0523760",
                "tokenizer_config.json": "127e3d4776f2a9c69fcf68c733c9bfc1",
                "vocab.json": "dd4b406c4c6166670db5aedd8460db1e"},
    SNAP_PR7: {"model.safetensors": "d916cdf8bf43d8263b9ad6fff161e978"},
}
VOCAB = os.path.join(MODEL_ROOT, "snapshots", SNAP_MAIN, "vocab.json")
BLANK_ID = 0          # config.json の pad_token_id（CTC の空白）
VOCAB_SIZE = 392

# CMU 発音辞書（ホストの cmudict パッケージ 1.1.3。表と仮説を作るのはホストの CPU だけ）
CMUDICT_MD5 = "5837aa6e49fd070d482b8ca0525f28ef"

# ---------------------------------------------------------------------------
# 書き込み先
# ---------------------------------------------------------------------------
WORK_ROOT = os.path.join(ER, "_appk_enjudge")
TABLE_PATH = os.path.join(WORK_ROOT, "en_reading_table_v1.json")
TABLE_PATH_V2 = os.path.join(WORK_ROOT, "en_reading_table_v2.json")
TABLE_PATHS = {"v1": TABLE_PATH, "v2": TABLE_PATH_V2}
EMISSION_CHECK_NAME = "symbol_emission_check_{version}.json"
DEFAULT_STAGE = os.path.join(WORK_ROOT, "judge_ctc")
SELFTEST_ROOT = os.path.join(ER, "_regen2_dryrun", "appk_enjudge_selftest")
MANIFEST = "stage_manifest.json"
JUDGE_OUT = "en_ctc_lr_judge.json"
RUN_META_PREFIX = "enjudge_run_meta_"
ESPEAK_OUT = "espeak_check.json"
REPORT = "en_injection_success.json"

WAV_SIDES = ("bl", "inj")
LP_FIELDS = ("bl_lp_src", "bl_lp_tgt", "inj_lp_src", "inj_lp_tgt")
INFEASIBLE_LP = -1e29      # 判定器の NEG = -1e30。これより小さい値は「音素列が音声のフレームに収まらない」
CTC_XCHECK_TOL = 0.01      # 判定器の関数と torch の ctc_loss の差の許容（中国語の値の許容 0.01 と同じ）
ZH_REF_BL_SRC_PCT = 95.0   # 中国語の判定器が書いている「BL が src を選ぶ率」の合格水準（英語では報告だけ）


def wav_name(i, side):
    return f"s{i:04d}_{side}.wav"


def all_wav_names():
    return [wav_name(i, s) for i in range(N) for s in WAV_SIDES]


def md5_file(p):
    return K.md5_file(p)


def md5_json(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def proj_rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, PROJ) if (p + "/").startswith(PROJ + "/") else p


def to_here(p):
    """記録のパス（コンテナの /mnt/... かホストの /data/...）を、今動いている側のパスに直す."""
    if p is None:
        return None
    if PROJ != "/mnt" and (p == "/mnt" or p.startswith("/mnt/")):
        return HOST_PROJ + p[len("/mnt"):]
    if PROJ == "/mnt" and (p == HOST_PROJ or p.startswith(HOST_PROJ + "/")):
        return "/mnt" + p[len(HOST_PROJ):]
    return p


def resolve(p):
    p = to_here(p)
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(PROJ, p))


def assert_write(p):
    """書き込み先の確認: _appk_enjudge/ か自己検査の置き場の下だけ。本番の音声の置き場・原稿・中国語の置き場には書かない."""
    p = os.path.abspath(p)
    for bad in ("/naacl2027/", "/paper_draft/", "/arr_resubmission/", "/_appk_zhjudge/", "/cjk_n300_zh_v2/"):
        if bad in p + "/":
            raise SystemExit(f"書き込み禁止の場所: {p}")
    if (p + "/").startswith(PROD_DIR + "/"):
        raise SystemExit(f"本番の EN の置き場には書かない: {p}")
    if not any((p + "/").startswith(r + "/") and p != r for r in (WORK_ROOT, SELFTEST_ROOT)):
        raise SystemExit(f"書き込みは {WORK_ROOT} か {SELFTEST_ROOT} の下だけ: {p}")
    if os.path.islink(p):
        raise SystemExit(f"symlink には書かない: {p}")
    return p


def write_json(obj, p, indent=1, backup=True):
    """書く前に既存のファイルを .bak_<時刻> に写す（消さない）."""
    assert_write(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if backup and os.path.exists(p):
        shutil.copy2(p, f"{p}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, p)
    return p


def check_pinned(tag, path):
    if not os.path.isfile(path):
        return False, None
    m = md5_file(path)
    return m == EXPECTED_MD5[tag], m


_MD5_CACHE = {}


def _md5_cached(p):
    st = os.stat(p)
    key = (os.path.realpath(p), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = md5_file(p)
    return _MD5_CACHE[key]


def check_model_files(full_weights_md5=True):
    """認識器の snapshot（refs と各ファイルの md5）が 2026-09-29 に確かめたものと同じか。戻り値: problems."""
    problems = []
    refs = {"main": SNAP_MAIN, os.path.join("refs", "pr", "7"): SNAP_PR7}
    for ref, want in refs.items():
        p = os.path.join(MODEL_ROOT, "refs", ref)
        got = open(p).read().strip() if os.path.isfile(p) else None
        if got != want:
            problems.append(f"認識器の refs/{ref} が {got}（{want} でない）")
    for snap, files in SNAP_MD5.items():
        for n, want in files.items():
            p = os.path.join(MODEL_ROOT, "snapshots", snap, n)
            if not os.path.isfile(p):
                problems.append(f"認識器のファイルがない: {snap[:8]}/{n}")
                continue
            if (n.endswith(".bin") or n.endswith(".safetensors")) and not full_weights_md5:
                continue
            got = _md5_cached(p)
            if got != want:
                problems.append(f"認識器の {snap[:8]}/{n} の md5 が違う: {got} != {want}")
    return problems


def load_vocab():
    v = read_json(VOCAB)
    if len(v) != VOCAB_SIZE or sorted(v.values()) != list(range(VOCAB_SIZE)) or v.get("<pad>") != BLANK_ID:
        raise SystemExit("認識器の vocab.json が想定（392 個、<pad>=0）と違う")
    return v


# ---------------------------------------------------------------------------
# 対の組（data/en_heteronyms_500.py の build_500_pairs() の先頭 300 = 30 語 × 10 対）
# ---------------------------------------------------------------------------
def load_data_pairs():
    """data ファイルを実行して、300 対の (語, 向き, src のラベル, tgt のラベル) と 30 語の読みの一覧を返す."""
    import runpy
    ns = runpy.run_path(DATA_FILE, run_name="appk_enjudge_data")
    pairs = ns["build_500_pairs"]()[:N]
    het = {e["word"]: e for e in ns["EN_HETERONYMS_50"]}
    out = []
    for i, p in enumerate(pairs):
        a, b = p["reading_a_label"], p["reading_b_label"]
        src, tgt = (a, b) if p["direction"] == "a_to_b" else (b, a)
        out.append({"pair_idx": i, "word": p["word_a"], "text_a": p["text_a"], "text_b": p["text_b"],
                    "direction": p["direction"], "src_label": src, "tgt_label": tgt})
    words = []
    for r in out:
        if r["word"] not in words:
            words.append(r["word"])
    readings = {w: {"a": het[w]["reading_a"]["label"], "b": het[w]["reading_b"]["label"],
                    "contexts_a": list(het[w]["reading_a"]["contexts"]),
                    "contexts_b": list(het[w]["reading_b"]["contexts"])} for w in words}
    return out, words, readings


def check_rows_vs_data(ckpt_rows, data_pairs):
    """checkpoint の 300 行が data ファイルの先頭 300 対と同じ並びか。戻り値: problems."""
    problems = []
    if len(ckpt_rows) != N:
        problems.append(f"checkpoint の行が {len(ckpt_rows)}（300 でない）")
    idx = [r.get("pair_idx") for r in ckpt_rows]
    if idx != list(range(N)):
        problems.append("checkpoint の pair_idx が 0..299 の順になっていない")
    bad = [r.get("pair_idx") for r, d in zip(ckpt_rows, data_pairs)
           if (r.get("word_a"), r.get("word_b"), r.get("text_a"), r.get("text_b"))
           != (d["word"], d["word"], d["text_a"], d["text_b"])]
    if bad:
        problems.append(f"checkpoint の語・文が data ファイルの対と違う: {bad[:10]}")
    return problems


# ---------------------------------------------------------------------------
# 読みの表（ラベルの強勢 → CMUdict の発音 → 認識器の音素単位）
# ---------------------------------------------------------------------------
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
VOWEL_UNITS = {"AA": "ɑː", "AE": "æ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "EY": "eɪ",
               "IH": "ɪ", "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː"}
STRESS_UNITS = {"AH": {"1": "ʌ", "2": "ʌ", "0": "ə"}, "ER": {"1": "ɜː", "2": "ɜː", "0": "ɚ"}}
CONS_UNITS = {"B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k",
              "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
              "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
R_MERGE = {"AA": "ɑːɹ", "AO": "ɔːɹ", "EH": "ɛɹ", "IH": "ɪɹ", "IY": "ɪɹ", "UH": "ʊɹ"}
REDUCED_FULL = {frozenset(("ə", "ʌ")), frozenset(("ɚ", "ɜː"))}   # 強勢だけで決まる弱い母音と強い母音の組
# 注意（22 時）: 下の v1 の "why" の文は、en_reading_table_v1.json（md5 c1d4224f）に書かれたまま残すために変えない。
# ただし内容は認識器の実際の出力と合わない（ɜː・ɔːɹ・ɑːɹ・ɛɹ・ɪɹ・ʊɹ・aɪɚ・aɪə は同じ 600 本で 0 回）。
# 照合は symbol_emission_check、認識器に合わせた版は MAPPING_RULES_V2。
MAPPING_RULES = {
    "consonants": CONS_UNITS,
    "vowels_any_stress": VOWEL_UNITS,
    "vowels_by_stress": STRESS_UNITS,
    "merges_inside_a_word": {"AA R": "ɑːɹ", "AO R": "ɔːɹ", "EH R": "ɛɹ", "IH R": "ɪɹ", "IY R": "ɪɹ", "UH R": "ʊɹ",
                             "AY ER0": "aɪɚ", "AY AH0": "aɪə",
                             "AH0 L (L at the word end or before a consonant)": "əl"},
    "why": ("the recognizer was trained on espeak-ng phonemes without stress marks; espeak-ng writes the r-coloured "
            "vowels, aɪɚ, aɪə and syllabic əl as single symbols, and marks stress on AH (ʌ/ə) and ER (ɜː/ɚ) only by "
            "vowel quality. All other vowels keep their CMUdict quality whatever the stress digit."),
}
POSSESSIVE_RULE = "word + 's: add s after p t k f θ, ɪ z after s z ʃ ʒ tʃ dʒ, z otherwise"

# v2 の置き換え（2026-09-29 22 時、どの結果も見る前に固定）。v1 との違いは次の 4 つだけ:
#   ER は強勢によらず ɚ / AO+R → oː / AA+R → ɑː / ほかの母音+R → その母音 + 別の ɹ。
# 理由は認識器が同じ 600 本に出した記号の数（IPA 変化の列の書き起こし 16,006 記号）: ɜː・ɔːɹ・ɑːɹ・ɛɹ・ɪɹ・ʊɹ は 0 回、
# ɚ 518 回・oː 182 回・ɑː 295 回・ɹ 821 回。ほかの規則（aɪɚ・aɪə・əl の結び、AO → ɔː、OY → ɔɪ など）は v1 のまま。
MAPPINGS = ("v1", "v2")
STRESS_UNITS_V2 = {"AH": {"1": "ʌ", "2": "ʌ", "0": "ə"}, "ER": {"1": "ɚ", "2": "ɚ", "0": "ɚ"}}
R_MERGE_V2 = {"AA": "ɑː", "AO": "oː"}
MAPPING_RULES_V2 = {
    "consonants": CONS_UNITS,
    "vowels_any_stress": VOWEL_UNITS,
    "vowels_by_stress": STRESS_UNITS_V2,
    "merges_inside_a_word": {"AA R": "ɑː", "AO R": "oː",
                             "other vowel + R": "the vowel's own unit, then a separate ɹ (EH R -> ɛ ɹ, IY R -> iː ɹ ...)",
                             "AY ER0": "aɪɚ", "AY AH0": "aɪə",
                             "AH0 L (L at the word end or before a consonant)": "əl"},
    "changes_from_v1": ["ER at any stress -> ɚ (v1: ɜː when stressed)", "AO + R -> oː (v1: ɔːɹ)",
                        "AA + R -> ɑː (v1: ɑːɹ)", "other vowel + R -> vowel + separate ɹ (v1: ɛɹ / ɪɹ / ʊɹ)"],
    "why": ("fixed before any result: in the recognizer's own greedy transcripts of the same 600 wavs "
            "(eval_results/_appk_redo/ipa_en_n300.json, 16,006 symbols) ɜː, ɔːɹ, ɑːɹ, ɛɹ, ɪɹ and ʊɹ occur 0 times, "
            "while ɚ occurs 518, oː 182, ɑː 295 and ɹ 821 times. Only these whole-file counts were used, never a "
            "pair's outcome. All other rules are the same as v1."),
}
MAPPING_RULES_BY = {"v1": MAPPING_RULES, "v2": MAPPING_RULES_V2}


def _mapping_parts(mapping):
    if mapping == "v1":
        return STRESS_UNITS, R_MERGE
    if mapping == "v2":
        return STRESS_UNITS_V2, R_MERGE_V2
    raise ValueError(f"置き換えの版が {mapping!r}（v1 か v2 だけ）")


def split_phone(ph):
    m = re.match(r"^([A-Z]+)([012]?)$", ph)
    if not m:
        raise ValueError(f"ARPAbet の記号でない: {ph}")
    return m.group(1), m.group(2)


def arpa_to_units(phones, mapping="v1"):
    """ARPAbet（強勢の数字つき）の 1 語 → 認識器の音素単位の列（上の規則だけ。例外なし）.

    mapping="v1" は最初の表の規則（既定。v1 の表と仮説はこれまでと 1 記号も変わらない）、"v2" は上の 4 つを変えた規則."""
    stress_units, r_merge = _mapping_parts(mapping)
    ph = [split_phone(p) for p in phones]
    out, i = [], 0
    while i < len(ph):
        base, st = ph[i]
        nxt = ph[i + 1] if i + 1 < len(ph) else None
        if base in r_merge and nxt is not None and nxt[0] == "R":
            out.append(r_merge[base])
            i += 2
            continue
        if base == "AY" and nxt is not None and nxt == ("ER", "0"):
            out.append("aɪɚ")
            i += 2
            continue
        if base == "AY" and nxt is not None and nxt == ("AH", "0"):
            out.append("aɪə")
            i += 2
            continue
        if base == "AH" and st == "0" and nxt is not None and nxt[0] == "L":
            after = ph[i + 2] if i + 2 < len(ph) else None
            if after is None or after[0] not in VOWELS:
                out.append("əl")
                i += 2
                continue
        if base in VOWELS:
            if not st:
                raise ValueError(f"母音に強勢の数字がない: {phones}")
            out.append(stress_units[base][st] if base in stress_units else VOWEL_UNITS[base])
        elif base in CONS_UNITS:
            out.append(CONS_UNITS[base])
        else:
            raise ValueError(f"規則にない ARPAbet の記号: {base}")
        i += 1
    return out


def possessive_units(units):
    last = units[-1]
    if last in ("p", "t", "k", "f", "θ"):
        return ["s"]
    if last in ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ"):
        return ["ɪ", "z"]
    return ["z"]


def label_stress_syllable(label):
    """ラベルの大文字の部分がある音節（0 始まり）。例 noun_REcord → 0、verb_reCORD → 1、noun_ESscort → 0."""
    part = label.split("_", 1)[1]
    m = re.search(r"[A-Z]", part)
    if not m:
        raise ValueError(f"ラベルに大文字（強勢の位置）がない: {label}")
    return len(re.findall(r"[aeiouy]+", part[:m.start()].lower()))


def primary_stress_syllable(phones):
    vow = [p for p in phones if split_phone(p)[0] in VOWELS]
    ones = [k for k, p in enumerate(vow) if split_phone(p)[1] == "1"]
    return ones[0] if len(ones) == 1 else None


def n_syllables(phones):
    return sum(1 for p in phones if split_phone(p)[0] in VOWELS)


def move_primary_stress(phones, syl):
    """第 1 強勢を音節 syl に移す（もとの第 1 強勢は 0 にする）。CMUdict に無い読みを作るときだけ使う."""
    out, k = [], -1
    for p in phones:
        b, s = split_phone(p)
        if b in VOWELS:
            k += 1
            if k == syl:
                s = "1"
            elif s == "1":
                s = "0"
            out.append(b + s)
        else:
            out.append(p)
    return out


def compare_units(u1, u2):
    """2 つの読みの単位列の違い: identical / reduced_vowel_only / segmental と、違う位置."""
    if u1 == u2:
        return "identical", []
    if len(u1) != len(u2):
        return "segmental", ["length differs"]
    diff = [(k, a, b) for k, (a, b) in enumerate(zip(u1, u2)) if a != b]
    if all(frozenset((a, b)) in REDUCED_FULL for _, a, b in diff):
        return "reduced_vowel_only", [f"{k}:{a}/{b}" for k, a, b in diff]
    return "segmental", [f"{k}:{a}/{b}" for k, a, b in diff]


# 表を作る前から決めてある、難しい語の注記と別の読み（結果を見て変えない）
HARD_CASES = {
    "export": {
        "hard_case": True,
        "note_ja": ("CMUdict に動詞の形が無い（名詞 EH1 K S P AO0 R T だけ）。名詞の第 1 強勢を 2 音節目に移して動詞を作った。"
                    "強勢の記号を除くと 2 つの読みは同じ単位列になり、この語の 10 対は必ず失敗に数えられる。"
                    "動詞の 1 音節目を弱い ɪ（ɪkˈspɔːrt）とする辞書もあるが、採らずに別の読みとして載せる。"),
        "alternatives": {"verb_exPORT": [["ɪ", "k", "s", "p", "ɔːɹ", "t"]]},
    },
    "research": {
        "hard_case": True,
        "note_ja": ("2 つの読みの違いは 2 音節目の ɚ（名詞）と ɜː（動詞）だけ。名詞 REsearch の 2 音節目は第 2 強勢の強い母音 ɜː で"
                    "読む人も多く、その読みでは違いがなくなる。動詞の 1 音節目を弱い ɪ とする辞書もある（ɹɪsˈɜːtʃ）。"),
        "alternatives": {"noun_REsearch": [["ɹ", "iː", "s", "ɜː", "tʃ"]],
                         "verb_reSEARCH": [["ɹ", "ɪ", "s", "ɜː", "tʃ"]]},
    },
    "upset": {
        "hard_case": True,
        "note_ja": ("2 つの読みの違いは 1 音節目の ʌ（adj_UPset）と ə（verb_upSET）だけ。しかも adj_UPset の 5 文のうち "
                    "4 文（felt / was / were upset）は述語の形容詞で、ふつうは upSET と読む。ラベルそのものが疑わしい。"
                    "対は除かない。"),
        "alternatives": {},
    },
}
EXTRA_ALTERNATIVES = {   # 難しい語には入れないが、表を見る人のために載せる別の読み
    "conduct": {"verb_conDUCT": [["k", "ə", "n", "d", "ʌ", "k", "t"]]},
}
# v2 の表での注記と別の読み（v1 の注記の記号を v2 の記号に直したもの。22 時に結果を見る前に固定）
HARD_CASES_V2 = {
    "export": {
        "hard_case": True,
        "note_ja": ("CMUdict に動詞の形が無い（名詞 EH1 K S P AO0 R T だけ）。名詞の第 1 強勢を 2 音節目に移して動詞を作った。"
                    "v2 では 2 つの読みとも ɛ k s p oː t になり、この語の 10 対は必ず失敗に数えられる（v1 と同じ）。"
                    "動詞の 1 音節目を弱い ɪ（ɪ k s p oː t）とする辞書もあるが、採らずに別の読みとして載せる。"),
        "alternatives": {"verb_exPORT": [["ɪ", "k", "s", "p", "oː", "t"]]},
    },
    "research": {
        "hard_case": True,
        "note_ja": ("v2 では ER を強勢によらず ɚ と書くので、名詞 REsearch も動詞 reSEARCH も ɹ iː s ɚ tʃ になり、"
                    "この語の 10 対は必ず失敗に数えられる。動詞の 1 音節目を弱い ɪ とする辞書の読み（ɹ ɪ s ɚ tʃ）を"
                    "別の読みとして載せる（使わない）。"),
        "alternatives": {"verb_reSEARCH": [["ɹ", "ɪ", "s", "ɚ", "tʃ"]]},
    },
    "upset": HARD_CASES["upset"],
}
HARD_CASES_BY = {"v1": HARD_CASES, "v2": HARD_CASES_V2}
# CMUdict に無い文脈の語（300 文で 1 語だけ）。CMUdict の reproduce（R IY2 P R AH0 D UW1 S）に -ible（AH0 B AH0 L）を付けた
CONTEXT_ADDITIONS = {"reproducible": ["R", "IY2", "P", "R", "AH0", "D", "UW1", "S", "AH0", "B", "AH0", "L"]}


def cmudict_lexicon():
    """ホストの cmudict パッケージ（md5 を確かめる）。コンテナでは使わない."""
    import cmudict
    path = os.path.join(os.path.dirname(cmudict.__file__), "data", "cmudict.dict")
    got = md5_file(path) if os.path.isfile(path) else None
    if got != CMUDICT_MD5:
        raise SystemExit(f"cmudict のデータの md5 が {got}（{CMUDICT_MD5} でない）")
    return cmudict.dict(), {"package": "cmudict", "version": _pkg_version("cmudict"), "dict_md5": got}


def _pkg_version(pkg):
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def build_table(lex, lex_info, words, readings, data_pairs, vocab, mapping="v1"):
    """60 個の読みの表を作る（ラベルの強勢に合う CMUdict の最初の発音 → 単位列）.

    mapping="v1" の出力は最初の表（en_reading_table_v1.json、作った時刻を除く）と同じ。"v2" は置き換えの規則だけが違い、
    各読みに v1 の単位列、各語に v1 の分け方も並べる（著者が見比べるため）."""
    _mapping_parts(mapping)
    hard_cases = HARD_CASES_BY[mapping]
    entries, word_info = [], {}
    for w in words:
        prons = [list(p) for p in lex.get(w, [])]
        if not prons:
            raise SystemExit(f"CMUdict に {w} が無い")
        units_by = {}
        for side in ("a", "b"):
            label = readings[w][side]
            syl = label_stress_syllable(label)
            match = [p for p in prons if primary_stress_syllable(p) == syl]
            derived = False
            if match:
                chosen = match[0]
            else:
                base = next(p for p in prons if n_syllables(p) >= syl + 1)
                chosen = move_primary_stress(base, syl)
                derived = True
            units = arpa_to_units(chosen, mapping)
            missing = [u for u in units if u not in vocab]
            if missing:
                raise SystemExit(f"{w} {label}: 認識器の語彙に無い単位 {missing}")
            alts = []
            for p in match[1:]:
                alts.append({"source": "other CMUdict pronunciation with the same stress",
                             "arpabet": " ".join(p), "units": arpa_to_units(p, mapping)})
            for u in (hard_cases.get(w, {}).get("alternatives", {}).get(label, [])
                      + EXTRA_ALTERNATIVES.get(w, {}).get(label, [])):
                alts.append({"source": "dictionary variant (listed for review, not used)", "units": u})
            ent = {"word": w, "reading": side, "label": label, "stress_syllable": syl,
                   "cmudict_all": [" ".join(p) for p in prons],
                   "cmudict_with_this_stress": [" ".join(p) for p in match],
                   "arpabet": " ".join(chosen), "derived_by_moving_stress": derived,
                   "units": units, "display": " ".join(units), "alternatives": alts}
            if mapping != "v1":
                ent["v1_units"] = arpa_to_units(chosen, "v1")
            entries.append(ent)
            units_by[side] = units
        cat, where = compare_units(units_by["a"], units_by["b"])
        flags = []
        if cat == "identical":
            flags.append("identical_units: the two readings give the same phoneme sequence -> all 10 pairs count as failures")
        if cat == "reduced_vowel_only":
            flags.append("reduced_vowel_only: the readings differ only by a reduced vs full vowel (weak difference)")
        if any(e["derived_by_moving_stress"] for e in entries if e["word"] == w):
            flags.append("derived: one reading is not in CMUdict; built by moving the primary stress")
        if any(len(e["cmudict_with_this_stress"]) > 1 for e in entries if e["word"] == w):
            flags.append("several_cmudict_candidates: the first one with the labelled stress is used; others listed")
        hc = hard_cases.get(w, {})
        word_info[w] = {"labels": [readings[w]["a"], readings[w]["b"]], "category": cat, "differences": where,
                        "hard_case": bool(hc.get("hard_case")), "note_ja": hc.get("note_ja"), "flags": flags,
                        "pairs": [d["pair_idx"] for d in data_pairs if d["word"] == w]}
        if mapping != "v1":
            v1e = {e["reading"]: e["v1_units"] for e in entries if e["word"] == w}
            v1cat, v1where = compare_units(v1e["a"], v1e["b"])
            word_info[w]["v1_category"] = v1cat
            word_info[w]["v1_differences"] = v1where
            if v1cat != cat or v1e["a"] != units_by["a"] or v1e["b"] != units_by["b"]:
                word_info[w]["mapping_note_ja"] = (
                    f"{mapping} で読みの記号が v1 から変わった（v1: {' '.join(v1e['a'])} / {' '.join(v1e['b'])}、"
                    f"{mapping}: {' '.join(units_by['a'])} / {' '.join(units_by['b'])}）。分け方 v1 {v1cat} → {mapping} {cat}")
    summ = {"n_entries": len(entries), "n_words": len(words)}
    for cat in ("identical", "reduced_vowel_only", "segmental"):
        ws = [w for w in words if word_info[w]["category"] == cat]
        summ[f"{cat}_words"] = ws
        summ[f"n_pairs_{cat}"] = sum(len(word_info[w]["pairs"]) for w in ws)
    summ["hard_cases"] = [w for w in words if word_info[w]["hard_case"]]
    table = {
        "name": "Appendix K English judge: reading table (30 heteronyms x 2 readings)",
        "version": "v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "scripts/regen2/appk_enjudge_table.py",
        "author_review": {"status": "draft, not yet approved by the author",
                          "how": ("edit 'units' of an entry to change a reading (the stage recomputes identical / "
                                  "reduced_vowel_only / segmental from the units), then run the stage with "
                                  "--table_approved_by_author")},
        "source": {"cmudict": lex_info, "data_file": proj_rel(DATA_FILE), "data_file_md5": EXPECTED_MD5["data_file"],
                   "recognizer": MODEL_ID, "vocab": proj_rel(VOCAB), "vocab_md5": SNAP_MD5[SNAP_MAIN]["vocab.json"]},
        "rules": {"stress_from_label": "the upper-case part of the label is the stressed syllable (noun_REcord -> 0, "
                                       "verb_reCORD -> 1)",
                  "choice": ("the first CMUdict pronunciation whose primary stress (digit 1) is on that syllable; "
                             "if there is none, the other reading with the primary stress moved there (flag 'derived')"),
                  "mapping": MAPPING_RULES,
                  "no_stress_symbols": "the recognizer vocabulary has no stress symbols, so stress shows only through vowel quality",
                  "categories": {"identical": "same unit sequence (d = 0 by construction, counted as failure)",
                                 "reduced_vowel_only": "differ only by ə/ʌ or ɚ/ɜː",
                                 "segmental": "any other difference"}},
        "context": {"other_words": ("first CMUdict pronunciation of each other word of the sentence, the same for both "
                                    "hypotheses (as pypinyin's default reading for the other characters in the Chinese judge)"),
                    "additions": {k: " ".join(v) for k, v in CONTEXT_ADDITIONS.items()},
                    "entity_token_rules": {"possessive": POSSESSIVE_RULE},
                    "tokenize": "words = [A-Za-z']+ (hyphens split words), lower-cased; no word separator symbol"},
        "entries": entries,
        "words": word_info,
        "summary": summ,
    }
    if mapping != "v1":
        n_ident = summ["n_pairs_identical"]
        table["version"] = mapping
        table["mapping_version"] = mapping
        table["rules"]["mapping"] = MAPPING_RULES_BY[mapping]
        table["rules"]["mapping_version"] = mapping
        table["rules"]["categories"]["reduced_vowel_only"] = "differ only by ə/ʌ (ɜː is not used in v2)"
        table["context"]["other_words"] += f"; the other words use the same {mapping} mapping as the entity"
        table["derived_from"] = {"v1_table": proj_rel(TABLE_PATH),
                                 "why": MAPPING_RULES_BY[mapping]["why"],
                                 "entries_v1_units": "each entry keeps its v1 units under 'v1_units' for comparison"}
        summ["max_primary_if_every_other_pair_succeeds"] = wilson(N - n_ident, N)
    return table


def table_mapping(table):
    """表の置き換えの版（v1 の表には版の欄が無いので version から決める）。文の中のほかの語も同じ版で置き換える."""
    m = table.get("mapping_version") or (table.get("rules") or {}).get("mapping_version")
    if m is None and table.get("version") == "v1":
        m = "v1"
    if m not in MAPPINGS:
        raise SystemExit(f"読みの表の置き換えの版が分からない: {m!r}（v1 か v2）")
    return m


def table_units(table):
    """表 → {(語, ラベル): 単位列}。表の形も確かめる."""
    out = {}
    for e in table["entries"]:
        key = (e["word"], e["label"])
        if key in out:
            raise SystemExit(f"表に同じ (語, ラベル) が 2 つある: {key}")
        out[key] = list(e["units"])
    return out


def check_table(table, words, readings, vocab):
    """表が 30 語 × 2 読みで、ラベルが data ファイルと同じで、単位がすべて認識器の語彙にあるか。戻り値: problems."""
    problems = []
    tu = table_units(table)
    want = {(w, readings[w][s]) for w in words for s in ("a", "b")}
    if set(tu) != want:
        problems.append(f"表の (語, ラベル) が data ファイルと違う: 多い {sorted(set(tu) - want)[:5]}、"
                        f"足りない {sorted(want - set(tu))[:5]}")
    for k, u in tu.items():
        if not u or not all(isinstance(x, str) for x in u):
            problems.append(f"表の単位列が空か文字列でない: {k}")
            continue
        bad = [x for x in u if x not in vocab or vocab[x] in (0, 1, 2, 3)]
        if bad:
            problems.append(f"表の {k} に認識器の語彙に無い単位: {bad}")
    return problems


def categories_from_units(tu, words, readings):
    return {w: compare_units(tu[(w, readings[w]["a"])], tu[(w, readings[w]["b"])])[0] for w in words}


# ---------------------------------------------------------------------------
# 仮説（文全体の音素列。対象の語だけ src の読み / tgt の読み）
# ---------------------------------------------------------------------------
def tokenize_text(text):
    return re.findall(r"[A-Za-z']+", text)


def entity_positions(tokens, word):
    return [k for k, t in enumerate(tokens) if t.lower() in (word, word + "'s")]


def context_units(tok, lex, mapping="v1"):
    t = tok.lower().strip("'")
    if t in CONTEXT_ADDITIONS:
        return arpa_to_units(CONTEXT_ADDITIONS[t], mapping), "addition"
    prons = lex.get(t)
    if not prons:
        return None, "missing"
    return arpa_to_units(list(prons[0]), mapping), "cmudict_first"


def build_hypotheses(data_pairs, tu, lex, vocab, mapping="v1"):
    """300 対の仮説（単位列とトークン id 列）。文の中のほかの語は表と同じ版（mapping）で置き換える。
    戻り値: (行のリスト, problems)."""
    rows, problems = [], []
    for d in data_pairs:
        toks = tokenize_text(d["text_a"])
        pos = entity_positions(toks, d["word"])
        if len(pos) != 1:
            problems.append(f"対 {d['pair_idx']}: 文の中の {d['word']} が {len(pos)} 個（1 個でない）")
            continue
        k = pos[0]
        ent_tok = toks[k].lower()
        poss = ent_tok.endswith("'s")
        src_u = list(tu[(d["word"], d["src_label"])])
        tgt_u = list(tu[(d["word"], d["tgt_label"])])
        if poss:
            src_u += possessive_units(src_u)
            tgt_u += possessive_units(tgt_u)
        left, right, how = [], [], {}
        ok = True
        for j, t in enumerate(toks):
            if j == k:
                continue
            u, h = context_units(t, lex, mapping)
            if u is None:
                problems.append(f"対 {d['pair_idx']}: 文脈の語 {t!r} の発音が無い")
                ok = False
                break
            if h != "cmudict_first":
                how[t.lower()] = h
            (left if j < k else right).extend(u)
        if not ok:
            continue
        units_src, units_tgt = left + src_u + right, left + tgt_u + right
        bad = [u for u in units_src + units_tgt if u not in vocab]
        if bad:
            problems.append(f"対 {d['pair_idx']}: 認識器の語彙に無い単位 {bad[:5]}")
            continue
        rows.append({"pair_idx": d["pair_idx"], "word": d["word"], "direction": d["direction"],
                     "src_label": d["src_label"], "tgt_label": d["tgt_label"], "text_a": d["text_a"],
                     "text_b": d["text_b"], "entity_token": toks[k], "entity_token_index": k,
                     "possessive": poss, "context_notes": how,
                     "entity_src_units": src_u, "entity_tgt_units": tgt_u,
                     "units_src": units_src, "units_tgt": units_tgt,
                     "ids_src": [vocab[u] for u in units_src], "ids_tgt": [vocab[u] for u in units_tgt],
                     "degenerate": units_src == units_tgt})
    return rows, problems


def hypotheses_digest(hyp_rows):
    return md5_json([[h["pair_idx"], h["ids_src"], h["ids_tgt"]] for h in hyp_rows])


# ---------------------------------------------------------------------------
# 表と仮説の記号を、認識器が同じ 600 本に実際に出した記号の数と照らす（CPU だけ。GPU・結果は使わない）
# ---------------------------------------------------------------------------
IPA_SIDES = ("bl_ipa", "inj_ipa")


def load_emission_inventory():
    """同じ認識器が同じ 600 本を貪欲に書き起こした記号の、ファイル全体での回数（bl_ipa と inj_ipa の合計）.

    使うのは記号ごとの総数だけで、どの対で何が出たか（対の成否）は使わない。
    書き起こしが今の 600 本のものであることは、記録の連なりで確かめる: 実行記録（md5 固定）が EN を done、
    600 本の md5 を反映後の無音の検査（md5 固定。stage はこの検査の md5 と今の音声を照合する）と照合済み、と書いている。
    戻り値: (回数の dict, 情報)."""
    for tag, p in (("ipa_en_transcripts", IPA_EN_TRANSCRIPTS), ("ipa_run_meta", IPA_RUN_META)):
        ok, m = check_pinned(tag, p)
        if not ok:
            raise SystemExit(f"{proj_rel(p)} の md5 {m} が {EXPECTED_MD5[tag]} でない")
    meta = read_json(IPA_RUN_META)
    en = (meta.get("langs") or {}).get("en") or {}
    bad = []
    if meta.get("model") != MODEL_ID or meta.get("hf_refs_main") != SNAP_MAIN:
        bad.append(f"認識器 {meta.get('model')} {meta.get('hf_refs_main')}")
    if resolve(meta.get("silent_scan") or "") != POST_APPLY_SCAN:
        bad.append(f"照合した無音の検査 {meta.get('silent_scan')}")
    if en.get("status") != "done" or en.get("md5_checked_against_scan") != 2 * N:
        bad.append(f"EN の status {en.get('status')} / 照合 {en.get('md5_checked_against_scan')}")
    if resolve(en.get("wav_dir") or "") != PROD_DIR or resolve(en.get("output_json") or "") != IPA_EN_TRANSCRIPTS:
        bad.append(f"EN の置き場 {en.get('wav_dir')} / {en.get('output_json')}")
    doc = read_json(IPA_EN_TRANSCRIPTS)
    rows = doc.get("per_pair", [])
    if (doc.get("summary") or {}).get("model") != MODEL_ID:
        bad.append(f"書き起こしの認識器 {(doc.get('summary') or {}).get('model')}")
    if [r.get("idx") for r in rows] != list(range(N)):
        bad.append("書き起こしの対が 0..299 の 300 対でない")
    elif any(os.path.basename(r["bl_wav"]) != wav_name(i, "bl") or os.path.basename(r["inj_wav"]) != wav_name(i, "inj")
             or resolve(os.path.dirname(r["bl_wav"])) != PROD_DIR or resolve(os.path.dirname(r["inj_wav"])) != PROD_DIR
             for i, r in enumerate(rows)):
        bad.append("書き起こしの音声の名前・置き場が本番の s####_bl / s####_inj でない")
    if bad:
        raise SystemExit(f"書き起こし {proj_rel(IPA_EN_TRANSCRIPTS)} を照合に使えない: {bad}")
    counts = {}
    for r in rows:
        for f in IPA_SIDES:
            for s in r[f].split():
                counts[s] = counts.get(s, 0) + 1
    info = {"file": proj_rel(IPA_EN_TRANSCRIPTS), "md5": EXPECTED_MD5["ipa_en_transcripts"],
            "run_meta": proj_rel(IPA_RUN_META), "run_meta_md5": EXPECTED_MD5["ipa_run_meta"],
            "fields": list(IPA_SIDES), "n_rows": len(rows), "n_tokens": sum(counts.values()), "n_distinct": len(counts),
            "uses": "whole-file count of each symbol only (never which pair it came from, never any pair's outcome)"}
    return counts, info


def diff_segments(a, b):
    """2 つの単位列の違う所（difflib の最長一致で並べた、equal 以外の区間）."""
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(tag, list(a[i1:i2]), list(b[j1:j2])) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def _sub_counts(big, small):
    out = {}
    for u in big:
        out[u] = out.get(u, 0) + 1
    for u in small:
        out[u] -= 1
    return {u: n for u, n in out.items() if n}


def symbol_emission_check(table, hyp_rows, words, readings, data_pairs, counts, inv_info):
    """表（60 個の読み）と仮説（300 文）のすべての記号を、書き起こしでの出た回数と並べる.

    語ごとに、2 つの読みの違う所に「認識器が 1 度も出さない記号」があるかを見る:
      only = 違う所のすべてにその記号がある（その記号が無ければ 2 つの読みは区別できない）
      part = 違う所の一部にその記号がある
      none = 違う所の記号はすべて出たことがある / identical = 2 つの読みが同じ（必ず失敗）
    対の成否は使わない（記号ごとの総数だけ）."""
    tu = table_units(table)
    mapping = table_mapping(table)
    em = lambda s: counts.get(s, 0)  # noqa: E731
    table_syms = {}
    for (w, lab), units in sorted(tu.items()):
        for u in units:
            d = table_syms.setdefault(u, {"emitted": em(u), "entries": []})
            if f"{w} {lab}" not in d["entries"]:
                d["entries"].append(f"{w} {lab}")
    ctx_occ, ctx_pairs = {}, {}
    for h in hyp_rows:
        for u, n in _sub_counts(h["units_src"], h["entity_src_units"]).items():
            ctx_occ[u] = ctx_occ.get(u, 0) + n
            ctx_pairs.setdefault(u, set()).add(h["pair_idx"])
    hyp_syms = {}
    for u in sorted(set(ctx_occ) | set(table_syms)):
        hyp_syms[u] = {"emitted": em(u), "in_entity_readings": u in table_syms,
                       "n_in_other_words": ctx_occ.get(u, 0), "n_pairs_in_other_words": len(ctx_pairs.get(u, ()))}
    per_word = {}
    for w in words:
        a, b = tu[(w, readings[w]["a"])], tu[(w, readings[w]["b"])]
        segs = diff_segments(a, b)
        pairs = [d["pair_idx"] for d in data_pairs if d["word"] == w]
        zero = [s for s in segs if any(em(u) == 0 for u in s[1] + s[2])]
        role = "identical" if not segs else ("none" if not zero else ("only" if len(zero) == len(segs) else "part"))
        per_word[w] = {"a": f"{readings[w]['a']}: {' '.join(a)}", "b": f"{readings[w]['b']}: {' '.join(b)}",
                       "differences": [f"{' '.join(x) or '-'} / {' '.join(y) or '-'}" for _, x, y in segs],
                       "unemitted_in_difference": sorted({u for s in segs for u in s[1] + s[2] if em(u) == 0}),
                       "role": role, "pairs": pairs}

    def pairs_of(role):
        return sum(len(v["pairs"]) for v in per_word.values() if v["role"] == role)

    n_ident = pairs_of("identical")
    summ = {"table": table.get("version"), "mapping": mapping,
            "n_symbols_in_table": len(table_syms), "n_symbols_in_hypotheses": len(hyp_syms),
            "table_symbols_never_emitted": sorted(u for u, d in table_syms.items() if d["emitted"] == 0),
            "other_word_symbols_never_emitted": {u: {"n_in_other_words": d["n_in_other_words"],
                                                     "n_pairs": d["n_pairs_in_other_words"]}
                                                 for u, d in hyp_syms.items() if d["emitted"] == 0
                                                 and d["n_in_other_words"] > 0},
            "n_pairs_readings_differ": N - n_ident,
            "n_pairs_contrast_uses_never_emitted": pairs_of("only") + pairs_of("part"),
            "contrast_only_never_emitted": {"words": [w for w in words if per_word[w]["role"] == "only"],
                                            "n_pairs": pairs_of("only")},
            "contrast_partly_never_emitted": {"words": [w for w in words if per_word[w]["role"] == "part"],
                                              "n_pairs": pairs_of("part")},
            "identical_words": [w for w in words if per_word[w]["role"] == "identical"],
            "n_pairs_identical": n_ident,
            "max_primary_if_every_other_pair_succeeds": wilson(N - n_ident, N)}
    return {"what": ("every symbol of the reading table and of the 300 hypotheses, with how often the recognizer "
                     "emitted it in its own greedy transcripts of the same 600 wavs"),
            "inventory": inv_info, "summary": summ,
            "emitted_counts_all": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
            "table_symbols": dict(sorted(table_syms.items(), key=lambda x: (x[1]["emitted"], x[0]))),
            "hypothesis_symbols": dict(sorted(hyp_syms.items(), key=lambda x: (x[1]["emitted"], x[0]))),
            "words": per_word}


def emission_check_lines(chk, full=True):
    """照合の表示（表の記号・仮説の記号とその回数、影響する語と対）."""
    s = chk["summary"]
    inv = chk["inventory"]
    out = [f"[symbol-emission] table {s['table']} (mapping {s['mapping']}) vs {inv['file']} "
           f"(md5 {inv['md5'][:8]}, {inv['n_tokens']} symbols, {inv['n_distinct']} distinct; whole-file counts only)"]
    if full:
        out.append("  table symbols (emitted count): " + ", ".join(
            f"{u} {d['emitted']}" for u, d in chk["table_symbols"].items()))
        out.append("  hypothesis symbols (emitted count / occurrences in the other words): " + ", ".join(
            f"{u} {d['emitted']}/{d['n_in_other_words']}" for u, d in chk["hypothesis_symbols"].items()))
    out.append(f"  table symbols never emitted: {s['table_symbols_never_emitted']}")
    out.append(f"  other-word symbols never emitted (same in both hypotheses): {s['other_word_symbols_never_emitted']}")
    out.append(f"  pairs whose src/tgt contrast uses a never-emitted symbol: {s['n_pairs_contrast_uses_never_emitted']}"
               f" of {s['n_pairs_readings_differ']} (only: {s['contrast_only_never_emitted']}, "
               f"part: {s['contrast_partly_never_emitted']})")
    for w, v in chk["words"].items():
        if v["role"] in ("only", "part"):
            out.append(f"    {w:9s} {v['role']:4s} {v['a']}  |  {v['b']}  diff {v['differences']}")
    m = s["max_primary_if_every_other_pair_succeeds"]
    out.append(f"  identical-reading words {s['identical_words']} ({s['n_pairs_identical']} pairs) -> primary at most "
               f"{m['k']}/300 = {m['paper_style']}")
    return out


# ---------------------------------------------------------------------------
# CTC の前向き計算（中国語の判定器の関数をそのまま使う）
# ---------------------------------------------------------------------------
ZH_CTC_FUNC = "ctc_logprob_batch"


def zh_ctc_source():
    """中国語の判定器（md5 を確かめる）から ctc_logprob_batch の文面を取り出す（1 文字も変えない）."""
    ok, m = check_pinned("zh_judge", ZH_JUDGE)
    if not ok:
        raise SystemExit(f"{proj_rel(ZH_JUDGE)} の md5 {m} が {EXPECTED_MD5['zh_judge']} でない")
    with open(ZH_JUDGE, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ZH_CTC_FUNC]
    if len(fn) != 1:
        raise SystemExit(f"{proj_rel(ZH_JUDGE)} に {ZH_CTC_FUNC} が 1 つだけでない")
    seg = ast.get_source_segment(src, fn[0])
    return seg, hashlib.md5(seg.encode("utf-8")).hexdigest()


def zh_ctc_function(torch_module):
    """取り出した関数を、torch だけを渡した名前空間で定義して返す."""
    seg, seg_md5 = zh_ctc_source()
    ns = {"torch": torch_module}
    exec(compile(seg, f"<{proj_rel(ZH_JUDGE)}:{ZH_CTC_FUNC}>", "exec"), ns)
    return ns[ZH_CTC_FUNC], seg_md5


# ---------------------------------------------------------------------------
# 数え方（中国語の 7/28 の集計コードと同じ式。appk_zhjudge_common の shift_of / wilson と同じ文面）
# ---------------------------------------------------------------------------
def shift_of(e):
    return (e["inj_lp_tgt"] - e["inj_lp_src"]) - (e["bl_lp_tgt"] - e["bl_lp_src"])


def shift_count_as_zh_original(rows):
    """中国語の 7/28 の集計コードの 3 行（ok / d / k）と同じ書き方で数える."""
    ok = [e for e in rows if "skip" not in e]
    d = [(e["inj_lp_tgt"]-e["inj_lp_src"]) - (e["bl_lp_tgt"]-e["bl_lp_src"]) for e in ok]
    k = sum(x > 0 for x in d)
    return k, len(ok), round(k/3, 2)


def wilson(k, n, z=1.96):
    """半幅は中国語の集計コードの wil() と同じ式。区間の両端は Wilson の中心 ± 半幅."""
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    return {"k": k, "n": n, "pct": round(100 * p, 2), "wilson_half_pct": round(100 * half, 2),
            "wilson95_lo_pct": round(100 * (center - half), 2), "wilson95_hi_pct": round(100 * (center + half), 2),
            "paper_style": f"{100 * p:.2f} ± {100 * half:.2f}"}
