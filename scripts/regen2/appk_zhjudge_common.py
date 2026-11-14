#!/usr/bin/env python3
"""付録 K 中国語の注入成功率 93.67 を、作り直した 5 対を含む音声で、元と同じ方法で判定し直す: 共通の定義（2026-09-29）.

torch も numpy も import しない（CPU 側の段取り・集計と、GPU 側のラッパーの両方から読む）。

元の方法（2026-07-28。変えない）:
  判定器   scripts/analysis/zh_ctc_lr_judge.py（md5 7e823d42…。1 バイトも変えずに runpy で呼ぶ）
           walston/whisper-pinyin（snapshot 7840ab2a…）の CTC 対数尤度を、文全体の期待 pinyin 列
           （対象の字を src 読みにしたもの / tgt 読みにしたもの）で BL 音声と注入音声（*_oeoi.wav）に対して計算する。
           gold 読み表は v2（eval_results/zh_polyphone_gold_readings_v2.json、md5 3d099397…）。
  集計     .perm_cmds.txt の 1 行（2026-07-28 18:14、保存されていない heredoc）。その文面を ORIGINAL_CODE に
           そのまま写し、改行を空白にした文字列が記録の 1 行と md5 で一致することを毎回確かめる。
           d = (inj_lp_tgt - inj_lp_src) - (bl_lp_tgt - bl_lp_src)、k = #(d > 0)、pct = round(k/3, 2)（N=300 固定）、
           ± は Wilson 95% の半幅（z = 1.96）。
  作り直しの前の値: 281/300 = 93.67 ± 2.79（eval_results/cjk_n300_zh_v2/llr_shift_summary.json、書き換えない）。

書いてよい場所は eval_results/_appk_zhjudge/ と eval_results/_regen2_dryrun/appk_zhjudge_selftest/ だけ。
本番の cjk_n300_zh_v2/ の中（llr_shift_summary.json・ctc_lr_judge.json・ctc_lr_v3_partial.json を含む）には書かない。
"""
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_common as K  # noqa: E402  （torch / numpy を import しない。silence_criterion は numpy を関数の中でだけ読む）

if K.TEST_ROOT:
    raise SystemExit("APPK_TEST_ROOT が設定されている。appk_zhjudge_* は本番の置き場だけを読む（検査は appk_zhjudge_selftest.py）。")

PROJ = K.PROJ
HOST_PROJ = os.environ.get("REPO_ROOT", "/mnt")
ER = K.REAL_ER
N = 300
THREADS = 13

# ---------------------------------------------------------------------------
# 元の判定に使ったファイル（md5 は 2026-09-29 にホストで計算。7/28 の実行から変わっていない）
# ---------------------------------------------------------------------------
PROD_DIR = os.path.join(ER, "cjk_n300_zh_v2")
STORED_JUDGE = os.path.join(PROD_DIR, "ctc_lr_judge.json")
STORED_PARTIAL = os.path.join(PROD_DIR, "ctc_lr_v3_partial.json")
STORED_SUMMARY = os.path.join(PROD_DIR, "llr_shift_summary.json")
PROD_CKPT = os.path.join(PROD_DIR, "checkpoint_zh.json")
JUDGE = os.path.join(PROJ, "scripts", "analysis", "zh_ctc_lr_judge.py")
JUDGE_REL = "scripts/analysis/zh_ctc_lr_judge.py"
GOLD_V2 = os.path.join(ER, "zh_polyphone_gold_readings_v2.json")
GOLD_V2_REL = "eval_results/zh_polyphone_gold_readings_v2.json"
MODEL_ID = "walston/whisper-pinyin"
SNAP_PARENT = os.path.join(PROJ, ".cache", "huggingface", "hub", "models--walston--whisper-pinyin", "snapshots")
SNAP_ID = "7840ab2afafecd7b03aa66c7517d756c180dbb43"
FORK = os.path.join(PROJ, "third_party", "whisper_pinyin")
CMD_LOG = os.path.join(PROJ, ".perm_cmds.txt")

EXPECTED_MD5 = {
    "judge": "7e823d4216967042148cd26690e50738",
    "gold_v2": "3d099397bc5e742659fe20968cf23081",
    "stored_judge": "cb2dc8beeb9499bd9fc5d12e24a5c8a7",
    "stored_partial": "e910feb2e646ebf3ee7b5beb7eed3a48",
    "stored_summary": "6c13022e0ac577502555b1a6c1d67418",
    # 作り直しの前の本番 checkpoint_zh.json（反映のあとは _appk_regen/original_seed42_backup/zh/ の写しがこの md5）
    "orig_checkpoint": "262087de707e18cdbf9e5f8a671ecd13",
    "cmd_log_line": "f89760a3b90b28afb09867a80d9faf6d",
}
STORED_SUMMARY_CONTENT = {"n": 300, "shift_tone_pos": 281, "shift_tone_pct": 93.67}
SNAP_MD5 = {
    ".gitattributes": "fb3cffda6ede87b50774979ebc01da2d",
    "README.md": "2d0c13bbfdf97a19773541cd3fc2fcb8",
    "checkpoint-epoch=0009.ckpt": "b3396deb7610c61c1dde28c472326ada",
    "config.json": "a1c98a1b26660361f65a68f6f3d63613",
    "configuration_whisper_pinyin.py": "7be201eed584fdb9b7b588d7ba3bb5fe",
    "modeling_whisper_pinyin.py": "9106bf469ab270376ee144e990d6034c",
    "tokens.txt": "1b64d4b128c24b35fe547c704019e531",
}
FORK_MD5 = {   # third_party/whisper_pinyin/whisper/（7/28 12:10 から変わっていない）
    "__init__.py": "0faabcd72cba767491d70ece8fe37b7e",
    "__main__.py": "b0a596a40b2cd1ab2f4643d8f25cc05c",
    "assets/gpt2.tiktoken": "8b1899fed9fa1619c609fa4ab4c10114",
    "assets/mel_filters.npz": "61b070b259b27b8f8550e632d5300c8b",
    "assets/multilingual.tiktoken": "da95f6601b7c4327d4464d081b9dcf09",
    "audio.py": "b93513d786bc6b5aa0b8e9b501425925",
    "decoding.py": "d34d7c2897a8bffbaac00eec608fab62",
    "model.py": "9c71c27cd410a36064566869a89169b5",
    "normalizers/__init__.py": "a380f6d803b604a5e9a38605fe093361",
    "normalizers/basic.py": "7b97d2adf109c8e0c2a1c320bf3ecdbf",
    "normalizers/english.py": "17bdd2d747048bf2c1b1c30fd8cf48c1",
    "timing.py": "6a3f6e1fd5e540a7eb30a56fa7b9b008",
    "tokenizer.py": "ebacd7ca928f1760251320b700bc894c",
    "transcribe.py": "8566a0f67f745b641b24aadc12a42969",
    "triton_ops.py": "3a8197be4dc7c38fcb626aa32e39f6e8",
    "utils.py": "32e0346b3d68c4e31250590da4f740c0",
    "version.py": "110d335e9cca4bd1fff281450b6b45ff",
}

# ---------------------------------------------------------------------------
# 書き込み先
# ---------------------------------------------------------------------------
WORK_ROOT = os.path.join(ER, "_appk_zhjudge")
DEFAULT_STAGE = os.path.join(WORK_ROOT, "rejudge_ctc")
SELFTEST_ROOT = os.path.join(ER, "_regen2_dryrun", "appk_zhjudge_selftest")
MANIFEST = "stage_manifest.json"
JUDGE_OUT = "ctc_lr_judge.json"          # 判定器が stage に書く（名前は判定器が決める）
JUDGE_PARTIAL = "ctc_lr_v3_partial.json"  # 判定器の途中保存（これが stage にあると判定器は続きから始めてしまう）
RUN_META_PREFIX = "zhjudge_run_meta_"
SUMMARY_NEW = "llr_shift_summary_rejudge.json"          # 元の集計コードが書く（新しい名前）
SUMMARY_STORED_RECHECK = "llr_shift_summary_stored_recheck.json"
REPORT = "zh_injection_success_rejudge.json"
STAGE_CKPT = "checkpoint_zh.json"

WAV_SIDES = (("bl", "bl"), ("inj", "oeoi"))   # (appk_common の side, ファイル名の接尾辞)。判定器は s####_bl.wav と s####_oeoi.wav を読む


def wav_names(i):
    return {side: f"s{i:04d}_{suf}.wav" for side, suf in WAV_SIDES}


def all_wav_names():
    return [wav_names(i)[s] for i in range(N) for s, _ in WAV_SIDES]


def md5_file(p):
    return K.md5_file(p)


def md5_json(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(p):
    with open(p) as f:
        return json.load(f)


def proj_rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, PROJ) if (p + "/").startswith(PROJ + "/") else p


def to_here(p):
    """記録に残ったパス（コンテナの /mnt/... かホストの /data/...）を、今動いている側のパスに直す."""
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
    """書き込み先の確認: _appk_zhjudge/ か自己検査の置き場の下だけ。本番の ZH の置き場と原稿には書かない."""
    p = os.path.abspath(p)
    for bad in ("/naacl2027/", "/paper_draft/", "/arr_resubmission/"):
        if bad in p + "/":
            raise SystemExit(f"書き込み禁止の場所: {p}")
    if (p + "/").startswith(PROD_DIR + "/") or p in (STORED_SUMMARY, STORED_JUDGE, STORED_PARTIAL):
        raise SystemExit(f"本番の ZH の置き場には書かない: {p}")
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
        bak = f"{p}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(p, bak)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, p)
    return p


def check_pinned(tag, path):
    """md5 が固定値と一致するか。戻り値: (ok, 実際の md5)."""
    if not os.path.isfile(path):
        return False, None
    m = md5_file(path)
    return m == EXPECTED_MD5[tag], m


_MD5_CACHE = {}


def _md5_cached(p):
    """大きいファイル（1.7 GB の ckpt）を同じ実行の中で何度も読まない（大きさと更新時刻が同じ間だけ使う）."""
    st = os.stat(p)
    key = (os.path.realpath(p), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = md5_file(p)
    return _MD5_CACHE[key]


def check_model_and_fork(full_ckpt_md5=True):
    """判定器のモデル（snapshot）と vendored whisper の fork が 7/28 と同じか。戻り値: problems のリスト."""
    problems = []
    if not os.path.isdir(SNAP_PARENT):
        return [f"snapshot の置き場がない: {SNAP_PARENT}"]
    snaps = sorted(os.listdir(SNAP_PARENT))
    if snaps != [SNAP_ID]:
        # 判定器は sorted(listdir)[-1] を使う。snapshot が増えると別のモデルを読む
        problems.append(f"snapshot が {SNAP_ID} だけではない: {snaps}")
    snap = os.path.join(SNAP_PARENT, SNAP_ID)
    for n, want in SNAP_MD5.items():
        p = os.path.join(snap, n)
        if not os.path.isfile(p):
            problems.append(f"snapshot のファイルがない: {n}")
            continue
        if n.endswith(".ckpt") and not full_ckpt_md5:
            continue
        got = _md5_cached(p)
        if got != want:
            problems.append(f"snapshot の {n} の md5 が違う: {got} != {want}")
    for n, want in FORK_MD5.items():
        p = os.path.join(FORK, "whisper", n)
        got = md5_file(p) if os.path.isfile(p) else None
        if got != want:
            problems.append(f"whisper fork の {n} の md5 が違う: {got} != {want}")
    return problems


# ---------------------------------------------------------------------------
# 元の集計コード（.perm_cmds.txt 18812 行目、2026-07-28 18:14。文面はそのまま。変えない）
# ---------------------------------------------------------------------------
ORIG_IN_LITERAL = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/cjk_n300_zh_v2/ctc_lr_judge.json")
ORIG_OUT_LITERAL = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/cjk_n300_zh_v2/llr_shift_summary.json")
ORIGINAL_CODE = r'''import json, math
from scipy.stats import wilcoxon

rows = json.load(open("$REPO_ROOT/eval_results/cjk_n300_zh_v2/ctc_lr_judge.json"))["rows"]
ok = [e for e in rows if "skip" not in e]

def wil(k, n, z=1.96):
    p = k/n
    return z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)*100

print(f"v2 N={len(ok)} (退化0 / 声調のみ {sum(e['tone_only'] for e in ok)} / 分節 {sum(not e['tone_only'] for e in ok)})")
# 声調ありシフト（全300）
d = [(e["inj_lp_tgt"]-e["inj_lp_src"]) - (e["bl_lp_tgt"]-e["bl_lp_src"]) for e in ok]
k = sum(x > 0 for x in d)
print(f"\n[主結果] ΔLLRシフト正（声調あり、N=300全対象）: {k}/300 = {k/3:.2f}% ±{wil(k,300):.2f}")
print(f"  Wilcoxon p={wilcoxon(d).pvalue:.3g} / ΔLLR中央値 {sorted(d)[150]:.2f}")
for grp, cond in [("声調のみ対", True), ("分節対", False)]:
    sub = [x for e, x in zip(ok, d) if e["tone_only"] == cond]
    kk = sum(x > 0 for x in sub)
    print(f"  {grp}: {kk}/{len(sub)} = {kk/len(sub)*100:.1f}%")
# 声調無視シフト（分節対）
seg = [e for e in ok if all(f in e for f in ("bl_lp_src_nt","bl_lp_tgt_nt","inj_lp_src_nt","inj_lp_tgt_nt"))]
dn = [(e["inj_lp_tgt_nt"]-e["inj_lp_src_nt"]) - (e["bl_lp_tgt_nt"]-e["bl_lp_src_nt"]) for e in seg]
kn = sum(x > 0 for x in dn)
print(f"\n声調無視シフト（分節対 n={len(seg)}）: {kn}/{len(seg)} = {kn/len(seg)*100:.2f}% ±{wil(kn,len(seg)):.2f} (p={wilcoxon(dn).pvalue:.3g})")
json.dump({"n": 300, "shift_tone_pos": k, "shift_tone_pct": round(k/3, 2)},
          open("$REPO_ROOT/eval_results/cjk_n300_zh_v2/llr_shift_summary.json", "w"), indent=1)'''


def original_code_as_logged():
    """コマンド記録は heredoc の改行を空白にして 1 行で残している。同じ形に直した文字列."""
    return "python3 - <<'EOF' " + ORIGINAL_CODE.replace("\n", " ") + " EOF"


def verify_original_code():
    """ORIGINAL_CODE が 7/28 の集計コードそのものか（記録の 1 行と md5 で照合）."""
    flat = original_code_as_logged()
    flat_md5 = hashlib.md5(flat.encode("utf-8")).hexdigest()
    out = {"source": f"{proj_rel(CMD_LOG)} (the only line that writes llr_shift_summary.json)",
           "flattened_md5": flat_md5, "expected_md5": EXPECTED_MD5["cmd_log_line"],
           "embedded_equals_pinned": flat_md5 == EXPECTED_MD5["cmd_log_line"]}
    if os.path.isfile(CMD_LOG):
        with open(CMD_LOG, encoding="utf-8") as f:
            hits = [(n + 1, line.rstrip("\n")) for n, line in enumerate(f)
                    if ORIG_OUT_LITERAL in line and "shift_tone_pos" in line]
        out["log_hits"] = [n for n, _ in hits]
        out["log_line_equals_embedded"] = [line == flat for _, line in hits]
        out["ok"] = out["embedded_equals_pinned"] and any(out["log_line_equals_embedded"])
    else:
        out["log_hits"] = None
        out["ok"] = out["embedded_equals_pinned"]
    for s in ('ok = [e for e in rows if "skip" not in e]',
              'd = [(e["inj_lp_tgt"]-e["inj_lp_src"]) - (e["bl_lp_tgt"]-e["bl_lp_src"]) for e in ok]',
              'k = sum(x > 0 for x in d)', '"shift_tone_pct": round(k/3, 2)'):
        assert ORIGINAL_CODE.count(s) == 1, s
    return out


def run_original_code(in_path, out_path):
    """元の集計コードを、入力と出力のパスの 2 か所だけを置き換えて、そのまま実行する.

    戻り値: (表示された文字列, 書かれた JSON)。out_path は _appk_zhjudge/ か自己検査の置き場の下に限る."""
    in_path = os.path.abspath(in_path)
    out_path = assert_write(out_path)
    for p in (in_path, out_path):
        if '"' in p or "\\" in p:
            raise SystemExit(f"パスに \" か \\ がある: {p}")
    code = ORIGINAL_CODE
    assert code.count(ORIG_IN_LITERAL) == 1 and code.count(ORIG_OUT_LITERAL) == 1
    code = code.replace('"' + ORIG_IN_LITERAL + '"', '"' + in_path + '"')
    code = code.replace('"' + ORIG_OUT_LITERAL + '"', '"' + out_path + '"')
    rows = read_json(in_path)["rows"]
    ok = [e for e in rows if "skip" not in e]
    if len(ok) != N:
        # 元のコードは N=300 を前提に k/3・sorted(d)[150]・wil(k,300) と書いている
        raise SystemExit(f"元の集計コードは N=300 前提。skip でない行が {len(ok)}（{in_path}）")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        shutil.copy2(out_path, f"{out_path}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<zh_llr_shift_original_20260728>", "exec"), {"__name__": "zh_llr_shift_original"})
    return buf.getvalue(), read_json(out_path)


# ---------------------------------------------------------------------------
# 数え方の検算（元のコードと同じ式を別に書いたもの。元のコードの結果と一致を確かめるだけに使う）
# ---------------------------------------------------------------------------
def shift_of(e):
    return (e["inj_lp_tgt"] - e["inj_lp_src"]) - (e["bl_lp_tgt"] - e["bl_lp_src"])


def shift_count(rows):
    ok = [e for e in rows if "skip" not in e]
    d = {e["pair_idx"]: shift_of(e) for e in ok}
    return sum(1 for v in d.values() if v > 0), len(ok), d


def wilson(k, n, z=1.96):
    """半幅は元のコードの wil() と同じ式。区間の両端は Wilson の中心 ± 半幅."""
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    return {"k": k, "n": n, "pct": round(100 * p, 2), "wilson_half_pct": round(100 * half, 2),
            "wilson95_lo_pct": round(100 * (center - half), 2), "wilson95_hi_pct": round(100 * (center + half), 2),
            "paper_style": f"{100 * p:.2f} ± {100 * half:.2f}"}


# ---------------------------------------------------------------------------
# 判定の行の照合
# ---------------------------------------------------------------------------
LP_FIELDS = ("bl_lp_src", "bl_lp_tgt", "inj_lp_src", "inj_lp_tgt",
             "bl_lp_src_nt", "bl_lp_tgt_nt", "inj_lp_src_nt", "inj_lp_tgt_nt")
META_FIELDS = ("char", "src", "tgt", "degenerate", "tone_only")
FLAG_DEFS = {"bl_choose_tgt": ("bl_lp_tgt", "bl_lp_src"), "inj_choose_tgt": ("inj_lp_tgt", "inj_lp_src"),
             "bl_choose_tgt_nt": ("bl_lp_tgt_nt", "bl_lp_src_nt"),
             "inj_choose_tgt_nt": ("inj_lp_tgt_nt", "inj_lp_src_nt")}
REQUIRED = ("pair_idx", "char", "src", "tgt", "degenerate", "tone_only",
            "bl_lp_src", "bl_lp_tgt", "inj_lp_src", "inj_lp_tgt")


def check_shape(rows, summary=None):
    """新しい判定の行が N=300 のそろった形か。戻り値: problems のリスト."""
    problems = []
    if len(rows) != N:
        problems.append(f"行が {len(rows)}（{N} でない）")
    idx = [r.get("pair_idx") for r in rows]
    if sorted(i for i in idx if isinstance(i, int)) != list(range(N)) or len(set(idx)) != len(idx):
        problems.append("pair_idx が 0..299 を 1 回ずつになっていない")
    skip = [r.get("pair_idx") for r in rows if "skip" in r]
    if skip:
        problems.append(f"skip の行がある: {skip[:10]}")
    deg = [r.get("pair_idx") for r in rows if r.get("degenerate")]
    if deg:
        problems.append(f"degenerate の行がある（7/28 は 0）: {deg[:10]}")
    miss = [r.get("pair_idx") for r in rows if "skip" not in r and any(k not in r for k in REQUIRED)]
    if miss:
        problems.append(f"必要な欄が欠けた行: {miss[:10]}")
    if summary is not None:
        for key, want in (("n_total", N), ("n_skip_tokenize", 0), ("n_degenerate", 0)):
            if summary.get(key) != want:
                problems.append(f"判定器の summary.{key}={summary.get(key)}（{want} でない）")
    return problems


def compare_rows(new_by, old_by, all_pairs, unchanged, value_tol=0.01, accept_value_drift=False):
    """作り直していない対で、7/28 の判定が再現されたか（採点の結果＝新しい k は見ない）.

    止める条件:
      * 主判定（d > 0 か）が 1 対でも変わった
      * 対の情報（char・src・tgt・degenerate・tone_only）が 1 対でも違う（300 対すべてで見る）
      * 欄の組が違う・行が無い
      * 尤度の値の差が value_tol を超えた（accept_value_drift のときは止めずに記録する）
      * 副判定（*_choose_tgt*）が、記録の差（余白）が雑音の幅（2 × value_tol、または許した最大差の 2 倍）を
        超えているのに変わった
    """
    res = {"n_pairs_checked_meta": len(all_pairs), "n_unchanged_compared": len(unchanged), "value_tol": value_tol,
           "missing": [], "meta_mismatch": [], "keyset_mismatch": [], "primary_flips": [],
           "values_over_tol": [], "max_abs_value_diff": 0.0, "n_values_compared": 0, "n_values_equal": 0,
           "secondary_flips_within_noise": [], "secondary_flips_beyond_noise": []}
    for i in all_pairs:
        n, o = new_by.get(i), old_by.get(i)
        if n is None or o is None:
            res["missing"].append(i)
            continue
        bad = {f: [o.get(f), n.get(f)] for f in META_FIELDS if n.get(f) != o.get(f)}
        if bad:
            res["meta_mismatch"].append({"pair_idx": i, "stored_vs_new": bad})
    flips = []
    for i in unchanged:
        n, o = new_by.get(i), old_by.get(i)
        if n is None or o is None:
            continue
        if set(n) != set(o):
            res["keyset_mismatch"].append({"pair_idx": i, "only_stored": sorted(set(o) - set(n)),
                                           "only_new": sorted(set(n) - set(o))})
        if all(k in n for k in LP_FIELDS[:4]):
            do, dn = shift_of(o), shift_of(n)
            if (do > 0) != (dn > 0):
                res["primary_flips"].append({"pair_idx": i, "stored_d": round(do, 3), "new_d": round(dn, 3)})
        for f in LP_FIELDS:
            if f in o and f in n:
                diff = abs(n[f] - o[f])
                res["n_values_compared"] += 1
                res["n_values_equal"] += int(n[f] == o[f])
                res["max_abs_value_diff"] = max(res["max_abs_value_diff"], diff)
                if diff > value_tol + 1e-9:
                    res["values_over_tol"].append({"pair_idx": i, "field": f, "stored": o[f], "new": n[f],
                                                   "abs_diff": round(diff, 4)})
        for flag, (a, b) in FLAG_DEFS.items():
            if (flag in o or flag in n) and n.get(flag) != o.get(flag):
                margin = abs(o[a] - o[b]) if (a in o and b in o) else None
                flips.append({"pair_idx": i, "flag": flag, "stored": o.get(flag), "new": n.get(flag),
                              "stored_margin": None if margin is None else round(margin, 4)})
    res["max_abs_value_diff"] = round(res["max_abs_value_diff"], 4)
    noise = 2 * (res["max_abs_value_diff"] if accept_value_drift else value_tol)
    res["noise_band_for_secondary_flags"] = round(noise, 4)
    for f in flips:
        key = "secondary_flips_within_noise" if (f["stored_margin"] is not None
                                                 and f["stored_margin"] <= noise + 1e-9) else "secondary_flips_beyond_noise"
        res[key].append(f)
    res["decisions_reproduced"] = not (res["missing"] or res["meta_mismatch"] or res["keyset_mismatch"]
                                       or res["primary_flips"] or res["secondary_flips_beyond_noise"])
    res["values_reproduced"] = not res["values_over_tol"]
    res["accept_value_drift"] = bool(accept_value_drift)
    res["pass"] = res["decisions_reproduced"] and (res["values_reproduced"] or bool(accept_value_drift))
    return res


# ---------------------------------------------------------------------------
# stage（判定器に渡す新しいフォルダ）の照合
# ---------------------------------------------------------------------------
def verify_stage_files(stage_dir, manifest):
    """stage の 600 本の wav と checkpoint_zh.json が manifest の md5 のままか。戻り値: 違うファイルのリスト."""
    bad = []
    for n, rec in manifest["files"].items():
        p = os.path.join(stage_dir, n)
        got = md5_file(p) if os.path.isfile(p) else None
        if got != rec["md5"]:
            bad.append({"file": n, "manifest_md5": rec["md5"], "now": got})
    p = os.path.join(stage_dir, STAGE_CKPT)
    got = md5_file(p) if os.path.isfile(p) else None
    if got != manifest["checkpoint"]["md5"]:
        bad.append({"file": STAGE_CKPT, "manifest_md5": manifest["checkpoint"]["md5"], "now": got})
    return bad


def prior_judge_outputs(stage_dir):
    """判定器の出力が stage に既にあるか（あれば判定器を走らせない。途中保存があると続きから始めてしまう）."""
    return [n for n in (JUDGE_PARTIAL, JUDGE_OUT) if os.path.exists(os.path.join(stage_dir, n))]
