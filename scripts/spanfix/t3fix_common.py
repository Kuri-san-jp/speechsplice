#!/usr/bin/env python3
"""t3fix: Table 3 の T5Gemma-TTS の行を直す作業の共通部品（2026-10-02）.

著者の決定（2026-10-02。note/2026-10-02_gpuneeds2_report.md の項目名）:
  (2b) B1・B2: Table 3 (a)（At our replacement point）の T5Gemma-TTS JA・ZH で、古い範囲の関数のせいで位置が
       誤っていた 4 対（JA 59,99,164,179 / ZH 15,145,246,295）の INJ を、l*=26 の掃引の音声
       （_layerabl/t5gemma_{ja,zh}_L25。BL・TGT は (a) と同じ inode）に差し替え、v1 PIB と whole MCD を
       出し直す。CPU だけ（t3fix_b12_apply.py）。値は eval_results/_audit_spanB_20261002/ と同じになるはず。
  (3)  GPU 表の 2・3: Table 3 (b)（Layer sweep）の T5Gemma-TTS EN で使えない 7 対（112,118,128,138,141,158,238。
       BL・TGT と 5 点の INJ）と、ZH の対 196（BL・TGT・(a) の INJ・5 点の INJ）を、対ごと seed 43、
       だめなら 44 で作り直す（t3fix_regen_gpu.py）。出力は新しい置き場だけ。差し替えは別の段階
       （t3fix_regen_apply.py。退避あり。hard link を通して書かない: 新しいファイルを作って os.replace する）。
  凍結・対象外: T5Gemma-TTS EN の (a)（_cc_wavs_a/t5gemma_en、45.33）、_cc_wavs_b/t5gemma_en、
       _layerabl/t5gemma_en_prog_*（EN の BL・TGT は掃引 5 点とこれらで同じ inode を持つ）、原稿の置き場。

決まり（著者の決定と precedent）:
  * N = 300。使えるか = scripts/regen2/silence_criterion.py（無音）＋ 生成の失敗 ＋ 位置合わせの失敗
    （Table 3 の採点 speechsplice_release/src/phoneme_per.py が行を出せない）。seed は 42 → 43 → 44 を
    使えるかどうかだけで選ぶ（PIB・MCD の値は見ない）。
  * pyopenjtalk はこちらのコードでは読まない（読み込みを塞ぐ）。T5Gemma-TTS 同梱の長さ推定器が使うのは
    著者が認めているが、作り直す EN・ZH の文では言語判定が ja にならないので呼ばれない（呼ばれたら止める）。
  * 新しいファイルは eval_results/_t3fix/ の下だけ。既存の結果は差し替えの段階（--apply）でだけ書き換え、
    その前に退避を取る。何も消さない。
  * 差し替えの段階は帳面（_t3fix/<段階>/apply_journal.json）を先に書き、途中で止まっても同じ --apply で続きから
    流せる（run_journal）。
  * 式の自己確認（formula_selftest）は反映の段階を見る: 反映の前は原稿の今の値、B1・B2 の後は監査の値、
    作り直しの反映の後は最後の反映の記録の値と照合する（expected_table3）。原稿の値との一致は記録だけ。

このモジュールは import のときに標準ライブラリと scripts/regen2/regen2_common.py（標準ライブラリだけ）しか読まない。
"""
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time

# ---------------------------------------------------------------- 置き場
HOST_ROOT = os.environ.get("REPO_ROOT", "/mnt")
IN_CONTAINER = os.path.isdir("/mnt/scripts")
PROJECT_ROOT = "/mnt" if IN_CONTAINER else HOST_ROOT
PROD_EV = os.path.join(PROJECT_ROOT, "eval_results")
OUT_NAME = "_t3fix"
SELFTEST_BASE = os.path.join(PROD_EV, OUT_NAME, "selftest")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))


def P(rel):
    return os.path.join(PROJECT_ROOT, rel)


def ev_root(sandbox=None):
    """採点・音声の置き場の根（本番は eval_results。試験では eval_results/_t3fix/selftest/ の下の写し）."""
    if not sandbox:
        return PROD_EV
    s = os.path.abspath(sandbox)
    if not s.startswith(SELFTEST_BASE + os.sep):
        raise SystemExit("[t3fix] --sandbox は %s の下に置く: %s" % (SELFTEST_BASE, s))
    return s


def out_root(ev):
    return os.path.join(ev, OUT_NAME)


def is_sandbox(ev):
    return os.path.abspath(ev) != os.path.abspath(PROD_EV)


# ---------------------------------------------------------------- 環境（numpy・torch・HF を読む前に呼ぶ）
THREADS = 13
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
HOST_CPUS = set(range(13))   # CPU 0-12
HOST_NICE = 10


def setup_env():
    """スレッド 13、HF はプロジェクトの .cache でオフライン、NUMBA の cache もプロジェクトの下。
    ホストでは GPU を見せず、CPU を 0-12 に絞り（今の affinity との共通部分）、nice を 10 にする。
    slurm_run.sh は HF_HOME=/tmp/... を渡すので setdefault ではなく上書きする。起動したときの値を返す."""
    keys = THREAD_VARS + ("GEN_THREADS", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                          "TORCH_HOME", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "CUDA_VISIBLE_DEVICES")
    launch = {k: os.environ.get(k) for k in keys}
    for v in THREAD_VARS:
        os.environ[v] = str(THREADS)
    os.environ["GEN_THREADS"] = str(THREADS)
    r = PROJECT_ROOT
    os.environ.update({"HF_HOME": r + "/.cache/huggingface", "HF_HUB_CACHE": r + "/.cache/huggingface/hub",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TORCH_HOME": r + "/.cache/torch",
                       "XDG_CACHE_HOME": r + "/.cache", "NUMBA_CACHE_DIR": r + "/.cache/numba"})
    if not IN_CONTAINER:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cur = os.sched_getaffinity(0)
        want = HOST_CPUS & cur
        if not want:
            raise SystemExit("[t3fix] CPU 0-12 がこのプロセスに許されていない（affinity %s）" % sorted(cur)[:20])
        if want != cur:
            os.sched_setaffinity(0, want)
        os.nice(max(0, HOST_NICE - os.nice(0)))
    sys.dont_write_bytecode = True   # 既存の scripts/ の下に .pyc を新しく作らない
    return launch


def host_state():
    return {"affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0),
            "threads": {v: os.environ.get(v) for v in THREAD_VARS},
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"), "in_container": IN_CONTAINER}


# regen2 共通の部品（無音の基準の数値の確認、pib_summary / mcd_summary、凍結した EN (a) の digest、失敗の分類）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "regen2"))
import regen2_common as RC  # noqa: E402

# ---------------------------------------------------------------- 対とセル
SWEEP_LAYERS = (0, 7, 14, 21, 25)          # encoder の層の番号。Table 3 の l* = 層 + 1（1/8/15/22/26）
REGEN_TARGETS = {"en": (112, 118, 128, 138, 141, 158, 238), "zh": (196,)}
CHECK_PAIRS = {"en": 113, "zh": 197}       # seed 42 の再現確認（対象外・どのセルでも無音なし・範囲は今と 5/17 で同じ）
SEEDS = (43, 44)
CHECK_SEED = 42
B12_PAIRS = {"ja": (59, 99, 164, 179), "zh": (15, 145, 246, 295)}
ZH_A_L25_NONSPAN_DIFF = (30,)              # (a) と L25 の INJ が違うが範囲とは関係ない対（差し替えない。記録のみ）
# B1・B2 の 4 対の今の音声の md5（差し替える前の (a) の INJ = 5/17 の古い関数、差し替えに使う L25 の INJ）
B12_PINS = {
    "ja": {59: ("c421189742c44a1fd911237828a347b0", "da6fb671fc282a08c7c033f721373414"),
           99: ("960d54456a29405f3126f4713ae64f39", "254f78b777e41d020177f7f1d17c09f6"),
           164: ("5a8908b24fdee854319f0f0684506345", "e3b414dbe5380b981d9a7d93a96637da"),
           179: ("d1bd69a8098053683014cd14b92f2fd6", "aeb26a0c3fda7123e227065d96bafa49")},
    "zh": {15: ("31fd3ff2cb021da13d41d61c1c7fa5f3", "bd141f1a86571d559c5c61afc3b08e65"),
           145: ("a4ff62e0227fe0567ef4c3d8fd561a38", "d5641ddf727ee083d2c98d513cf1e9ce"),
           246: ("38e457ad376887acd24006331087e828", "700d277d6737adc58272624dc5914b26"),
           295: ("4ef68183d3c386c2e722b1e7e2f7379c", "1b40c8e4a909c3dd439d12e86659bf96")},
}

A_DIR = {"ja": "_cc_wavs_a/t5gemma_ja", "zh": "_cc_wavs_a/t5gemma_zh"}
A_PIB = {"ja": "_v1all_t5gemma_ja.json", "zh": "_v1all_t5gemma_zh.json"}
A_MCD = {"ja": "_caseSPTK_whole_t5gemma_ja.json", "zh": "_caseSPTK_whole_t5gemma_zh.json"}
AUDIT_MCD_L25 = {"ja": "_audit_spanB_20261002/mcd_whole_t5gemma_ja_L25.json",
                 "zh": "_audit_spanB_20261002/mcd_whole_t5gemma_zh_L25.json"}
EN_B_DIR = "_cc_wavs_b/t5gemma_en"
EN_PROG_DIRS = tuple("_layerabl/t5gemma_en_prog_L%dto25" % L for L in (0, 7, 14, 21))
FROZEN_EN_A_DIR = "_cc_wavs_a/t5gemma_en"
FROZEN_EN_FILES = ("_v1all_t5gemma_en.json", "_caseSPTK_whole_t5gemma_en.json", "_caseSPTK_phperPER_t5gemma_en.json")
FROZEN_DIGEST_PREFIX = "0c9b56e5"


def sweep_dir(lang, L):
    return "_layerabl/t5gemma_%s_L%d" % (lang, L)


def sweep_pib(lang, L):
    return "_v1L_t5gemma_%s_L%d.json" % (lang, L)


def sweep_meta(lang, L):
    pre = {"en": "meta_L", "ja": "meta_ja_L", "zh": "meta_zh_L"}[lang]
    return "_layerabl/%s%d/layer_inject_meta.json" % (pre, L)


def regen_cells(lang):
    """作り直す言語のセル: [{"cell", "layer", "use_hook", "dir", "pib", "mcd"}]。
    掃引は本番どおり 5 点とも hook の経路（L25 も final_via_hook）。ZH の (a) は L25 で hook なし
    （= eval_zh_crossculture_t5gemma_fixed.py の t5_synth_inj と同じ関数・同じ引数）。"""
    cells = []
    if lang == "zh":
        cells.append({"cell": "A", "layer": 25, "use_hook": False, "dir": A_DIR["zh"], "pib": A_PIB["zh"],
                      "mcd": A_MCD["zh"], "lstar": 26})
    for L in SWEEP_LAYERS:
        cells.append({"cell": "L%d" % L, "layer": L, "use_hook": True, "dir": sweep_dir(lang, L),
                      "pib": sweep_pib(lang, L), "mcd": None, "lstar": L + 1})
    return cells


def wav_names(lang):
    """1 対で作る音声: bl, tgt と、セルごとの inj_<cell>."""
    return ["bl", "tgt"] + ["inj_%s" % c["cell"] for c in regen_cells(lang)]


def prod_wav(ev, cell, pidx, cond):
    return os.path.join(ev, cell["dir"], "s%04d_%s.wav" % (pidx, cond))


# ---------------------------------------------------------------- 原稿の今の値（読むだけ。naacl2027/main.tex 297-299 行）
MANUSCRIPT_TEX = "naacl2027/main.tex"
MANUSCRIPT_NOW = {
    "(a) JA": {"MCD": "8.23 \\pm 0.21", "PIB": "51.33 \\pm 5.62", "line": 298},
    "(a) ZH": {"MCD": "8.96 \\pm 0.25", "PIB": "54.33 \\pm 5.60", "line": 299},
    "(b) EN": {"range": "56.33--59.67", "line": 297},
    "(b) JA": {"range": "48.00--52.00", "line": 298},
    "(b) ZH": {"range": "52.33--55.33", "line": 299},
}


def manuscript_check():
    """原稿の 297-299 行に今の値がそのまま書かれているか（読むだけ）."""
    p = P(MANUSCRIPT_TEX)
    if not os.path.exists(p):
        return {"readable": False}
    lines = open(p, encoding="utf-8").read().split("\n")
    out = {"readable": True}
    for k, v in MANUSCRIPT_NOW.items():
        ln = lines[v["line"] - 1] if len(lines) >= v["line"] else ""
        want = [v[x] for x in ("MCD", "PIB", "range") if x in v]
        out[k] = all(w in ln for w in want)
    return out


# ---------------------------------------------------------------- コードの md5（本番のファイルは変えない）
PROD_CODE = {
    "scripts/layer_ablation/eval_en_crossculture_t5gemma_layer.py": "6950076abd83d09ca9eeddad6e2fda00",
    "scripts/encoder_injection_t5gemma.py": "4f36fa570145cde805072438d8a39bb1",   # 7/11 に直した範囲の関数
    "src/phoneme_per.py": "8ba4518514d80de1fce493c24187a40c",  # v1 PIB
    "src/mcd_pysptk.py": "c785eaa02cd53339e00874a01ce7061c",   # whole MCD
    "data/en_cross_culture_names.py": "4bbb35b331d21aea27389b6f2f49af3d",
    "data/zh_cross_culture_names.py": "1260a6110cdff2591aa5bb0e44836ae2",
    "data/ja_cross_culture_names.py": "a74383c4f910c8388031deb2d68662a7",
    "scripts/regen2/silence_criterion.py": "56b84ca2f09425ac95fe91a85b120bc6",
    "scripts/nontemplate/gen_t5gemma_pairs_module.py": "53eb4bfc1008beb361a23c6adfc429ba",  # HF offline の読み替え
}
LAYER_REL = "scripts/layer_ablation/eval_en_crossculture_t5gemma_layer.py"
PER_REL = "src/phoneme_per.py"
MCD_REL = "src/mcd_pysptk.py"
LEGACY_SPAN_REL = "release_artifacts/scripts/encoder_injection_t5gemma.py.bak_spanfix_20260917_095951"
OWN_CODE = ("scripts/spanfix/t3fix_common.py", "scripts/spanfix/t3fix_regen_gpu.py",
            "scripts/spanfix/t3fix_regen_apply.py", "scripts/spanfix/t3fix_b12_apply.py",
            "scripts/spanfix/spanfix_common.py", "scripts/regen2/regen2_common.py")
VENDOR_FILES = ("duration_estimator.py", "inference_tts_utils.py")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_pin_errors():
    errs = []
    for rel, want in PROD_CODE.items():
        got = md5_file(P(rel)) if os.path.exists(P(rel)) else None
        if got != want:
            errs.append("%s の md5 が固定の値と違う（今 %s / 固定 %s）" % (rel, got, want))
    return errs


def code_md5_record():
    rec = {rel: (md5_file(P(rel)) if os.path.exists(P(rel)) else None) for rel in tuple(PROD_CODE) + OWN_CODE}
    for f in VENDOR_FILES:
        p = os.path.join(T5GEMMA_ROOT, f)
        rec["vendor:" + f] = md5_file(p) if os.path.exists(p) else None
    return rec


def own_code_drift(rec):
    """記録した自前のコードの md5 が今と違うもの（著者の判断）."""
    out = []
    for rel in OWN_CODE:
        cur = md5_file(P(rel)) if os.path.exists(P(rel)) else None
        if (rec or {}).get(rel) != cur:
            out.append("%s: 記録 %s / 今 %s" % (rel, (rec or {}).get(rel), cur))
    return out


# ---------------------------------------------------------------- 小さな道具
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


_PART_SEQ = [0]


def _part_path(path):
    """path と同じ置き場の一時ファイル名（.<名前>.<pid>.<番号>.part）。読む側はこの名前を見ない."""
    _PART_SEQ[0] += 1
    return os.path.join(os.path.dirname(path) or ".", ".%s.%d.%d.part" % (os.path.basename(path), os.getpid(),
                                                                          _PART_SEQ[0]))


def write_json_new(obj, path, indent=1):
    """新しいファイルとしてだけ書く（既にあれば止まる）。同じ置き場の一時ファイルに書いて fsync してから名前を
    付けるので、途中で止まっても半端な JSON は残らない（一時ファイルは残ることがあるが、どこからも読まない）。
    同じ名前を同時に書く者はいない前提（どのスクリプトも書き手は 1 つ）."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(path)
    tmp = _part_path(path)
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        raise FileExistsError(path)
    os.replace(tmp, path)


def next_attempt_dir(base):
    """base/aNN（NN は既存の最大 + 1）を新しく作って返す（前の試行は消さない）."""
    os.makedirs(base, exist_ok=True)
    nums = [int(d[1:]) for d in os.listdir(base) if d.startswith("a") and d[1:].isdigit()]
    n = (max(nums) + 1) if nums else 1
    d = os.path.join(base, "a%02d" % n)
    os.makedirs(d)
    return d


def attempt_dirs(base):
    if not os.path.isdir(base):
        return []
    ds = [d for d in os.listdir(base) if d.startswith("a") and d[1:].isdigit()]
    return [os.path.join(base, d) for d in sorted(ds, key=lambda x: int(x[1:]))]


def stat_info(p, with_md5=True):
    st = os.stat(p)
    d = {"ino": st.st_ino, "nlink": st.st_nlink, "dev": st.st_dev, "size": st.st_size}
    if with_md5:
        d["md5"] = md5_file(p)
    return d


def wav_check(p):
    """regen2 共通の判定（無い・読めない・無音は bad）."""
    return RC.wav_check(p)


def load_pairs(lang):
    d = P("data")
    if d not in sys.path:
        sys.path.insert(0, d)
    mod = importlib.import_module("%s_cross_culture_names" % lang)
    pairs = mod.build_300_pairs()[:300]
    assert [p["pair_idx"] for p in pairs] == list(range(300)), lang
    return {p["pair_idx"]: p for p in pairs}


def load_function_from_source(path, name, extra_globals=None):
    """ファイルを import せず、関数定義だけを取り出して exec する（副作用なし）."""
    import ast
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (path, name, len(nodes))
    ns = dict(extra_globals or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------- 書き込みの安全装置
FORBIDDEN_DIRS = ("naacl2027", "paper_draft", "arr_resubmission", "icassp2027")


def protected_paths(ev):
    """書いてはいけない置き場（前方一致）と、ファイル."""
    dirs = [os.path.join(ev, FROZEN_EN_A_DIR) + os.sep, os.path.join(ev, "_cc_wavs_b") + os.sep]
    dirs += [os.path.join(ev, d) + os.sep for d in EN_PROG_DIRS]
    files = [os.path.join(ev, f) for f in FROZEN_EN_FILES]
    return dirs, files


def assert_write_ok(path, ev, planned):
    """書き込み先の確認: 原稿の置き場・凍結した EN (a)・_cc_wavs_b・prog には書かない。
    本番の置き場では、予定した書き込み先（planned）だけ。_t3fix の下は自由."""
    p = os.path.abspath(path)
    for d in FORBIDDEN_DIRS:
        if ("/%s/" % d) in p:
            raise SystemExit("[t3fix] 書き込み禁止の置き場: %s" % p)
    dirs, files = protected_paths(ev)
    if any(p.startswith(d) for d in dirs) or p in files:
        raise SystemExit("[t3fix] 凍結・対象外のファイルには書かない: %s" % p)
    if p.startswith(out_root(ev) + os.sep):
        return p
    if p not in planned:
        raise SystemExit("[t3fix] 予定に無い書き込み先: %s" % p)
    return p


def copy_new(src, dst):
    """dst を新しいファイルとして作り src の中身を写す（既にあれば止まる。hard link は作らない）."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(src, "rb") as fi, open(dst, "xb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    return dst


def copy_new_atomic(src, dst):
    """copy_new と同じだが、同じ置き場の一時ファイルに写してから名前を付ける（途中で止まっても dst は
    無いか完全かのどちらか）。退避に使う."""
    if os.path.exists(dst):
        raise FileExistsError(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = copy_new(src, _part_path(dst))
    if os.path.exists(dst):
        raise FileExistsError(dst)
    os.replace(tmp, dst)
    return dst


_TMP_SEQ = [0]


def _tmp_path(tmp_dir, base):
    _TMP_SEQ[0] += 1
    return os.path.join(tmp_dir, "%s.%d.%d.%d.tmp" % (base, os.getpid(), int(time.time() * 1000), _TMP_SEQ[0]))


def _check_same_fs(tmp_dir, dest):
    if os.stat(tmp_dir).st_dev != os.stat(os.path.dirname(dest)).st_dev:
        raise SystemExit("[t3fix] 一時ファイルの置き場 %s と %s が別のファイルシステム（os.replace が使えない）"
                         % (tmp_dir, dest))


def replace_with_copy(src, dest, tmp_dir, ev, planned):
    """dest の中身を src にする。既存の dest（とその hard link 先）には書かない:
    新しいファイルを tmp_dir に作り（open 'xb'）、os.replace で dest の名前をそれに付け替える。
    古い inode は、ほかの hard link（_cc_wavs_b など）から見て中身が変わらない."""
    assert_write_ok(dest, ev, planned)
    os.makedirs(tmp_dir, exist_ok=True)
    _check_same_fs(tmp_dir, dest)
    before = stat_info(dest)
    want = md5_file(src)
    tmp = copy_new(src, _tmp_path(tmp_dir, os.path.basename(dest)))
    os.replace(tmp, dest)
    after = stat_info(dest)
    errs = []
    if after["md5"] != want:
        errs.append("md5 が写した元と違う")
    if after["ino"] == before["ino"]:
        errs.append("inode が変わっていない（hard link を通して書いた恐れ）")
    if after["nlink"] != 1:
        errs.append("nlink が 1 でない（%d）" % after["nlink"])
    if errs:
        raise SystemExit("[t3fix] 差し替えの確認に失敗: %s %s" % (dest, errs))
    return {"dest": dest, "src": src, "before": before, "after": after}


def write_json_replace(obj, dest, tmp_dir, ev, planned, indent=2):
    """JSON を新しいファイルに書いてから os.replace（途中で止まっても半端な JSON を残さない）."""
    assert_write_ok(dest, ev, planned)
    os.makedirs(tmp_dir, exist_ok=True)
    _check_same_fs(tmp_dir, dest)
    before = stat_info(dest) if os.path.exists(dest) else None
    tmp = _tmp_path(tmp_dir, os.path.basename(dest))
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    return {"dest": dest, "before": before, "after": stat_info(dest)}


def backup_copy(src, ev, backup_root):
    """src を backup_root/<ev からの相対パス> に新しいファイルとして写す（既にあれば止まる）."""
    rel = os.path.relpath(os.path.abspath(src), os.path.abspath(ev))
    if rel.startswith(".."):
        raise SystemExit("[t3fix] 退避の元が置き場の外: %s" % src)
    dst = os.path.join(backup_root, rel)
    copy_new_atomic(src, dst)
    if md5_file(dst) != md5_file(src):
        raise SystemExit("[t3fix] 退避の md5 が違う: %s" % src)
    return dst


def digest_files(paths):
    """ファイルの並び（名前と md5）の digest."""
    h = hashlib.md5()
    for p in sorted(paths):
        h.update(p.encode())
        h.update((md5_file(p) if os.path.exists(p) else "missing").encode())
    return h.hexdigest()


def frozen_en_a_digest(ev):
    """凍結した T5Gemma EN (a) の音声と採点の digest。本番は regen2_common.frozen_digest（0c9b56e5…）と同じ."""
    if not is_sandbox(ev):
        return RC.frozen_digest()
    d = os.path.join(ev, FROZEN_EN_A_DIR)
    files = sorted(os.path.join(d, n) for n in os.listdir(d)) if os.path.isdir(d) else []
    files += [os.path.join(ev, f) for f in FROZEN_EN_FILES]
    return {"n_wav_files": len(files), "digest": digest_files(files)}


# ---------------------------------------------------------------- 表の値
def pib_cell(summary):
    """Table 3 の PIB: inj_better_pct ± Wilson 95% の半幅（phoneme_per.py の summary）."""
    return {"n_valid": summary["n_valid"], "k": summary["inj_better_n"], "pct": summary["inj_better_pct"],
            "half": summary["inj_better_ci95_half_pp"],
            "tex": "%.2f \\pm %.2f" % (summary["inj_better_pct"], summary["inj_better_ci95_half_pp"])}


def mcd_cell(rows):
    """Table 3 の MCD: BL–INJ の whole MCD の平均 ± t_{0.975,n-1}·sd/√n（sd は ddof=1）。
    今の原稿の 3 セル（EN 9.20 ± 0.28、JA 8.23 ± 0.21、ZH 8.96 ± 0.25）をこの式で再現することを確かめた
    （1.96 を使うと JA が 0.20 になり合わない）."""
    from scipy import stats
    v = [r["mcd_inj_vs_bl"] for r in rows if r.get("mcd_inj_vs_bl") is not None
         and r.get("mcd_inj_vs_tgt") is not None and r.get("mcd_bl_vs_tgt") is not None]
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n_valid": n, "mean": m, "half": half, "tex": "%.2f \\pm %.2f" % (m, half)}


def table3_view(ev, docs=None):
    """置き場 ev の採点ファイル（docs で一部を差し替えて見ることもできる）から、Table 3 の T5Gemma-TTS の
    (a) JA・ZH と (b) EN・JA・ZH の値を作る。docs: {相対パス: JSON}（予行で「差し替えた後」を見るため）."""
    docs = docs or {}

    def doc(rel):
        return docs[rel] if rel in docs else read_json(os.path.join(ev, rel))

    out = {}
    for lang in ("ja", "zh"):
        key = "(a) %s" % lang.upper()
        if os.path.exists(os.path.join(ev, A_PIB[lang])) or A_PIB[lang] in docs:
            out[key] = {"MCD": mcd_cell(doc(A_MCD[lang])["rows"]), "PIB": pib_cell(doc(A_PIB[lang])["summary"])}
    for lang in ("en", "ja", "zh"):
        pts = {}
        for L in SWEEP_LAYERS:
            rel = sweep_pib(lang, L)
            if rel in docs or os.path.exists(os.path.join(ev, rel)):
                pts[L + 1] = pib_cell(doc(rel)["summary"])
        if len(pts) == len(SWEEP_LAYERS):
            pcts = [v["pct"] for v in pts.values()]
            out["(b) %s" % lang.upper()] = {"points": {str(k): v["pct"] for k, v in pts.items()},
                                            "n_valid": {str(k): v["n_valid"] for k, v in pts.items()},
                                            "range": "%.2f--%.2f" % (min(pcts), max(pcts))}
    return out


def view_lines(before, after, title):
    """Table 3 の前後を人が読む行にする."""
    lines = ["[before -> after] %s" % title]
    for key in ("(a) JA", "(a) ZH"):
        if key in before or key in after:
            b, a = before.get(key, {}), after.get(key, {})
            for m in ("MCD", "PIB"):
                bt = (b.get(m) or {}).get("tex", "n/a")
                at = (a.get(m) or {}).get("tex", "n/a")
                bn = (b.get(m) or {}).get("n_valid")
                an = (a.get(m) or {}).get("n_valid")
                extra = ""
                if m == "PIB" and a.get(m):
                    extra = "  (%d/%d)" % (a[m]["k"], a[m]["n_valid"])
                lines.append("  %-7s %-4s %-16s -> %-16s n %s -> %s%s%s" % (
                    key, m, bt, at, bn, an, extra, "" if bt == at else "   CHANGED"))
    for key in ("(b) EN", "(b) ZH", "(b) JA"):
        if key in before or key in after:
            b, a = before.get(key, {}), after.get(key, {})
            lines.append("  %-7s range %s -> %s%s" % (key, b.get("range", "n/a"), a.get("range", "n/a"),
                                                       "" if b.get("range") == a.get("range") else "   CHANGED"))
            for ls in sorted(set(b.get("points", {})) | set(a.get("points", {})), key=int):
                bp, ap = b.get("points", {}).get(ls), a.get("points", {}).get(ls)
                lines.append("      l*=%-2s %6s -> %6s  (n %s -> %s)%s" % (
                    ls, bp, ap, b.get("n_valid", {}).get(ls), a.get("n_valid", {}).get(ls),
                    "" if bp == ap else "   CHANGED"))
    return lines


def summaries_equal(a, b):
    """summary の比較（float は相対 1e-9。p 値の最後の桁のずれを許す。merge_regen2_scores._same と同じ考え方）."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(summaries_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def jsonable(obj):
    """JSON に書いて読み戻したときと同じ形（記録との比較に使う）."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- 反映の段階と、その段階で表がとるはずの値
APPLY_STEPS = ("b12", "regen_apply")      # eval_results/_t3fix/<段階>/apply_record.json
# B1・B2 を反映した後の (a)（監査 eval_results/_audit_spanB_20261002/ の値。t3fix_b12_apply.AUDIT_EXPECT と同じ）
POST_B12 = {"(a) JA": {"MCD": "8.24 \\pm 0.21", "PIB": "52.00 \\pm 5.62"},
            "(a) ZH": {"MCD": "8.95 \\pm 0.25", "PIB": "54.67 \\pm 5.60"}}
REGEN_DEPENDENT = ("(a) ZH", "(b) EN", "(b) ZH")   # 作り直しの反映で、新しい音声しだいで変わるセル


def apply_paths(base):
    return {"journal": os.path.join(base, "apply_journal.json"), "record": os.path.join(base, "apply_record.json"),
            "backup": os.path.join(base, "backup_before_apply"), "tmp": os.path.join(base, "tmp")}


def apply_state(ev):
    """各段階が 反映済み（記録あり）/ 途中で止まった（帳面か退避があり、記録が無い）/ まだ、のどれか."""
    st = {}
    for step in APPLY_STEPS:
        jp = apply_paths(os.path.join(out_root(ev), step))
        applied = os.path.exists(jp["record"])
        st[step] = {"applied": applied, "record": jp["record"],
                    "interrupted": (not applied) and (os.path.exists(jp["journal"]) or os.path.exists(jp["backup"]))}
    return st


def expected_table3(ev):
    """今の置き場の反映の段階から、Table 3 の T5Gemma-TTS のセルがとるはずの値を決める.
      * 反映なし: 原稿の今の値（MANUSCRIPT_NOW）。
      * B1・B2 を反映: (a) JA は POST_B12（作り直しは JA に触れない）。(a) ZH は、作り直しをまだ反映していなければ
        POST_B12。
      * 作り直しを反映: (a) ZH と (b) EN・ZH は新しい音声しだいなので、最後に書かれた反映の記録の after
        （書いた後のファイルから計算した値。どちらの段階の記録でも、その時点の全セルを持つ）と照合する。
      * (b) JA はどの段階でも原稿の値。
    2 つの段階が両方済んでいれば、先の記録の after と後の記録の before が同じことも確かめる（間に別の変更が無い）。
    返り値: (期待値 {セル: {"kind", "want", "source"}}, 反映の順, 問題の並び)."""
    st = apply_state(ev)
    recs = []
    for step in APPLY_STEPS:
        if st[step]["applied"]:
            p = st[step]["record"]
            r = read_json(p)
            recs.append((str(r.get("date", "")), os.path.getmtime(p), step, r))
    recs.sort(key=lambda x: (x[0], x[1]))
    exp = {}
    for k, v in MANUSCRIPT_NOW.items():
        if "range" in v:
            exp[k] = {"kind": "range", "want": v["range"], "source": "manuscript (naacl2027/main.tex %d)" % v["line"]}
        else:
            exp[k] = {"kind": "tex", "want": [v["MCD"], v["PIB"]],
                      "source": "manuscript (naacl2027/main.tex %d)" % v["line"]}
    b12, rg = st["b12"]["applied"], st["regen_apply"]["applied"]
    if b12:
        exp["(a) JA"] = {"kind": "tex", "want": [POST_B12["(a) JA"]["MCD"], POST_B12["(a) JA"]["PIB"]],
                         "source": "after B1 (audit _audit_spanB_20261002)"}
        if not rg:
            exp["(a) ZH"] = {"kind": "tex", "want": [POST_B12["(a) ZH"]["MCD"], POST_B12["(a) ZH"]["PIB"]],
                             "source": "after B2 (audit _audit_spanB_20261002)"}
    if rg:
        last_step, last = recs[-1][2], recs[-1][3]
        for k in REGEN_DEPENDENT:
            exp[k] = {"kind": "cell", "want": (last.get("after") or {}).get(k),
                      "source": "after of _t3fix/%s/apply_record.json (latest apply)" % last_step}
    issues = []
    for step, s in st.items():
        if s["interrupted"]:
            issues.append("%s: 反映が途中で止まっている（帳面か退避があり、記録が無い）。同じ --apply で続きから流す" % step)
    if len(recs) == 2:
        first, second = recs[0][3], recs[1][3]
        fa, sb = first.get("after") or {}, second.get("before") or {}
        bad = sorted(k for k in set(fa) & set(sb) if fa[k] != sb[k])
        if bad:
            issues.append("反映の記録がつながらない（%s の after と %s の before が違う）: %s" % (recs[0][2], recs[1][2], bad))
    return exp, [r[2] for r in recs], issues


def formula_selftest(ev):
    """(1) regen2_common の pib_summary / mcd_summary が今の採点ファイルの summary を再現するか、
    (2) mcd_cell・pib_cell で計算した Table 3 の値が、今の反映の段階でとるはずの値（expected_table3）と同じか、
    (3) 反映が途中で止まっていないか。読むだけ。原稿の今の値との一致は記録だけ（matches_manuscript_now）."""
    out = {"pib_summary": {}, "mcd_summary": {}}
    rels = [A_PIB["ja"], A_PIB["zh"]] + [sweep_pib(l, L) for l in ("en", "zh") for L in SWEEP_LAYERS]
    for rel in rels:
        d = read_json(os.path.join(ev, rel))
        out["pib_summary"][rel] = summaries_equal(RC.pib_summary(d["rows"], d["summary"]), d["summary"])
    for rel in (A_MCD["ja"], A_MCD["zh"]):
        d = read_json(os.path.join(ev, rel))
        out["mcd_summary"][rel] = summaries_equal(RC.mcd_summary(d["rows"], d["summary"]), d["summary"])
    v = jsonable(table3_view(ev))
    exp, order, issues = expected_table3(ev)
    out["apply_state"] = {k: {"applied": s["applied"], "interrupted": s["interrupted"]}
                          for k, s in apply_state(ev).items()}
    out["apply_order"] = order
    out["state_issues"] = issues
    out["expected"] = {}
    for k, e in exp.items():
        got = v.get(k)
        if e["kind"] == "tex":
            g = [got["MCD"]["tex"], got["PIB"]["tex"]] if got else None
        elif e["kind"] == "range":
            g = got["range"] if got else None
        else:
            g = got
        out["expected"][k] = {"source": e["source"], "want": e["want"], "got": g,
                              "ok": e["want"] is not None and g == e["want"]}
    out["values_reproduced"] = {k: x["ok"] for k, x in out["expected"].items()}
    out["matches_manuscript_now"] = {   # 記録だけ（反映の後は違うのが正しい）
        k: (([v[k]["MCD"]["tex"], v[k]["PIB"]["tex"]] == [m["MCD"], m["PIB"]]) if "range" not in m
            else v[k]["range"] == m["range"]) if k in v else None
        for k, m in MANUSCRIPT_NOW.items()}
    out["ok"] = (all(out["pib_summary"].values()) and all(out["mcd_summary"].values())
                 and all(out["values_reproduced"].values()) and not issues)
    return out


def formula_fail_reasons(fst):
    """formula_selftest の失敗を人が読む短い並びにする."""
    r = ["pib_summary が再現しない: %s" % k for k, ok in fst.get("pib_summary", {}).items() if not ok]
    r += ["mcd_summary が再現しない: %s" % k for k, ok in fst.get("mcd_summary", {}).items() if not ok]
    for k, x in (fst.get("expected") or {}).items():
        if not x["ok"]:
            r.append("%s: 今 %s / 期待 %s（%s）" % (k, x["got"], x["want"], x["source"]))
    return r + list(fst.get("state_issues") or [])


# ---------------------------------------------------------------- 反映の帳面（途中で止まっても同じ --apply で続きから）
def build_journal(ev, base, ops, extra):
    """反映の前に、操作の並びと、書き換える前の中身（md5・inode）・退避の置き場を固定する.
    ops: [{"kind": "wav"|"doc", "dest": 本番のパス, "src": 新しい中身のファイル（_t3fix の下）, ...}]。
    doc の src は、入れ替え後の採点ファイルを work の下に書いたもの（反映ではこの中身をそのまま写す）."""
    jp = apply_paths(base)
    jops = []
    for i, o in enumerate(ops):
        dest, src = os.path.abspath(o["dest"]), os.path.abspath(o["src"])
        rel = os.path.relpath(dest, os.path.abspath(ev))
        if rel.startswith(".."):
            raise SystemExit("[t3fix] 書き込み先が置き場の外: %s" % dest)
        st = stat_info(dest)
        sm = md5_file(src)
        if o.get("src_md5") is not None and o["src_md5"] != sm:
            raise SystemExit("[t3fix] 差し替えに使う中身の md5 が記録と違う: %s" % src)
        if sm == st["md5"]:
            raise SystemExit("[t3fix] 差し替えの前後で中身が同じ（済んだ操作と区別できない）: %s" % dest)
        jops.append(dict(o, i=i, dest=dest, src=src, src_md5=sm, orig_md5=st["md5"], orig_ino=st["ino"],
                         orig_nlink=st["nlink"], backup=os.path.join(jp["backup"], rel)))
    dests = [o["dest"] for o in jops]
    if len(set(dests)) != len(dests):
        raise SystemExit("[t3fix] 同じ書き込み先が 2 回ある")
    return dict(extra, journal_created=now(), ev=os.path.abspath(ev), ops=jops, planned_dests=dests,
                backup=jp["backup"])


def journal_states(j):
    """各操作の今の状態: todo（元の中身のまま）/ done（新しい中身）/ unknown（どちらでもない）."""
    out = []
    for o in j["ops"]:
        cur = md5_file(o["dest"]) if os.path.exists(o["dest"]) else None
        out.append("todo" if cur == o["orig_md5"] else ("done" if cur == o["src_md5"] else "unknown"))
    return out


def run_journal(j, ev, base, after_op=None, stop_after=None):
    """帳面の操作を行う。何度流しても同じ結果になる（済んだ操作は md5 で見分けて飛ばす）。
      1. 退避: 無いものだけ作る（一時ファイル → 名前を付ける）。あるものは md5 が元の中身と同じことを確かめる。
      2. 差し替え: 帳面の順に、todo だけを replace_with_copy（新しいファイル → os.replace。hard link を通して
         書かない）。済んだものは、新しい inode で nlink 1 であることを確かめる。
    unknown が 1 つでもあれば、何もせずに止まる。stop_after は試験用（その件数の差し替えの後で止める）."""
    jp = apply_paths(base)
    planned = set(j["planned_dests"])
    states = journal_states(j)
    unknown = [o["dest"] for o, s in zip(j["ops"], states) if s == "unknown"]
    if unknown:
        raise SystemExit("[t3fix] 元の中身でも新しい中身でもないファイルがある（帳面 %s）: %s" % (jp["journal"], unknown[:5]))
    for o, s in zip(j["ops"], states):
        if os.path.exists(o["backup"]):
            if md5_file(o["backup"]) != o["orig_md5"]:
                raise SystemExit("[t3fix] 退避の md5 が元の中身と違う: %s" % o["backup"])
        elif s == "todo":
            assert_write_ok(o["backup"], ev, planned)
            copy_new_atomic(o["dest"], o["backup"])
            if md5_file(o["backup"]) != o["orig_md5"]:
                raise SystemExit("[t3fix] 退避の md5 が元の中身と違う: %s" % o["backup"])
        else:
            raise SystemExit("[t3fix] 差し替え済みなのに退避が無い: %s" % o["dest"])
    results, n_now = [], 0
    for o in j["ops"]:
        cur = stat_info(o["dest"])
        if cur["md5"] == o["src_md5"]:
            if cur["ino"] == o["orig_ino"] or cur["nlink"] != 1:
                raise SystemExit("[t3fix] 済んだ差し替えの inode・nlink が想定と違う: %s %s" % (o["dest"], cur))
            results.append({"i": o["i"], "dest": o["dest"], "state": "done_before", "after": cur})
            continue
        if cur["md5"] != o["orig_md5"]:
            raise SystemExit("[t3fix] 書き換える前の中身が帳面と違う: %s" % o["dest"])
        if md5_file(o["src"]) != o["src_md5"]:
            raise SystemExit("[t3fix] 差し替えに使う中身が帳面と違う: %s" % o["src"])
        r = replace_with_copy(o["src"], o["dest"], jp["tmp"], ev, planned)
        if after_op is not None:
            after_op(o)
        results.append({"i": o["i"], "dest": o["dest"], "state": "done_now", "before": r["before"], "after": r["after"]})
        n_now += 1
        if stop_after and n_now >= stop_after:
            raise SystemExit("[t3fix] 試験: %d 件の差し替えの後で止めた（帳面 %s）" % (n_now, jp["journal"]))
    return results


def merge_rows(prod_doc, new_rows, pairs):
    """本番の行のうち pairs の行だけを new_rows（{pair_idx: row}）で入れ替えた行の並び."""
    rows = list(prod_doc["rows"])
    idx = {r["pair_idx"]: j for j, r in enumerate(rows)}
    for p in pairs:
        if p not in new_rows:
            raise SystemExit("[t3fix] 対 %d の新しい行が無い" % p)
        if p not in idx:
            raise SystemExit("[t3fix] 本番の採点ファイルに対 %d の行が無い" % p)
        rows[idx[p]] = new_rows[p]
    return rows


def pib_row_ok(r):
    return bool(r) and r.get("available") is True and r.get("inj_per_vs_tgt") is not None


def mcd_row_ok(r):
    return bool(r) and all(r.get(k) is not None for k in ("mcd_inj_vs_bl", "mcd_inj_vs_tgt", "mcd_bl_vs_tgt"))


def run_mcd_cpu(staging_dir, lang, out_json, log_path):
    """speechsplice_release/src/mcd_pysptk.py --mode whole を 1 行も変えずにホストの CPU で回す
    （このプロセスの affinity・nice・スレッド上限は子にそのまま引き継がれる）。{pair_idx: row} を返す."""
    import subprocess
    if IN_CONTAINER:
        raise SystemExit("[t3fix] MCD はホストの CPU で回す")
    env = dict(os.environ, REPO_ROOT=PROJECT_ROOT)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(log_path, "x") as lf:
        rc = subprocess.run([sys.executable, "-u", P(MCD_REL), "--wav_dir", staging_dir, "--lang", lang,
                             "--mode", "whole", "--n", "300", "--output_json", out_json],
                            cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(out_json):
        raise SystemExit("[t3fix] mcd_pysptk.py が失敗（rc=%s、ログ %s）" % (rc, log_path))
    return {r["pair_idx"]: r for r in read_json(out_json)["rows"] if r.get("available")}
