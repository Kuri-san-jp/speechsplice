#!/usr/bin/env python3
"""spanfix (B): 付録 B（tab:crossattn_mask）英語 300 対の作り直しの共通部品（2026-10-01）.

標準ライブラリだけを読む（numpy・torch・transformers は読まない）。
GPU 側（spanfix_appB_en_gen.py）と CPU 側（spanfix_span_check_appB_en.py, spanfix_appB_en_collect.py）が使う。

著者の決定（2026-10-01）:
  * 英語の 300 対すべてを、5/20 の本番と同じスクリプト・同じ設定で作り直す
    （scripts/projection/eval_crossattn_entity_dtw.py --lang en --cross_culture --inject_mode length_preserve
     --n 300。seed 42。.perm_cmds.txt 16714 行）。
  * 違いは実体語の範囲を求める関数だけ（scripts/encoder_injection_t5gemma.py の find_entity_token_range。
    7/11 に修正済み。今のファイルが修正後の関数を持つので、本番のスクリプトは 1 バイトも変えない）。
  * 出力は新しい置き場（eval_results/_spanfix_appB_en/）だけ。既存の結果は上書きしない。何も消さない。
  * 使えない対（無音・生成の失敗・位置合わせの失敗）は、対ごと seed 43、次に seed 44 で作り直す。
    選ぶ基準は有効性だけ（数値は見ない）。すべて記録する。最後は N=300。
  * 日本語・中国語の行は変えない。naacl2027/ ほか原稿の置き場には書かない。

状態（1 回の実行の中の 1 対の結果。G3 = scripts/regen2/g3_common.py と同じ意味）:
  valid               位置合わせ成功（5 本すべてで Whisper が実体語を見つけた）、5 本の frames、5 本の wav が
                      決まった順で写し取れ、どれも無音でなく、実行時の範囲が修正後の関数の範囲と一致
  -- 対として使えない（PAIR_INVALID）: 次の seed に進む --
  unaligned           Whisper が実体語を見つけられなかった（Whisper の例外は無い）
  generation_failed   生成が音声を返さなかった（本番 main() は行を作らない）
  silent              位置合わせは成功したが、5 本のどれかが無音（scripts/regen2/silence_criterion.py）
  -- 対の良し悪しではない（RETRY_SAME_SEED）: 同じ seed をもう一度流す --
  run_error           対の処理中の例外、main() の例外、T5Gemma の層が CPU/disk に置かれた
  alignment_exception Whisper の呼び出しが例外で落ちた（同じ seed で 2 回続いたら著者の判断）
  capture_error       行はあるが wav の写し取りが確かめに通らない（順序・本数）
  no_row_unexplained  実行は最後まで終わったのに、行も例外も None も無い
  crashed_no_record   実行が途中で止まり（時間切れなど）、この対の行が保存される前だった
  -- 止まって著者の判断 --
  span_mismatch       実行時の範囲が、事前に求めた修正後の関数の範囲と違う
  checkpoint_unreadable  本番 main() の checkpoint.json が読めない
"""
import ast
import glob
import hashlib
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------- 置き場
HOST_ROOT = os.environ.get("REPO_ROOT", "/mnt")
CONTAINER_ROOT = "/mnt"
IN_CONTAINER = os.path.isdir("/mnt/scripts")
PROJECT_ROOT = CONTAINER_ROOT if IN_CONTAINER else HOST_ROOT

EVAL_REL = "scripts/projection/eval_crossattn_entity_dtw.py"
INJ_REL = "scripts/encoder_injection_t5gemma.py"
FA_REL = "scripts/projection/forced_alignment_utils.py"
SHIM_REL = "scripts/nontemplate/gen_t5gemma_pairs_module.py"
SILENCE_REL = "scripts/regen2/silence_criterion.py"
# 5/20 の本番の時点の範囲の関数（7/11 の修正の前の論理）。9/17 に release_artifacts を直す前に取った控え。
# 今の INJ_REL の後半（legacy logic, fallback）と同じ論理（diff で確かめた: 違いは T5GEMMA_ROOT と範囲の関数の前半だけ）。
LEGACY_SPAN_REL = "release_artifacts/scripts/encoder_injection_t5gemma.py.bak_spanfix_20260917_095951"
GEN_REL = "scripts/spanfix/spanfix_appB_en_gen.py"
COMMON_REL = "scripts/spanfix/spanfix_common.py"
DATA_REL = "data"
PAIRS_MODULE = "en_cross_culture_names"

# 本番のファイル（変えない）。GPU の前と取り込みの時に、この md5 を求める（G3 の PINNED_MD5 と同じ値）
PINNED_MD5 = {
    EVAL_REL: "c20083b9142439e6a46f88f3f0b2cfe4",  # 5/20 11:53 から変わっていない
    INJ_REL: "4f36fa570145cde805072438d8a39bb1",   # 範囲の関数の修正後（7/11）。9/27 の控えとも一致
    FA_REL: "be2b300a061e1374509bd1aa75f41b5d",    # Whisper の整列
    SHIM_REL: "53eb4bfc1008beb361a23c6adfc429ba",  # HF の offline の読み替え（9/28 の前例）
}
LEGACY_SPAN_MD5 = "a7aca46875c8b405a89bc3dbfbabddb4"
OWN_CODE = (GEN_REL, COMMON_REL)  # 記録の md5 が取り込みの時点と違えば著者の判断（G3 と同じ扱い）
CODE_FILES = tuple(PINNED_MD5) + (SILENCE_REL, LEGACY_SPAN_REL) + OWN_CODE

# 5/20 の本番（読むだけ）と、今の表の出どころ（9/29 に対 237 を入れ替えたもの。読むだけ）
PROD_DIR_REL = "eval_results/crossattn_entity_dtw_encc_lp_n300"
PROD_MD5 = {"results.json": "f32dd293df04abf4a99ca5cf83ade871",
            "checkpoint.json": "84bdf01ddd81b71a50bb7101f3d77793"}
CUR_TABLE_SRC_REL = "eval_results/_regen2_crossattn_merged/encc_lp_n300/results.json"

OUT_REL = "eval_results/_spanfix_appB_en"
DRYRUN_REL = OUT_REL + "/dryrun"
LOGS_REL = OUT_REL + "/logs"

TOKENIZER_ID = "google/t5gemma-2b-2b-ul2"   # Aratako/T5Gemma-TTS-2b-2b の config.text_tokenizer_name
T5GEMMA_TTS_ID = "Aratako/T5Gemma-TTS-2b-2b"
CONTAINER_HF_HOME = "/mnt/.cache/huggingface"

N_PAIRS = 300
CONDS = ("bl", "inj", "bl_mask", "inj_mask", "tgt")
SEEDS = (42, 43, 44)
PRODUCER = "spanfix_appB_en_gen.py"
MODE_GPU = "gpu"
MODE_FAKE = "FAKE_selftest"
# seed 42 は 4 つに分けて流す（1 つ約 75 分。SLURM_TIME=02:00:00 に収まる）。対の番号は 0 始まり
PARTS = {1: (0, 75), 2: (75, 150), 3: (150, 225), 4: (225, 300)}


def prod_args():
    """本番の引数（.perm_cmds.txt 16714 行）。この後に --output_dir <新しい置き場> だけを足す."""
    return ["--lang", "en", "--cross_culture", "--inject_mode", "length_preserve", "--n", str(N_PAIRS)]


def part_pairs(k):
    a, b = PARTS[k]
    return list(range(a, b))


# ---------------------------------------------------------------- 環境変数（numpy・torch・HF を読む前に呼ぶ）
THREADS = 13
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TORCH_HOME", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR")


def cache_env():
    r = PROJECT_ROOT
    return {"HF_HOME": r + "/.cache/huggingface", "HF_HUB_CACHE": r + "/.cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1", "TORCH_HOME": r + "/.cache/torch", "XDG_CACHE_HOME": r + "/.cache",
            "NUMBA_CACHE_DIR": r + "/.cache/numba"}


def setup_env():
    """スレッド上限 13、HF とキャッシュの置き場、ホストでは GPU を見せない。起動したときの値を返す.

    setdefault ではなく上書き（slurm_run.sh は HF_HOME=/tmp/.cache/huggingface を渡すので）。
    """
    launch = {k: os.environ.get(k) for k in THREAD_VARS + CACHE_VARS + ("CUDA_VISIBLE_DEVICES",)}
    for k in THREAD_VARS:
        os.environ[k] = str(THREADS)
    for k, v in cache_env().items():
        os.environ[k] = v
    if not IN_CONTAINER:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ホスト（CPU だけ）では GPU に触れない
    sys.dont_write_bytecode = True  # 既存の scripts/ 以下に .pyc を新しく作らない
    return launch


# ---------------------------------------------------------------- 小さな道具
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


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def current_code_md5(root=None):
    return {rel: (md5_file(P(rel, root)) if os.path.isfile(P(rel, root)) else None) for rel in CODE_FILES}


def pinned_file_errors(root=None):
    errs = []
    for rel, want in PINNED_MD5.items():
        got = md5_file(P(rel, root))
        if got != want:
            errs.append(f"{rel} の md5 が固定の値と違う（今 {got} / 固定 {want}）")
    got = md5_file(P(LEGACY_SPAN_REL, root))
    if got != LEGACY_SPAN_MD5:
        errs.append(f"{LEGACY_SPAN_REL} の md5 が違う（今 {got} / 固定 {LEGACY_SPAN_MD5}）")
    return errs


def load_function_from_source(path, name):
    """ファイルを import せず（副作用なし）、関数定義だけを取り出して exec する."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f"{name} が {path} に 1 つだけあるはず（{len(nodes)}）"
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns[name]


def span_functions(root=None):
    """(修正後の関数, 5/20 の時点の関数)."""
    return (load_function_from_source(P(INJ_REL, root), "find_entity_token_range"),
            load_function_from_source(P(LEGACY_SPAN_REL, root), "find_entity_token_range"))


def load_pairs(root=None):
    """eval_crossattn_entity_dtw.py の main() と同じ組み立て（cross_culture, [:300]）."""
    d = P(DATA_REL, root)
    if d not in sys.path:
        sys.path.insert(0, d)
    mod = __import__(PAIRS_MODULE)
    cc = mod.build_300_pairs()[:N_PAIRS]
    pairs = [{"text_a": q["text_a"], "text_b": q["text_b"], "word_a": q["word_a"], "word_b": q["word_b"],
              "pair_idx": q["pair_idx"]} for q in cc]
    assert len(pairs) == N_PAIRS and [q["pair_idx"] for q in pairs] == list(range(N_PAIRS))
    # 文は対ごとに一意（写し取りで呼び出しを対に結びつけるのに使う）、実体語は文に 1 回だけ
    assert len({q["text_a"] for q in pairs}) == N_PAIRS and len({q["text_b"] for q in pairs}) == N_PAIRS
    for q in pairs:
        assert q["text_a"].count(q["word_a"]) == 1 and q["text_b"].count(q["word_b"]) == 1, q
    return pairs


def name_position(pair):
    return "sentence_initial" if pair["text_a"].startswith(pair["word_a"]) else "mid_sentence"


# ---------------------------------------------------------------- tokenizer（HF offline の読み替えは 9/28 の前例）
def load_shim_module(root=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_spanfix_offline_shim", P(SHIM_REL, root))
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    return W


def setup_offline(W):
    """HF_HUB_OFFLINE=1 で、生成の経路で読む repo が手元にあることを確かめ、tokenizer の読み替えを入れる."""
    info = {"hf_offline": bool(W.hf_offline()), "offline_preflight_ok": None, "offline_preflight": None,
            "offline_tokenizer_shim": []}
    if not info["hf_offline"]:
        raise SystemExit("[spanfix] HF_HUB_OFFLINE=1 で流す")
    ok, pre = W.offline_preflight()
    info["offline_preflight_ok"] = bool(ok)
    info["offline_preflight"] = pre
    if not ok:
        raise SystemExit(f"[spanfix] 生成の経路で読む repo が手元のキャッシュに無い: {pre}")
    cfg = read_json(os.path.join(pre[T5GEMMA_TTS_ID]["snapshot"], "config.json"))
    tok_name = cfg.get("text_tokenizer_name") or cfg.get("t5gemma_model_name")
    info["text_tokenizer_name_in_config"] = tok_name
    if tok_name != TOKENIZER_ID:
        raise SystemExit(f"[spanfix] T5Gemma-TTS の config の tokenizer が {tok_name}（{TOKENIZER_ID} のはず）")
    W.install_offline_tokenizer_shim(info["offline_tokenizer_shim"])
    return info


# ---------------------------------------------------------------- 範囲の確かめ（tokenizer だけ）
def independent_span(tok, text, word):
    """修正後の関数とは別の求め方: 実体語を（前に空白を付けたものも）符号化して、文の符号の中で探す."""
    full = tok.encode(text, add_special_tokens=False)
    for cand in (word, " " + word):
        et = tok.encode(cand, add_special_tokens=False)
        for s in range(len(full) - len(et) + 1):
            if full[s:s + len(et)] == et:
                return [s, s + len(et)]
    return None


def span_points_at_name(tok, text, word, span):
    """範囲 [s, e) の token が、文の中の実体語の文字だけを覆い、実体語の文字をすべて覆うか.

    文字の位置は tokenizer の offset mapping（tokenizer の出す文字位置）で見る。範囲の前後の token が
    実体語の文字に掛かっていないことも確かめる。復号した文字列（前後の空白を除く）が実体語と同じことも求める。
    """
    out = {"ok": False}
    if span is None:
        out["why"] = "範囲が None"
        return out
    s, e = span
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = text.find(word)
    end = pos + len(word)
    out.update({"char_range": [pos, end], "n_tokens_text": len(ids)})
    if not (0 <= s < e <= len(ids)) or pos < 0:
        out["why"] = f"範囲が文の外 ({s},{e}) / {len(ids)} token"
        return out
    covered = set()
    outside = []
    for i in range(s, e):
        a, b = offs[i]
        for c in range(a, b):
            if pos <= c < end:
                covered.add(c)
            elif not text[c].isspace():
                outside.append(c)
    neighbors = [i for i in (s - 1, e) if 0 <= i < len(ids) and offs[i][0] < end and offs[i][1] > pos]
    decoded = tok.decode(ids[s:e]).strip()
    out.update({"covers_all_chars": len(covered) == len(word), "chars_outside_name": len(outside),
                "neighbor_overlaps_name": neighbors, "decoded": decoded,
                "pieces": tok.convert_ids_to_tokens(ids[s:e])})
    out["ok"] = bool(out["covers_all_chars"] and not outside and not neighbors and decoded == word)
    if not out["ok"]:
        out["why"] = "実体語を指していない"
    return out


def span_table(tok, pairs, fixed_fn, legacy_fn=None):
    """対ごとの範囲（修正後・5/20 の時点・別の求め方）と、実体語を指しているか."""
    rows = {}
    for p in pairs:
        r = {"pair_idx": p["pair_idx"], "position": name_position(p)}
        for side, text, word in (("a", p["text_a"], p["word_a"]), ("b", p["text_b"], p["word_b"])):
            fx = fixed_fn(tok, text, word, add_bos=0)
            fx = list(fx) if fx else None
            ind = independent_span(tok, text, word)
            d = {"text": text, "word": word, "fixed": fx, "independent": ind,
                 "fixed_equals_independent": fx is not None and fx == ind,
                 "fixed_check": span_points_at_name(tok, text, word, fx)}
            if legacy_fn is not None:
                lg = legacy_fn(tok, text, word, add_bos=0)
                lg = list(lg) if lg else None
                d["legacy_0520"] = lg
                d["legacy_check"] = span_points_at_name(tok, text, word, lg)
                d["changed_vs_0520"] = lg != fx
            r[side] = d
        r["fixed_ok"] = bool(r["a"]["fixed_check"]["ok"] and r["b"]["fixed_check"]["ok"]
                             and r["a"]["fixed_equals_independent"] and r["b"]["fixed_equals_independent"])
        if legacy_fn is not None:
            r["changed_vs_0520"] = bool(r["a"]["changed_vs_0520"] or r["b"]["changed_vs_0520"])
            r["legacy_pointed_at_name"] = bool(r["a"]["legacy_check"]["ok"] and r["b"]["legacy_check"]["ok"])
        rows[p["pair_idx"]] = r
    return rows


# ---------------------------------------------------------------- 状態
STATUS_VALID = "valid"
STATUS_ALIGN_EXC = "alignment_exception"
CRASHED_NO_RECORD = "crashed_no_record"
PAIR_INVALID = ("unaligned", "generation_failed", "silent")
RETRY_SAME_SEED = ("run_error", STATUS_ALIGN_EXC, "capture_error", "no_row_unexplained", CRASHED_NO_RECORD)
STOP_STATUSES = ("span_mismatch", "checkpoint_unreadable")
ALL_STATUSES = (STATUS_VALID,) + PAIR_INVALID + RETRY_SAME_SEED + STOP_STATUSES


def run_dir_rel(seed, tag, stamp, uniq):
    return f"{OUT_REL}/seed{seed}/{tag}_{stamp}_{uniq}"


def list_runs(data_root, seed):
    """seed の実行の置き場（run_record.json があるもの）を、始めた時刻の順に返す。無いものは警告に."""
    base = os.path.join(data_root, OUT_REL, f"seed{seed}")
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


def read_events(path):
    ev, bad = [], 0
    if not os.path.isfile(path):
        return ev, bad
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except Exception:  # noqa: BLE001  時間切れで書きかけの最後の行
                bad += 1
    return ev, bad


def load_run(data_root, rel, rec=None):
    """1 回の実行の記録をすべて読む（書かない）."""
    d = os.path.join(data_root, rel)
    rec = rec if rec is not None else read_json(os.path.join(d, "run_record.json"))
    run = {"rel": rel, "record": rec, "seed": rec.get("seed"), "todo": list(rec.get("todo") or []),
           "errors": [], "rows": {}, "ckpt_ok": True, "captures": {}, "pair_errors": {}, "whisper_errors": {},
           "other_events": [], "end": None}
    ck = os.path.join(d, "checkpoint.json")
    try:
        rows = read_json(ck).get("rows", [])
        run["rows"] = {r["pair_idx"]: r for r in rows if not r.get("placeholder")}
        ph = sorted(r["pair_idx"] for r in rows if r.get("placeholder"))
        want_ph = sorted(set(range(N_PAIRS)) - set(run["todo"]))
        if ph != want_ph:
            run["errors"].append("checkpoint.json の placeholder が todo 以外の対になっていない")
        extra = sorted(set(run["rows"]) - set(run["todo"]))
        if extra:
            run["errors"].append(f"todo に無い対の行がある: {extra[:5]}")
    except Exception as e:  # noqa: BLE001
        run["ckpt_ok"] = False
        run["errors"].append(f"checkpoint.json が読めない: {type(e).__name__}")
    rs = os.path.join(d, "results.json")
    if os.path.isfile(rs) and run["ckpt_ok"]:
        try:
            rrows = {r["pair_idx"]: r for r in read_json(rs).get("rows", []) if not r.get("placeholder")}
            if rrows != run["rows"]:
                run["errors"].append("results.json の行が checkpoint.json の行と違う")
        except Exception as e:  # noqa: BLE001
            run["errors"].append(f"results.json が読めない: {type(e).__name__}")
    for p in glob.glob(os.path.join(d, "capture", "p*.json")):
        try:
            c = read_json(p)
            run["captures"][c["pair_idx"]] = c
        except Exception as e:  # noqa: BLE001
            run["errors"].append(f"{os.path.relpath(p, data_root)} が読めない: {type(e).__name__}")
    ev, bad = read_events(os.path.join(d, "events.jsonl"))
    if bad:
        run["other_events"].append({"type": "unparsable_event_lines", "n": bad})
    for e in ev:
        t, i = e.get("type"), e.get("pair_idx")
        if t == "pair_error" and i is not None:
            run["pair_errors"].setdefault(i, []).append(e.get("msg"))
        elif t == "whisper_error" and i is not None:
            run["whisper_errors"].setdefault(i, []).append(e.get("msg"))
        elif t not in ("pair_done",):
            run["other_events"].append(e)
    endp = os.path.join(d, "run_end.json")
    if os.path.isfile(endp):
        try:
            run["end"] = read_json(endp)
        except Exception as e:  # noqa: BLE001
            run["errors"].append(f"run_end.json が読めない: {type(e).__name__}")
    return run


def pair_status(run, idx):
    """実行 run の中の対 idx の状態（上の表）。数値（MCD・類似度）は見ない."""
    if idx not in run["todo"]:
        return None, {}
    row = run["rows"].get(idx)
    cap = run["captures"].get(idx) or {}
    pe = run["pair_errors"].get(idx, [])
    we = run["whisper_errors"].get(idx, [])
    end = run["end"]
    info = {"row_present": row is not None, "aligned": row.get("aligned") if row else None,
            "frames": [row.get(f"frames_{c}") for c in CONDS] if row else None,
            "pair_errors": pe, "whisper_errors": we, "capture_complete": cap.get("complete"),
            "capture_ok": cap.get("capture_ok"), "span_match": cap.get("span_match"),
            "any_silent": cap.get("any_silent"), "any_returned_none": cap.get("any_returned_none"),
            "run_finished": end is not None}
    if not run["ckpt_ok"]:
        return "checkpoint_unreadable", info
    if end is not None and (end.get("main_error") or end.get("device_offload")):
        info["run_level_error"] = {"main_error": end.get("main_error"), "device_offload": end.get("device_offload")}
        return "run_error", info
    if cap and cap.get("span_match") is False:
        return "span_mismatch", info
    if pe:
        return "run_error", info
    if we:
        return STATUS_ALIGN_EXC, info
    if row is None:
        if cap.get("complete") and cap.get("any_returned_none"):
            return "generation_failed", info
        return ("no_row_unexplained" if end is not None else CRASHED_NO_RECORD), info
    if row.get("aligned") is not True:
        return "unaligned", info
    if any(row.get(f"frames_{c}") is None for c in CONDS):
        return "run_error", info
    if not cap or not cap.get("capture_ok") or len(cap.get("wavs") or {}) != len(CONDS):
        return "capture_error", info
    if cap.get("any_silent"):
        return "silent", info
    return STATUS_VALID, info


# ---------------------------------------------------------------- 出どころ
def run_identity_errors(run, seed):
    rec = run["record"]
    errs = []
    if rec.get("producer") != PRODUCER:
        errs.append(f"producer が {PRODUCER} でない: {rec.get('producer')!r}")
    if rec.get("seed") != seed:
        errs.append(f"seed が置き場（seed{seed}）と違う: {rec.get('seed')}")
    if rec.get("run_dir") != run["rel"]:
        errs.append(f"run_dir が置き場と違う: {rec.get('run_dir')!r} != {run['rel']!r}")
    return errs


def run_provenance_errors(run, seed, code_root=None, fake_ok=False):
    """判断に使う実行（valid の取り込み・次の seed の根拠）が、本物の GPU の実行で決まりどおりかを確かめる.

    OWN_CODE（このラッパーと共通部品）の md5 の違いは別に返す（著者の判断。G3 と同じ扱い）。
    """
    rec = run["record"]
    errs = run_identity_errors(run, seed) + list(run["errors"])
    env = rec.get("env") or {}
    if not fake_ok:
        if rec.get("mode") != MODE_GPU:
            errs.append(f"mode が {MODE_GPU} でない: {rec.get('mode')!r}")
        if env.get("project_root") != CONTAINER_ROOT:
            errs.append(f"コンテナ（/mnt）の中で作られていない: {env.get('project_root')!r}")
        if not env.get("cuda_device"):
            errs.append("GPU の記録が無い（cuda_device）")
        if env.get("HF_HUB_OFFLINE") != "1" or env.get("HF_HOME") != CONTAINER_HF_HOME:
            errs.append(f"HF の環境が決まりと違う: HF_HOME={env.get('HF_HOME')!r} HF_HUB_OFFLINE={env.get('HF_HUB_OFFLINE')!r}")
        if env.get("offline_preflight_ok") is not True:
            errs.append("offline_preflight が通っていない")
        if not any(s.get("repo_id") == TOKENIZER_ID and s.get("snapshot") for s in env.get("offline_tokenizer_shim") or []):
            errs.append(f"{TOKENIZER_ID} を手元の snapshot から読んだ記録が無い")
        want_argv = prod_args() + ["--output_dir", f"{CONTAINER_ROOT}/{run['rel']}"]
        if rec.get("main_argv") != want_argv:
            errs.append(f"main_argv が本番の引数と違う: {rec.get('main_argv')}")
    pt = rec.get("print_taps") or {}
    if pt.get("eval_module_tapped") is not True or pt.get("forced_alignment_utils_tapped") is not True:
        errs.append("本番 main() と forced_alignment_utils の print を記録した証拠が無い")
    if rec.get("span_tap_installed") is not True:
        errs.append("実行時の範囲を記録する差し替えが入っていない")
    sp = rec.get("span_preflight") or {}
    if sorted(int(k) for k in sp) != sorted(run["todo"]) or not all(v.get("fixed_ok") for v in sp.values()):
        errs.append("span_preflight が todo のすべての対で「実体語を指す」になっていない")
    cm = rec.get("code_md5") or {}
    for rel, want in PINNED_MD5.items():
        if cm.get(rel) != want:
            errs.append(f"本番のファイル {rel} の md5 が固定の値と違う（記録 {cm.get(rel)} / 固定 {want}）")
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


# ---------------------------------------------------------------- seed の決まり
def all_runs(data_root):
    runs, warns = {}, []
    for s in SEEDS:
        lst, w = list_runs(data_root, s)
        warns += w
        runs[s] = [load_run(data_root, rel, rec) for rel, rec in lst]
    return runs, warns


def chain(runs_s, idx):
    """seed の中で対 idx を扱った実行の連なり: [(run, status, info)]（始めた順）."""
    out = []
    for run in runs_s:
        st, info = pair_status(run, idx)
        if st is not None:
            out.append((run, st, info))
    return out


def _chain_errors(ch, seed):
    errs = []
    for k, (run, st, _) in enumerate(ch[:-1]):
        if st not in RETRY_SAME_SEED:
            errs.append(f"seed {seed}: {run['rel']} の状態 {st} の後に別の実行がある（結果で選ぶことになるので使わない）")
    return errs


def _repeated_align_exc(ch):
    sts = [st for _, st, _ in ch]
    return len(sts) >= 2 and sts[-1] == STATUS_ALIGN_EXC and sts[-2] == STATUS_ALIGN_EXC


def next_action(runs, idx, code_root=None, check_provenance=True):
    """対 idx について、今の記録から次にすることを 1 つ返す.

    action: run_seed42 / rerun_seed{S} / confirm_stopped_seed{S} / use_seed{S} / run_seed43 / run_seed44 /
            stop_author_decision / stop_error
    """
    out = {"pair_idx": idx, "errors": [], "use": None, "chain": {}}
    chains = {s: chain(runs[s], idx) for s in SEEDS}
    for s in SEEDS:
        out["chain"][str(s)] = [{"run": r["rel"], "status": st,
                                 **{k: v for k, v in info.items() if k in ("aligned", "frames", "any_silent",
                                                                           "pair_errors", "whisper_errors")}}
                                for r, st, info in chains[s]]
    decided_at = SEEDS[-1]
    for s in SEEDS:
        ch = chains[s]
        decided_at = s
        if not ch:
            out["action"] = f"run_seed{s}"
            break
        out["errors"] += _chain_errors(ch, s)
        run, st, info = ch[-1]
        if st in RETRY_SAME_SEED:
            out["errors"] += run_identity_errors(run, s)
            if _repeated_align_exc(ch):
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
            out["errors"] += run_provenance_errors(run, s, code_root)
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
        # 対として使えない: 次の seed へ
    later = [s for s in SEEDS if s > decided_at and chains[s]]
    if later:
        out["errors"].append(f"seed {later} の実行があるが、seed {decided_at} で対として使えないと決まっていない")
    if out["errors"]:
        out["action"] = "stop_error"
    elif out.get("code_drift"):
        out["action"] = "stop_author_decision"
        out["note"] = "このラッパーか共通部品の md5 が GPU の実行の後に変わった（著者の判断）"
    return out


def allowed_todo(runs, seed, candidates, confirm_stopped=(), code_root=None, check_provenance=True):
    """seed で流してよい対（candidates の中）と、流さない理由.

    seed 42: まだ流していない対、または最後の回が対の良し悪しでない（RETRY_SAME_SEED）対。
    seed 43 / 44: 前の seed の最後の回が「対として使えない」で、この seed をまだ流していないか、最後の回が RETRY_SAME_SEED。
    最後の回が終わっていない実行（crashed_no_record）なら、その置き場が confirm_stopped にあるときだけ。
    """
    ok, why = [], {}
    for idx in candidates:
        na = next_action(runs, idx, code_root, check_provenance)
        a = na["action"]
        if a in (f"run_seed{seed}", f"rerun_seed{seed}"):
            ok.append(idx)
        elif a == f"confirm_stopped_seed{seed}" and na.get("unfinished_run") in confirm_stopped:
            ok.append(idx)
        else:
            why[idx] = {"action": a, "errors": na["errors"][:3], "note": na.get("note")}
    return ok, why


# ---------------------------------------------------------------- GPU の時間の見積もり
ALLOWED_SLURM = ("00:20:00", "00:40:00", "01:00:00", "02:00:00", "04:00:00", "06:00:00", "08:00:00", "12:00:00")
MIN_PER_PAIR = 279.87 / 300  # 5/20 の本番のログ（elapsed_min 279.87, 300 対）= 0.933 分
LOAD_MIN = 5.0               # 3 つのモデルの読み込み（5/20 は 1 分未満。余裕を見て 5 分）
MARGIN = 1.5


def slurm_time_for(n_pairs):
    need = (LOAD_MIN + MIN_PER_PAIR * n_pairs) * MARGIN
    for t in ALLOWED_SLURM:
        h, m, _ = (int(x) for x in t.split(":"))
        if h * 60 + m >= need:
            return t, round(LOAD_MIN + MIN_PER_PAIR * n_pairs, 1)
    return None, round(LOAD_MIN + MIN_PER_PAIR * n_pairs, 1)


# ---------------------------------------------------------------- 表の値（今の表と同じ式）
TABLE_KEYS = {
    "MCD (dB)": {"intact": "mcd_bl_inj", "blocked": "mcd_bl_mask_inj_mask"},
    # 付録 B の定義: 実体語の区間で時間平均した log-mel の cosine 類似度（9/30 の確認: 表は tmean_*_mel）
    "MEL Similarity": {"intact": "tmean_bl_inj_mel", "blocked": "tmean_bl_mask_inj_mask_mel"},
}
# 今の原稿の英語の値（naacl2027/main.tex 894-895 行、explanation_of_revisions.tex 279 行）。読むだけ。書かない
CURRENT_TABLE_EN = {"MCD (dB)": ("32.52 \\pm 0.58", "0.00 \\pm 0.00"),
                    "MEL Similarity": ("0.90 \\pm 0.01", "1.00 \\pm 0.00")}
CURRENT_EOR_L279_EN_MCD = "32.52\\pm0.58"
CI_METHOD = "mean ± 1.96 × np.std(ddof=0) / sqrt(n)（aligned の行の、値が None でないもの。今の表と同じ）"


def table_values(rows):
    import math
    import numpy as np
    aligned = [r for r in rows if r.get("aligned")]
    out = {}
    for metric, cols in TABLE_KEYS.items():
        out[metric] = {}
        for col, key in cols.items():
            vs = np.array([r[key] for r in aligned if r.get(key) is not None], dtype=np.float64)
            n = int(vs.size)
            mean = float(vs.mean())
            sd0 = float(vs.std())
            sd1 = float(vs.std(ddof=1)) if n > 1 else float("nan")
            ci0 = 1.96 * sd0 / math.sqrt(n)
            out[metric][col] = {"key": key, "n": n, "mean": mean, "std_ddof0": sd0, "ci95_half": ci0,
                                "std_ddof1_reference": sd1,
                                "tex": f"{mean:.2f} \\pm {ci0:.2f}"}
    return out


def eor_tex(tv):
    v = tv["MCD (dB)"]["intact"]
    return f"{v['mean']:.2f}\\pm{v['ci95_half']:.2f}"
