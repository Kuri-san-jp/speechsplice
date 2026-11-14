#!/usr/bin/env python3
"""付録 C 日本語の作り直し: 対の出どころと、直したスパン関数が実体語を指すかを CPU で確かめる（2026-10-02）.

トークナイザだけを読む（モデルは読まない。transformers が torch を読み込むが GPU は見せない。pyopenjtalk は塞ぐ）。
ホストで直接流す:
  taskset -c 0-12 python3 scripts/spanfix/appc_ja_span_check.py

確かめること:
  A. 対（付録 K の日本語 300 対）
     1. 正本 eval_results/cjk_n300_rewav/checkpoint_ja.json の results 300 行（appk_common.SETS["ja"] が指すもの）
     2. eval_results/cjk_n300/checkpoint_ja.json の 300 行と (pair_idx, text_a, text_b, word_a, word_b) が同じ並び
     3. 元の定義 scripts/projection/eval_cjk_n300.py の load_ja_pairs(300)（data/entities_10000.jsonl, seed 42）と
        テンプレートから作り直した 300 対と同じ並び（関数は ast で取り出して実行。ファイルは import しない）
     4. 300 対がすべて異なる（対、text_a、text_b、(word_a, word_b)、word_a、word_b、pair_idx）
     5. 参考: 5/11 の日本語の対（crossattn_mask_n300_filtered_v2 の先頭 300）との重なり
  B. 本番 main() への渡し方
     6. 対のファイル pairs/appk_ja_300_v2format.json を作る（既にあれば中身が同じことを確かめる）
     7. 本番 main() の JA の分岐の文そのもの（ast で取り出す）を、open をこのファイルに向けて実行すると、
        付録 K の 300 対と 300/300 同じ並びになる
  C. スパン（対ごと、text_a/word_a と text_b/word_b の両方）
     8. 本番と同じ入り口: テキストは vendor の normalize_text_with_lang(text, "ja")（_normalize_japanese_text）を通す。
        add_bos は config の add_bos_to_text（0）、エンコーダの入力の末尾は add_eos_to_text
     9. 今の find_entity_token_range（7/11 修正後）が範囲を返し、その範囲が §3.2 の定義どおり
        「文字位置が実体語に重なるトークンすべて」とちょうど一致し（別に計算した集合と比べる）、実体語を全部覆い、
        範囲の外のトークンは実体語に掛からない。実体語は文中にちょうど 1 回だけ現れる
    10. 範囲の端のトークンが実体語の外の文字も含む対（例: 「子は」）を数えて記録する（定義どおりで失敗ではない）
    11. 修正前の関数（release_artifacts の bak）の範囲と比べ、違う対を記録する（参考）
    12. 置換（a と b の範囲の長さが同じ）か継ぎ合わせ（違う）か、遮断する列の期待値（BL_mask, INJ_mask）
出力: eval_results/_spanfix_appC_ja/span_check/span_check_<日時>.json（新規。上書きしない）
     全部通れば終了コード 0、通らなければ 1。
"""
import os
import sys

THREADS = 13
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = str(THREADS)
_R = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
for _k, _v in {"HF_HOME": _R + "/.cache/huggingface", "HF_HUB_CACHE": _R + "/.cache/huggingface/hub",
               "HF_HUB_OFFLINE": "1", "NUMBA_CACHE_DIR": _R + "/.cache/numba",
               "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[_k] = _v
if not os.path.isdir("/mnt/scripts"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ホストでは GPU に触れない
sys.dont_write_bytecode = True
sys.modules["pyopenjtalk"] = None  # 自分のコードでは使わない。間接の import も塞ぐ

import argparse  # noqa: E402
import ast  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from typing import Optional, Tuple  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appc_ja_common as C  # noqa: E402

OWN_REL = "scripts/spanfix/appc_ja_span_check.py"
OWN_CODE = (OWN_REL, "scripts/spanfix/appc_ja_common.py")
VENDOR_NORM_NAMES = ("_REPLACE_MAP", "_FULLWIDTH_ALPHA_TO_HALFWIDTH", "_HALFWIDTH_KATAKANA_CHARS",
                     "_FULLWIDTH_KATAKANA_CHARS", "_HALFWIDTH_KATAKANA_TO_FULLWIDTH", "_FULLWIDTH_DIGITS_TO_HALFWIDTH")


def load_function_from_source(path, name, extra_ns=None):
    """ファイルを import せず（副作用なし）、関数定義だけを取り出して exec する（英語と同じ方法）."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f"{name} が {path} に 1 つだけあるはず（{len(nodes)}）"
    ns = dict(extra_ns or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns[name], ast.get_source_segment(src, nodes[0])


def _not_called(*a, **k):
    raise RuntimeError("lang='ja' を渡しているので呼ばれないはずの関数（detect_language）が呼ばれた")


def load_vendor_normalizer():
    """vendor の normalize_text_with_lang と _normalize_japanese_text を、定数ごと ast で取り出す（import しない）."""
    path = os.path.join(C.VENDOR_ROOT, "inference_tts_utils.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in VENDOR_NORM_NAMES for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name in ("_normalize_japanese_text", "normalize_text_with_lang"))]
    got = sorted(getattr(n, "name", None) or n.targets[0].id for n in nodes)
    assert got == sorted(VENDOR_NORM_NAMES + ("_normalize_japanese_text", "normalize_text_with_lang")), got
    ns = {"re": re, "Optional": Optional, "Tuple": Tuple, "detect_language": _not_called}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)

    def norm(t):
        out, lang = ns["normalize_text_with_lang"](t, C.LANG)  # get_encoder_output(..., lang=args.lang) と同じ
        assert lang == C.LANG
        return out
    return norm


def resolve_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tts_snap = snapshot_download(C.T5GEMMA_TTS_ID, local_files_only=True)
    cfg = C.read_json(os.path.join(tts_snap, "config.json"))
    tok_name = cfg.get("text_tokenizer_name") or cfg.get("t5gemma_model_name")
    assert tok_name == C.TOKENIZER_ID, tok_name
    tok_snap = snapshot_download(tok_name, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(tok_snap)
    info = {"t5gemma_tts_snapshot": os.path.basename(tts_snap), "tokenizer_id": tok_name,
            "tokenizer_snapshot": os.path.basename(tok_snap), "tokenizer_class": type(tok).__name__,
            "is_fast": bool(getattr(tok, "is_fast", False)),
            "add_bos_to_text": int(cfg.get("add_bos_to_text", 0)),
            "add_eos_to_text": int(cfg.get("add_eos_to_text", 0))}
    return tok, info


# ---------------------------------------------------------------- A. 対の出どころ
def rebuild_from_definition(root):
    """eval_cjk_n300.py の load_ja_pairs(300) と、main() と同じテンプレートの当てはめで 300 対を作り直す."""
    path = C.P(C.CJK_GEN_REL, root)
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    seeds = [ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SEED" for t in n.targets)]
    assert seeds == [42], seeds
    # main() の JA のテキストの作り方が想定どおりか（文字列で確かめる）
    for s in ('tmpl = pair["template_a"]', "text_a = tmpl.format(word=word_a)", "text_b = tmpl.format(word=word_b)"):
        assert s in src, s
    fn, _ = load_function_from_source(path, "load_ja_pairs",
                                      {"os": os, "json": json, "random": random, "PROJECT_ROOT": root,
                                       "SEED": seeds[0]})
    raw = fn(C.N_PAIRS)
    out = []
    for i, p in enumerate(raw):
        tmpl = p["template_a"]
        out.append({"pair_idx": i, "text_a": tmpl.format(word=p["word_a"]), "text_b": tmpl.format(word=p["word_b"]),
                    "word_a": p["word_a"], "word_b": p["word_b"]})
    return out


def appk_sets_ja(root):
    src = open(C.P(C.APPK_COMMON_REL, root), encoding="utf-8").read()
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SETS" for t in n.targets):
            return ast.literal_eval(n.value)["ja"]
    return None


def seq(rows):
    return [tuple(r.get(k) for k in C.PAIR_KEYS) for r in rows]


def check_pair_sources(pairs, root):
    rep = {"canonical": C.APPK_JA_CKPT_REL, "canonical_md5": C.md5_file(C.P(C.APPK_JA_CKPT_REL, root))}
    fails = []
    for rel, want in C.PAIR_SOURCE_MD5.items():
        got = C.md5_file(C.P(rel, root))
        rep[f"md5:{rel}"] = got
        if got != want:
            fails.append(f"{rel} の md5 が 10/02 の確かめの時点と違う")
    sj = appk_sets_ja(root)
    rep["appk_common_SETS_ja"] = sj
    rep["appk_common_md5"] = C.md5_file(C.P(C.APPK_COMMON_REL, root))
    if not sj or os.path.join("eval_results", sj["dir"], sj["ckpt"]) != C.APPK_JA_CKPT_REL or sj["rows_key"] != "results":
        fails.append(f"appk_common.SETS['ja'] が {C.APPK_JA_CKPT_REL} の results を指していない: {sj}")
    cjk = C.read_json(C.P(C.CJK_N300_CKPT_REL, root))["results"]
    rep["cjk_n300_same_order"] = seq(cjk) == seq(pairs)
    if not rep["cjk_n300_same_order"]:
        fails.append(f"{C.CJK_N300_CKPT_REL} と並びが違う")
    reb = rebuild_from_definition(root)
    rep["definition_rebuild_same_order"] = seq(reb) == seq(pairs)
    rep["definition"] = f"{C.CJK_GEN_REL} load_ja_pairs(300) + template '{{word}}は、日本で広く知られています。'"
    if not rep["definition_rebuild_same_order"]:
        first = next((i for i, (a, b) in enumerate(zip(seq(reb), seq(pairs))) if a != b), None)
        fails.append(f"元の定義から作り直した対と並びが違う（最初の違い {first}）")
    rep["distinctness"] = C.distinctness(pairs)
    if not rep["distinctness"]["all_distinct"]:
        fails.append(f"300 対がすべて異なるとは言えない: {rep['distinctness']}")
    old = C.read_json(C.P(C.OLD_PAIR_SRC_REL, root))["results"][:C.N_PAIRS]
    rep["overlap_with_0511_ja_pairs"] = {
        "source": C.OLD_PAIR_SRC_REL + " (first 300; the pairs of the current Table A2 JA row)",
        "same_sentence_a": len({r["text"] for r in old} & {p["text_a"] for p in pairs}),
        "same_name_pair": len({(r["kanji"], r["kata"]) for r in old} & {(p["word_a"], p["word_b"]) for p in pairs}),
        "old_distinct_pairs": len({(r["text"], r["kanji"], r["kata"]) for r in old}),
    }
    rep["fails"] = fails
    rep["ok"] = not fails
    return rep


# ---------------------------------------------------------------- B. 本番 main() への渡し方
def ensure_pair_file(pairs, root):
    obj = C.pairs_v2format(pairs)
    path = C.P(C.PAIR_FILE_REL, root)
    blob = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
    if os.path.exists(path):
        with open(path, "rb") as f:
            have = f.read()
        if have != blob:
            raise SystemExit(f"[span_check] {C.PAIR_FILE_REL} が既にあり、中身が今の対と違う（消さずに著者の判断）")
        created = False
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "xb") as f:
            f.write(blob)
        created = True
    return {"path": C.PAIR_FILE_REL, "md5": C.md5_file(path), "created_now": created}


def check_production_loader(pairs, root):
    target = C.P(C.PAIR_FILE_REL, root)
    calls = []

    def open_redirect(file, mode="r", *a, **k):
        calls.append([file, mode])
        if file == C.OLD_PAIR_SRC_REL and mode in ("r", "rt"):
            return open(target, mode, *a, encoding="utf-8", **k)
        raise AssertionError(f"JA の分岐が想定外のファイルを開いた: {file}")

    prod = C.run_production_ja_loader(open_redirect, root=root)
    _, branch_src = C.production_ja_branch_source(root)
    same = seq(prod) == seq(pairs)
    return {"branch_source": branch_src, "open_calls": calls, "n_loaded": len(prod),
            "same_order_as_appk": same, "keys": sorted(prod[0]) if prod else None,
            "ok": same and len(prod) == C.N_PAIRS and calls == [[C.OLD_PAIR_SRC_REL, "r"]]}


# ---------------------------------------------------------------- C. スパン
def check_side(tok, cur_fn, old_fn, norm, text_raw, word, add_bos, add_eos):
    text = norm(text_raw)
    rec = {"text": text, "word": word, "normalized_changed": text != text_raw, "word_normalized_changed": norm(word) != word}
    off = 1 if add_bos else 0
    cur = cur_fn(tok, text, word, add_bos=add_bos)
    old = old_fn(tok, text, word, add_bos=add_bos)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    model_ids = ([add_bos] if add_bos else []) + ids + ([add_eos] if add_eos else [])  # get_encoder_output と同じ並び
    rec["n_text_tokens"] = len(ids)
    rec["n_encoder_tokens"] = len(model_ids)
    rec["fixed"] = list(cur) if cur else None
    rec["legacy"] = list(old) if old else None
    pieces = tok.convert_ids_to_tokens(ids)
    fails = []
    if tok.encode(text, add_special_tokens=False) != ids:
        fails.append("encode と offset 付きの並びが違う")
    n_occ = text.count(word)
    rec["word_occurrences"] = n_occ
    if n_occ != 1:
        fails.append(f"語の出現が {n_occ} 回")
    if rec["word_normalized_changed"]:
        fails.append("語が正規化で変わる（文中で見つからない恐れ）")
    ws = text.find(word)
    we = ws + len(word)
    rec["word_char_span"] = [ws, we]
    overlap = [i for i, (a, b) in enumerate(offs) if a < we and b > ws]  # §3.2 の定義（別に計算する）
    rec["overlap_tokens"] = overlap
    if cur is None:
        fails.append("範囲が返らない")
    else:
        s, e = cur[0] - off, cur[1] - off
        if not (0 <= s < e <= len(ids)):
            fails.append(f"範囲が文の外 {cur}")
        else:
            rec["fixed_tokens"] = pieces[s:e]
            rec["fixed_decoded"] = tok.decode(ids[s:e])
            if model_ids[cur[0]:cur[1]] != ids[s:e]:
                fails.append("エンコーダ入力の並びでの位置がずれる")
            if list(range(s, e)) != overlap:
                fails.append(f"範囲 {list(range(s, e))} が実体語に重なるトークンの集合 {overlap} と違う")
            cs = min(offs[i][0] for i in range(s, e))
            ce = max(offs[i][1] for i in range(s, e))
            rec["fixed_char_span"] = [cs, ce]
            if not (cs <= ws and ce >= we):
                fails.append(f"範囲の文字 [{cs},{ce}) が語 [{ws},{we}) を覆わない")
            if any(offs[i][1] <= offs[i][0] for i in range(s, e)):
                fails.append("範囲に長さ 0 のトークンがある")
            if word not in rec["fixed_decoded"].replace(" ", ""):
                fails.append(f"decode に語が含まれない: {rec['fixed_decoded']!r}")
            rec["extra_left"] = text[cs:ws]
            rec["extra_right"] = text[we:ce]
            rec["straddle"] = bool(rec["extra_left"] or rec["extra_right"])
    if old is not None:
        so, eo = old[0] - off, old[1] - off
        rec["legacy_tokens"] = pieces[max(0, so):max(0, eo)] if 0 <= so <= len(ids) else None
    rec["legacy_differs"] = (rec["fixed"] != rec["legacy"])
    rec["fails"] = fails
    rec["ok"] = not fails
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="出力 JSON（既定: span_check/span_check_<日時>.json）")
    args = ap.parse_args()
    root = C.PROJECT_ROOT
    errs = C.pinned_errors(root) + C.old_result_errors(root)
    if errs:
        raise SystemExit("[span_check] " + " / ".join(errs))

    pairs = C.load_pairs(root)
    src_rep = check_pair_sources(pairs, root)
    pair_file = ensure_pair_file(pairs, root)
    loader = check_production_loader(pairs, root)

    cur_fn, cur_src = load_function_from_source(C.P(C.INJ_REL), "find_entity_token_range")
    old_fn, old_src = load_function_from_source(C.P(C.LEGACY_SPAN_REL), "find_entity_token_range")
    assert "return_offsets_mapping=True" in cur_src and "return_offsets_mapping" not in old_src
    norm = load_vendor_normalizer()
    tok, tinfo = resolve_tokenizer()
    if not tinfo["is_fast"]:
        raise SystemExit("[span_check] fast tokenizer でない（offset_mapping が使えず旧ロジックに落ちる）")
    add_bos, add_eos = tinfo["add_bos_to_text"], tinfo["add_eos_to_text"]
    per_pair = []
    for p in pairs:
        a = check_side(tok, cur_fn, old_fn, norm, p["text_a"], p["word_a"], add_bos, add_eos)
        b = check_side(tok, cur_fn, old_fn, norm, p["text_b"], p["word_b"], add_bos, add_eos)
        la = (a["fixed"][1] - a["fixed"][0]) if a["fixed"] else None
        lb = (b["fixed"][1] - b["fixed"][0]) if b["fixed"] else None
        ref = {"a": a["fixed"], "b": b["fixed"]}
        per_pair.append({
            "pair_idx": p["pair_idx"], "word_a": p["word_a"], "word_b": p["word_b"], "a": a, "b": b,
            "ok": a["ok"] and b["ok"], "same_token_count": la == lb,  # 同じなら置換、違えば継ぎ合わせ
            "expected_masked_cols": ({"bl_mask": C.expected_masked_cols(ref, "bl_mask"),
                                      "inj_mask": C.expected_masked_cols(ref, "inj_mask")}
                                     if a["fixed"] and b["fixed"] else None),
            "legacy_affected_a": a["legacy_differs"], "legacy_affected_b": b["legacy_differs"]})

    n_ok = sum(r["ok"] for r in per_pair)
    strad = {s: [r["pair_idx"] for r in per_pair if r[s].get("straddle")] for s in ("a", "b")}
    extra = collections.Counter((s, r[s].get("extra_left", ""), r[s].get("extra_right", ""))
                                for r in per_pair for s in ("a", "b") if r[s].get("straddle"))
    aff_a = sum(r["legacy_affected_a"] for r in per_pair)
    aff_b = sum(r["legacy_affected_b"] for r in per_pair)
    aff_any = sum(r["legacy_affected_a"] or r["legacy_affected_b"] for r in per_pair)
    by_cond = {"bl": 0, "tgt": 0, "bl_mask": aff_a, "inj": aff_any, "inj_mask": aff_any}
    examples = [{"pair_idx": r["pair_idx"], "text_a": r["a"]["text"], "fixed_tokens_a": r["a"].get("fixed_tokens"),
                 "legacy_tokens_a": r["a"].get("legacy_tokens"), "text_b": r["b"]["text"],
                 "fixed_tokens_b": r["b"].get("fixed_tokens"), "legacy_tokens_b": r["b"].get("legacy_tokens")}
                for r in per_pair[:: max(1, len(per_pair) // 10)]]
    norm_changed = [r["pair_idx"] for r in per_pair if r["a"]["normalized_changed"] or r["b"]["normalized_changed"]]
    all_ok = (n_ok == len(per_pair) == C.N_PAIRS and src_rep["ok"] and loader["ok"])

    out = {
        "item": "Appendix C tab:ja_crossattn_phoneme, JA rows rerun on the Appendix K JA pairs (fixed span)",
        "producer": OWN_REL, "date": C.stamp(),
        "pairs": f"{C.APPK_JA_CKPT_REL} results (Appendix K JA)", "pairs_digest": C.pairs_digest(pairs),
        "pair_sources": src_rep, "pair_file": pair_file, "production_loader": loader,
        "tokenizer": tinfo, "code_md5": C.code_md5(root, own=OWN_CODE),
        "n_pairs": len(per_pair), "n_ok": n_ok, "all_ok": all_ok,
        "normalization_changed_pairs": norm_changed,
        "straddle": {"a_side": strad["a"], "b_side": strad["b"], "n_a": len(strad["a"]), "n_b": len(strad["b"]),
                     "extra_chars": [[s, l, r, n] for (s, l, r), n in extra.most_common()],
                     "note": "§3.2 (main.tex L124): every token whose offset range overlaps the entity is taken. "
                             "A boundary token that also carries a following particle (e.g. '子は') is inside the "
                             "span by definition; with splicing, INJ then carries no separate encoder row for it."},
        "legacy_affected": {"a_side": aff_a, "b_side": aff_b, "either": aff_any, "by_condition": by_cond,
                            "note": "legacy = the span function before the 7/11 fix, applied to these App K pairs "
                                    "(reference only; the 5/11 run used other pairs)"},
        "n_same_token_count": sum(r["same_token_count"] for r in per_pair),
        "fixed_spans": {str(r["pair_idx"]): {"a": r["a"]["fixed"], "b": r["b"]["fixed"],
                                             "norm_text_a": r["a"]["text"], "norm_text_b": r["b"]["text"]}
                        for r in per_pair},
        "examples": examples,
        "per_pair": per_pair,
    }
    path = args.out or C.P(f"{C.SPAN_CHECK_DIR_REL}/span_check_{C.stamp()}.json")
    C.write_json_new(out, path)
    d = src_rep["distinctness"]
    print(f"[span_check] 対: {C.APPK_JA_CKPT_REL}（md5 {src_rep['canonical_md5']}）digest={out['pairs_digest']}")
    print(f"[span_check]   cjk_n300 と同じ並び={src_rep['cjk_n300_same_order']}、元の定義（{C.CJK_GEN_REL} "
          f"load_ja_pairs）から作り直した 300 対と同じ並び={src_rep['definition_rebuild_same_order']}、"
          f"appk_common.SETS['ja']={src_rep['appk_common_SETS_ja']['dir']}/{src_rep['appk_common_SETS_ja']['ckpt']}")
    print(f"[span_check]   異なる数: 対 {d['pair_tuple']}, text_a {d['text_a']}, text_b {d['text_b']}, "
          f"(word_a,word_b) {d['word_pair']}, word_a {d['word_a']}, word_b {d['word_b']} / {d['n']} "
          f"→ all_distinct={d['all_distinct']}")
    o = src_rep["overlap_with_0511_ja_pairs"]
    print(f"[span_check]   参考: 5/11 の JA の対との重なり: 文 {o['same_sentence_a']}/300、名前の組 {o['same_name_pair']}/300"
          f"（5/11 の対の異なる数 {o['old_distinct_pairs']}）")
    print(f"[span_check] 対のファイル {pair_file['path']} md5={pair_file['md5']} 作成={pair_file['created_now']}；"
          f"本番 main() の JA の分岐で読むと付録 K と同じ並び={loader['same_order_as_appk']} "
          f"(open の呼び出し {loader['open_calls']})")
    print(f"[span_check] tokenizer={tinfo['tokenizer_class']} snapshot={tinfo['tokenizer_snapshot']} "
          f"add_bos={add_bos} add_eos={add_eos}；正規化で文が変わる対 {len(norm_changed)}")
    print(f"[span_check] スパン ok={n_ok}/{len(per_pair)}（a・b の両側で、範囲 = 実体語に重なるトークンすべて）")
    print(f"[span_check]   端のトークンが実体語の外の文字も含む対: a 側 {len(strad['a'])}, b 側 {len(strad['b'])}；"
          f"外の文字 {out['straddle']['extra_chars'][:6]}")
    print(f"[span_check]   a と b の範囲のトークン数が同じ（置換）対: {out['n_same_token_count']}/{len(per_pair)}"
          f"（残りは継ぎ合わせ）")
    print(f"[span_check]   参考: 修正前の関数なら範囲が違う対: a 側 {aff_a}, b 側 {aff_b}, どちらか {aff_any}")
    for ex in examples[:4]:
        print(f"  p{ex['pair_idx']:03d} {ex['text_a']!r}: fixed={ex['fixed_tokens_a']} legacy={ex['legacy_tokens_a']}"
              f" | {ex['text_b']!r}: fixed={ex['fixed_tokens_b']}")
    for r in [r for r in per_pair if not r["ok"]][:20]:
        print(f"  [NG] p{r['pair_idx']:03d} a={r['a']['fails']} b={r['b']['fails']}")
    for f in src_rep["fails"]:
        print(f"  [NG] {f}")
    print(f"[span_check] all_ok={all_ok}  wrote {os.path.relpath(path, root)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
