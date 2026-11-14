#!/usr/bin/env python3
"""LongCat-AudioDiT-1B: 1 文中の 2 箇所の実体語を同時に差し替える (BL/INJ_1/INJ_2/INJ_12).

**新規ファイル。既存ファイルは 1 行も変更していない。** 差し替え・生成・wav 書き出しは
すべて既存の関数を import してそのまま呼ぶ:

    scripts/projection/splice_multi.py
        build_spliced_hidden_multi / resolve_spans_with_cursor /
        verify_span_roundtrip / check_span_sets / span_deltas
    scripts/longcat/longcat_splice.py            (L)
        encode_for_text / duration_frames / generate_from_embedding /
        entity_token_span / build_spliced_embedding / AdaLNSourceOverride /
        normalize_text / load_longcat / seed_all
    scripts/longcat/run_longcat_eval.py          (R)
        save_wav / atomic_json_dump

差し替える対象は LongCat では H = LN(h_L) + LN(h_0) (凍結 umt5-base 符号化器の
最終出力。manifest の kappa.H_cross_attn_input がこの値の cosine)。
splice_multi は last_hidden_state という鍵名で [1, T, d] を受けるだけなので、
{"last_hidden_state": parts["H"]} という薄い包みを渡す。テンソルの実体は同一で、
build_spliced_hidden_multi は arithmetic を一切しない (slice + torch.cat 1 回) ので
k=1 に縮退させると LongCat 既存の build_spliced_embedding(splice_part="both") と
bit 一致する (scripts/longcat/_cpu_multientity/check_spans_cpu.py の (D))。

4 条件 (1 文あたり)
-------------------
    bl     差し替えなし。H = H^src
    inj1   枠 1 だけ差し替え。ref = text_refs[0] の H
    inj2   枠 2 だけ差し替え。ref = text_refs[1] の H
    inj12  両方差し替え。枠 i の ref は ref_i (**枠ごとに別の forward**)

枠ごとに別の ref を使うのは、両方置換した 1 本の文から 2 枠を取ると
干渉の原因 (splice の相互作用 vs. ref 側の文脈が既に汚れている) が
分離できなくなるため。encoder forward は 1 文あたり 3 回、生成は 4 回。

固定事項
--------
* duration は必ず source テキスト (text_a) の正規化文字数から決める
  (L.duration_frames)。4 条件すべて同じ値になるので初期ノイズ
  y0 = randn(duration, latent_dim) は 4 条件で bit 単位で同一。
* seed は生成の直前に毎回戻る (L.generate_from_embedding の先頭が seed_all)。
  4 条件で乱数の消費順序が揃う。
* 系列長は差し替えで T -> T + Σ(m_i - n_i) に変わる。**2 番目の枠の index を
  自分でずらしてはいけない。** splice_multi が src の「元の」index だけを読んで
  一括 concat するので、ずらすと二重補正になる。
* 区間は splice_multi.resolve_spans_with_cursor で**まとめて**解決し、
  verify_span_roundtrip で往復一致を確認する。落ちた文は丸ごと skip。
  「末尾 N トークンを実体語とみなす」種類の fallback は一切用意しない。
* 文字スパンは LongCat 既存の語境界正規表現 (L.entity_char_span と同じパターン)
  で決め、char_starts として resolve_spans_with_cursor に渡す。さらに語が文中で
  一意な場合は既存の L.entity_token_span とトークン区間が一致することを
  実行時に照合する (一致しなければ AssertionError で落ちる)。

使い方 (GPU)
------------
  # 本番 (EN 300 文 x 4 条件)。中国語には多 entity データが無いので EN のみ。
  SLURM_TIME=06:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/longcat/run_multientity_longcat.py \\
      --pairs_module en_multientity_300 --lang en \\
      --splice_part both --cfg_strength 4 --guidance_method cfg

  # 単一区間の md5 等価性試験 (n=5)
  SLURM_TIME=00:30:00 slurm_run.sh \\
    python3 -u /mnt/scripts/longcat/run_multientity_longcat.py \\
      --equiv_single --lang en --n 5 \\
      --equiv_ref_dir /mnt/eval_results/_cc_wavs_a/longcat_en_cfg4_cfg_both

出力: <out_dir>/{results.json, checkpoint.json, s{idx:04d}_{cond}.wav}
採点: scripts/analysis/score_multientity.py --dir <out_dir> --lang en
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

# --------------------------------------------------------------------------
# HF cache は setdefault では効かない (slurm_run.sh がコンテナに
# HF_HOME=/tmp/.cache/huggingface を渡してくるので必ず代入で上書きする)。
# transformers / torch より先に済ませる。
# --------------------------------------------------------------------------
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")

# **スレッド上限。torch を読む前に置く必要がある。**
# 2026-09-18、この種のジョブを並べたらログインノードの CPU を埋めて
# 他の利用者に迷惑をかけた（note/2026-09-18_gpuoverrun_debug.md）。
# **上限は 128 スレッドの 1/4 = 32**（2026-09-18 ユーザ指示）。
# GPU を 3 枚まで同時に使うので、1 本あたり 10 にして合計 30 に収める。
# 実測では GPU 律速で 1 コアしか使わないが、上限として置く。
_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

PROJECT_ROOT = ROOT
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "longcat"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "projection"))

import longcat_splice as L  # noqa: E402
import run_longcat_eval as R  # noqa: E402
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi, check_span_sets, resolve_spans_with_cursor,
    span_deltas, verify_span_roundtrip,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

# 条件名 -> 使う枠の index (0-based)
COND_SLOTS_2 = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}
COND_SLOTS_1 = {"bl": [], "inj1": [0]}


# --------------------------------------------------------------------------
# 区間の解決
# --------------------------------------------------------------------------
def _regex_char_span(norm_text, word, cursor=0, require_unique=True):
    """LongCat 既存 L.entity_char_span と同じ語境界パターンで文字スパンを取る.

    違いは cursor 以降だけを候補にする点のみ。cursor=0 かつ
    require_unique=True のとき L.entity_char_span:261-275 と同値
    (同じ正規表現・同じ「複数ヒットは例外」規則・同じ normalize_text)。
    同じ表層形が 2 回出るデータ (en_multientity_repeat_40) のために
    cursor を持つが、en_multientity_300 は 1 文に同じ語が出ない設計なので
    既定では require_unique=True のまま使う。
    """
    w = L.normalize_text(word).strip()
    if not w:
        raise ValueError(f"empty entity word: {word!r}")
    pat = r"(?<![0-9a-z])" + re.escape(w) + r"(?![0-9a-z])"
    hits = [m for m in re.finditer(pat, norm_text) if m.start() >= cursor]
    if not hits:
        raise ValueError(f"entity {w!r} not found in normalized text "
                         f"{norm_text!r} at or after char {cursor}")
    if require_unique and len(hits) > 1:
        raise ValueError(f"entity {w!r} occurs {len(hits)} times in {norm_text!r}")
    m = hits[0]
    return (m.start(), m.end()), w


def _assert_contiguous(tokenizer, norm, spans, char_spans):
    """既存 L.entity_token_span:295-296 の「非連続なら例外」を再現する.

    resolve_spans_with_cursor は交差するトークンの min..max+1 を取るだけで
    連続性も offset の退化 (b <= a) も見ない。ここを補わないと、
    既存の単一 entity 経路より広い区間を黙って掴む可能性が残る。
    数値を合わせるための補正ではなく、既存と同じ条件で落とすためのガード。
    """
    enc = tokenizer(norm, return_offsets_mapping=True, add_special_tokens=False)
    offs = [tuple(x) for x in enc["offset_mapping"]]
    for i, ((t_s, t_e), (c0, c1)) in enumerate(zip(spans, char_spans)):
        bad = [j for j in range(t_s, t_e)
               if not (offs[j][1] > offs[j][0]
                       and offs[j][0] < c1 and offs[j][1] > c0)]
        if bad:
            raise ValueError(
                f"span {i}: entity token span is not contiguous: tokens {bad} "
                f"in [{t_s}, {t_e}) do not overlap chars [{c0}, {c1}) of {norm!r}")


def _crosscheck_single_path(tokenizer, text, word, span):
    """語が文中で一意なら、既存の単一 entity 経路と区間が一致することを照合する.

    L.entity_token_span は add_special_tokens=True、
    resolve_spans_with_cursor は False を使う。umt5 は EOS を末尾に足すだけで
    先頭に特殊トークンを付けないので実トークンの index は一致する
    (_cpu_multientity/check_spans_cpu.py の (A) で 実測)。その前提が崩れたら
    ここで落ちる。
    """
    try:
        ref = L.entity_token_span(tokenizer, text, word)
    except ValueError as e:
        # 複数出現など、単一 entity 経路では扱えない文。照合を省く。
        return {"checked": False, "reason": f"{e}"[:200]}
    want = (int(ref["tok_start"]), int(ref["tok_end"]))
    got = (int(span[0]), int(span[1]))
    if got != want:
        raise AssertionError(
            f"multi-entity span {got} != single-entity path {want} "
            f"(L.entity_token_span) for word={word!r} in {text!r}")
    return {"checked": True, "tok": list(want),
            "tokens": ref["entity_tokens"],
            "roundtrip_ok": bool(ref["roundtrip_ok"]),
            "n_tokens_total": int(ref["n_tokens_total"])}


def resolve_spans(tokenizer, text, words, require_unique=True, crosscheck=True,
                  span_extension=False, peer_texts=None, peer_words=None,
                  side="a"):
    """k 個の実体語を splice_multi の関数で**まとめて**トークン区間へ解決する.

    返り値 (spans, surfaces, info)
      spans    : [(t_s, t_e)] x k。add_special_tokens=True 側の index と同じ空間。
      surfaces : 正規化後の表層形
      info     : 診断 (norm_text / char_spans / per_span / roundtrip / crosscheck)
    失敗は例外で落とす (fallback 無し)。
    """
    norm = L.normalize_text(text)
    surfaces, starts, char_spans = [], [], []
    cursor = 0
    for w in words:
        (c0, c1), nw = _regex_char_span(norm, w, cursor=cursor,
                                        require_unique=require_unique)
        surfaces.append(nw)
        starts.append(c0)
        char_spans.append((c0, c1))
        cursor = c1

    # **最小対称拡張（2026-09-20 追加）。**中国語で次の語が実体語に癒着し、
    # 往復確認が落ちる対が 14/300 あった:
    #   covered '东京都' != surface '东京'   （「都」が癒着）
    #   covered '墨尔本的' != surface '墨尔本' （「的」が癒着）
    # 規則の実装は scripts/projection/span_extend.py の 1 箇所だけで、
    # T5Gemma / Qwen3 が既に同じものを使っている。ここでは呼ぶだけ。
    # **既定は False。**語そのままで往復一致する対には plain=True が返り、
    # 解決されるトークン区間は既定経路と同一になる。
    pnorm = None
    if peer_texts is not None:
        pnorm = [L.normalize_text(t) for t in peer_texts]
    spans, sinfo = resolve_spans_with_cursor(
        tokenizer, norm, surfaces, add_bos=0, char_starts=starts,
        allow_symmetric_extension=bool(span_extension),
        peer_texts=pnorm, peer_surfaces=peer_words, side=side)
    if spans is None:
        raise ValueError(f"resolve_spans_with_cursor failed: {sinfo['reason']}")
    # **連続性の照合も拡張後の char 範囲で行う。**拡張が起きた対では
    # 掴むトークンが「东京」から「东京都」へ広がるので、拡張前の char_spans
    # で見ると必ず落ちる（往復確認と同じ理由）。per_span は拡張後の値を持つ。
    eff_char = list(char_spans)
    for i, ps in enumerate(sinfo.get("per_span") or []):
        if i < len(eff_char) and ps.get("char_start") is not None \
                and ps.get("char_end") is not None:
            eff_char[i] = (int(ps["char_start"]), int(ps["char_end"]))
    _assert_contiguous(tokenizer, norm, spans, eff_char)
    # **往復確認は拡張後の表層で行う。**拡張が起きた対では要求する表層が
    # 変わっているので、元の語で見ると必ず落ちる（Qwen3 で 2026-09-20 に踏んだ）。
    check = list(surfaces)
    for i, e in enumerate(sinfo.get("extension") or []):
        if i < len(check) and e.get("surface"):
            check[i] = e["surface"]
    ok, bad = verify_span_roundtrip(tokenizer, norm, spans, check, add_bos=0)

    info = {
        "norm_text": norm,
        "surfaces": surfaces,
        "char_spans": [list(x) for x in char_spans],
        "per_span": sinfo["per_span"],
        "roundtrip_ok": bool(ok),
        "roundtrip_bad": bad,
        "crosscheck": [],
    }
    if crosscheck:
        # **照合も拡張後の語で行う。**拡張が起きた対で元の語（'东京'）を
        # 単一 entity 経路に渡すと (4, 6) が返り、こちらの (4, 7) と食い違う。
        # 拡張後の語（'东京都'）で呼べば、既存経路と同じ条件のまま照合できる。
        for i, w in enumerate(words):
            e = (sinfo.get("extension") or [None] * len(words))
            w_eff = w
            if i < len(e) and e[i] and e[i].get("surface"):
                w_eff = e[i]["surface"]
            info["crosscheck"].append(
                _crosscheck_single_path(tokenizer, text, w_eff, spans[i]))
    return spans, surfaces, info


# --------------------------------------------------------------------------
# データ
# --------------------------------------------------------------------------
def load_items(args):
    """多 entity 本番と単一区間等価性試験の両方を同じレコード形に揃える."""
    if args.equiv_single:
        # **言語ごとに刺激を選ぶ（2026-09-20）。**以前は EN 固定だったが、
        # 中国語にも同じ作りの cross-culture 刺激（300 対）があり、
        # 既存本番 _cc_wavs_a/longcat_zh_cfg4_cfg_both はそれで作られている。
        # 照合先は --equiv_ref_dir で渡すので、ここで言語を縛る理由は無い。
        import importlib
        _cc = {"en": "en_cross_culture_names", "zh": "zh_cross_culture_names"}
        if args.lang not in _cc:
            raise SystemExit("--equiv_single は %s のみ"
                             % " / ".join(sorted(_cc)))
        raw = importlib.import_module(_cc[args.lang]).build_300_pairs()
        items = []
        for i, p in enumerate(raw):
            items.append({
                "pair_idx": p.get("pair_idx", i),
                "template_id": None, "lang": args.lang,
                "text_a": p["text_a"], "text_refs": [p["text_b"]],
                "text_both": p["text_b"],
                "words_a": [p["word_a"]], "words_b": [p["word_b"]],
                "gap_words": None, "slot1_at_start": None, "k": 1,
            })
        return items[:args.n] if args.n else items

    mod = __import__(args.pairs_module)
    raw = mod.build_pairs()
    items = []
    for it in raw:
        assert len(it["spans"]) == 2, \
            f"pair {it['pair_idx']}: 2 枠のデータであること (got {len(it['spans'])})"
        assert len(it["text_refs"]) == 2
        items.append({
            "pair_idx": it["pair_idx"],
            "template_id": it.get("template_id"),
            "lang": args.lang,
            "text_a": it["text_a"], "text_refs": list(it["text_refs"]),
            "text_both": it["text_both"],
            "words_a": [sp["word_a"] for sp in it["spans"]],
            "words_b": [sp["word_b"] for sp in it["spans"]],
            "char_start_a_raw": [sp["char_start_a"] for sp in it["spans"]],
            "char_start_b_raw": [sp["char_start_b"] for sp in it["spans"]],
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "k": 2,
        })
    return items[:args.n] if args.n else items


# --------------------------------------------------------------------------
# 1 条件の H を作る
# --------------------------------------------------------------------------
def build_condition_hidden(src_parts, ref_parts, s_ranges, r_ranges, slots):
    """slots (使う枠の index) に対応する H~ / delta_tok / out_ranges を返す.

    slots が空 (bl) のときは差し替えをせず H^src をそのまま返す。
    それ以外は splice_multi.build_spliced_hidden_multi に丸投げする。
    **s_ranges を自分でずらさない。** 一括 concat 側が src の元 index だけを
    読むので、ずらすと二重補正になる。
    """
    if not slots:
        return src_parts["H"], 0, [], int(src_parts["H"].shape[1])
    sub_s = [s_ranges[i] for i in slots]
    sub_r = [r_ranges[i] for i in slots]
    sub_ref = [{"last_hidden_state": ref_parts[i]["H"]} for i in slots]
    modified, delta_tok, out_ranges = build_spliced_hidden_multi(
        {"last_hidden_state": src_parts["H"]}, sub_ref, sub_s, sub_r)
    return modified, int(delta_tok), [list(x) for x in out_ranges], \
        int(modified.shape[1])


# --------------------------------------------------------------------------
# md5 等価性の照合
# --------------------------------------------------------------------------
def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_md5(out_dir, ref_dir, rows):
    """単一区間モードの wav を既存の単一 entity 経路の wav と md5 照合する.

    bl  <-> s{idx:04d}_bl.wav
    inj1<-> s{idx:04d}_inj.wav
    """
    pairs = []
    for r in rows:
        for cond, ref_name in (("bl", "bl"), ("inj1", "inj")):
            c = (r.get("conds") or {}).get(cond)
            if not c:
                continue
            mine = os.path.join(out_dir, c["wav"])
            theirs = os.path.join(ref_dir, f"s{r['pair_idx']:04d}_{ref_name}.wav")
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                pairs.append({"pair_idx": r["pair_idx"], "cond": cond,
                              "status": "missing", "mine": mine, "ref": theirs})
                continue
            a, b = md5_of(mine), md5_of(theirs)
            pairs.append({"pair_idx": r["pair_idx"], "cond": cond,
                          "status": "match" if a == b else "MISMATCH",
                          "md5_mine": a, "md5_ref": b,
                          "mine": os.path.basename(mine),
                          "ref": os.path.basename(theirs)})
    n_match = sum(1 for p in pairs if p["status"] == "match")
    n_bad = sum(1 for p in pairs if p["status"] != "match")
    print(f"\n=== md5 等価性 (単一区間) : 一致 {n_match} / 不一致・欠落 {n_bad} ===")
    for p in pairs:
        if p["status"] != "match":
            print(f"  [{p['pair_idx']}] {p['cond']}: {p['status']} "
                  f"{p.get('md5_mine')} vs {p.get('md5_ref')}")
    if n_bad == 0 and n_match:
        print("  すべて bit 一致。移植は単一区間で既存経路と同一。")
    return {"n_match": n_match, "n_bad": n_bad, "ref_dir": ref_dir,
            "pairs": pairs}


# --------------------------------------------------------------------------
def condition_tag(args):
    base = f"equiv_single_{args.lang}" if args.equiv_single else args.pairs_module
    tag = f"{base}_cfg{args.cfg_strength:g}_{args.guidance_method}_" \
          f"{args.splice_part}"
    if args.adaln_source:
        tag += "_adalnsrc"
    if args.steps != L.DEFAULT_STEPS:
        tag += f"_nfe{args.steps}"
    return tag


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs_module", default="en_multientity_300",
                   help="data/<name>.py の build_pairs() (2 枠形式)")
    # **zh を許可した（2026-09-20）。**この制限を書いた時点では中国語の
    # 多 entity データが無かったが、data/zh_multientity_300_v2.py を作った
    # （中国地名 -> 中国地名、50 対 x 6 文型、EN/JA と同じ同種内の入れ替え）。
    # 入口は en_multientity_300 と同じ build_pairs() / spans 2 枠 / text_refs 2 本。
    # **実験条件は何も変えていない。**言語の選択肢を増やしただけ。
    p.add_argument("--lang", default="en", choices=["en", "zh"],
                   help="多 entity データがある言語: EN は en_multientity_300、"
                        "ZH は zh_multientity_300_v2")
    # **中国語の癒着対策（2026-09-20）。**既定 False で従来と完全に同じ経路。
    p.add_argument("--span_extension", action="store_true",
                   help="往復一致しない対に最小対称拡張を試す "
                        "(scripts/projection/span_extend.py の規則)")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--ckpt_every", type=int, default=10)
    # 単一区間の等価性試験
    p.add_argument("--equiv_single", action="store_true",
                   help="cross-culture 300 対を 1 区間で複数 entity 経路に流し、"
                        "既存 wav と md5 照合する")
    p.add_argument("--equiv_ref_dir", default=None,
                   help="既存単一 entity の wav ディレクトリ "
                        "(例 eval_results/_cc_wavs_a/longcat_en_cfg4_cfg_both)")
    # model / decode (既定は run_longcat_eval.py と同じ)
    p.add_argument("--model_id", default=L.DEFAULT_MODEL_ID)
    p.add_argument("--tokenizer_id", default=L.DEFAULT_TOKENIZER_ID)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no_vae_half", action="store_true")
    p.add_argument("--load_mode", default="meta", choices=["meta", "cpu"])
    p.add_argument("--seed", type=int, default=L.SEED)
    p.add_argument("--steps", type=int, default=L.DEFAULT_STEPS)
    p.add_argument("--cfg_strength", type=float, default=L.DEFAULT_CFG)
    p.add_argument("--guidance_method", default=L.DEFAULT_GUIDANCE,
                   choices=["cfg", "apg"])
    p.add_argument("--splice_part", default="both", choices=["both"],
                   help="LongCat の担当条件は both のみ。hL_only は k>=2 で "
                        "h_0 の写像が枠ごとに絡むので本スクリプトでは扱わない")
    p.add_argument("--adaln_source", action="store_true",
                   help="AdaLN の text 条件を source に固定する統制 (既定 off)")
    p.add_argument("--allow_repeat_surface", action="store_true",
                   help="同じ表層形が 1 文に 2 回出るデータを許す "
                        "(左から順に cursor で割り当てる)")
    p.add_argument("--utmos", action="store_true",
                   help="UTMOS22 も測る (torch.hub の SpeechMOS キャッシュが必要)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    args = p.parse_args()

    conds_map = COND_SLOTS_1 if args.equiv_single else COND_SLOTS_2
    conditions = list(conds_map.keys())

    tag = condition_tag(args)
    if args.out_dir is None:
        args.out_dir = os.path.join(ROOT, "eval_results", f"multientity_longcat_{tag}")
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(ROOT, args.out_dir)
    if args.equiv_ref_dir and not os.path.isabs(args.equiv_ref_dir):
        args.equiv_ref_dir = os.path.join(ROOT, args.equiv_ref_dir)
    if args.equiv_ref_dir and os.path.realpath(args.equiv_ref_dir) == \
            os.path.realpath(args.out_dir):
        raise SystemExit("--out_dir と --equiv_ref_dir が同じ。既存 wav を "
                         "上書きしてはいけない。")
    os.chdir(ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    items = load_items(args)
    print(f"[data] {'equiv_single(cross_culture)' if args.equiv_single else args.pairs_module}"
          f" lang={args.lang} n={len(items)} k={items[0]['k']} conds={conditions}",
          flush=True)
    print(f"[data] out_dir={args.out_dir}", flush=True)
    print(f"[data] HF_HOME={os.environ['HF_HOME']}", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    t_load = time.time()
    bundle = L.load_longcat(model_id=args.model_id, tokenizer_id=args.tokenizer_id,
                            device=args.device, dtype=dtype,
                            vae_half=not args.no_vae_half,
                            load_mode=args.load_mode)
    print(f"[load] {bundle.model_dir} in {time.time() - t_load:.1f}s "
          f"missing={bundle.load_report['missing_keys']} "
          f"unexpected={bundle.load_report['unexpected_keys']}", flush=True)
    print(f"[gen] steps={args.steps} cfg={args.cfg_strength} "
          f"guidance={args.guidance_method} seed={args.seed} "
          f"splice_part={args.splice_part} adaln_source={args.adaln_source}",
          flush=True)
    tok = bundle.tokenizer

    utmos = None
    if args.utmos:
        print("[load] UTMOS22 ...", flush=True)
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(args.device).eval()

    ckpt_path = os.path.join(args.out_dir, "checkpoint.json")
    rows, skipped = [], []
    if os.path.exists(ckpt_path) and not args.overwrite:
        st = json.load(open(ckpt_path))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != "splice_multi_longcat"]
        assert not bad, (f"checkpoint mode mismatch in {ckpt_path}: pairs "
                         f"{bad[:5]} は別のスクリプトの出力。--out_dir を分けること")
        print(f"[resume] rows={len(rows)} skipped={len(skipped)}", flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}

    t0 = time.time()
    n_new = 0
    for it in items:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        k = it["k"]

        # ---- 1) 区間の解決 -> 往復確認 -> 重なり検査 (生成の前に済ませる) ----
        try:
            s_ranges, s_surf, s_info = resolve_spans(
                tok, it["text_a"], it["words_a"],
                require_unique=not args.allow_repeat_surface,
                span_extension=args.span_extension,
                peer_texts=it["text_refs"], peer_words=it["words_b"], side="a")
            if not s_info["roundtrip_ok"]:
                raise ValueError(f"src span roundtrip failed: {s_info['roundtrip_bad']}")
            r_ranges, r_info = [], []
            for i in range(k):
                rs, _, ri = resolve_spans(tok, it["text_refs"][i],
                                          [it["words_b"][i]],
                                          require_unique=not args.allow_repeat_surface,
                                          span_extension=args.span_extension,
                                          peer_texts=[it["text_a"]],
                                          peer_words=[it["words_a"][i]], side="b")
                if not ri["roundtrip_ok"]:
                    raise ValueError(f"ref[{i}] span roundtrip failed: "
                                     f"{ri['roundtrip_bad']}")
                r_ranges.append(rs[0])
                r_info.append(ri)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:400]
            print(f"  [{pidx}] SKIP span: {msg}", flush=True)
            skipped.append({"pair_idx": pidx, "stage": "span", "reason": msg,
                            "words_a": it["words_a"], "words_b": it["words_b"]})
            done.add(pidx)
            if args.stop_on_error:
                raise
            continue

        # ---- 2) encoder forward (src 1 回 + ref 枠ごとに 1 回) ----
        try:
            src = L.encode_for_text(bundle, it["text_a"])
            ref_parts = [L.encode_for_text(bundle, t) for t in it["text_refs"]]
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:400]
            print(f"  [{pidx}] SKIP enc: {msg}", flush=True)
            skipped.append({"pair_idx": pidx, "stage": "encode", "reason": msg})
            done.add(pidx)
            if args.stop_on_error:
                raise
            continue

        T_src = int(src["H"].shape[1])
        # 重なり検査は build_spliced_hidden_multi 内の check_span_sets でも
        # 走るが、生成前に全枠まとめて落としたいのでここで明示的に呼ぶ。
        try:
            check_span_sets(T_src, [int(rp["H"].shape[1]) for rp in ref_parts],
                            s_ranges, r_ranges)
        except AssertionError as e:
            msg = f"span sets invalid: {e}"[:400]
            print(f"  [{pidx}] SKIP {msg}", flush=True)
            skipped.append({"pair_idx": pidx, "stage": "span_sets", "reason": msg})
            done.add(pidx)
            if args.stop_on_error:
                raise
            continue

        # ---- 3) duration は必ず source テキスト由来 (4 条件で共有) ----
        dur, dur_sec_est = L.duration_frames(bundle, src["norm_text"])

        d_per, d_tot = span_deltas(s_ranges, r_ranges)
        span_tok = {
            "src": [list(x) for x in s_ranges],
            "ref": [list(x) for x in r_ranges],
            "n": [s_ranges[i][1] - s_ranges[i][0] for i in range(k)],
            "m": [r_ranges[i][1] - r_ranges[i][0] for i in range(k)],
            "delta_per_span": [int(x) for x in d_per],
            "gap_tokens": (s_ranges[1][0] - s_ranges[0][1]) if k == 2 else None,
            "len_imbalance": abs((r_ranges[0][1] - r_ranges[0][0])
                                 - (r_ranges[1][1] - r_ranges[1][0]))
            if k == 2 else None,
            "src_tokens": [s["covered"] for s in s_info["per_span"]],
            "ref_tokens": [ri["per_span"][0]["covered"] for ri in r_info],
            "src_resolve_via": [s["via"] for s in s_info["per_span"]],
            "src_crosscheck": s_info["crosscheck"],
            "ref_crosscheck": [ri["crosscheck"][0] for ri in r_info],
            "norm_text_src": s_info["norm_text"],
            "norm_text_refs": [ri["norm_text"] for ri in r_info],
        }

        row = {
            "pair_idx": pidx, "mode": "splice_multi_longcat",
            "method": "longcat", "lang": args.lang,
            "template_id": it.get("template_id"),
            "text_a": it["text_a"], "text_refs": it["text_refs"],
            "text_both": it["text_both"],
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "words_a": list(it["words_a"]), "words_b": list(it["words_b"]),
            "span_tok": span_tok,
            "T_src": T_src,
            "T_refs": [int(rp["H"].shape[1]) for rp in ref_parts],
            "duration_frames": int(dur),
            "duration_sec_est": round(float(dur_sec_est), 4),
            "conds": {},
        }

        # ---- 4) 4 条件を生成。seed は generate_from_embedding の先頭で毎回戻る ----
        ok = True
        for cond in conditions:
            slots = conds_map[cond]
            try:
                modified, delta_tok, out_ranges, T_out = build_condition_hidden(
                    src, ref_parts, s_ranges, r_ranges, slots)
            except Exception as e:
                msg = f"{cond} splice: {type(e).__name__}: {e}"[:400]
                print(f"  [{pidx}] {msg}", flush=True)
                skipped.append({"pair_idx": pidx, "stage": "splice", "reason": msg})
                ok = False
                break

            try:
                if not slots:
                    # 既存 run_longcat_eval.py:239 (BL) と同じ呼び方。
                    # BL は差し替えが無いので AdaLN 統制の対象にもならない。
                    audio = L.generate_from_embedding(
                        bundle, modified, dur, seed=args.seed, steps=args.steps,
                        cfg_strength=args.cfg_strength,
                        guidance_method=args.guidance_method)
                else:
                    # 既存 L.splice_synthesize:615-620 と同じ包み方。
                    # adaln_source=False (既定) では __enter__ が何もしないので
                    # hook は登録されない。
                    override = L.AdaLNSourceOverride(
                        bundle, modified, src["H"], enabled=bool(args.adaln_source))
                    with override:
                        audio = L.generate_from_embedding(
                            bundle, modified, dur, seed=args.seed, steps=args.steps,
                            cfg_strength=args.cfg_strength,
                            guidance_method=args.guidance_method)
            except Exception as e:
                msg = f"{cond} synth: {type(e).__name__}: {e}"[:400]
                print(f"  [{pidx}] {msg}", flush=True)
                skipped.append({"pair_idx": pidx, "stage": "synth", "reason": msg})
                ok = False
                if args.stop_on_error:
                    raise
                break

            wav_name = f"s{pidx:04d}_{cond}.wav"
            wav_path = os.path.join(args.out_dir, wav_name)
            try:
                n_smp, sec, clip = R.save_wav(wav_path, audio, bundle.sampling_rate)
            except Exception as e:
                msg = f"{cond} save: {type(e).__name__}: {e}"[:400]
                print(f"  [{pidx}] {msg}", flush=True)
                skipped.append({"pair_idx": pidx, "stage": "save", "reason": msg})
                ok = False
                break

            c = {
                "wav": wav_name,
                "delta_tok": int(delta_tok),
                "T_out": int(T_out),
                "out_ranges": out_ranges,
                # LongCat は splice 後のテキストを作らない (duration は文字数から
                # 決まり、系列は H のまま渡す)。ここは「実際に duration と
                # source 系列を決めたテキスト」= 正規化済み text_a を記録する。
                "target_text": src["norm_text"],
                "dur_sec": round(float(sec), 4),
                "n_samples": int(n_smp),
                "clip_ratio": round(float(clip), 6),
                "duration_frames": int(dur),
                "slots": list(slots),
            }
            if utmos is not None:
                try:
                    import soundfile as sf
                    w, sr = sf.read(wav_path, dtype="float32")
                    with torch.no_grad():
                        c["utmos"] = round(float(utmos(
                            torch.from_numpy(w).unsqueeze(0).to(args.device), sr)), 4)
                except Exception as e:
                    c["utmos_err"] = f"{type(e).__name__}: {e}"[:160]
            row["conds"][cond] = c

        done.add(pidx)
        if not ok:
            continue
        # 系列長の総ずれ。長さが変わるので inj12 の delta_tok を代表値にする。
        row["total_position_drift"] = row["conds"][conditions[-1]]["delta_tok"]
        assert row["total_position_drift"] == int(d_tot), (
            f"delta_tok bookkeeping broken: {row['total_position_drift']} != {d_tot}")
        rows.append(row)
        n_new += 1

        if n_new % args.log_every == 0:
            el = time.time() - t0
            rate = el / max(1, n_new)
            print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                  f"gap_tok={span_tok['gap_tokens']} "
                  f"n={span_tok['n']} m={span_tok['m']} "
                  f"delta={row['total_position_drift']} "
                  f"T_out={[row['conds'][c2]['T_out'] for c2 in conditions]} "
                  f"{el / 60:.1f}min ({rate:.1f}s/文)", flush=True)
        if n_new % args.ckpt_every == 0:
            R.atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt_path)

    R.atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt_path)

    def _mean(key, cond):
        v = [r["conds"][cond][key] for r in rows
             if r["conds"].get(cond, {}).get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    summary = {
        "method": "longcat", "mode": "splice_multi_longcat",
        "pairs_module": "en_cross_culture_names" if args.equiv_single
        else args.pairs_module,
        "equiv_single": bool(args.equiv_single),
        "lang": args.lang, "conditions": conditions,
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "model_id": args.model_id, "model_dir": bundle.model_dir,
        "tokenizer_id": args.tokenizer_id,
        "seed": args.seed, "steps_nfe": args.steps,
        "cfg_strength": args.cfg_strength, "guidance_method": args.guidance_method,
        "splice_part": args.splice_part, "adaln_source": bool(args.adaln_source),
        "dtype": args.dtype, "vae_half": not args.no_vae_half,
        "sampling_rate": bundle.sampling_rate, "latent_hop": bundle.latent_hop,
        "spliced_tensor": "H = LN(h_L) + LN(h_0) of the frozen umt5-base encoder "
                          "(same tensor as kappa.H_cross_attn_input)",
        "controls": {
            "duration_from_source_text": True,
            "seed_reset_immediately_before_each_generation": True,
            "ref_per_slot_separate_forward": True,
            "index_shift_done_by_splice_multi_only": True,
        },
        "mean_dur_sec": {c: _mean("dur_sec", c) for c in conditions},
        "mean_delta_tok": {c: _mean("delta_tok", c) for c in conditions},
        "mean_abs_position_drift": round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3) if rows else None,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    if utmos is not None:
        summary["utmos"] = {c: _mean("utmos", c) for c in conditions}
    if args.equiv_single and args.equiv_ref_dir:
        summary["md5_equivalence"] = compare_md5(args.out_dir, args.equiv_ref_dir, rows)

    R.atomic_json_dump({"summary": summary, "rows": rows, "skipped": skipped},
                       os.path.join(args.out_dir, "results.json"))
    print(f"\n=== multi-entity longcat {tag} ===")
    print(json.dumps({k2: v for k2, v in summary.items()
                      if k2 != "md5_equivalence"}, ensure_ascii=False, indent=1))
    if skipped:
        print(f"[Warn] {len(skipped)} 件 skip。results.json['skipped'] を見ること")
    print(f"[out] {os.path.join(args.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
