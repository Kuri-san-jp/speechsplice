#!/usr/bin/env python3
"""付録 K 英語の判定: 30 語 × 2 読み = 60 個の読みの表を作る（2026-09-29、ホストの CPU のみ。モデルは使わない）.

作り方（前もって決めた規則だけ。結果を見て変えない）:
  1. ラベルの大文字の部分を強勢のある音節とする（noun_REcord → 1 音節目、verb_reCORD → 2 音節目）。
  2. CMU 発音辞書（ホストの cmudict 1.1.3、md5 5837aa6e…）で、第 1 強勢がその音節にある最初の発音を選ぶ。
     無ければ、もう一方の読みの第 1 強勢をその音節に移して作る（export の動詞だけ。derived と記す）。
  3. ARPAbet を認識器（wav2vec2-xlsr-53-espeak-cv-ft）の音素単位に直す（appk_enjudge_common の規則）。
     v1（最初の表）: 強勢で変わるのは AH（ʌ/ə）と ER（ɜː/ɚ）だけ。r の付いた母音は ɑːɹ・ɔːɹ・ɛɹ・ɪɹ・ʊɹ の 1 記号。
     v2（22 時に、どの結果も見る前に固定）: ER は強勢によらず ɚ、AO+R → oː、AA+R → ɑː、ほかの母音+R → 母音 + 別の ɹ。
     ほかは v1 と同じ。文の中のほかの語も、表と同じ版で置き換える（stage が表の版を読む）。
  4. 2 つの読みの単位列を比べ、同じ（identical）/ 弱い母音と強い母音の違いだけ（reduced_vowel_only）/
     それ以外（segmental）に分ける。難しい語（export・research・upset）には注記と別の読みを付ける。
  5. 記号の照合（CPU だけ）: 表と 300 文の仮説のすべての記号を、同じ認識器が同じ 600 本に出した記号の数
     （eval_results/_appk_redo/ipa_en_n300.json の bl_ipa と inj_ipa、16,006 記号。md5 固定）と並べて表示し、
     2 つの読みの違う所に「1 度も出ない記号」がある語と対を数える。使うのは記号ごとのファイル全体の回数だけで、
     どの対の成否も使わない。記録は eval_results/_appk_enjudge/symbol_emission_check_<版>.json。
書き先: eval_results/_appk_enjudge/en_reading_table_v1.json / en_reading_table_v2.json（著者が見て直せる JSON）。
既にあれば、中身（作った時刻を除く）が同じなら何も書かない。違えば上書きせずに止まる（著者が直した表を消さない）。
著者が直した表の照合だけをするときは --check_only（表は作らず、--out の表を読む）。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_table.py                 （v1 と照合）
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_table.py --mapping v2    （v2 と照合）
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_table.py --check_only --out <表>
  --dry_run を付けると何も書かない（表示だけ）。
モデルも Hugging Face も numba も読まないが、決まりどおり HF_HOME などを最初に設定する。
"""
import os
import sys

sys.dont_write_bytecode = True
# numpy / torch を読む前に（このスクリプトは読まないが、決まりどおり最初に設定する。appk_enjudge_run_gpu.py と同じ）
_PROJ = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
if _PROJ != "/mnt":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ホストでは GPU を見せない
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
_HF = os.path.join(_PROJ, ".cache", "huggingface")
os.environ["HF_HOME"] = _HF
os.environ["HF_HUB_CACHE"] = os.path.join(_HF, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(_PROJ, ".cache", "numba")
sys.modules["pyopenjtalk"] = None   # 自分のコードでは pyopenjtalk を使わない
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_enjudge_common as C  # noqa: E402

C.K.cap_threads()

import argparse  # noqa: E402
import time  # noqa: E402


def make_table(mapping="v1"):
    lex, info = C.cmudict_lexicon()
    ok, m = C.check_pinned("data_file", C.DATA_FILE)
    if not ok:
        raise SystemExit(f"{C.proj_rel(C.DATA_FILE)} の md5 {m} が {C.EXPECTED_MD5['data_file']} でない")
    data_pairs, words, readings = C.load_data_pairs()
    if len(words) != C.N_WORDS:
        raise SystemExit(f"語が {len(words)}（30 でない）")
    vocab = C.load_vocab()
    return C.build_table(lex, info, words, readings, data_pairs, vocab, mapping=mapping)


def emission_check(table):
    """表と 300 文の仮説の記号を、認識器が同じ 600 本に出した記号の数と照らす（CPU だけ。対の成否は使わない）."""
    lex, _ = C.cmudict_lexicon()
    data_pairs, words, readings = C.load_data_pairs()
    vocab = C.load_vocab()
    probs = C.check_table(table, words, readings, vocab)
    if probs:
        raise SystemExit(f"表の形が合わない: {probs[:3]}")
    tu = C.table_units(table)
    hyp, hp = C.build_hypotheses(data_pairs, tu, lex, vocab, mapping=C.table_mapping(table))
    if hp or len(hyp) != C.N:
        raise SystemExit(f"仮説を 300 対組めない: {hp[:3]}")
    counts, inv = C.load_emission_inventory()
    chk = C.symbol_emission_check(table, hyp, words, readings, data_pairs, counts, inv)
    chk["hypotheses_md5"] = C.hypotheses_digest(hyp)
    return chk


def comparable(t):
    return {k: v for k, v in t.items() if k != "created"}


def print_table(t):
    print(f"[enjudge-table] version {t.get('version')} (mapping {C.table_mapping(t)}) {t['summary']}")
    for w, info in t["words"].items():
        ent = [e for e in t["entries"] if e["word"] == w]
        s = "  |  ".join(f"{e['label']}: {e['display']}" + (" (derived)" if e["derived_by_moving_stress"] else "")
                         for e in ent)
        mark = " HARD" if info["hard_case"] else ""
        v1 = f"  (v1: {info['v1_category']})" if info.get("v1_category") not in (None, info["category"]) else ""
        print(f"  {w:9s} {info['category']:18s}{mark} {s}  {info['differences']}{v1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", choices=C.MAPPINGS, default="v1",
                    help="置き換えの版（v1 = 最初の表、v2 = 認識器が出す記号に合わせた版）")
    ap.add_argument("--out", default=None, help="表の置き場（既定 v1: en_reading_table_v1.json、v2: en_reading_table_v2.json）")
    ap.add_argument("--check_only", action="store_true", help="表は作らず、--out の表の記号の照合だけをする")
    ap.add_argument("--dry_run", action="store_true", help="表と照合を表示するだけ（何も書かない）")
    args = ap.parse_args()
    out = C.resolve(args.out) if args.out else C.TABLE_PATHS[args.mapping]
    if args.check_only:
        if not os.path.isfile(out):
            raise SystemExit(f"表がない: {C.proj_rel(out)}")
        t = C.read_json(out)
        print(f"[enjudge-table] check only: {C.proj_rel(out)} (md5 {C.md5_file(out)})")
        print_table(t)
    else:
        t = make_table(args.mapping)
        print_table(t)
        if args.dry_run:
            print("  dry run: table not written")
        else:
            C.assert_write(out)
            if os.path.exists(out):
                old = C.read_json(out)
                if C.md5_json(comparable(old)) != C.md5_json(comparable(t)):
                    raise SystemExit(f"{C.proj_rel(out)} が既にあり、中身が違う（著者が直した表かもしれない）。上書きしない。"
                                     f"直した表の照合は --check_only --out {C.proj_rel(out)}。")
                print(f"  unchanged: {C.proj_rel(out)} (nothing written)")
                t = old
            else:
                C.write_json(t, out, backup=False)
                print(f"  written: {C.proj_rel(out)} (md5 {C.md5_file(out)})")
                t = C.read_json(out)
    chk = emission_check(t)
    for line in C.emission_check_lines(chk, full=True):
        print(line)
    if args.dry_run:
        print("  dry run: symbol check not written")
        return
    chk = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "table_file": C.proj_rel(out), "table_md5": C.md5_file(out) if os.path.isfile(out) else None, **chk}
    cp = os.path.join(os.path.dirname(out), C.EMISSION_CHECK_NAME.format(version=t.get("version")))
    C.assert_write(cp)
    if os.path.exists(cp):
        old = C.read_json(cp)
        if C.md5_json({k: v for k, v in old.items() if k != "date"}) == C.md5_json(
                {k: v for k, v in chk.items() if k != "date"}):
            print(f"  symbol check unchanged: {C.proj_rel(cp)} (nothing written)")
            return
    C.write_json(chk, cp)   # 前の版は .bak_<時刻> に残す
    print(f"  symbol check written: {C.proj_rel(cp)} (md5 {C.md5_file(cp)})")


if __name__ == "__main__":
    main()
