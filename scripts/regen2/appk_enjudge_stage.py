#!/usr/bin/env python3
"""付録 K 英語の判定: GPU に渡す一覧（stage_manifest.json）を作る（2026-09-29、ホストの CPU のみ）.

音声は写さない（本番の utmos_speaker_fixed_en_n300_splice/ を GPU の側がそのまま読む。md5 で照合する）。
確かめること（1 つでも外れたら一覧を書かない）:
  1. data/en_heteronyms_500.py と本番の checkpoint.json が決めた md5 のままで、checkpoint の 300 行が
     data ファイルの build_500_pairs() の先頭 300 対（30 語 × 10 対）と同じ並び。
  2. 反映の記録 regen2_appk_pairs.json（md5 固定）: 差し替えは対 23 だけ、seed 44、未解決 0、
     記録の新しい md5 が今の s0023_bl.wav / s0023_inj.wav と同じ。
  3. 反映後の無音の検査（_appk_redo/silent_scan_n300.json、md5 固定）の EN の 600 本の md5 が今の音声と同じで、
     使えない対 0。さらに、regen2 共通の無音の基準（silence_criterion.assess_file）で 600 本を調べ直して無音 0。
  4. 読みの表（eval_results/_appk_enjudge/en_reading_table_v1.json）が 30 語 × 2 読みで、ラベルが data ファイルと同じ、
     単位がすべて認識器の語彙にある。2 つの読みが同じ単位列の語（identical）はここで単位列から決め直す。
  5. 300 対の仮説（文全体の単位列。対象の語だけ src / tgt の読み。ほかの語は CMUdict の最初の発音を、表と同じ版
     （v1 か v2）の規則で置き換えたもの）が組め、文の中の対象の語がちょうど 1 個、src と tgt が同じ仮説になる対 =
     identical の語の対（v1 は 60 対、v2 は 90 対）。
  6. 表と仮説のすべての記号を、同じ認識器が同じ 600 本に出した記号の数（_appk_redo/ipa_en_n300.json、md5 固定）と
     照らし、一覧に記録して表示する（記号ごとの総数だけを使う。止める条件ではなく、著者が表を認める前に見るもの）。
書くのは --table_approved_by_author を付けたときだけ（著者が表を認めてから）。--dry_run は何も書かない。
同じ場所に別の中身の一覧、または GPU の出力が既にあれば書かない（新しい --stage_dir を使う）。
モデルも Hugging Face も numba も読まないが、決まりどおり HF_HOME などを最初に設定する。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定する）:
  予行: taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_stage.py --dry_run
  本番（著者が v1 の表を認めたとき）:
        taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_stage.py --table_approved_by_author
  本番（著者が v2 の表を認めたとき）:
        taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_stage.py --table eval_results/_appk_enjudge/en_reading_table_v2.json --table_approved_by_author
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

K = C.K
K.cap_threads()

import argparse  # noqa: E402
import time  # noqa: E402


def check_inputs(silence=True):
    """手順 1–3。戻り値: (情報, problems)."""
    info, problems = {}, []
    for tag, path in (("data_file", C.DATA_FILE), ("checkpoint", C.PROD_CKPT), ("apply_record", C.APPLY_RECORD),
                      ("post_apply_scan", C.POST_APPLY_SCAN)):
        ok, m = C.check_pinned(tag, path)
        info[f"{tag}_md5"] = m
        if not ok:
            problems.append(f"{C.proj_rel(path)} の md5 {m} が {C.EXPECTED_MD5[tag]} でない")
    if problems:
        return info, problems
    ckpt_rows = C.read_json(C.PROD_CKPT)["rows"]
    data_pairs, words, readings = C.load_data_pairs()
    problems += C.check_rows_vs_data(ckpt_rows, data_pairs)
    if len(words) != C.N_WORDS or any(sum(d["word"] == w for d in data_pairs) != C.PAIRS_PER_WORD for w in words):
        problems.append("30 語 × 10 対になっていない")
    info.update(data_pairs=data_pairs, words=words, readings=readings,
                checkpoint_rows_digest=C.md5_json(ckpt_rows))

    # 2. 反映の記録
    rec = C.read_json(C.APPLY_RECORD)
    cur = {n: C.md5_file(os.path.join(C.PROD_DIR, n)) for n in C.APPLY_EXPECT["new_md5"]}
    rec_md5 = {f["file"]: f["new_md5"] for f in rec.get("files", [])}
    ar = {"pairs_replaced": rec.get("pairs_replaced"), "seeds": rec.get("seeds"),
          "unresolved": rec.get("unresolved_pairs_not_replaced"), "date": rec.get("date"),
          "record_new_md5": rec_md5, "current_md5": cur}
    info["apply_record"] = ar
    if rec.get("lang") != "en" or rec.get("pairs_replaced") != C.APPLY_EXPECT["pairs_replaced"] \
            or rec.get("seeds") != C.APPLY_EXPECT["seeds"] \
            or rec.get("unresolved_pairs_not_replaced") != C.APPLY_EXPECT["unresolved_pairs_not_replaced"]:
        problems.append(f"反映の記録が「対 23 だけ seed 44、未解決 0」でない: {ar}")
    if rec_md5 != C.APPLY_EXPECT["new_md5"] or cur != C.APPLY_EXPECT["new_md5"]:
        problems.append(f"対 23 の音声の md5 が反映の記録と合わない: 記録 {rec_md5}、今 {cur}")

    # 3. 反映後の無音の検査との照合と、無音の調べ直し
    scan = C.read_json(C.POST_APPLY_SCAN)
    en = scan["langs"]["en"]
    per = scan["per_wav"]["en"]
    files, changed = {}, []
    for n in C.all_wav_names():
        p = os.path.join(C.PROD_DIR, n)
        if not os.path.isfile(p):
            problems.append(f"音声がない: {n}")
            continue
        m = C.md5_file(p)
        files[n] = {"md5": m, "size": os.path.getsize(p)}
        if (per.get(n) or {}).get("md5") != m:
            changed.append(n)
    if len(files) != 2 * C.N:
        problems.append(f"音声が {len(files)} 本（600 でない）")
    if changed:
        problems.append(f"反映後の無音の検査（{scan.get('date')}）のあとで音声が変わった: {changed[:5]}（{len(changed)} 本）")
    if en.get("n_unusable_pairs") != 0 or en.get("n_silent_pairs") != 0 or en.get("n_ckpt_rows") != C.N:
        problems.append(f"反映後の無音の検査で EN に使えない対がある: {en.get('n_unusable_pairs')}")
    pac = C.read_json(C.POST_APPLY_CHECK) if os.path.isfile(C.POST_APPLY_CHECK) else {}
    pac_en = (pac.get("langs") or {}).get("en") or {}
    if not pac_en.get("ok"):
        problems.append("反映後の照合の記録（post_apply_check）で EN が ok でない")
    info["files"] = files
    info["post_apply_scan"] = {"path": C.proj_rel(C.POST_APPLY_SCAN), "date": scan.get("date"),
                               "n_unusable_pairs_en": en.get("n_unusable_pairs"),
                               "post_apply_check_en_ok": pac_en.get("ok"), "wavs_changed_since_scan": changed}
    if silence:
        bad = []
        for n in C.all_wav_names():
            p = os.path.join(C.PROD_DIR, n)
            if os.path.isfile(p):
                w = K.wav_check(p)
                if w.get("bad"):
                    bad.append({"file": n, "why": w.get("why")})
        info["silence_recheck"] = {"n_checked": len(files), "n_bad": len(bad), "bad": bad,
                                   "criterion": K.SILENCE_CRITERION}
        if bad:
            problems.append(f"無音・読めない音声がある: {bad[:5]}")
    return info, problems


def build_stage(table_path, approved, silence=True):
    """手順 1–5 をすべて行い、一覧（manifest）を組む（書かない）。戻り値: (manifest, problems, waits)."""
    waits = []
    info, problems = check_inputs(silence=silence)
    if problems:
        return None, problems, waits
    vocab = C.load_vocab()
    if not os.path.isfile(table_path):
        return None, [f"読みの表がない: {C.proj_rel(table_path)}（appk_enjudge_table.py を先に流す）"], waits
    table = C.read_json(table_path)
    words, readings, data_pairs = info["words"], info["readings"], info["data_pairs"]
    problems += C.check_table(table, words, readings, vocab)
    if problems:
        return None, problems, waits
    tu = C.table_units(table)
    try:
        mapping = C.table_mapping(table)
    except SystemExit as e:
        return None, [str(e)], waits
    cats = C.categories_from_units(tu, words, readings)
    stored_cats = {w: (table.get("words") or {}).get(w, {}).get("category") for w in words}
    cat_note = {w: [stored_cats[w], cats[w]] for w in words if stored_cats[w] != cats[w]}
    if cat_note:
        waits.append(f"表の words.category と単位列から決め直した分類が違う語（単位列の方を使う）: {cat_note}")
    lex, lex_info = C.cmudict_lexicon()
    hyp, hp = C.build_hypotheses(data_pairs, tu, lex, vocab, mapping=mapping)
    problems += hp
    if len(hyp) != C.N:
        problems.append(f"仮説を組めた対が {len(hyp)}（300 でない）")
    for h in hyp:
        h["category"] = cats[h["word"]]
    deg_pairs = [h["pair_idx"] for h in hyp if h["degenerate"]]
    ident_pairs = [h["pair_idx"] for h in hyp if cats[h["word"]] == "identical"]
    if deg_pairs != ident_pairs:
        problems.append(f"src と tgt が同じ仮説の対 {deg_pairs[:10]} が、読みが同じ語の対 {ident_pairs[:10]} と一致しない")
    if any(min(h["ids_src"] + h["ids_tgt"]) < 4 for h in hyp):
        problems.append("仮説に特別な記号（空白・文頭・文末・不明）の id が入っている")
    # CTC が成り立つか: 認識器のフレーム数（16 kHz で 320 標本ごと、窓 400）≥ 単位の数 + 同じ単位の続き
    per = C.read_json(C.POST_APPLY_SCAN)["per_wav"]["en"]
    ratios = []
    for h in hyp:
        for side in C.WAV_SIDES:
            n = int(round(per[C.wav_name(h["pair_idx"], side)]["dur"] * 16000))
            frames = (n - 400) // 320 + 1
            for ids in (h["ids_src"], h["ids_tgt"]):
                need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)
                ratios.append((frames / need, h["pair_idx"], side))
    ratios.sort()
    info["ctc_frames_margin"] = {"min_frames_over_needed": round(ratios[0][0], 2), "at": list(ratios[0][1:])}
    if ratios[0][0] < 1.0:
        problems.append(f"音素列が認識器のフレームに収まらない対がある: {ratios[:5]}")
    # 6. 表と仮説の記号を、認識器が同じ 600 本に出した記号の数と照らす（記号ごとの総数だけ。記録と表示）
    try:
        counts, inv = C.load_emission_inventory()
        info["symbol_emission"] = C.symbol_emission_check(table, hyp, words, readings, data_pairs, counts, inv)
    except SystemExit as e:
        problems.append(str(e))
    if not approved:
        waits.append("著者が読みの表を認めていない（認めたら --table_approved_by_author を付けて流す）")
    if problems:
        return None, problems, waits
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "created_by": "scripts/regen2/appk_enjudge_stage.py",
           "mode": "post_apply", "selftest": False,
           "what": ("Appendix K English injection success with the Chinese-row judge: CTC log-likelihood of the "
                    "whole-sentence phoneme sequence with the target vs the source reading at the entity, on "
                    f"{C.MODEL_ID}, for BL and INJ of the 300 EN pairs"),
           "table": {"path": C.proj_rel(table_path), "md5": C.md5_file(table_path), "author_approved": bool(approved),
                     "version": table.get("version"), "mapping_version": mapping},
           "context_mapping": mapping,
           "prod_dir": C.proj_rel(C.PROD_DIR), "files": info["files"],
           "checkpoint": {"path": C.proj_rel(C.PROD_CKPT), "md5": info["checkpoint_md5"],
                          "rows_digest": info["checkpoint_rows_digest"]},
           "data_file": {"path": C.proj_rel(C.DATA_FILE), "md5": info["data_file_md5"]},
           "apply_record": {"path": C.proj_rel(C.APPLY_RECORD), "md5": info["apply_record_md5"], **info["apply_record"]},
           "post_apply_scan": info["post_apply_scan"], "silence_recheck": info.get("silence_recheck"),
           "recognizer": {"model": C.MODEL_ID, "snapshot_main": C.SNAP_MAIN, "snapshot_pr7": C.SNAP_PR7,
                          "vocab_md5": C.md5_file(C.VOCAB), "blank_id": C.BLANK_ID},
           "cmudict": lex_info, "context_additions": {k: " ".join(v) for k, v in C.CONTEXT_ADDITIONS.items()},
           "categories": cats, "degenerate_pairs": deg_pairs,
           "degenerate_words": [w for w in words if cats[w] == "identical"],
           "replaced_pairs": C.REPLACED_PAIRS, "ctc_frames_margin": info["ctc_frames_margin"],
           "symbol_emission": info["symbol_emission"],
           "hypotheses_md5": C.hypotheses_digest(hyp), "hypotheses": hyp}
    return man, problems, waits


def comparable(man):
    return {k: v for k, v in man.items() if k not in ("created", "silence_recheck")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_dir", default=C.proj_rel(C.DEFAULT_STAGE))
    ap.add_argument("--table", default=C.proj_rel(C.TABLE_PATH))
    ap.add_argument("--table_approved_by_author", action="store_true",
                    help="著者が読みの表を認めたときだけ付ける（一覧に記録する）")
    ap.add_argument("--dry_run", action="store_true", help="確かめるだけ（何も書かない）")
    args = ap.parse_args()
    stage_dir = C.resolve(args.stage_dir)
    table_path = C.resolve(args.table)
    t0 = time.time()
    man, problems, waits = build_stage(table_path, args.table_approved_by_author)
    print(f"[enjudge-stage] stage={C.proj_rel(stage_dir)} table={C.proj_rel(table_path)} "
          f"approved={args.table_approved_by_author} dry_run={args.dry_run} ({time.time() - t0:.1f} s)")
    if man:
        ar = man["apply_record"]
        print(f"  inputs: 300 rows = data pairs, apply record pairs {ar['pairs_replaced']} seeds {ar['seeds']}, "
              f"600 wavs = post-apply scan md5, silence recheck bad {man['silence_recheck']['n_bad']}")
        print(f"  table md5 {man['table']['md5']}, hypotheses md5 {man['hypotheses_md5']}, "
              f"degenerate (identical readings) {len(man['degenerate_pairs'])} pairs: {man['degenerate_words']}")
        lens = [len(h["ids_src"]) for h in man["hypotheses"]]
        print(f"  hypothesis length (units) min {min(lens)} max {max(lens)}; possessive pairs "
              f"{[h['pair_idx'] for h in man['hypotheses'] if h['possessive']]}; context additions used "
              f"{[h['pair_idx'] for h in man['hypotheses'] if h['context_notes']]}; other words mapped with "
              f"{man['context_mapping']}")
        for line in C.emission_check_lines(man["symbol_emission"], full=False):
            print("  " + line)
    for w in waits:
        print(f"  WAIT: {w}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    if problems:
        sys.exit(1)
    if args.dry_run:
        print("  dry run: READY (nothing written)" if not waits or all("認めていない" in w for w in waits)
              else "  dry run: READY with notes (nothing written)")
        return
    if not args.table_approved_by_author:
        print("  not written: the author has not approved the reading table yet")
        sys.exit(1)
    mp = os.path.join(stage_dir, C.MANIFEST)
    C.assert_write(mp)
    if os.path.exists(os.path.join(stage_dir, C.JUDGE_OUT)):
        sys.exit(f"  {C.proj_rel(stage_dir)} に GPU の出力が既にある。新しい --stage_dir を使う。")
    if os.path.exists(mp):
        old = C.read_json(mp)
        if C.md5_json(comparable(old)) == C.md5_json(comparable(man)):
            print(f"  unchanged: {C.proj_rel(mp)} (nothing written)")
            return
        sys.exit(f"  {C.proj_rel(mp)} に別の中身の一覧がある。上書きしない。新しい --stage_dir を使う。")
    C.write_json(man, mp, backup=False)
    print(f"  written: {C.proj_rel(mp)} (md5 {C.md5_file(mp)})")
    print("  next (GPU, from the project root): SLURM_TIME=00:20:00 slurm_run.sh "
          f"python3 -u scripts/regen2/appk_enjudge_run_gpu.py --stage_dir {C.proj_rel(stage_dir)}")


if __name__ == "__main__":
    main()
