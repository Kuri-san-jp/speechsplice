#!/usr/bin/env python3
"""付録 K の IPA 変化の採点で、JA・ZH の音声を採点器が読める名前に並べ直す（2026-09-29、CPU のみ）.

採点器 scripts/projection/score_phoneme_asr_from_dir.py（変更しない）は、1 つのフォルダの中の
s{i:04d}_bl.wav と s{i:04d}_inj.wav の組を読む。EN/FR/DE/KO（utmos_speaker_fixed_*_n300_splice）は
この名前のままなので直接渡せる。JA（cjk_n300_rewav）と ZH（cjk_n300_zh_v2）は注入側が s{i:04d}_oeoi.wav
なので、そのままでは 1 対も読まれない。

そこで、新しいフォルダ（--stage_root/stage_{ja,zh}/）に相対パスの symlink を置く:
  s{i:04d}_bl.wav  -> <元の集合>/s{i:04d}_bl.wav
  s{i:04d}_inj.wav -> <元の集合>/s{i:04d}_oeoi.wav
相対パスなので、ホスト（$REPO_ROOT）でもコンテナ（/mnt）でも同じ実体を指す。
元の音声はコピーも書き換えもしない。対応表と md5 を stage_manifest.json に残す。

最後に、採点器と同じ探し方（os.listdir で *_bl.wav を集め、同じ番号の *_inj.wav があるか）で
300 対が見つかること、symlink の先の md5 が元の音声と一致することを確かめる。

既にフォルダがある場合: 中身が今回作るものと同じなら何もしない。違えば止まる（上書きしない）。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定するので、env や nice は付けない。
相対パスの --stage_root はプロジェクト直下を基準に解く）:
  本番: taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_prepare_stage.py --stage_root eval_results/_appk_redo
  試し: taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_prepare_stage.py --stage_root eval_results/_regen2_dryrun/appk_stage
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
if not os.path.isdir("/mnt/scripts"):   # ホストで動くときだけ優先度を 10 に下げる（何度呼んでも 10 のまま）
    try:
        _cur = os.nice(0)
        if _cur < 10:
            os.nice(10 - _cur)
    except OSError:
        pass

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from appk_silent_scan import SETS, N_EXPECTED, PROJ, ER  # noqa: E402

STAGED_LANGS = ("ja", "zh")  # 注入側の名前が _inj でない集合だけ並べ直す


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_dir_of(stage_root, lang):
    return os.path.join(stage_root, f"stage_{lang}")


def planned_links(lang, sdir):
    spec = SETS[lang]
    src_dir = os.path.join(ER, spec["dir"])
    links = {}
    for i in range(N_EXPECTED):
        for side, stage_suffix in (("bl", "bl"), ("inj", "inj")):
            src = os.path.join(src_dir, f"s{i:04d}_{spec[side]}.wav")
            links[f"s{i:04d}_{stage_suffix}.wav"] = (src, os.path.relpath(src, sdir))
    return src_dir, links


def discover_like_scorer(wav_dir):
    """score_phoneme_asr_from_dir.py の 103–116 行と同じ探し方."""
    files = sorted([f for f in os.listdir(wav_dir) if f.endswith("_bl.wav")])
    found = []
    for bl_name in files:
        idx = bl_name.split("_")[0]
        if os.path.exists(os.path.join(wav_dir, f"{idx}_inj.wav")):
            found.append(idx)
    return files, found


def build(lang, stage_root, check_md5=True):
    sdir = stage_dir_of(stage_root, lang)
    src_dir, links = planned_links(lang, sdir)
    missing_src = [s for s, _ in links.values() if not os.path.isfile(s)]
    if missing_src:
        raise SystemExit(f"[{lang}] 元の wav が欠けている: {missing_src[:5]} ... ({len(missing_src)})")
    if os.path.isdir(sdir):
        existing = sorted(f for f in os.listdir(sdir) if f.endswith(".wav"))
        same = existing == sorted(links) and all(
            os.path.islink(os.path.join(sdir, n)) and os.readlink(os.path.join(sdir, n)) == rel
            for n, (_, rel) in links.items())
        if not same:
            raise SystemExit(f"[{lang}] {sdir} は既にあり、中身が今回の計画と違う。上書きしないので止める。")
        print(f"[{lang}] {sdir} は既に同じ内容で作られている（作り直さない）")
    else:
        os.makedirs(sdir)
        for n, (_, rel) in links.items():
            os.symlink(rel, os.path.join(sdir, n))
        print(f"[{lang}] {sdir} に symlink {len(links)} 本を作った")

    # 確認 1: 採点器と同じ探し方で 300 対
    files, found = discover_like_scorer(sdir)
    assert len(files) == N_EXPECTED and len(found) == N_EXPECTED, (lang, len(files), len(found))
    assert found == [f"s{i:04d}" for i in range(N_EXPECTED)], lang
    # 確認 2: symlink の先が元の wav と同じ実体
    manifest = {"lang": lang, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stage_dir": sdir, "source_dir": src_dir,
                "source_bl_suffix": SETS[lang]["bl"], "source_inj_suffix": SETS[lang]["inj"],
                "links": {}}
    n_md5 = 0
    for n, (src, rel) in sorted(links.items()):
        p = os.path.join(sdir, n)
        assert os.path.realpath(p) == os.path.realpath(src), (p, src)
        ent = {"target_rel": rel, "source": src}
        if check_md5:
            m1, m2 = md5_file(p), md5_file(src)
            assert m1 == m2, (p, src)
            ent["md5"] = m1
            n_md5 += 1
        manifest["links"][n] = ent
    mpath = os.path.join(sdir, "stage_manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        if not check_md5:
            print(f"[{lang}] stage_manifest.json は既にある（--no_md5 なので md5 は照合しない）。書き換えない。")
        else:
            diff = sorted(k for k, v in manifest["links"].items()
                          if old["links"].get(k, {}).get("md5") != v.get("md5"))
            if not diff:
                print(f"[{lang}] stage_manifest.json は既にあり、md5 も一致。書き換えない。")
            else:
                # 本番の wav を（退避のうえ）差し替えた後は symlink の先の中身が変わる。symlink はそのまま使えるので、
                # 古い manifest は残し、今の md5 を別名で記録する。
                mpath = os.path.join(sdir, f"stage_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json")
                manifest["changed_since_first_manifest"] = diff
                with open(mpath, "w") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=1)
                print(f"[{lang}] 元の音声が最初の manifest から {len(diff)} 本変わっている（例 {diff[:4]}）。"
                      f"今の md5 を {os.path.basename(mpath)} に記録した。")
    else:
        with open(mpath, "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[{lang}] 確認済み: 採点器の探し方で {len(found)} 対、symlink の先と元の md5 一致 {n_md5} 本 -> {mpath}")
    return sdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_root", required=True,
                    help="並べ直したフォルダを置く親（例: eval_results/_appk_redo）。プロジェクト直下からの相対でも可。")
    ap.add_argument("--langs", default=",".join(STAGED_LANGS))
    ap.add_argument("--no_md5", action="store_true")
    args = ap.parse_args()
    root = args.stage_root if os.path.isabs(args.stage_root) else os.path.join(PROJ, args.stage_root)
    root = os.path.normpath(root)
    assert root.startswith(os.path.join(PROJ, "eval_results") + os.sep), f"eval_results の下に置く: {root}"
    for lang in args.langs.split(","):
        assert lang in STAGED_LANGS, f"{lang} は並べ直し不要（採点器にそのまま渡す）"
        build(lang, root, check_md5=not args.no_md5)


if __name__ == "__main__":
    main()
