"""Table A9: name items on which the listener majority and the Whisper judgment disagree.

Joins the item-level listening counts (human_eval/results/sheet3_item_level.csv) with
the automatic verdict on the INJ clip of the same pair, as the pre-specified analysis
(human_eval/analysis/analyze_preference_primary.py, load_whisper_item_results) does:

  EN  eval_results/table2_system_comparison/whisper_inj_large_v3.json  (Whisper large-v3)
  JA  eval_results/table1_cross_lingual/ja_judge.json                  (Furigana Whisper)
  ZH  eval_results/table1_cross_lingual/zh_judge.json                  (Whisper large-v3 + pinyin)

An item counts as accepted by the listeners when more than half of its judgments chose
INJ. The replacement JA item 7001 is not one of the 300 pairs and has no automatic
judgment, so it is left out (Section 4.2).

Run:
  REPO_ROOT=$PWD python3 scripts/aggregate_failure_cases.py
"""
import csv, json, os

REPO = os.environ.get("REPO_ROOT", os.getcwd())
EV = os.path.join(REPO, "eval_results")


def jload(*p):
    with open(os.path.join(EV, *p), encoding="utf-8") as f:
        return json.load(f)


auto = {}
for r in jload("table2_system_comparison", "whisper_inj_large_v3.json")["systems"]["speechsplice"]["rows"]:
    auto[("en", r["idx"])] = (bool(r["hit"]), r["transcript"])
for lang in ("ja", "zh"):
    for r in jload("table1_cross_lingual", f"{lang}_judge.json")["rows"]:
        auto[(lang, r["pair_idx"])] = (bool(r["inj_hit"]), r["inj_tr"])

rows = list(csv.DictReader(open(os.path.join(REPO, "human_eval/results/sheet3_item_level.csv"), encoding="utf-8")))
n_items, mismatch = {}, []
for r in rows:
    lang = r["language"]
    if lang not in ("en", "ja", "zh") or r["item_type"] != "main":
        continue
    key = (lang, int(r["item_id"]))
    if key not in auto:
        print(f"  (no automatic judgment: {lang.upper()} item {r['item_id']} {r['target_word']})")
        continue
    n_items[lang] = n_items.get(lang, 0) + 1
    human_ok = int(r["inj_n"]) > int(r["n"]) / 2
    hit, tr = auto[key]
    if human_ok != hit:
        mismatch.append((lang, r["item_id"], r["target_word"], tr, int(r["inj_n"]), int(r["nopref_n"]),
                         int(r["n"]), hit))

print("Items compared: " + ", ".join(f"{k.upper()} {v}" for k, v in sorted(n_items.items())))
print(f"Disagreements: {len(mismatch)}")
for lang, iid, tgt, tr, inj, nopref, n, hit in mismatch:
    side = "ASR rejected, listeners accepted" if not hit else "ASR accepted, listeners rejected"
    lis = f"{inj}/{n} INJ" if inj > n / 2 else f"{nopref}/{n} neither"
    print(f"  {lang.upper()} {iid}  {tgt:<10} transcript: {tr:<40} listeners: {lis:<12} ({side})")

# the automatic-verdict column of sheet3 must agree with the judge files
bad = [r["item_id"] for r in rows if r["language"] in ("en", "ja", "zh") and r["auto_inj_says_target"]
       and (r["language"], int(r["item_id"])) in auto
       and (r["auto_inj_says_target"] == "True") != auto[(r["language"], int(r["item_id"]))][0]]
print("sheet3 auto_inj_says_target consistent with the judge files:", "yes" if not bad else f"no ({bad})")
