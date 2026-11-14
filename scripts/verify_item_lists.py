"""Check that the item lists in data/ rebuild the items the shipped outputs were run on (CPU, stdlib).

  Table A10 JA   300 (kanji, katakana) pairs sampled from data/entities_10000.jsonl with
                 random.Random(42), as scripts/projection/eval_cjk_n300.py:load_ja_pairs
  Table A10 ZH   300 polyphone context pairs from data/zh_polyphones_v2.json, as
                 scripts/projection/eval_cjk_n300_zhv2.py:load_zh_pairs (characters compared)
  Table 3        the two-entity sentence sets (data/{en,ja}_multientity_300.py,
                 data/zh_multientity_300_v2.py): word pairs and template of every sentence
  Table A3       data/bench_hetnum77.py and data/bench_ext83.py (benchmark id and target)

Run:
  python3 scripts/verify_item_lists.py
"""
import ast, json, os, random, sys

REPO = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
EV = os.path.join(REPO, "eval_results")
sys.path.insert(0, DATA)


def rows(*p):
    return json.load(open(os.path.join(EV, *p), encoding="utf-8"))["rows"]


def function_from(path, name, env):
    """Take one top-level function from a generator script without importing it (no torch needed)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), env)
    return env[name]


def report(name, ok, n):
    print(f"  {name:<44} {'OK' if ok == n else 'MISMATCH'}  ({ok}/{n})")
    return ok == n


gen = os.path.join(REPO, "scripts", "projection", "eval_cjk_n300_zhv2.py")
env = {"os": os, "json": json, "random": random, "SEED": 42, "PROJECT_ROOT": REPO,
       "ZH_POLYPHONES": json.load(open(os.path.join(DATA, "zh_polyphones_v2.json"), encoding="utf-8"))}
all_ok = True
print("Item lists vs. the items in the shipped outputs:")

ja = function_from(gen, "load_ja_pairs", env)(300)
out = {r["pair_idx"]: r for r in rows("table_a10_six_language", "ja_injection_furigana.json")}
all_ok &= report("Table A10 JA (entities_10000.jsonl)",
                 sum((p["word_a"], p["word_b"]) == (out[i]["word_a"], out[i]["word_b"]) for i, p in enumerate(ja)), 300)

zh = function_from(gen, "load_zh_pairs", env)(300)
out = {r["pair_idx"]: r for r in rows("table_a10_six_language", "zh_injection_judge.json")}
all_ok &= report("Table A10 ZH (zh_polyphones_v2.json)", sum(p["word_a"] == out[i]["char"] for i, p in enumerate(zh)), 300)

for lang, mod in (("en", "en_multientity_300"), ("ja", "ja_multientity_300"), ("zh", "zh_multientity_300_v2")):
    built = {p["pair_idx"]: p for p in __import__(mod).build_pairs()}
    out = rows("table3_cross_arch", "two_entity", f"t5gemma_{lang}.json")
    ok = sum(1 for r in out
             if [s["word_a"] for s in built[r["pair_idx"]]["spans"]] == list(r["words_a"])
             and [s["word_b"] for s in built[r["pair_idx"]]["spans"]] == list(r["words_b"])
             and built[r["pair_idx"]]["template_id"] == r.get("template_id", built[r["pair_idx"]]["template_id"]))
    all_ok &= report(f"Table 3 two-entity {lang.upper()} ({mod}.py)", ok, len(out))

for mod, sub in (("bench_hetnum77", "hetnum77"), ("bench_ext83", "ext83")):
    built = __import__(mod).build_pairs()
    out = {r["idx"]: r for r in rows("table_a3_ja_reading", f"{sub}_bl.json")}
    ok = sum(1 for i, p in enumerate(built) if p.get("bench_id", p.get("id")) == out[i]["bench_id"]
             and p["word_a"] == out[i]["target"])
    all_ok &= report(f"Table A3 {sub} ({mod}.py)", ok, len(out))

print("all item lists match" if all_ok else "some item lists do not match")
sys.exit(0 if all_ok else 1)
