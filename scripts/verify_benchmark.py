"""Materialize every benchmark item list in data/ and report its size.

This is the quickest end-to-end check that the dataset half of this repository
is complete and self-contained. It needs no GPU, no model download, and no
dependency beyond the standard library.

Run:
  python3 scripts/verify_benchmark.py
  python3 scripts/verify_benchmark.py --dump out/            # write JSONL per set

Each module exposes a builder (build_pairs / build_300_pairs / build_500_pairs)
returning a list of dicts with at least word_a, word_b, text_a, text_b.
"""
import argparse, importlib, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
sys.path.insert(0, DATA)

# module -> (role, expected number of pairs). None means "report, do not assert".
SETS = [
    ("en_wordrepl300_pairs", "EN word-replacement benchmark (main)", 300),
    ("ja_wordrepl300_pairs", "JA word-replacement benchmark (main)", 300),
    ("zh_wordrepl300_pairs", "ZH word-replacement benchmark (main)", 300),
    ("en_cross_culture_names", "EN cross-culture names", 300),
    ("ja_cross_culture_names", "JA cross-culture names", 300),
    ("zh_cross_culture_names", "ZH cross-culture names", 300),
    ("en_heteronyms_500", "EN heteronyms", 500),
    ("zh_polyphones_300", "ZH polyphones", 300),
    ("ko_hanja_300_strict", "KO hanja polyphones (strict)", 300),
    ("fr_heteronyms_300", "FR heteronyms", 300),
    ("de_heteronyms_300", "DE heteronyms", 300),
]

BUILDERS = ("build_pairs", "build_300_pairs", "build_500_pairs")


def build(mod):
    m = importlib.import_module(mod)
    for fn in BUILDERS:
        if hasattr(m, fn):
            return getattr(m, fn)(), fn
    raise AttributeError(f"{mod}: no builder among {BUILDERS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", metavar="DIR", help="write each set to DIR/<module>.jsonl")
    args = ap.parse_args()
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    print(f"{'set':<24}{'role':<38}{'builder':<18}{'pairs':>7}{'':>4}")
    print("-" * 95)
    total, failures = 0, []
    for mod, role, expect in SETS:
        try:
            pairs, fn = build(mod)
        except Exception as e:
            failures.append(f"{mod}: {type(e).__name__}: {e}")
            print(f"{mod:<24}{role:<38}{'-':<18}{'FAIL':>7}")
            continue
        n = len(pairs)
        total += n
        mark = "ok" if (expect is None or n == expect) else f"!= {expect}"
        if expect is not None and n != expect:
            failures.append(f"{mod}: built {n} pairs, expected {expect}")
        print(f"{mod:<24}{role:<38}{fn + '()':<18}{n:>7}  {mark}")
        if args.dump:
            out = os.path.join(args.dump, f"{mod}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("-" * 95)
    print(f"{'TOTAL':<24}{'':<38}{'':<18}{total:>7}")
    if args.dump:
        print(f"\nwrote {len(SETS)} JSONL files to {args.dump}/")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll benchmark item lists built successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
