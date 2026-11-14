"""Aggregate the appendix tables A1, A2, A5, A6, A10, and A12 from per-sample files.

Reads eval_results/table_a{1,2,5,6,10,12}_*/ (one manifest.json per folder
names the experiment file each copy came from). Every printed value is
recomputed from per-pair rows:

  A1   acoustic cross-attention mask ablation (T5Gemma-TTS, EN/JA/ZH).
       Rows with aligned=True; MCD = mcd_bl_inj (Intact) and
       mcd_bl_mask_inj_mask (Blocked); MEL Similarity = the time-averaged
       tmean_*_mel fields. Mean +- 1.96 * std (ddof=0) / sqrt(n).
  A2   phoneme-level mask ablation (EN/JA): target-match rate (PER < 0.3) with
       a Wilson 95% half-width (upper bound only when the count is 0), and the
       mean entity PER +- its standard deviation (ddof=0).
  A5   conventional baselines on JA entity replacement: TRM and G2P-CER Match
       counts with Wilson 95% intervals.
  A6   full-sentence CER (mean over 300 pairs). EN uses Whisper medium.en
       transcripts, JA Furigana Whisper kana transcripts against pykakasi kana
       of the reference (needs pykakasi; skipped if it is missing), ZH Whisper
       large-v3 transcripts on Han characters only.
  A10  six-language same-surface test: injection success (EN and ZH: CTC
       log-likelihood shift toward the target reading, d > 0; JA: Furigana
       Whisper transcript contains the target reading after kana
       normalization), and IPA change over the pairs whose BL and INJ IPA
       transcripts are both non-empty.
  A12  UTMOS of the two-entity sentences: BL, INJ (one) = mean of
       (inj1 + inj2) / 2, INJ (both), and the paired INJ (both) - BL difference
       with 1.96 * SD (ddof=1) / sqrt(n).
"""
import ast, importlib.util, json, os, re, unicodedata
from math import sqrt

REPO = os.environ.get("REPO_ROOT", os.getcwd())
EV = os.path.join(REPO, "eval_results")
DATA = os.path.join(REPO, "data")
Z = 1.96


def load(*parts):
    return json.load(open(os.path.join(EV, *parts), encoding="utf-8"))


def mean_sd(v, ddof):
    n = len(v)
    m = sum(v) / n
    return n, m, sqrt(sum((x - m) ** 2 for x in v) / (n - ddof))


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h, h


def fmt_rate(k, n):
    p, lo, hi, h = wilson(k, n)
    return "0.00 +%.2f" % (100 * hi) if k == 0 else "%.2f +- %.2f" % (100 * p, 100 * h)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def literal(path, name):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def table_a1():
    print("Table A1  acoustic cross-attention mask ablation (T5Gemma-TTS)")
    cols = [("MCD Intact", "mcd_bl_inj"), ("MCD Blocked", "mcd_bl_mask_inj_mask"),
            ("MEL Intact", "tmean_bl_inj_mel"), ("MEL Blocked", "tmean_bl_mask_inj_mask_mel")]
    for lang in ("en", "ja", "zh"):
        rows = [r for r in load("table_a1_crossattn_mask", lang + ".json")["rows"] if r.get("aligned")]
        cells = []
        for _, key in cols:
            n, m, sd = mean_sd([r[key] for r in rows if r.get(key) is not None], 0)
            cells.append("%.2f +- %.2f" % (m, Z * sd / sqrt(n)))
        print("  %-3s " % lang.upper() + "   ".join("%s %s" % (c, v) for (c, _), v in zip(cols, cells)))
    print()


def table_a2():
    print("Table A2  phoneme-level mask ablation (target-match rate %, mean entity PER)")
    for lang in ("ja", "en"):
        rows = load("table_a2_phoneme_mask", lang + ".json")["final_rows"]
        n = len(rows)
        out = []
        for c in ("bl", "inj", "bl_mask", "inj_mask"):
            k = sum(1 for r in rows if r[c + "_match"])
            _, m, sd = mean_sd([r[c + "_per"] for r in rows if r.get(c + "_per") is not None], 0)
            out.append("%s %s / %.3f +- %.3f" % (c.upper(), fmt_rate(k, n), m, sd))
        print("  %-3s " % lang.upper() + "   ".join(out))
    print()


def table_a5():
    print("Table A5  conventional baselines, JA entity replacement (TRM %, G2P-CER Match %)")
    rows = load("table_a5_baselines", "ja.json")["rows"]
    n = len(rows)
    for label, key in (("Baseline", "baseline"), ("G2P Baseline", "g2p"), ("Text Hint", "text_hint"),
                       ("ROME/MEMIT", "rome"), ("SpeechSplice", "speechsplice")):
        k_trm = sum(1 for r in rows if r[key]["trm"])
        k_g2p = sum(1 for r in rows if r[key]["g2p_cer_match"])
        if key == "speechsplice":
            def asym(k):
                p, lo, hi, _ = wilson(k, n)
                return "%.2f (+%.2f/-%.2f)" % (100 * p, 100 * (hi - p), 100 * (p - lo))
            print("  %-13s TRM %-20s G2P-CER Match %s" % (label, asym(k_trm), asym(k_g2p)))
        else:
            print("  %-13s TRM %-20s G2P-CER Match %s" % (label, fmt_rate(k_trm, n), fmt_rate(k_g2p, n)))
    print()


def table_a6():
    print("Table A6  full-sentence CER (BL vs source, BL vs target, INJ vs target, INJ vs source)")
    halves = []

    def cell(v):
        n, m, sd = mean_sd(v, 1)
        halves.append(Z * sd / sqrt(n))
        return "%.3f" % m

    # EN: references from data/en_wordrepl_500.py, pair index = template * 50 + pair
    en_data = os.path.join(DATA, "en_wordrepl_500.py")
    tm = list(literal(en_data, "TEMPLATES").values())
    bl = load("table_a6_cer", "en_bl.json")["rows"]
    inj = load("table_a6_cer", "en_inj.json")["rows"]

    def nen(s):
        s = unicodedata.normalize("NFKC", s).lower()
        return re.sub(r" +", " ", re.sub(r"[^a-z0-9 ]", "", s)).strip()

    src = [nen(r["text_a"]) for r in bl]
    tgt = [nen(tm[i // 50].format(word=r["word_b"])) for i, r in enumerate(bl)]
    hb = [nen(r["bl"]["whisper"]) for r in bl]
    hi = [nen(r["whisper"]) for r in inj]
    en = [cell([lev(h, s) / len(s) for h, s in zip(hyp, ref)])
          for hyp, ref in ((hb, src), (hb, tgt), (hi, tgt), (hi, src))]
    print("  EN  " + "  ".join(en))

    # JA: kana CER; references need pykakasi
    try:
        import pykakasi
    except ImportError:
        print("  JA  n/a (pip install pykakasi to recompute)")
    else:
        kk = pykakasi.kakasi()
        spec = importlib.util.spec_from_file_location("ja_pairs", os.path.join(DATA, "ja_wordrepl300_pairs.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pairs = {p["pair_idx"]: p for p in mod.build_pairs()}

        def kata(s):
            return re.sub(r"[^ァ-ヶー]", "", unicodedata.normalize("NFKC", s))

        def kana(s):
            return "".join(x["kana"] for x in kk.convert(s))

        rows = load("table_a6_cer", "ja.json")["rows"]
        s_, t_, b_, i_ = [], [], [], []
        for r in rows:
            p = pairs[int(r["pair_idx"])]
            s_.append(kata(kana(p["text_a"].replace(p["word_a"], r["read_a"]))))
            t_.append(kata(kana(p["text_b"].replace(p["word_b"], r["read_b"]))))
            b_.append(kata(r["bl_tr"]))
            i_.append(kata(r["inj_tr"]))
        ja = [cell([lev(h, s) / len(s) for h, s in zip(hyp, ref)])
              for hyp, ref in ((b_, s_), (b_, t_), (i_, t_), (i_, s_))]
        print("  JA  " + "  ".join(ja))

    # ZH: Han characters only
    zh_data = os.path.join(DATA, "zh_wordrepl300_pairs.py")
    zp = [(a, b, t.format(n=a), t.format(n=b)) for a, b in literal(zh_data, "_PAIRS")
          for t in literal(zh_data, "_TEMPLATES")]

    def han(s):
        s = unicodedata.normalize("NFKC", s)
        return "".join(c for c in s if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")

    rows = load("table_a6_cer", "zh.json")["rows"]
    zh = []
    for cond, ri in (("bl", 2), ("bl", 3), ("inj", 3), ("inj", 2)):
        zh.append(cell([lev(han(r[cond + "_tr"]), han(zp[r["pair_idx"]][ri])) / len(han(zp[r["pair_idx"]][ri]))
                        for r in rows]))
    print("  ZH  " + "  ".join(zh))
    print("  95%% CI half-widths: %.3f to %.3f" % (min(halves), max(halves)))
    print()


HIRA_VOWEL = {}
for _v, _cs in {"あ": "あかさたなはまやらわがざだばぱぁゃゎ", "い": "いきしちにひみりぎじぢびぴぃ",
                "う": "うくすつぬふむゆるぐずづぶぷぅゅゔっ", "え": "えけせてねへめれげぜでべぺぇ",
                "お": "おこそとのほもよろをごぞどぼぽぉょ"}.items():
    for _c in _cs:
        HIRA_VOWEL[_c] = _v
VARIANT = str.maketrans({"ぢ": "じ", "づ": "ず", "ゔ": "ぶ", "ゐ": "い", "ゑ": "え", "を": "お"})


def norm_hira(s):
    out = []
    for c in s:
        o = ord(c)
        if 0x30A1 <= o <= 0x30F6:
            c = chr(o - 0x60)
            o = ord(c)
        if 0x3041 <= o <= 0x3096 or c == "ー":
            out.append(c)
    return "".join(out)


def norm_ja(s):
    """Kana unification, long-vowel unification (ー expansion, おう = おお, えい = ええ), variant kana."""
    s = norm_hira(s)
    out = []
    for c in s:
        out.append(HIRA_VOWEL.get(out[-1], c) if (c == "ー" and out) else c)
    return "".join(out).replace("おう", "おお").replace("えい", "ええ").translate(VARIANT)


def table_a10():
    print("Table A10  six-language same-surface test (injection success %, IPA change %)")
    succ = {}
    for lang in ("en", "zh"):
        rows = load("table_a10_six_language", lang + "_injection_judge.json")["rows"]
        k = sum(1 for r in rows if (r["inj_lp_tgt"] - r["inj_lp_src"]) - (r["bl_lp_tgt"] - r["bl_lp_src"]) > 0)
        succ[lang] = fmt_rate(k, len(rows))
    rows = load("table_a10_six_language", "ja_injection_furigana.json")["rows"]
    k = sum(1 for r in rows if norm_ja(r["word_b"]) in norm_ja(r["oeoi_tr"]))
    succ["ja"] = fmt_rate(k, len(rows))
    for lang in ("en", "ja", "zh", "ko", "fr", "de"):
        pp = load("table_a10_six_language", "ipa_%s.json" % lang)["per_pair"]
        used = [q for q in pp if (q.get("bl_ipa") or "").strip() and (q.get("inj_ipa") or "").strip()]
        k = sum(1 for q in used if q["ipa_changed"])
        print("  %-3s injection success %-14s IPA change %.2f  (n=%d)" % (
            lang.upper(), succ.get(lang, "--"), 100 * k / len(used), len(used)))
    print()


def table_a12():
    print("Table A12  UTMOS of the two-entity sentences (BL, INJ one, INJ both, INJ both - BL)")
    man = json.load(open(os.path.join(EV, "table3_cross_arch", "manifest.json")))
    for e in man["rows"]:
        s = load("table_a12_two_entity_utmos", e["key"] + ".json")["scores"]
        keys = sorted(s["bl"])
        n = len(keys)
        bl = sum(s["bl"][k] for k in keys) / n
        one = sum((s["inj1"][k] + s["inj2"][k]) / 2 for k in keys) / n
        both = sum(s["inj12"][k] for k in keys) / n
        _, m, sd = mean_sd([s["inj12"][k] - s["bl"][k] for k in keys], 1)
        print("  %-17s %-3s %.2f  %.2f  %.2f  %+.4f +- %.4f" % (
            e["model"], e["lang"], bl, one, both, m, Z * sd / sqrt(n)))
    print()


if __name__ == "__main__":
    for fn in (table_a1, table_a2, table_a5, table_a6, table_a10, table_a12):
        fn()
