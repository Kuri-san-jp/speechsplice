"""Appendix J, paired accuracy difference of the two-entity test (CPU only).

For every condition of Table 3, average the two per-slot verdicts of each
two-entity sentence and take INJ (both) - INJ (one) per sentence. Prints the
mean difference in percentage points with a 95% percentile bootstrap interval
over the 300 sentences (10,000 resamples, seed 20261124), and writes the same
numbers to the JSON file given as the first argument
(default: eval_results/table3_cross_arch/two_entity_paired.json).

Run from the repository root:  python3 scripts/aggregate_two_entity_paired.py
"""
import json, random, sys
BASE='eval_results/table3_cross_arch/'
m=json.load(open(BASE+'manifest.json'))
random.seed(20261124)
def load(rel): return json.load(open(BASE+rel))
out=[]
for e in m['rows']:
    te=e['two_entity']; rows=load(te['file'])['rows']; suf='_pinyin' if te['judge']=='pinyin' else ''
    g=lambda r,f: 1 if r.get(f+suf) else 0
    n=len(rows)
    bl=[(g(r,'bl_slot1_ok')+g(r,'bl_slot2_ok'))/2 for r in rows]
    one=[(g(r,'inj1_slot1_ok')+g(r,'inj2_slot2_ok'))/2 for r in rows]
    both=[(g(r,'inj12_slot1_ok')+g(r,'inj12_slot2_ok'))/2 for r in rows]
    d=[b-o for b,o in zip(both,one)]
    mean=lambda x: 100*sum(x)/len(x)
    def boot(x,B=10000):
        v=[]
        for _ in range(B):
            s=0.0
            for _ in range(n): s+=x[random.randrange(n)]
            v.append(100*s/n)
        v.sort(); return v[int(0.025*B)], v[int(0.975*B)-1]
    lo,hi=boot(d); ol,oh=boot(one); bl_,bh=boot(both)
    out.append(dict(model=e['model'],lang=e['lang'],family=e['family'],n=n,bl=mean(bl),one=mean(one),both=mean(both),diff=mean(d),lo=lo,hi=hi,one_lo=ol,one_hi=oh,both_lo=bl_,both_hi=bh,nonzero=sum(1 for x in d if x!=0)))
json.dump(out,open(sys.argv[1] if len(sys.argv)>1 else BASE+'two_entity_paired.json','w'),indent=1)
for r in out:
    print(f"{r['model']:17s}{r['lang']} {r['family'][:7]} n={r['n']} BL {r['bl']:6.2f} one {r['one']:6.2f} [{r['one_lo']:.2f},{r['one_hi']:.2f}] both {r['both']:6.2f} [{r['both_lo']:.2f},{r['both_hi']:.2f}] diff {r['diff']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] changed={r['nonzero']}")
