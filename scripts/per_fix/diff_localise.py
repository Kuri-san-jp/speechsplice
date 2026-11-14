#!/usr/bin/env python3
"""音素列の突き合わせで実体語区間を取る（時刻に依存しない方式）。

BL と TGT は文脈が同一で実体語だけが違う。したがって2つの全体音素列を
整列すれば、食い違う区間がそのまま実体語になる。
Whisper のタイムスタンプも espeak も強制アライメントも使わないので、
日本語・中国語の等分割問題が原理的に発生せず、必ず答えが出る。

手順:
  1. BL と TGT の全体音素列を編集距離で整列し、食い違う区間を求める
  2. その区間の TGT 側が「目標の読み」、BL 側が「元の読み」
  3. INJ と TGT を整列し、同じ TGT 区間に対応する INJ 側を取る
  4. PER = 編集距離 / TGT 区間長

3条件すべてで「TGT を基準に整列する」という同一手続きが適用される。
"""
import json, sys, argparse, os


def align(a, b):
    """a, b を整列し、(i, j) の対応表を返す。編集距離の traceback。"""
    n, m = len(a), len(b)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        di, dp = d[i], d[i-1]
        for j in range(1, m+1):
            di[j] = min(dp[j]+1, di[j-1]+1, dp[j-1] + (ai != b[j-1]))
    # traceback: ops = [(i_a, i_b, kind)]
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (a[i-1] != b[j-1]):
            ops.append((i-1, j-1, 'M' if a[i-1] == b[j-1] else 'S')); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            ops.append((i-1, None, 'D')); i -= 1
        else:
            ops.append((None, j-1, 'I')); j -= 1
    ops.reverse()
    return d[n][m], ops


def diff_block(a, b, pad=0, gap=3):
    """a と b の食い違う最大の連続区間を (a_s, a_e, b_s, b_e) で返す。"""
    _, ops = align(a, b)
    # 一致しない op の連なりを拾う
    blocks, cur = [], None
    for k, (ia, ib, kind) in enumerate(ops):
        if kind == 'M':
            if cur is not None: blocks.append(cur); cur = None
        else:
            if cur is None: cur = [k, k]
            else: cur[1] = k
    if cur is not None: blocks.append(cur)
    if not blocks:
        return None
    # 近接する塊を結合する。間に挟まる一致が gap 個以下なら同じ実体語とみなす。
    # 実体語の中の音が偶然一致すると塊が分断されるため（例: dʒɑːn と jɑːməɡɑː の ɑː）。
    merged = [list(blocks[0])]
    for blk in blocks[1:]:          # 変数名は b を避ける（引数 b を隠さないため）
        if blk[0] - merged[-1][1] - 1 <= gap:
            merged[-1][1] = blk[1]
        else:
            merged.append(list(blk))
    blocks = merged
    # 一番長い塊を実体語とみなす
    best = max(blocks, key=lambda x: x[1]-x[0]+1)
    ks, ke = best
    ai = [ops[k][0] for k in range(ks, ke+1) if ops[k][0] is not None]
    bi = [ops[k][1] for k in range(ks, ke+1) if ops[k][1] is not None]
    # 端の文脈を少し含める（pad>0 のとき）
    a_s = (min(ai) if ai else _nearest_a(ops, ks)) - pad
    a_e = (max(ai)+1 if ai else _nearest_a(ops, ks)) + pad
    b_s = (min(bi) if bi else _nearest_b(ops, ks)) - pad
    b_e = (max(bi)+1 if bi else _nearest_b(ops, ks)) + pad
    return (max(a_s,0), min(a_e,len(a)), max(b_s,0), min(b_e,len(b)))


# **塊を 2 つ取る方式は採らない（2026-09-20 に検討して却下）。**
# 目標文（2 枠とも差し替えた文）を 1 本だけ合成し、基線との差分の塊を
# 2 つ取れば枠ごとに測れる、と考えた。しかし実際の刺激では 300 対中 50 対で
# 2 つの実体語が隣接しており（間の語数 0）、gap=3 の結合規則で 1 つの塊に
# 融合する。gap を下げると今度は実体語の中で塊が割れる。
# **枠ごとに目標文を合成する**（tgt1 = 枠 1 だけ差し替えた文、tgt2 = 枠 2）。
# それぞれ基線との差が 1 箇所なので、下の diff_block がそのまま使え、
# 本体セルと手続きが完全に一致する。合成量は 2 倍になるが、
# 採点の手続きを表 1・表 2 と揃えられる方を採る。

def _nearest_a(ops, k):
    for q in range(k, -1, -1):
        if ops[q][0] is not None: return ops[q][0]+1
    return 0

def _nearest_b(ops, k):
    for q in range(k, -1, -1):
        if ops[q][1] is not None: return ops[q][1]+1
    return 0


def per(hyp, ref):
    if not ref: return None
    dd, _ = align(hyp, ref)
    return dd/len(ref)


def _span_against(seq, tgt, ts, te):
    """seq を tgt に整列し、tgt の [ts, te) に対応する区間を返す。"""
    _, ops = align(seq, tgt)
    ii = [ia for ia, ib, _k in ops if ib is not None and ts <= ib < te and ia is not None]
    return seq[min(ii):max(ii) + 1] if ii else ""


def entity_from_pair(bl, inj, tgt, pad=0, gap=3):
    """(bl_ent, inj_ent, tgt_ent) を返す。

    **BL と INJ は必ず同じ手続きで取る。**
    以前は BL だけ「TGT と最も食い違う区間」をそのまま使い、INJ は整列で
    対応区間を取っていた。BL の区間は定義上いちばん合わない場所なので、
    注入側だけが有利になり、効果が無いはずのモデルでも改善が出た
    （2026-09-17、Qwen3 が 3 言語とも有意になって発覚）。

    いまは TGT 側の実体語区間 [ts, te) を BL と TGT の差から一度だけ決め、
    その窓に対して BL と INJ を同じ整列手続きで当てる。
    """
    r = diff_block(bl, tgt, pad, gap)
    if r is None:
        return None
    _bs, _be, ts, te = r
    tgt_ent = tgt[ts:te]
    bl_ent = _span_against(bl, tgt, ts, te)
    inj_ent = _span_against(inj, tgt, ts, te)
    return bl_ent, inj_ent, tgt_ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="v2 の結果 JSON（全体音素列を持つもの）")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--gap", type=int, default=3, help="この数以下の一致で隔たった食い違いは結合する")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    rows = json.load(open(a.input_json))["rows"]
    out, n_ok, n_skip = [], 0, 0
    for r in rows:
        bl, inj, tgt = r.get("bl_full_ipa"), r.get("inj_full_ipa"), r.get("tgt_full_ipa")
        rec = {"pair_idx": r.get("pair_idx"), "word_a": r.get("word_a"), "word_b": r.get("word_b")}
        if not (bl and inj and tgt):
            rec["skip"] = "missing_full_ipa"; n_skip += 1; out.append(rec); continue
        e = entity_from_pair(bl, inj, tgt, a.pad, a.gap)
        if e is None:
            rec["skip"] = "no_diff_block"; n_skip += 1; out.append(rec); continue
        b_e, i_e, t_e = e
        rec.update(bl_ent_ipa=b_e, inj_ent_ipa=i_e, tgt_ent_ipa=t_e,
                   bl_per_vs_tgt=per(b_e, t_e), inj_per_vs_tgt=per(i_e, t_e))
        if rec["bl_per_vs_tgt"] is None or rec["inj_per_vs_tgt"] is None:
            rec["skip"] = "empty_tgt_entity"; n_skip += 1; out.append(rec); continue
        rec["delta_per"] = rec["bl_per_vs_tgt"] - rec["inj_per_vs_tgt"]
        rec["inj_better"] = rec["inj_per_vs_tgt"] < rec["bl_per_vs_tgt"]
        n_ok += 1
        out.append(rec)

    print(f"有効 {n_ok}/{len(rows)} ({n_ok/len(rows)*100:.1f}%)  除外 {n_skip}")
    for r in out[:a.show]:
        if r.get("skip"): print("  skip", r["skip"]); continue
        print(f"  {r['word_a']}->{r['word_b']}  BL={r['bl_ent_ipa']!r} INJ={r['inj_ent_ipa']!r} "
              f"TGT={r['tgt_ent_ipa']!r}  PER {r['bl_per_vs_tgt']:.3f}->{r['inj_per_vs_tgt']:.3f}")
    if a.output_json:
        json.dump({"summary": {"n_valid": n_ok, "n_total": len(rows),
                               "scorer": "diff_localise"}, "rows": out},
                  open(a.output_json, "w"), ensure_ascii=False, indent=1)
        print("wrote", a.output_json)


if __name__ == "__main__":
    main()
