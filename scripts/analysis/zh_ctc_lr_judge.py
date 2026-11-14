#!/usr/bin/env python3
"""ZH多音字: CTC強制照合（尤度比）による2択読み判定（GPU）.

自由書き起こしの再現率に依存する判定は、低品質域の合成音声で過小計上する
（BL既知読みですら検出64-84%）。本判定は「対象文字の読みが src と tgt の
どちらか」という2択に落とし、期待pinyin列（gold読み・文単位）の src版/tgt版
それぞれの CTC 対数尤度を walston/whisper-pinyin の logits 上で計算して比較する。

検証: BL音声で src 側を選ぶ率（≥95%が合格水準）
判定: INJ音声で tgt 側を選ぶ率（= 注入成功率、N=300）

出力: eval_results/cjk_n300_zh_rewav/ctc_lr_judge.json
Usage:
  SLURM_TIME=01:30:00 slurm_run.sh \
      python scripts/analysis/zh_ctc_lr_judge.py
"""
import json
import math
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
_hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(_hf, "modules")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "whisper_pinyin"))

import torch

GEN = os.environ.get("ZH_GEN_DIR",
                     os.path.join(PROJECT_ROOT, "eval_results", "cjk_n300_zh_rewav"))
GOLD = json.load(open(os.environ.get("ZH_GOLD_PATH",
    os.path.join(PROJECT_ROOT, "eval_results", "zh_polyphone_gold_readings.json"))))
HAN = re.compile(r"[一-鿿]")


def local_snapshot(hub_id):
    d = os.path.join(os.environ["HF_HUB_CACHE"],
                     f"models--{hub_id.replace('/', '--')}", "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


def load_tokens(snap):
    sym2id = {}
    for line in open(os.path.join(snap, "tokens.txt")):
        parts = line.split()
        if len(parts) == 2:
            sym2id[parts[0]] = int(parts[1])
    return sym2id


def tokenize_syllable(syl, sym2id):
    """声調つき音節（例 zhuan4）→ トークンid列（貪欲最長一致 + 声調数字）."""
    m = re.match(r"^([a-zü]+)([1-5])$", syl)
    base, tone = (m.group(1), m.group(2)) if m else (syl, None)
    ids, i = [], 0
    while i < len(base):
        for j in range(len(base), i, -1):
            piece = base[i:j]
            if piece in sym2id:
                ids.append(sym2id[piece])
                i = j
                break
        else:
            return None  # 分解不能
    if tone is not None:
        if tone not in sym2id:
            return None
        ids.append(sym2id[tone])
    return ids


def sent_token_ids(text, ch, reading, sym2id):
    """文全体の期待pinyin列（対象文字=reading）をトークンid列へ."""
    from pypinyin import pinyin, Style
    hanzi = [c for c in text if HAN.match(c)]
    py = [p[0] for p in pinyin("".join(hanzi), style=Style.TONE3, heteronym=False)]
    idx = text.find(ch)
    hidx = len([c for c in text[:idx] if HAN.match(c)])
    if hidx < len(py):
        py[hidx] = reading
    ids = []
    for s in py:
        # 軽声等で声調数字がない場合は5扱い（両仮説共通なので比較には影響しない）
        if not re.search(r"[1-5]$", s):
            s = s + "5"
        t = tokenize_syllable(s, sym2id)
        if t is None:
            return None
        ids.extend(t)
    return ids


def ctc_logprob_batch(logp, ids_list, blank=0):
    """複数仮説のCTC前向きを一括計算。logp: (T,V)。返り値: list[float]."""
    exts, lens = [], []
    for ids in ids_list:
        ext = [blank]
        for i in ids:
            ext.extend([i, blank])
        exts.append(ext)
        lens.append(len(ext))
    S = max(lens)
    B = len(exts)
    dev = logp.device
    ext_t = torch.zeros((B, S), dtype=torch.long, device=dev)
    for b, e in enumerate(exts):
        ext_t[b, :len(e)] = torch.tensor(e, device=dev)
    allow2 = (ext_t != blank)
    allow2[:, 2:] &= (ext_t[:, 2:] != ext_t[:, :-2])
    allow2[:, :2] = False
    NEG = -1e30
    T = logp.shape[0]
    alpha = torch.full((B, S), NEG, device=dev)
    alpha[:, 0] = logp[0, blank]
    for b, e in enumerate(exts):
        if len(e) > 1:
            alpha[b, 1] = logp[0, e[1]]
    negcol = torch.full((B, 1), NEG, device=dev)
    for t in range(1, T):
        s1 = torch.cat([negcol, alpha[:, :-1]], dim=1)
        s2 = torch.cat([negcol, negcol, alpha[:, :-2]], dim=1)
        s2 = torch.where(allow2, s2, torch.full_like(s2, NEG))
        alpha = torch.logsumexp(torch.stack([alpha, s1, s2]), dim=0) + logp[t, ext_t]
    outs = []
    for b, L in enumerate(lens):
        outs.append(torch.logsumexp(alpha[b, L-2:L], dim=0).item())
    return outs


def ctc_logprob(logp, ids, blank=0):
    """標準CTC前向きアルゴリズム。logp: (T,V) log_softmax済み."""
    ext = [blank]
    for i in ids:
        ext.extend([i, blank])
    S = len(ext)
    T = logp.shape[0]
    NEG = -1e30
    alpha = torch.full((S,), NEG, device=logp.device)
    alpha[0] = logp[0, blank]
    if S > 1:
        alpha[1] = logp[0, ext[1]]
    for t in range(1, T):
        prev = alpha
        shifted1 = torch.cat([torch.tensor([NEG], device=logp.device), prev[:-1]])
        shifted2 = torch.cat([torch.tensor([NEG, NEG], device=logp.device), prev[:-2]])
        # 同一ラベル連続 or blank へのスキップ禁止を適用
        ext_t = torch.tensor(ext, device=logp.device)
        allow2 = (ext_t != blank)
        allow2[2:] &= (ext_t[2:] != ext_t[:-2])
        allow2[:2] = False
        cand = torch.stack([prev, shifted1,
                            torch.where(allow2, shifted2, torch.full_like(shifted2, NEG))])
        alpha = torch.logsumexp(cand, dim=0) + logp[t, ext_t]
    return torch.logsumexp(alpha[-2:], dim=0).item()


def wilson_half(k, n, z=1.96):
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) * 100


def main():
    global GEN, GOLD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default=None, help="生成ディレクトリ（既定: cjk_n300_zh_rewav）")
    ap.add_argument("--gold", default=None, help="gold読み表（既定: v1）")
    args = ap.parse_args()
    if args.gen_dir:
        GEN = args.gen_dir if os.path.isabs(args.gen_dir) else os.path.join(PROJECT_ROOT, args.gen_dir)
    if args.gold:
        gp = args.gold if os.path.isabs(args.gold) else os.path.join(PROJECT_ROOT, args.gold)
        GOLD = json.load(open(gp))
    print(f"[config] GEN={GEN}", flush=True)
    import whisper as wsp  # fork（third_party/whisper_pinyin）
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snap = local_snapshot("walston/whisper-pinyin")
    model = AutoModel.from_pretrained(snap, trust_remote_code=True).to(device).eval()
    sym2id = load_tokens(snap)

    ck = json.load(open(os.path.join(GEN, "checkpoint_zh.json")))["results"]
    part_path = os.path.join(GEN, "ctc_lr_v3_partial.json")
    rows = []
    done = set()
    if os.path.exists(part_path):
        rows = json.load(open(part_path))
        done = {e["pair_idx"] for e in rows}
        print(f"[resume] {len(rows)}件を再開読み込み", flush=True)
    skip_tok = 0
    for r in ck:
        if r.get("status") != "success":
            continue
        pidx = r["pair_idx"]
        if pidx in done:
            continue
        ch = r["word_a"]
        src = GOLD[r["text_a"]]["reading"]
        tgt = GOLD[r["text_b"]]["reading"]
        ids_src = sent_token_ids(r["text_a"], ch, src, sym2id)
        ids_tgt = sent_token_ids(r["text_a"], ch, tgt, sym2id)
        entry = {"pair_idx": pidx, "char": ch, "src": src, "tgt": tgt,
                 "degenerate": src == tgt,
                 "tone_only": src != tgt and re.sub(r"\d", "", src) == re.sub(r"\d", "", tgt)}
        if ids_src is None or ids_tgt is None:
            skip_tok += 1
            entry["skip"] = "tokenize"
            rows.append(entry)
            continue
        src_base = re.sub(r"\d", "", src)
        tgt_base = re.sub(r"\d", "", tgt)
        # 仮説一覧を構築（src指定声調、tgt指定声調、各基底の声調1-5）
        hyp_ids, hyp_keys = [ids_src, ids_tgt], ["src", "tgt"]
        for base, tag in [(src_base, "srcnt"), (tgt_base, "tgtnt")]:
            if tag == "tgtnt" and tgt_base == src_base:
                continue
            for t in "12345":
                ids = sent_token_ids(r["text_a"], ch, base + t, sym2id)
                if ids is not None:
                    hyp_ids.append(ids)
                    hyp_keys.append(f"{tag}{t}")
        for cond, fn in [("bl", f"s{pidx:04d}_bl.wav"), ("inj", f"s{pidx:04d}_oeoi.wav")]:
            audio = wsp.load_audio(os.path.join(GEN, fn), sr=16000)
            audio = wsp.pad_or_trim(audio)
            mel = wsp.log_mel_spectrogram(audio, n_mels=80).to(device)
            with torch.no_grad():
                enc = model.whisper_model.encoder(mel.unsqueeze(0))
                feats = enc[0] if isinstance(enc, tuple) else enc
                _, logits = model.whisper_model.ctc_head(feats)
                logp = torch.log_softmax(logits.squeeze(0).float(), dim=-1)
            lps = ctc_logprob_batch(logp, hyp_ids)
            d = dict(zip(hyp_keys, lps))
            entry[f"{cond}_lp_src"] = round(d["src"], 3)
            entry[f"{cond}_lp_tgt"] = round(d["tgt"], 3)
            entry[f"{cond}_choose_tgt"] = d["tgt"] > d["src"]
            ls = max((v for k, v in d.items() if k.startswith("srcnt")), default=None)
            lt = max((v for k, v in d.items() if k.startswith("tgtnt")), default=None)
            if ls is not None:
                entry[f"{cond}_lp_src_nt"] = round(ls, 3)
            if ls is not None and lt is not None:
                entry[f"{cond}_lp_tgt_nt"] = round(lt, 3)
                entry[f"{cond}_choose_tgt_nt"] = lt > ls
        rows.append(entry)
        if len(rows) % 25 == 0:
            json.dump(rows, open(part_path, "w"), ensure_ascii=False)
            print(f"  {len(rows)}/300", flush=True)

    ok = [e for e in rows if "skip" not in e and not e["degenerate"]]
    n = len(ok)
    bl_src = sum(not e["bl_choose_tgt"] for e in ok)
    inj_tgt = sum(e["inj_choose_tgt"] for e in ok)
    seg = [e for e in ok if "bl_choose_tgt_nt" in e]
    nseg = len(seg)
    bl_src_nt = sum(not e["bl_choose_tgt_nt"] for e in seg)
    inj_tgt_nt = sum(e["inj_choose_tgt_nt"] for e in seg)
    summary = {
        "n_total": len(rows), "n_judged_nondegenerate": n,
        "n_skip_tokenize": skip_tok,
        "n_degenerate": sum(1 for e in rows if e.get("degenerate")),
        "validation_bl_choose_src": {"k": bl_src, "n": n,
                                     "pct": round(bl_src / n * 100, 2),
                                     "wilson_half": round(wilson_half(bl_src, n), 2)},
        "inj_choose_tgt": {"k": inj_tgt, "n": n, "pct": round(inj_tgt / n * 100, 2),
                           "wilson_half": round(wilson_half(inj_tgt, n), 2)},
        "toneless_validation_bl_choose_src": {"k": bl_src_nt, "n": nseg,
            "pct": round(bl_src_nt / nseg * 100, 2) if nseg else None},
        "toneless_inj_choose_tgt": {"k": inj_tgt_nt, "n": nseg,
            "pct": round(inj_tgt_nt / nseg * 100, 2) if nseg else None,
            "wilson_half": round(wilson_half(inj_tgt_nt, nseg), 2) if nseg else None},
    }
    for grp, cond in [("tone_only", True), ("segmental", False)]:
        sub = [e for e in ok if e["tone_only"] == cond]
        if sub:
            k = sum(e["inj_choose_tgt"] for e in sub)
            kb = sum(not e["bl_choose_tgt"] for e in sub)
            summary[f"{grp}"] = {"n": len(sub),
                                 "bl_choose_src_pct": round(kb / len(sub) * 100, 1),
                                 "inj_choose_tgt_pct": round(k / len(sub) * 100, 1)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(GEN, "ctc_lr_judge.json"), "w"),
              ensure_ascii=False, indent=1)
    print("-> ctc_lr_judge.json")


if __name__ == "__main__":
    main()
