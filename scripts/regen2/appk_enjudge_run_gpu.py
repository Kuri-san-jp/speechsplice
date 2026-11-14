#!/usr/bin/env python3
"""付録 K 英語の判定: 300 対の BL と注入の音声で、src の読み / tgt の読みの CTC 対数尤度を計算する（GPU、2026-09-29）.

中国語の行の判定（scripts/analysis/zh_ctc_lr_judge.py）と同じ手順を英語の認識器で行う:
  * 認識器 facebook/wav2vec2-xlsr-53-espeak-cv-ft（付録 K の IPA 変化の列と同じ読み込み方: HF_HUB_OFFLINE=1 で
    Wav2Vec2FeatureExtractor / Wav2Vec2ForCTC.from_pretrained(MODEL_ID)）。音声は torchaudio で読み、16 kHz に
    変換し、特徴抽出器を通す（scripts/projection/score_phoneme_asr_from_dir.py と同じ前処理）。
  * logits の log_softmax（T × 392）に対して、仮説 2 つ（stage_manifest.json の ids_src / ids_tgt）の CTC 対数尤度を、
    中国語の判定器の関数 ctc_logprob_batch（md5 を確かめてから ast で取り出す。1 文字も変えない）で計算する。
    torch の ctc_loss でも同じ計算をして差を記録する（集計は差が 0.01 を超えたら止まる）。
  * 行の形は中国語の判定器と同じ（bl_lp_src / bl_lp_tgt / inj_lp_src / inj_lp_tgt は小数 3 桁、
    *_choose_tgt は丸める前の値の比較）。成功率はここでは数えない（集計 appk_enjudge_aggregate.py が確かめてから数える）。
torch を読む前に: スレッド上限 13、HF_HOME=/mnt/.cache/huggingface、HF_HUB_CACHE、HF_HUB_OFFLINE=1、
TRANSFORMERS_OFFLINE=1、NUMBA_CACHE_DIR=/mnt/.cache/numba、sys.modules['pyopenjtalk']=None。ホストでは GPU を見せない。

走らせる前の確認（1 つでも外れたら GPU を使う前に止まる）:
  コンテナの中・CUDA が使える、pyopenjtalk が塞がれている、中国語の判定器の md5 と関数の取り出し（合成の値で
  torch の ctc_loss と一致）、認識器の refs と全ファイルの md5、一覧（manifest）が著者の認めた表で作られ、
  600 本の音声の md5・読みの表・語彙・仮説の md5 が一覧と今のファイルで同じ、出力がまだ無い。

GPU（プロジェクト直下から。GPU 1 枚。見込み 3〜5 分）:
  SLURM_TIME=00:20:00 slurm_run.sh python3 -u scripts/regen2/appk_enjudge_run_gpu.py --stage_dir eval_results/_appk_enjudge/judge_ctc
予行（ホストの CPU。モデルの重みは読まない。何も書かない。一覧がまだ無ければ、その場で一覧を組んで確かめる）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_run_gpu.py --dry_run
著者が認めたときだけ付ける追加（判定には使わない記録）:
  --espeak_check  採点のあとで、コンテナの espeak-ng（phonemizer）が 300 文の対象の語をどう読むかを記録し、表と比べる。
"""
import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules or "numpy" in sys.modules:
    raise SystemExit("[threads] torch / numpy がスレッド上限の前に import されている")
_PROJ = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
if _PROJ != "/mnt":
    # ホスト（ログインノード）では GPU を見せない。予行と自己検査は GPU に一切触らない
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
_HF = os.path.join(_PROJ, ".cache", "huggingface")
os.environ["HF_HOME"] = _HF
os.environ["HF_HUB_CACHE"] = os.path.join(_HF, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(_PROJ, ".cache", "numba")
if _PROJ == "/mnt":   # コンテナの中だけ作る（ホストの予行では何も書かない）
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
sys.modules["pyopenjtalk"] = None  # 自分のコードでは pyopenjtalk を使わない（判定にも要らない）

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_enjudge_common as C  # noqa: E402

C.K.cap_threads()   # ホストでは優先度も 10 下げる（コンテナでは下げない）

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata as _md  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(C.THREADS)


def _ver(pkg):
    try:
        return _md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def environment():
    cuda = torch.cuda.is_available()   # ホストでは CUDA_VISIBLE_DEVICES="" なので常に False（初期化しない）
    return {"python": platform.python_version(), "host": platform.node(), "proj": C.PROJ,
            "in_container": C.PROJ == "/mnt", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None, "cuda_available": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32, "torch_threads": torch.get_num_threads(),
            "torchaudio": _ver("torchaudio"), "transformers": _ver("transformers"), "numpy": _ver("numpy"),
            "phonemizer": _ver("phonemizer"),
            "pyopenjtalk_blocked": sys.modules.get("pyopenjtalk", "absent") is None,
            "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                   "NUMBA_CACHE_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                   "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


# ---------------------------------------------------------------------------
# CTC（中国語の判定器の関数）と torch の ctc_loss
# ---------------------------------------------------------------------------
def ctc_torch(logp, ids_list):
    """torch の ctc_loss（CPU、空白 0）で同じ対数尤度を計算する（照合用）."""
    import torch.nn.functional as F
    lp = logp.detach().float().cpu()
    T = lp.shape[0]
    out = []
    for ids in ids_list:
        loss = F.ctc_loss(lp.unsqueeze(1), torch.tensor([ids], dtype=torch.long), torch.tensor([T]),
                          torch.tensor([len(ids)]), blank=C.BLANK_ID, reduction="none", zero_infinity=False)
        out.append(-float(loss[0]))
    return out


def ctc_selfcheck(fn):
    """合成の logits（CPU）で、中国語の判定器の関数と torch の ctc_loss が一致するか（同じ記号の続きも含める）."""
    g = torch.Generator().manual_seed(20260929)
    cases = [(60, [[5, 7, 7, 9, 4], [5, 7, 9, 4]]), (40, [[4, 4, 4], [30, 31, 30, 31, 30, 31, 30]]),
             (120, [list(range(4, 40)), list(range(4, 38)) + [5, 5]]), (25, [[10], [10, 10]])]
    worst = 0.0
    for T, hyps in cases:
        logp = torch.log_softmax(torch.randn(T, C.VOCAB_SIZE, generator=g) * 3, dim=-1)
        a = fn(logp, hyps)
        b = ctc_torch(logp, hyps)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return {"n_cases": len(cases), "max_abs_diff": worst, "ok": worst < 1e-3}


# ---------------------------------------------------------------------------
# 一覧（manifest）の照合
# ---------------------------------------------------------------------------
def load_or_build_manifest(stage_dir, dry_run):
    mp = os.path.join(stage_dir, C.MANIFEST)
    if os.path.isfile(mp):
        return C.read_json(mp), C.md5_file(mp), None
    if not dry_run:
        return None, None, f"一覧がない: {C.proj_rel(mp)}（appk_enjudge_stage.py --table_approved_by_author を先に流す）"
    import appk_enjudge_stage as S   # 予行だけ: その場で一覧を組む（cmudict はホストにだけある）
    man, problems, _ = S.build_stage(C.TABLE_PATH, approved=False)
    if problems:
        return None, None, f"一覧をその場で組めない: {problems[:3]}"
    return man, None, f"一覧がまだ無い: {C.proj_rel(mp)}（予行はその場で組んだ一覧で確かめた）"


def check_manifest(man, stage_dir, dry_run, allow_selftest=False):
    """一覧が今のファイルと合うか。戻り値: (problems, waits)."""
    problems, waits = [], []
    if not man["table"].get("author_approved"):
        (waits if dry_run else problems).append("一覧が著者の認めていない読みの表で作られている")
    if man.get("mode") != "post_apply":
        problems.append(f"一覧の mode={man.get('mode')}")
    if man.get("selftest") and not allow_selftest:
        problems.append("自己検査の一覧は判定に使わない")
    if sorted(man.get("files", {})) != sorted(C.all_wav_names()):
        problems.append("一覧の音声が 600 本そろっていない")
    bad = []
    for n, rec in man.get("files", {}).items():
        p = os.path.join(C.PROD_DIR, n)
        if (C.md5_file(p) if os.path.isfile(p) else None) != rec["md5"]:
            bad.append(n)
    if bad:
        problems.append(f"本番の音声が一覧と違う（{len(bad)} 本、例 {bad[:3]}）。一覧を作り直す")
    tp = C.resolve(man["table"]["path"])
    if (C.md5_file(tp) if os.path.isfile(tp) else None) != man["table"]["md5"]:
        problems.append("読みの表が一覧を作ったあとで変わった")
    if C.md5_file(C.VOCAB) != man["recognizer"]["vocab_md5"]:
        problems.append("認識器の vocab.json が一覧と違う")
    hyp = man.get("hypotheses", [])
    if [h["pair_idx"] for h in hyp] != list(range(C.N)):
        problems.append("一覧の仮説が 0..299 の 300 対になっていない")
    if C.hypotheses_digest(hyp) != man.get("hypotheses_md5"):
        problems.append("一覧の仮説の md5 が合わない")
    if any(not (4 <= i < C.VOCAB_SIZE) for h in hyp for i in h["ids_src"] + h["ids_tgt"]):
        problems.append("仮説の id が語彙の範囲（4..391）の外")
    if any(h["degenerate"] != (h["ids_src"] == h["ids_tgt"]) for h in hyp):
        problems.append("一覧の degenerate と仮説が合わない")
    if os.path.exists(os.path.join(stage_dir, C.JUDGE_OUT)):
        problems.append(f"{C.proj_rel(stage_dir)} に出力が既にある（上書きしない。新しい --stage_dir を使う）")
    return problems, waits


def preflight(stage_dir, dry_run, allow_selftest=False):
    pf = {"stage_dir": C.proj_rel(stage_dir), "dry_run": dry_run, "problems": [], "waits": []}
    if not dry_run and C.PROJ != "/mnt" and not allow_selftest:
        pf["problems"].append("GPU の実行はコンテナの中（slurm_run.sh 経由）だけ。ホストでは --dry_run のみ")
    if not dry_run and not torch.cuda.is_available() and not allow_selftest:
        pf["problems"].append("CUDA が使えない")
    if sys.modules.get("pyopenjtalk", "absent") is not None:
        pf["problems"].append("pyopenjtalk が塞がれていない")
    try:
        fn, fn_md5 = C.zh_ctc_function(torch)
        pf["zh_judge_md5"] = C.md5_file(C.ZH_JUDGE)
        pf["ctc_function_md5"] = fn_md5
        pf["ctc_selfcheck"] = ctc_selfcheck(fn)
        if not pf["ctc_selfcheck"]["ok"]:
            pf["problems"].append(f"中国語の判定器の関数と torch の ctc_loss が合わない: {pf['ctc_selfcheck']}")
    except SystemExit as e:
        pf["problems"].append(str(e))
    pf["problems"] += C.check_model_files(full_weights_md5=True)
    man, man_md5, msg = load_or_build_manifest(stage_dir, dry_run)
    if msg:
        (pf["waits"] if (dry_run and man is not None) else pf["problems"]).append(msg)
    if man is not None:
        pf["manifest_md5"] = man_md5
        p, w = check_manifest(man, stage_dir, dry_run, allow_selftest)
        pf["problems"] += p
        pf["waits"] += w
        pf["n_degenerate"] = len(man.get("degenerate_pairs", []))
        pf["hypotheses_md5"] = man.get("hypotheses_md5")
    return pf, man


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------
def load_recognizer(device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(C.MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(C.MODEL_ID).to(device).eval()
    cfg = model.config
    if cfg.pad_token_id != C.BLANK_ID or cfg.vocab_size != C.VOCAB_SIZE or feat_ext.sampling_rate != 16000:
        raise SystemExit(f"認識器の設定が想定と違う: pad={cfg.pad_token_id} vocab={cfg.vocab_size} sr={feat_ext.sampling_rate}")
    w = model.lm_head.weight.detach().float().cpu().numpy().tobytes()
    fp = {"n_params": int(sum(p.numel() for p in model.parameters())),
          "lm_head_weight_md5": hashlib.md5(w).hexdigest(), "name_or_path": getattr(cfg, "_name_or_path", None)}
    return feat_ext, model, fp


def make_logprob_fn(feat_ext, model, device):
    import torchaudio

    def logprobs(path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != feat_ext.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, feat_ext.sampling_rate)
        audio = wav.squeeze().numpy()
        with torch.no_grad():
            x = feat_ext(audio, sampling_rate=feat_ext.sampling_rate, return_tensors="pt").input_values.to(device)
            logits = model(x).logits
            logp = torch.log_softmax(logits.squeeze(0).float(), dim=-1)
        return logp, int(audio.shape[0])
    return logprobs


def score_pairs(man, logprob_fn, ctc_fn, pairs=None, wav_dir=None, log_every=25):
    """一覧の仮説で、各対の BL と注入の音声の尤度を計算する。戻り値: (行, 記録)."""
    wav_dir = wav_dir or C.PROD_DIR
    hyp = man["hypotheses"] if pairs is None else [h for h in man["hypotheses"] if h["pair_idx"] in pairs]
    rows, worst, infeasible, deg_unequal = [], 0.0, [], []
    t0 = time.time()
    for h in hyp:
        e = {"pair_idx": h["pair_idx"], "word": h["word"], "direction": h["direction"],
             "src_label": h["src_label"], "tgt_label": h["tgt_label"], "category": h["category"],
             "degenerate": h["degenerate"]}
        full = {}
        for side in C.WAV_SIDES:
            logp, n16 = logprob_fn(os.path.join(wav_dir, C.wav_name(h["pair_idx"], side)))
            lps = ctc_fn(logp, [h["ids_src"], h["ids_tgt"]])
            xs = ctc_torch(logp, [h["ids_src"], h["ids_tgt"]])
            e[f"{side}_lp_src"] = round(lps[0], 3)
            e[f"{side}_lp_tgt"] = round(lps[1], 3)
            e[f"{side}_choose_tgt"] = lps[1] > lps[0]
            full[side] = {"lp_src": lps[0], "lp_tgt": lps[1], "torch_ctc_src": xs[0], "torch_ctc_tgt": xs[1],
                          "n_frames": int(logp.shape[0]), "n_samples_16k": n16}
            if min(lps) < C.INFEASIBLE_LP:
                infeasible.append([h["pair_idx"], side])
            else:
                worst = max(worst, abs(lps[0] - xs[0]), abs(lps[1] - xs[1]))
            if h["degenerate"] and lps[0] != lps[1]:
                deg_unequal.append([h["pair_idx"], side, lps[0], lps[1]])
        e["full"] = full
        rows.append(e)
        if log_every and len(rows) % log_every == 0:
            print(f"  {len(rows)}/{len(hyp)}  ({time.time() - t0:.0f} s)", flush=True)
    summ = {"n_total": len(rows), "n_degenerate": sum(1 for r in rows if r["degenerate"]),
            "n_infeasible": len(infeasible), "infeasible": infeasible,
            "degenerate_unequal": deg_unequal, "ctc_crosscheck_max_abs_diff": worst,
            "elapsed_sec": round(time.time() - t0, 1)}
    return rows, summ


def fake_logprob_fn(seed=0):
    """予行・自己検査用: 音声の長さに合わせた決まった乱数の logits（モデルは使わない）."""
    import soundfile as sf

    def logprobs(path):
        info = sf.info(path)
        n16 = int(round(info.frames * 16000 / info.samplerate))
        T = (n16 - 400) // 320 + 1
        g = torch.Generator().manual_seed(seed + int(hashlib.md5(os.path.basename(path).encode()).hexdigest()[:6], 16))
        return torch.log_softmax(torch.randn(T, C.VOCAB_SIZE, generator=g) * 3, dim=-1), n16
    return logprobs


# ---------------------------------------------------------------------------
# espeak-ng との比べ（著者が認めたときだけ。判定には使わない）
# ---------------------------------------------------------------------------
def espeak_check(man):
    rec = {"what": ("informational only, not used for scoring: how espeak-ng (en-us, via phonemizer) reads the "
                    "entity in each source sentence, compared with the reading table"), "status": None}
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        be = EspeakBackend("en-us", with_stress=True, preserve_punctuation=False)
        try:
            rec["espeak_version"] = ".".join(str(x) for x in EspeakBackend.version())
        except Exception:  # noqa: BLE001
            rec["espeak_version"] = None
        hyp = man["hypotheses"]
        outs = be.phonemize([h["text_a"] for h in hyp], separator=Separator(phone="", word=" ", syllable=""),
                            strip=True, njobs=1)
        rows = []
        for h, o in zip(hyp, outs):
            ew = o.split()
            ok_len = len(ew) == len(C.tokenize_text(h["text_a"]))
            ent = ew[h["entity_token_index"]] if ok_len else None
            plain = None if ent is None else ent.replace("ˈ", "").replace("ˌ", "")
            rows.append({"pair_idx": h["pair_idx"], "word": h["word"], "src_label": h["src_label"],
                         "espeak_sentence": o, "espeak_entity": ent,
                         "equals_table_src": plain == "".join(h["entity_src_units"]),
                         "equals_table_tgt": plain == "".join(h["entity_tgt_units"])})
        per = {}
        for r in rows:
            k = f"{r['word']} {r['src_label']}"
            d = per.setdefault(k, {"n": 0, "n_equals_table": 0, "n_equals_other_reading": 0, "espeak_forms": []})
            d["n"] += 1
            d["n_equals_table"] += int(r["equals_table_src"])
            d["n_equals_other_reading"] += int(r["equals_table_tgt"])
            if r["espeak_entity"] not in d["espeak_forms"]:
                d["espeak_forms"].append(r["espeak_entity"])
        rec.update(status="done", per_table_entry=per, rows=rows,
                   n_equals_table=sum(r["equals_table_src"] for r in rows), n=len(rows))
    except Exception as e:  # noqa: BLE001  判定には使わないので、失敗しても採点の結果は変わらない
        rec["status"] = f"error: {e!r}"[:400]
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


# ---------------------------------------------------------------------------
def dry_run_extra(man):
    """予行の追加: 認識器の設定（小さい json だけ）と、決まった乱数の logits で 3 対の採点の流れを通す."""
    out = {}
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
    cfg = Wav2Vec2Config.from_pretrained(C.MODEL_ID)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(C.MODEL_ID)
    out["config"] = {"pad_token_id": cfg.pad_token_id, "vocab_size": cfg.vocab_size,
                     "sampling_rate": fe.sampling_rate, "do_normalize": fe.do_normalize,
                     "conv_kernel": list(cfg.conv_kernel), "conv_stride": list(cfg.conv_stride)}
    ok = (cfg.pad_token_id == C.BLANK_ID and cfg.vocab_size == C.VOCAB_SIZE and fe.sampling_rate == 16000
          and list(cfg.conv_stride) == [5, 2, 2, 2, 2, 2, 2] and list(cfg.conv_kernel) == [10, 3, 3, 3, 3, 2, 2])
    out["config_ok"] = ok
    fn, _ = C.zh_ctc_function(torch)
    probe = [0, 23, 164, 190]
    rows, summ = score_pairs(man, fake_logprob_fn(), fn, pairs=probe, log_every=0)
    out["fake_scoring"] = {"pairs": probe, "summary": summ,
                           "degenerate_equal": all(r["bl_lp_src"] == r["bl_lp_tgt"] and r["inj_lp_src"] == r["inj_lp_tgt"]
                                                   for r in rows if r["degenerate"]),
                           "fields": sorted(rows[0])}
    out["ok"] = (ok and summ["n_total"] == len(probe) and not summ["infeasible"] and not summ["degenerate_unequal"]
                 and summ["ctc_crosscheck_max_abs_diff"] < C.CTC_XCHECK_TOL and out["fake_scoring"]["degenerate_equal"])
    # 本物の前処理（torchaudio で読み、16 kHz、特徴抽出器）を 1 本だけ通す（モデルは通さない）
    import torchaudio
    wav, sr = torchaudio.load(os.path.join(C.PROD_DIR, C.wav_name(23, "inj")))
    wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, fe.sampling_rate)
    x = fe(wav.squeeze().numpy(), sampling_rate=fe.sampling_rate, return_tensors="pt").input_values
    out["frontend_probe"] = {"file": C.wav_name(23, "inj"), "sr_in": sr, "n_samples_16k": int(x.shape[-1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_dir", default=C.proj_rel(C.DEFAULT_STAGE))
    ap.add_argument("--dry_run", action="store_true", help="確認だけ（モデルの重みを読まない。何も書かない）")
    ap.add_argument("--espeak_check", action="store_true",
                    help="採点のあとで espeak-ng の読みを記録する（著者が認めたときだけ。判定には使わない）")
    args = ap.parse_args()
    stage_dir = C.resolve(args.stage_dir)
    t0 = time.time()
    env = environment()
    pf, man = preflight(stage_dir, args.dry_run)
    print(f"[enjudge-gpu] stage={C.proj_rel(stage_dir)} dry_run={args.dry_run} proj={C.PROJ} "
          f"cuda={env['cuda_available']} gpu={env['gpu_name']} torch={env['torch']} "
          f"transformers={env['transformers']}", flush=True)
    print(f"  zh judge md5 {pf.get('zh_judge_md5')}, ctc function md5 {pf.get('ctc_function_md5')}, "
          f"selfcheck {pf.get('ctc_selfcheck')}", flush=True)
    print(f"  hypotheses md5 {pf.get('hypotheses_md5')}, degenerate pairs {pf.get('n_degenerate')}", flush=True)
    if args.dry_run and man is not None and not pf["problems"]:
        ex = dry_run_extra(man)
        pf["dry_run_extra"] = ex
        fs = ex["fake_scoring"]["summary"]
        print(f"  recognizer config {ex['config']} ok={ex['config_ok']}", flush=True)
        print(f"  fake-logits scoring of pairs {ex['fake_scoring']['pairs']}: infeasible {fs['n_infeasible']}, "
              f"degenerate equal {ex['fake_scoring']['degenerate_equal']}, ctc cross-check max |diff| "
              f"{fs['ctc_crosscheck_max_abs_diff']:.2e}; front end {ex['frontend_probe']}", flush=True)
        if not ex["ok"]:
            pf["problems"].append("予行の追加の確認（設定・合成の採点の流れ）が通らない")
    for w in pf["waits"]:
        print(f"  WAIT: {w}", flush=True)
    for p in pf["problems"]:
        print(f"  PROBLEM: {p}", flush=True)
    if args.dry_run:
        print(f"  dry run: {'OK' if not pf['problems'] else 'PROBLEMS'} (nothing written)", flush=True)
        sys.exit(1 if pf["problems"] else 0)
    if pf["problems"]:
        sys.exit(1)

    started = time.strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(stage_dir, f"{C.RUN_META_PREFIX}{started}.json")
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running", "stage_dir": C.proj_rel(stage_dir),
            "stage_manifest_md5": pf["manifest_md5"], "zh_judge": C.proj_rel(C.ZH_JUDGE),
            "zh_judge_md5": pf["zh_judge_md5"], "ctc_function_md5": pf["ctc_function_md5"],
            "model": C.MODEL_ID, "snapshot_main": C.SNAP_MAIN, "snapshot_pr7": C.SNAP_PR7,
            "environment": env, "preflight": pf, "espeak_check_requested": bool(args.espeak_check)}
    C.write_json(meta, meta_path, backup=False)
    out = os.path.join(stage_dir, C.JUDGE_OUT)
    try:
        device = "cuda"
        feat_ext, model, fp = load_recognizer(device)
        meta["model_fingerprint"] = fp
        print(f"[enjudge-gpu] model loaded: {fp}", flush=True)
        ctc_fn, _ = C.zh_ctc_function(torch)
        rows, summ = score_pairs(man, make_logprob_fn(feat_ext, model, device), ctc_fn)
        doc = {"summary": summ, "rows": rows,
               "method": {"judge": "CTC log-likelihood of the whole-sentence phoneme sequence, entity = src vs tgt reading",
                          "ctc_function": f"{C.proj_rel(C.ZH_JUDGE)}:{C.ZH_CTC_FUNC}",
                          "ctc_function_md5": pf["ctc_function_md5"], "recognizer": C.MODEL_ID,
                          "front_end": "torchaudio.load -> mono -> torchaudio.functional.resample to 16 kHz -> "
                                       "Wav2Vec2FeatureExtractor -> Wav2Vec2ForCTC logits -> log_softmax",
                          "rounding": "lp values rounded to 3 decimals as in the Chinese judge; *_choose_tgt from "
                                      "unrounded values; unrounded values under 'full'",
                          "stage_manifest_md5": pf["manifest_md5"]}}
        C.write_json(doc, out, backup=False)
    except BaseException as e:  # noqa: BLE001  SystemExit も含めて記録する
        meta.update(status="failed", error=repr(e)[:500], traceback=traceback.format_exc()[-4000:],
                    elapsed_min=round((time.time() - t0) / 60, 2))
        C.write_json(meta, meta_path, backup=False)
        print(f"[enjudge-gpu] FAILED: {e!r} (meta {C.proj_rel(meta_path)})", flush=True)
        sys.exit(1)
    changed = [n for n, rec in man["files"].items()
               if C.md5_file(os.path.join(C.PROD_DIR, n)) != rec["md5"]]
    status = "done" if (not changed and summ["n_total"] == C.N) else "failed"
    meta.update(status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                elapsed_min=round((time.time() - t0) / 60, 2), judge_output=C.proj_rel(out),
                judge_output_md5=C.md5_file(out), n_rows=summ["n_total"], judge_summary=summ,
                stage_files_changed_during_run=changed)
    C.write_json(meta, meta_path, backup=False)
    print(f"[enjudge-gpu] {status} in {meta['elapsed_min']} min, rows {summ['n_total']}, infeasible "
          f"{summ['n_infeasible']}, ctc cross-check max |diff| {summ['ctc_crosscheck_max_abs_diff']:.2e}, "
          f"output md5 {meta['judge_output_md5']} -> meta {C.proj_rel(meta_path)}", flush=True)
    if args.espeak_check:
        ec = espeak_check(man)
        C.write_json(ec, os.path.join(stage_dir, C.ESPEAK_OUT))
        print(f"[enjudge-gpu] espeak check: {ec['status']} (equals table {ec.get('n_equals_table')}/{ec.get('n')})",
              flush=True)
    print("  next (CPU on the host): taskset -c 0-12 python3 "
          f"$REPO_ROOT/scripts/regen2/appk_enjudge_aggregate.py --stage_dir {C.proj_rel(stage_dir)}",
          flush=True)
    sys.exit(0 if status == "done" else 1)


if __name__ == "__main__":
    main()
