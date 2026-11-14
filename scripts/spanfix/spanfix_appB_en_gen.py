#!/usr/bin/env python3
"""spanfix (B): 付録 B（tab:crossattn_mask）英語 300 対を、修正後の範囲の関数で作り直す（GPU, 2026-10-01）.

本番スクリプト scripts/projection/eval_crossattn_entity_dtw.py を 1 バイトも変えずに main() を呼ぶ。
引数は 5/20 の本番と同じ（.perm_cmds.txt 16714 行: --lang en --cross_culture --inject_mode length_preserve --n 300）で、
--output_dir（新しい置き場）だけを足す。seed は main 冒頭の seed_all と各生成の前の seed_everything が使う
モジュール変数 SEED を差し替える（42 = 本番と同じ。43 / 44 は作り直し）。
範囲の関数は、本番スクリプトが main() の中で from-import する scripts/encoder_injection_t5gemma.py の
find_entity_token_range（7/11 に修正済み）がそのまま使われる。これが 5/20 との唯一の違い。

1 回の起動（1 つの GPU ジョブ）で扱う対（todo）以外は、新しい置き場の checkpoint.json に「済み」（placeholder）と
先に書いておくので、本番の resume の仕組みで todo の対だけが生成・採点される（G3 と同じ型）。

差し替え（どれも呼び出しをそのまま通し、記録だけを足す。出力の数値は変わらない）:
  * encoder_injection_t5gemma.generate_with_modified_encoder: 5 本（BL, INJ, BL_mask, INJ_mask, TGT）の音声を
    写し取り、対ごとに順序（文 a/b、遮断の有無、memory の差）を確かめて wav（float32）に保存し、
    regen2 共通の無音の基準（scripts/regen2/silence_criterion.py）で判定する。
  * encoder_injection_t5gemma.find_entity_token_range: 実行時に使われた範囲を記録し、事前に求めた
    修正後の関数の範囲（+ add_bos）と一致するかを確かめる。遮断した列も実体語の範囲と一致するかを確かめる。
  * 本番 main() と forced_alignment_utils の print: そのまま出力し、握りつぶされた例外の行
    （"[i] err: ..." と "[Whisper] transcribe failed: ..."）を events.jsonl に記録する。
状態の判定は spanfix_common.pair_status（G3 と同じ意味）。数値（MCD・類似度）は見ない。

seed の決まり（spanfix_common.next_action / allowed_todo。始める前に確かめ、外れた対は流さない）:
  * seed 42: --part k（1〜4。各 75 対）。まだ流していない対と、最後の回が対の良し悪しでない状態
    （run_error / alignment_exception / capture_error / no_row_unexplained / crashed_no_record）の対。
  * seed 43: seed 42 の最後の回が「対として使えない」（unaligned / generation_failed / silent）の対だけ。
  * seed 44: seed 43 の最後の回が「対として使えない」の対だけ。45 以上は流さない（著者の判断）。
  * 最後まで終わっていない実行（run_end.json が無い）の対は、squeue でジョブが動いていないことを確かめてから
    --confirm_stopped <その置き場> を付けたときだけ、同じ seed でもう一度流す。

置き場（新規。既存の結果には書かない。何も消さない）:
  eval_results/_spanfix_appB_en/seed<S>/<part{k}|regen>_<日時>_<ジョブ番号>/
    checkpoint.json results.json（本番 main() が書く）、run_record.json（始める前）、events.jsonl（追記）、
    capture/p<idx>.json（対ごと）、wav/s<idx>_<条件>.wav、run_end.json（終わったとき）

環境変数（このスクリプトが numpy・torch・HF を読む前に入れる。コマンドに env を付けない）:
  スレッド上限 13（OMP / MKL / OPENBLAS / NUMEXPR / VECLIB）、HF_HOME=/mnt/.cache/huggingface、HF_HUB_CACHE、
  HF_HUB_OFFLINE=1、TORCH_HOME、XDG_CACHE_HOME、NUMBA_CACHE_DIR=/mnt/.cache/numba。ホストでは CUDA_VISIBLE_DEVICES=''。

GPU（プロジェクト直下から。1 ジョブ = GPU 1 枚。投入の前に squeue で利用者全体の合計 2 枚の上限を確かめる）:
  SLURM_TIME=02:00:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/spanfix_appB_en_gen.py \
    --seed 42 --part 1 >> eval_results/_spanfix_appB_en/logs/seed42_part1.log 2>&1
  （part 2〜4 も同じ。seed 43 / 44 と同じ seed のやり直しは spanfix_appB_en_collect.py --status が示したときだけ）
CPU の試運転（モデルは読まない。tokenizer と偽の生成で配管を確かめる。ホストで直接）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/spanfix_appB_en_gen.py --seed 42 --part 1 --dry_run
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import spanfix_common as C  # noqa: E402  標準ライブラリだけを読む

LAUNCH_ENV = C.setup_env()  # numpy・torch・HF を読む前に

import argparse  # noqa: E402
import ast  # noqa: E402
import builtins  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402

sys.path.insert(0, C.P("scripts/regen2"))
import silence_criterion as SC  # noqa: E402  regen2 共通の無音の基準（numpy は呼ぶときに読む）

PROJECT_ROOT = C.PROJECT_ROOT
EVAL_SCRIPT = C.P(C.EVAL_REL)
INJ_SCRIPT = C.P(C.INJ_REL)
FA_SCRIPT = C.P(C.FA_REL)
CONDS = C.CONDS
PAIR_ERR_RE = re.compile(r"^\s*\[(\d+)\] err: ")
WHISPER_ERR_RE = re.compile(r"^\s*\[Whisper\] transcribe failed: ")
FA_WHISPER_FAIL_PRINT = 'print(f"  [Whisper] transcribe failed: {str(e)[:160]}")'
SILENCE_PARAMS = {"RMS_TH": 0.005, "PEAK_TH": 0.05, "FRAME_MS": 20, "FRAME_RMS_TH": 0.01, "ACTIVE_FRAC_TH": 0.05}
SILENCE_ID = "regen2-silence-2026-09-29"
# 各条件の生成の直前に呼ばれる範囲の関数（run_condition の中の順）。a = (text_a, word_a), b = (text_b, word_b)
EXPECTED_SPAN_CALLS = {"bl": ["a"], "inj": ["a", "b"], "bl_mask": ["a"], "inj_mask": ["a", "b"], "tgt": []}


def silence_param_errors():
    errs = [f"silence_criterion.{k} = {getattr(SC, k, None)!r}（{v} のはず）"
            for k, v in SILENCE_PARAMS.items() if getattr(SC, k, None) != v]
    if SC.CRITERION_ID != SILENCE_ID:
        errs.append(f"silence_criterion.CRITERION_ID = {SC.CRITERION_ID!r}")
    return errs


# ---------------------------------------------------------------- 記録
class EventLog:
    """events.jsonl に 1 行ずつ追記する（時間切れで止まっても、それまでの行は残る）."""

    def __init__(self, path):
        self.path = path

    def __call__(self, type_, **kw):
        rec = {"t": C.now(), "type": type_, **kw}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


class SpanTap:
    """find_entity_token_range をそのまま呼び、使われた範囲を記録する（次の生成の呼び出しに付ける）."""

    def __init__(self, orig):
        self.orig = orig
        self.sig = inspect.signature(orig)
        self.pending = []

    def __call__(self, *args, **kwargs):
        out = self.orig(*args, **kwargs)
        try:
            b = self.sig.bind(*args, **kwargs)
            b.apply_defaults()
            a = b.arguments
            self.pending.append({"text": a["full_text"], "word": a["entity_text"],
                                 "add_bos": int(a.get("add_bos") or 0), "span": list(out) if out else None})
        except Exception as e:  # noqa: BLE001  記録の失敗で本番の処理を止めない
            self.pending.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        return out

    def take(self):
        p, self.pending = self.pending, []
        return p


class MultiCapture:
    """generate_with_modified_encoder をそのまま呼び、対ごとに 5 本の音声を写し取って確かめ、保存する.

    呼び出しを対に結びつけるのは target_text（文は対ごとに一意。spanfix_common.load_pairs で確かめる）。
    1 つの対の呼び出しは main() の決まった順（BL, INJ, BL_mask, INJ_mask は文 a、TGT は文 b）で 5 回来る。
    5 回目で確かめて保存する。途中で別の対の呼び出しが来たら（対の処理中の例外など）、前の対は未完として記録する。
    """

    def __init__(self, orig, todo_pairs, expected_spans, run_abs, run_rel, log, span_tap, abort_on_offload):
        self.orig = orig
        self.sig = inspect.signature(orig)
        self.by_text = {}
        for p in todo_pairs:
            self.by_text[p["text_a"]] = (p, "a")
            self.by_text[p["text_b"]] = (p, "b")
        self.expected = expected_spans  # {idx: {"a": [s,e], "b": [s,e]}}（add_bos=0）
        self.run_abs, self.run_rel = run_abs, run_rel
        self.log = log
        self.span_tap = span_tap
        self.abort_on_offload = abort_on_offload
        self.cur = None
        self.done = set()
        self.last_idx = None
        self.device_map = None
        self.n_unmatched = 0

    def current_idx(self):
        return self.last_idx

    def device_offload(self):
        return any(v in ("cpu", "disk") for v in (self.device_map or {}).values())

    def __call__(self, *args, **kwargs):
        spans = self.span_tap.take()
        try:
            b = self.sig.bind(*args, **kwargs)
            b.apply_defaults()
            a = b.arguments
        except Exception as e:  # noqa: BLE001  記録の失敗で本番の処理を止めない（その対は写し取れず capture_error）
            a = None
            self.log("capture_internal_error", pair_idx=self.last_idx, msg=f"bind: {type(e).__name__}: {str(e)[:200]}")
        if a is not None and self.device_map is None:
            dm = getattr(a["model"], "hf_device_map", None)
            self.device_map = {str(k): str(v) for k, v in dm.items()} if isinstance(dm, dict) else {}
            if self.device_offload() and self.abort_on_offload:
                # 層が CPU/disk に置かれた実行はすべて run_error になるので、GPU の時間を使う前に止める
                self.log("device_offload", pair_idx=None, hf_device_map=self.device_map)
                raise SystemExit(f"[spanfix] T5Gemma の層が CPU/disk に置かれた: {self.device_map}")
        out = self.orig(*args, **kwargs)
        if a is not None:
            try:
                self._record(out, a, spans)
            except Exception as e:  # noqa: BLE001  写し取りの失敗で本番の処理を止めない
                print(f"[spanfix] capture failed: {type(e).__name__}: {e}", flush=True)
                self.log("capture_internal_error", pair_idx=self.last_idx,
                         msg=f"{type(e).__name__}: {str(e)[:300]}")
        return out

    def _record(self, out, a, spans):
        text = a["target_text"]
        cur = self.cur
        if cur is None or cur["finalized"] or text not in (cur["pair"]["text_a"], cur["pair"]["text_b"]) \
                or len(cur["calls"]) >= len(CONDS):
            if cur is not None and not cur["finalized"]:
                self._finalize(complete=False)
            hit = self.by_text.get(text)
            if hit is None or hit[0]["pair_idx"] in self.done:
                self.n_unmatched += 1
                self.log("capture_unmatched_call", pair_idx=None, msg=text[:120])
                return
            self.cur = cur = {"pair": hit[0], "calls": [], "audio": [], "mems": [], "finalized": False}
        p = cur["pair"]
        attn = a["encoder_attention_mask"]
        zero = (attn[0] == 0).nonzero().flatten().tolist() if attn is not None else []
        audio = None
        if out is not None:
            audio = out[0].detach().float().cpu().numpy().reshape(-1)  # run_condition / synth_simple と同じく先頭
        cur["calls"].append({"text_role": "a" if text == p["text_a"] else "b", "masked": bool(zero),
                             "masked_cols": zero, "enc_len": int(a["modified_memory"].shape[1]),
                             "sr": int(a["codec_audio_sr"]), "returned_none": out is None, "spans": spans})
        cur["audio"].append(audio)
        cur["mems"].append(a["modified_memory"].detach().float().cpu())
        self.last_idx = p["pair_idx"]
        if len(cur["calls"]) == len(CONDS):
            self._finalize(complete=True)

    def _span_checks(self, idx, calls, labels):
        """実行時の範囲が修正後の関数の範囲（+ add_bos）と一致するか、遮断した列が実体語の範囲か."""
        exp = self.expected.get(idx)
        problems = []
        if exp is None:
            return False, ["事前の範囲が無い"]
        pair_txt = self.cur["pair"]
        for c, lab in zip(calls, labels):
            if lab not in EXPECTED_SPAN_CALLS:
                continue
            want_sides = EXPECTED_SPAN_CALLS[lab]
            got = c["spans"]
            if len(got) != len(want_sides):
                problems.append(f"{lab}: 範囲の呼び出しが {len(got)} 回（{len(want_sides)} 回のはず）")
                continue
            for g, side in zip(got, want_sides):
                text, word = (pair_txt["text_a"], pair_txt["word_a"]) if side == "a" else \
                    (pair_txt["text_b"], pair_txt["word_b"])
                want = [exp[side][0] + g.get("add_bos", 0), exp[side][1] + g.get("add_bos", 0)]
                if g.get("text") != text or g.get("word") != word or g.get("span") != want:
                    problems.append(f"{lab}/{side}: 実行時 {g} / 期待 {want}")
            if lab in ("bl_mask", "inj_mask"):
                ab = (got[0].get("add_bos", 0) if got else 0)
                want_cols = list(range(exp["a"][0] + ab, exp["a"][1] + ab))
                if c["masked_cols"] != want_cols:
                    problems.append(f"{lab}: 遮断した列 {c['masked_cols']} / 実体語の範囲 {want_cols}")
        return not problems, problems

    def _finalize(self, complete):
        cur = self.cur
        cur["finalized"] = True
        p = cur["pair"]
        idx = p["pair_idx"]
        self.done.add(idx)
        calls, mems = cur["calls"], cur["mems"]

        def maxdiff(i, j):
            x, y = mems[i], mems[j]
            if x is None or y is None or x.shape != y.shape:
                return None
            return float((x - y).abs().max().item())

        chk = {"n_calls": len(calls), "text_roles": [c["text_role"] for c in calls],
               "masked": [c["masked"] for c in calls], "returned_none": [c["returned_none"] for c in calls]}
        ok = complete and chk["text_roles"] == ["a", "a", "a", "a", "b"] \
            and chk["masked"] == [False, False, True, True, False]
        if len(mems) >= 4:
            d_inj_bl, d_blm_bl, d_injm_inj = maxdiff(1, 0), maxdiff(2, 0), maxdiff(3, 1)
            chk.update({"maxdiff_inj_vs_bl": d_inj_bl, "maxdiff_blmask_vs_bl": d_blm_bl,
                        "maxdiff_injmask_vs_inj": d_injm_inj})
            # GPU の非決定性で完全一致しないことがあるので、INJ-BL の差の 1% 以下を「同じ」とみなす（G3 と同じ）
            ok = ok and None not in (d_inj_bl, d_blm_bl, d_injm_inj) and d_inj_bl > 0 \
                and d_blm_bl <= 0.01 * d_inj_bl and d_injm_inj <= 0.01 * d_inj_bl
        else:
            ok = False
        chk["order_ok"] = bool(ok)
        labels = list(CONDS) if ok else [f"call{k}" for k in range(len(calls))]
        span_ok, span_problems = self._span_checks(idx, calls, labels) if ok else (None, ["順序が確かめに通らない"])
        wavs = {}
        save_error = None
        if ok and all(x is not None for x in cur["audio"]):
            try:
                import soundfile as sf
                wdir = os.path.join(self.run_abs, "wav")
                os.makedirs(wdir, exist_ok=True)
                for lab, c, x in zip(labels, calls, cur["audio"]):
                    rel = os.path.join(self.run_rel, "wav", f"s{idx:04d}_{lab}.wav")
                    path = os.path.join(wdir, f"s{idx:04d}_{lab}.wav")
                    assert not os.path.exists(path), f"既にある wav は上書きしない: {rel}"
                    sf.write(path, x, c["sr"], subtype="FLOAT")  # 評価に使った値をそのまま残す（無損失）
                    wavs[lab] = {"path": rel, "sr": c["sr"], "md5": C.md5_file(path), **SC.assess_file(path)}
            except Exception as e:  # noqa: BLE001
                save_error = f"{type(e).__name__}: {str(e)[:300]}"
        for k, c in enumerate(calls):
            c["label"] = labels[k]
        rec = {"pair_idx": idx, "text_a": p["text_a"], "text_b": p["text_b"], "word_a": p["word_a"],
               "word_b": p["word_b"], "complete": bool(complete and len(calls) == len(CONDS)),
               "capture_ok": bool(ok and len(wavs) == len(CONDS) and save_error is None),
               "checks": chk, "span_match": span_ok, "span_problems": span_problems,
               "expected_span_add_bos0": self.expected.get(idx),
               "calls": calls, "wavs": wavs, "save_error": save_error,
               "any_silent": any(v["silent"] for v in wavs.values()),
               "any_returned_none": any(c["returned_none"] for c in calls),
               "silence_criterion_id": SC.CRITERION_ID, "written_at": C.now()}
        C.write_json_new(rec, os.path.join(self.run_abs, "capture", f"p{idx:04d}.json"))
        self.log("pair_captured", pair_idx=idx, complete=rec["complete"], capture_ok=rec["capture_ok"],
                 span_match=span_ok, any_silent=rec["any_silent"], any_returned_none=rec["any_returned_none"])
        cur["audio"], cur["mems"] = [], []

    def close(self):
        if self.cur is not None and not self.cur["finalized"]:
            self._finalize(complete=False)


class PrintTap:
    """本番の print をそのまま出し、握りつぶされた例外の行だけを events.jsonl に記録する."""

    def __init__(self, log, cap):
        self.log = log
        self.cap = cap

    def __call__(self, *args, **kwargs):
        builtins.print(*args, **kwargs)
        try:
            s = " ".join(str(x) for x in args)
            m = PAIR_ERR_RE.match(s)
            if m:
                self.log("pair_error", pair_idx=int(m.group(1)), msg=s.strip()[:400])
            elif WHISPER_ERR_RE.match(s):
                # Whisper は対の 5 本の生成の後に呼ばれるので、最後に生成した対のもの
                self.log("whisper_error", pair_idx=self.cap.current_idx(), msg=s.strip()[:400])
        except Exception:  # noqa: BLE001
            pass


def install_print_taps(M, FA, tap, require_real_fa=True):
    info = {}
    if require_real_fa:
        fa_file = os.path.realpath(FA.__file__)
        assert fa_file == os.path.realpath(FA_SCRIPT), f"forced_alignment_utils が本番のファイルでない: {fa_file}"
        assert C.md5_file(fa_file) == C.PINNED_MD5[C.FA_REL], "forced_alignment_utils の md5 が固定の値と違う"
        info["forced_alignment_utils_file"] = fa_file
    M.print = tap
    FA.print = tap
    info.update({"eval_module_tapped": getattr(M, "print", None) is tap,
                 "forced_alignment_utils_tapped": getattr(FA, "print", None) is tap,
                 "pair_error_regex": PAIR_ERR_RE.pattern, "whisper_error_regex": WHISPER_ERR_RE.pattern})
    return info


# ---------------------------------------------------------------- 1 回の実行
class _FakeKill(BaseException):
    """試運転だけ: 時間切れで止まった実行を作る（run_end.json を書かない）."""


def run_one(M, EI, FA, *, seed, tag, todo, pairs, span_pre, data_root, mode, env_info, policy, on_gpu,
            uniq=None, require_real_fa=True):
    import torch
    stamp = time.strftime("%Y%m%d_%H%M%S")
    uniq = uniq or (os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}")
    rel = C.run_dir_rel(seed, tag, stamp, uniq)
    d = os.path.join(data_root, rel)
    rp = os.path.realpath(d)
    assert "/_spanfix_appB_en/" in rp + "/" and "/crossattn_entity_dtw_" not in rp, d
    os.makedirs(d, exist_ok=False)
    rows = [{"pair_idx": i, "placeholder": True} for i in range(C.N_PAIRS) if i not in set(todo)]
    with open(os.path.join(d, "checkpoint.json"), "x") as f:
        json.dump({"rows": rows}, f)
    done = {r["pair_idx"] for r in rows}
    assert [q["pair_idx"] for q in pairs if q["pair_idx"] not in done] == sorted(todo), "todo だけが生成される設定でない"

    log = EventLog(os.path.join(d, "events.jsonl"))
    expected = {i: {"a": span_pre[i]["a"]["fixed"], "b": span_pre[i]["b"]["fixed"]} for i in todo}
    orig_gen, orig_span = EI.generate_with_modified_encoder, EI.find_entity_token_range
    span_tap = SpanTap(orig_span)
    cap = MultiCapture(orig_gen, [pairs[i] for i in todo], expected, d, rel, log, span_tap,
                       abort_on_offload=(mode == C.MODE_GPU))
    EI.generate_with_modified_encoder = cap
    EI.find_entity_token_range = span_tap
    tap = PrintTap(log, cap)
    taps = install_print_taps(M, FA, tap, require_real_fa)
    seed_before = M.SEED
    M.SEED = seed
    argv = C.prod_args() + ["--output_dir", d]
    env = {"project_root": PROJECT_ROOT, "host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__,
           "cuda_device": torch.cuda.get_device_name(0) if on_gpu else None,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
           "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"), "torch_num_threads": torch.get_num_threads(),
           "thread_env": {k: os.environ.get(k) for k in C.THREAD_VARS},
           **{k: os.environ.get(k) for k in C.CACHE_VARS},
           "env_at_launch": LAUNCH_ENV, "env_set_by_script": C.cache_env()}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        env["transformers"] = None
    env.update(env_info)
    rec = {"item": "spanfix (B) Appendix B tab:crossattn_mask EN, fixed entity span",
           "producer": C.PRODUCER, "mode": mode, "seed": seed, "tag": tag, "run_dir": rel,
           "started": C.now(), "todo": sorted(todo), "n_todo": len(todo),
           "main_argv": argv,  # コンテナの中では --output_dir が /mnt/<rel>（出どころの確かめがこれを求める）
           "module_SEED": {"before": seed_before, "after": M.SEED},
           "span_preflight": {str(i): span_pre[i] for i in todo},
           "span_tap_installed": EI.find_entity_token_range is span_tap,
           "generate_tap_installed": EI.generate_with_modified_encoder is cap,
           "print_taps": taps, "seed_policy": policy, "code_md5": C.current_code_md5(PROJECT_ROOT),
           "silence_criterion": {"id": SC.CRITERION_ID, "text": SC.CRITERION_TEXT,
                                 "params": {k: getattr(SC, k) for k in SILENCE_PARAMS}},
           "valid_criterion": "row aligned (Whisper found the entity in all 5 wavs, no Whisper exception), 5 frames, "
                              "5 wavs captured in the checked order, runtime spans equal the fixed function's spans, "
                              "none silent under " + SC.CRITERION_TEXT,
           "env": env}
    if mode == C.MODE_GPU:
        assert d == f"{C.CONTAINER_ROOT}/{rel}", (d, rel)
    C.write_json_new(rec, os.path.join(d, "run_record.json"))
    log("run_start", pair_idx=None, n_todo=len(todo))
    print(f"[spanfix] run {rel}: seed={seed} todo={len(todo)} pairs ({todo[0]}..{todo[-1]}) argv={argv}", flush=True)
    saved_argv = sys.argv
    sys.argv = [EVAL_SCRIPT] + argv
    t0 = time.time()
    err = None
    killed = False
    try:
        M.main()
    except _FakeKill:
        killed = True
    except BaseException as ex:  # noqa: BLE001  例外でも、写し取った wav と checkpoint と記録は残す
        err = f"{type(ex).__name__}: {str(ex)[:500]}"
        print(f"[spanfix] main() ended with {err}", flush=True)
        traceback.print_exc()
    finally:
        sys.argv = saved_argv
        M.SEED = seed_before
        EI.generate_with_modified_encoder, EI.find_entity_token_range = orig_gen, orig_span
    cap.close()
    if killed:  # 試運転だけ: 時間切れで止まった実行（run_end.json なし）
        return rel, None
    elapsed = round((time.time() - t0) / 60, 2)
    end = {"finished": C.now(), "elapsed_min": elapsed, "main_error": err,
           "hf_device_map": cap.device_map, "device_offload": cap.device_offload(),
           "n_unmatched_generate_calls": cap.n_unmatched}
    run = C.load_run(data_root, rel)
    run["end"] = end
    st = {i: C.pair_status(run, i)[0] for i in todo}
    end["statuses"] = {str(i): s for i, s in st.items()}
    end["status_counts"] = {s: sum(1 for v in st.values() if v == s) for s in sorted(set(st.values()))}
    end["load_errors"] = run["errors"]
    C.write_json_new(end, os.path.join(d, "run_end.json"))
    log("run_end", pair_idx=None, main_error=err, elapsed_min=elapsed)
    print(f"[spanfix] RESULT {rel}: elapsed={elapsed}min main_error={err} statuses={end['status_counts']}",
          flush=True)
    print("[spanfix] NEXT: ホストで spanfix_appB_en_collect.py --status を流す", flush=True)
    return rel, end


# ---------------------------------------------------------------- 流す対を決める（seed の決まり）
def select_todo(seed, part, pairs_arg, confirm_stopped, data_root, check_provenance=True):
    runs, warns = C.all_runs(data_root)
    if seed == 42:
        if part is None or pairs_arg:
            raise SystemExit("[spanfix] seed 42 は --part k（1〜4）で流す（--pairs は使わない）")
        cand = C.part_pairs(part)
        tag = f"part{part}"
    else:
        if part is not None:
            raise SystemExit("[spanfix] seed 43 / 44 に --part は使わない（--pairs で絞れる）")
        cand = sorted(set(int(x) for x in pairs_arg.split(","))) if pairs_arg else list(range(C.N_PAIRS))
        tag = "regen"
    todo, why = C.allowed_todo(runs, seed, cand, confirm_stopped, PROJECT_ROOT, check_provenance)
    if seed != 42 and pairs_arg:
        refused = sorted(set(cand) - set(todo))
        if refused:
            raise SystemExit(f"[spanfix] --pairs のうち seed {seed} で流せない対: "
                             f"{ {i: why[i] for i in refused[:5]} }")
    policy = {"seed": seed, "tag": tag, "candidates": cand, "todo": todo,
              "not_run_here": {str(i): v for i, v in why.items()}, "confirm_stopped": list(confirm_stopped),
              "record_warnings": warns}
    return todo, tag, policy


# ---------------------------------------------------------------- 試運転（CPU, モデルなし）
def static_patch_checks():
    """本番スクリプトに、差し替えに使う箇所があるかを ast で確かめる（import しない）."""
    src = open(EVAL_SCRIPT, encoding="utf-8").read()
    tree = ast.parse(src)
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    lazy = {n2.name for n in ast.walk(main) if isinstance(n, ast.ImportFrom)
            and n.module == "encoder_injection_t5gemma" for n2 in n.names}
    top_import = any(isinstance(n, (ast.ImportFrom, ast.Import)) and "encoder_injection" in ast.dump(n)
                     for n in tree.body)
    res = {
        "eval_md5_pinned": C.md5_file(EVAL_SCRIPT) == C.PINNED_MD5[C.EVAL_REL],
        "pinned_errors": C.pinned_file_errors(),
        "module_SEED_assign": any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SEED"
                                                                    for t in n.targets) for n in tree.body),
        "SEED_uses_in_main": sum(1 for n in ast.walk(main) if isinstance(n, ast.Name) and n.id == "SEED"),
        "main_lazy_imports": sorted(lazy),
        "no_top_level_import_of_injection": not top_import,
        "resume_skips_done_pairs": "if pair_idx in done: continue" in src,
        "pair_loop_err_print_present": 'print(f"  [{i}] err: {str(e)[:200]}"); continue' in src,
        "enumerate_pairs_loop": "for i, pair in enumerate(pairs):" in src,
        "mask_uses_entity_range": "attn[0, entity_start:entity_end] = 0" in src,
        "tokenizer_loaded_by_repo_id_in_main": "t5_tok = AutoTokenizer.from_pretrained(tok_name)" in src,
        "fa_whisper_fail_print_present": FA_WHISPER_FAIL_PRINT in open(FA_SCRIPT, encoding="utf-8").read(),
    }
    inj_tree = ast.parse(open(INJ_SCRIPT, encoding="utf-8").read())
    fns = {n.name: [a.arg for a in n.args.args] for n in inj_tree.body if isinstance(n, ast.FunctionDef)}
    res["generate_params"] = fns.get("generate_with_modified_encoder")
    res["span_params"] = fns.get("find_entity_token_range")
    ok = (res["eval_md5_pinned"] and not res["pinned_errors"] and res["module_SEED_assign"]
          and res["SEED_uses_in_main"] >= 4
          and {"generate_with_modified_encoder", "find_entity_token_range"} <= set(lazy)
          and res["no_top_level_import_of_injection"] and res["resume_skips_done_pairs"]
          and res["pair_loop_err_print_present"] and res["enumerate_pairs_loop"] and res["mask_uses_entity_range"]
          and res["fa_whisper_fail_print_present"]
          and all(p in (res["generate_params"] or []) for p in ("model", "modified_memory", "encoder_attention_mask",
                                                                 "target_text", "codec_audio_sr"))
          and (res["span_params"] or [])[:3] == ["text_tokenizer", "full_text", "entity_text"])
    assert ok, res
    return res


def fake_selftest(tok, fixed_fn, legacy_fn, pairs, span_all, base):
    """偽の main()（本番と同じ resume・呼び出し順・print）で、写し取り・状態・seed の決まりを確かめる.

    数値の行は 5/20 の本番の行を写しただけの偽物。出力は今回の試運転の置き場（FAKE）だけ。
    """
    import numpy as np
    import torch
    prod = {r["pair_idx"]: r for r in C.read_json(C.P(C.PROD_DIR_REL + "/results.json"))["rows"]}
    data_root = os.path.join(base, "fake_root")
    os.makedirs(data_root, exist_ok=False)
    sr = 16000

    def fake_generate(model, text_tokenizer, audio_tokenizer, modified_memory, encoder_attention_mask,
                      encoder_position_ids, target_text, device, codec_audio_sr, codec_sr):
        beh = FakeM.behavior.get((target_text, bool((encoder_attention_mask == 0).any().item())), "tone")
        if beh == "none":
            return None
        amp = 0.0 if beh == "zeros" else 0.3
        t = torch.arange(int(0.6 * codec_audio_sr), dtype=torch.float32) / codec_audio_sr
        return (amp * torch.sin(2 * np.pi * 220 * t)).view(1, 1, -1)

    EI = types.SimpleNamespace(generate_with_modified_encoder=fake_generate, find_entity_token_range=fixed_fn)
    FA = types.SimpleNamespace(print=builtins.print)

    class FakeM:
        SEED = 42
        print = builtins.print
        cases = {}
        behavior = {}
        kill_after = None

        @staticmethod
        def main():
            argv = sys.argv
            out = argv[argv.index("--output_dir") + 1]
            ck = os.path.join(out, "checkpoint.json")
            rows = C.read_json(ck)["rows"]
            done = {r["pair_idx"] for r in rows}
            gen, span = EI.generate_with_modified_encoder, EI.find_entity_token_range  # main() と同じく呼び出し時に引く
            for i, pair in enumerate(C.load_pairs()):
                if pair["pair_idx"] in done:
                    continue
                case = FakeM.cases.get(i, "valid")
                L = 16
                src = torch.randn(1, L, 4)
                ones = torch.ones(1, L, dtype=torch.long)
                sa = fixed_fn(tok, pair["text_a"], pair["word_a"], add_bos=0)  # 記録しない（差し替えの外で求める）
                inj = src.clone()
                inj[0, sa[0]:sa[1]] += 1.0
                seq = [(src, ones, pair["text_a"], [("a",)]), (inj, ones, pair["text_a"], [("a",), ("b",)]),
                       (src.clone(), "mask", pair["text_a"], [("a",)]), (inj.clone(), "mask", pair["text_a"], [("a",), ("b",)]),
                       (torch.randn(1, L, 4), ones, pair["text_b"], [])]
                if case == "capture_error":  # INJ と BL_mask の順が入れ替わる
                    seq[1], seq[2] = seq[2], seq[1]
                outs = []
                try:
                    for k, (mem, attn, text, sp) in enumerate(seq):
                        for (side,) in sp:
                            if side == "a":
                                span(tok, pair["text_a"], pair["word_a"], add_bos=0)
                            else:
                                span(tok, pair["text_b"], pair["word_b"], add_bos=0)
                        if isinstance(attn, str):
                            attn = ones.clone()
                            s0, e0 = sa
                            if case == "span_mismatch":
                                s0, e0 = s0 + 1, e0 + 1  # 隣の token を遮断（旧の不具合と同じ形）
                            attn[0, s0:e0] = 0
                        outs.append(gen(None, None, None, mem, attn, None, text, "cpu", sr, 50))
                        if case == "run_error" and k == 1:
                            raise RuntimeError("FAKE CUDA out of memory (selftest)")
                except Exception as e:  # noqa: BLE001  本番 main() と同じ握りつぶし
                    FakeM.print(f"  [{i}] err: {str(e)[:200]}")
                    continue
                if FakeM.kill_after is not None and i > FakeM.kill_after:
                    raise _FakeKill()  # 行を保存する前に止まった（時間切れに見立てる）
                if any(o is None for o in outs):
                    continue
                if case == "alignment_exception":
                    FA.print("  [Whisper] transcribe failed: FAKE CUDA out of memory")
                row = copy.deepcopy(prod[i])
                row.update({k2: pair[k2] for k2 in ("pair_idx", "text_a", "text_b", "word_a", "word_b")})
                row["aligned"] = case not in ("unaligned", "alignment_exception")
                rows.append(row)
                done.add(pair["pair_idx"])
                with open(ck, "w") as f:  # 本番 main() と同じく checkpoint を書き直す（今回作った置き場の中だけ）
                    json.dump({"rows": rows}, f)
            with open(os.path.join(out, "results.json"), "w") as f:
                json.dump({"summary": {"FAKE": True}, "rows": rows}, f)

    def go(seed, tag, todo, cases, behavior=None, kill_after=None, uniq=None):
        FakeM.cases = cases
        FakeM.behavior = behavior or {}
        FakeM.kill_after = kill_after
        return run_one(FakeM, EI, FA, seed=seed, tag=tag, todo=todo, pairs=pairs, span_pre=span_all,
                       data_root=data_root, mode=C.MODE_FAKE, env_info={"FAKE": True}, policy={"FAKE": True},
                       on_gpu=False, uniq=uniq, require_real_fa=False)

    def beh(idx, cond):
        p = pairs[idx]
        return {"bl": (p["text_a"], False), "inj_mask": (p["text_a"], True), "bl_mask": (p["text_a"], True),
                "tgt": (p["text_b"], False)}[cond]

    R = {}
    # 1) seed 42 part1 の 1 回目: いろいろな状態
    cases1 = {2: "silent", 3: "generation_failed", 4: "run_error", 5: "alignment_exception", 6: "unaligned",
              7: "capture_error", 8: "span_mismatch"}
    b1 = {beh(2, "bl_mask"): "zeros", beh(3, "bl_mask"): "none"}
    rel1, end1 = go(42, "part1", list(range(0, 9)), cases1, b1, uniq="fake1")
    want1 = {0: "valid", 1: "valid", 2: "silent", 3: "generation_failed", 4: "run_error",
             5: "alignment_exception", 6: "unaligned", 7: "capture_error", 8: "span_mismatch"}
    got1 = {int(k): v for k, v in end1["statuses"].items()}
    assert got1 == want1, (got1, want1)
    R["run1_statuses"] = got1
    # 2) 時間切れで止まった実行（9 は行あり、10 は写し取りの後・行の前、11 は始まっていない）
    rel2, end2 = go(42, "part1", [9, 10, 11], {}, kill_after=9, uniq="fake2")
    assert end2 is None
    runs, _ = C.all_runs(data_root)
    st2 = {i: C.pair_status([r for r in runs[42] if r["rel"] == rel2][0], i)[0] for i in (9, 10, 11)}
    assert st2 == {9: "valid", 10: "crashed_no_record", 11: "crashed_no_record"}, st2
    R["killed_run_statuses"] = st2
    # 3) seed の決まり（出どころの確かめは偽物なので外す。別に、偽物が出どころで拒まれることを確かめる）
    na = {i: C.next_action(runs, i, PROJECT_ROOT, check_provenance=False)["action"] for i in range(0, 13)}
    want_na = {0: "use_seed42", 1: "use_seed42", 2: "run_seed43", 3: "run_seed43", 4: "rerun_seed42",
               5: "rerun_seed42", 6: "run_seed43", 7: "rerun_seed42", 8: "stop_author_decision", 9: "use_seed42",
               10: "confirm_stopped_seed42", 11: "confirm_stopped_seed42", 12: "run_seed42"}
    assert na == want_na, (na, want_na)
    R["next_actions"] = na
    t42, _ = C.allowed_todo(runs, 42, list(range(13)), (), PROJECT_ROOT, check_provenance=False)
    t42c, _ = C.allowed_todo(runs, 42, list(range(13)), (rel2,), PROJECT_ROOT, check_provenance=False)
    t43, _ = C.allowed_todo(runs, 43, list(range(13)), (), PROJECT_ROOT, check_provenance=False)
    assert t42 == [4, 5, 7, 12] and t42c == [4, 5, 7, 10, 11, 12] and t43 == [2, 3, 6], (t42, t42c, t43)
    R["allowed_todo"] = {"seed42": t42, "seed42_confirm_stopped": t42c, "seed43": t43}
    pe = C.run_provenance_errors([r for r in runs[42] if r["rel"] == rel1][0], 42, PROJECT_ROOT)
    assert any("mode" in e for e in pe) and any("コンテナ" in e for e in pe) and any("GPU" in e for e in pe), pe
    R["fake_refused_by_provenance"] = len(pe)
    # 4) seed 43（2 は有効、3 は無音、6 は例外）→ 2 は seed 43 を使う、3 は seed 44、6 は seed 43 をもう一度
    rel3, end3 = go(43, "regen", [2, 3, 6], {6: "run_error"}, {beh(3, "tgt"): "zeros"}, uniq="fake3")
    runs, _ = C.all_runs(data_root)
    na3 = {i: C.next_action(runs, i, PROJECT_ROOT, check_provenance=False)["action"] for i in (2, 3, 6)}
    assert na3 == {2: "use_seed43", 3: "run_seed44", 6: "rerun_seed43"}, na3
    R["after_seed43"] = na3
    # 5) 同じ seed で Whisper の例外が 2 回続く → 著者の判断
    go(42, "part1", [5], {5: "alignment_exception"}, uniq="fake4")
    runs, _ = C.all_runs(data_root)
    a5 = C.next_action(runs, 5, PROJECT_ROOT, check_provenance=False)["action"]
    assert a5 == "stop_author_decision", a5
    R["two_alignment_exceptions"] = a5
    # 6) 保存した wav が元と一致（無損失）
    import soundfile as sf
    cap0 = C.read_json(os.path.join(data_root, rel1, "capture", "p0000.json"))
    x, fs = sf.read(os.path.join(data_root, cap0["wavs"]["bl"]["path"]), dtype="float32")
    assert fs == sr and abs(float(np.max(np.abs(x))) - 0.3) < 1e-2 and cap0["span_match"] is True, cap0
    assert C.md5_file(os.path.join(data_root, cap0["wavs"]["bl"]["path"])) == cap0["wavs"]["bl"]["md5"]
    R["wav_roundtrip_and_span_match"] = True
    R["fake_root"] = os.path.relpath(data_root, PROJECT_ROOT)
    return R


def dry_run(args):
    assert not C.IN_CONTAINER, "試運転はホストで（CPU のみ）"
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = C.P(f"{C.DRYRUN_REL}/gen_{ts}_seed{args.seed}" + (f"_part{args.part}" if args.part else ""))
    os.makedirs(base, exist_ok=False)
    rep = {"date": C.now(), "args": vars(args), "env_at_launch": LAUNCH_ENV,
           "env_now": {k: os.environ.get(k) for k in C.CACHE_VARS + C.THREAD_VARS + ("CUDA_VISIBLE_DEVICES",)}}
    rep["static_patch_checks"] = static_patch_checks()
    rep["silence"] = {"params_errors": silence_param_errors(), "selftest": SC.selftest()}
    assert not rep["silence"]["params_errors"]
    W = C.load_shim_module()
    off = C.setup_offline(W)
    rep["offline"] = {k: off[k] for k in ("hf_offline", "offline_preflight_ok", "offline_tokenizer_shim")}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_ID)
    fixed_fn, legacy_fn = C.span_functions()
    pairs = C.load_pairs()
    span_all = C.span_table(tok, pairs, fixed_fn, legacy_fn)
    rep["span_all_fixed_ok"] = all(v["fixed_ok"] for v in span_all.values())
    assert rep["span_all_fixed_ok"]
    todo, tag, policy = select_todo(args.seed, args.part, args.pairs, tuple(args.confirm_stopped or ()),
                                    PROJECT_ROOT)
    st, est = C.slurm_time_for(len(todo))
    rep["policy_now"] = policy
    rep["would_run"] = {"tag": tag, "n_todo": len(todo), "estimated_min": est, "slurm_time": st,
                        "container_output_dir": f"{C.CONTAINER_ROOT}/{C.OUT_REL}/seed{args.seed}/{tag}_<日時>_<ジョブ番号>"}
    rep["fake_selftest"] = fake_selftest(tok, fixed_fn, legacy_fn, pairs, span_all, base)
    out = os.path.join(base, "dryrun_report.json")
    C.write_json_new(rep, out)
    print(json.dumps({k: rep[k] for k in ("would_run", "fake_selftest", "span_all_fixed_ok")},
                     ensure_ascii=False, indent=1))
    print(f"[spanfix] dry run OK; report: {os.path.relpath(out, PROJECT_ROOT)}")


# ---------------------------------------------------------------- main
def gpu_main(args):
    if not C.IN_CONTAINER:
        raise SystemExit("[spanfix] GPU の処理は slurm_run.sh のコンテナの中（/mnt）で流す（ホストは --dry_run）")
    if os.environ.get("HF_HOME") != C.CONTAINER_HF_HOME or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit(f"[spanfix] HF の環境が決まりと違う: {os.environ.get('HF_HOME')}")
    pe = C.pinned_file_errors()
    if pe:
        raise SystemExit("[spanfix] 本番のファイルが固定の md5 と違う:\n  - " + "\n  - ".join(pe))
    se = silence_param_errors()
    if se:
        raise SystemExit("[spanfix] 無音の基準の数値が決まりと違う:\n  - " + "\n  - ".join(se))
    todo, tag, policy = select_todo(args.seed, args.part, args.pairs, tuple(args.confirm_stopped or ()),
                                    PROJECT_ROOT)
    print("[spanfix] seed policy:", json.dumps({k: policy[k] for k in ("seed", "tag", "todo", "confirm_stopped")},
                                                ensure_ascii=False), flush=True)
    if not todo:
        print("[spanfix] 流す対が無い（すべて決着済み、または動いている／止まった実行の確認待ち）。終わる。", flush=True)
        return
    st, est = C.slurm_time_for(len(todo))
    print(f"[spanfix] {len(todo)} pairs; estimate {est} min; recommended SLURM_TIME {st}", flush=True)

    import torch
    assert torch.cuda.is_available(), "GPU で実行する（slurm_run.sh 経由）"
    torch.set_num_threads(C.THREADS)
    W = C.load_shim_module()
    env_info = C.setup_offline(W)  # 事前確認と本番 main() の両方に効く tokenizer の読み替え
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_ID)
    fixed_fn, legacy_fn = C.span_functions()
    pairs = C.load_pairs()
    span_pre = C.span_table(tok, [pairs[i] for i in todo], fixed_fn, legacy_fn)
    bad = [i for i in todo if not span_pre[i]["fixed_ok"]]
    if bad:
        raise SystemExit(f"[spanfix] 修正後の関数の範囲が実体語を指さない対: {bad}")

    # 本番スクリプトを module として読む（本番と同じ sys.path の並び）。このスクリプトの置き場と scripts/regen2
    # （無音の基準を読むために足した。他の作業のファイルも置かれる）は外し、名前の衝突で本番と違うモジュールを読まない
    drop = {os.path.realpath(_HERE), os.path.realpath(C.P("scripts/regen2"))}
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.getcwd()) not in drop]
    sys.path.insert(0, os.path.dirname(EVAL_SCRIPT))
    import eval_crossattn_entity_dtw as M
    import encoder_injection_t5gemma as EI  # main() が呼び出し時に from-import するので、先に差し替える
    assert os.path.realpath(M.__file__) == os.path.realpath(EVAL_SCRIPT), M.__file__
    assert os.path.realpath(EI.__file__) == os.path.realpath(INJ_SCRIPT), EI.__file__
    vendor = os.path.realpath(M.T5GEMMA_ROOT)
    vendor_md5 = {}
    for name in ("data.tokenizer", "duration_estimator", "inference_tts_utils"):
        f = os.path.realpath(sys.modules[name].__file__)
        assert f.startswith(vendor + os.sep), f"{name} が vendor 以外から読まれた: {f}"
        vendor_md5[name] = {"file": f, "md5": C.md5_file(f)}
    FA = sys.modules["forced_alignment_utils"]
    assert M.WhisperAligner is FA.WhisperAligner
    env_info["vendor_md5"] = vendor_md5
    env_info["tokenizer_for_preflight"] = {"class": type(tok).__name__, "name_or_path": tok.name_or_path}
    run_one(M, EI, FA, seed=args.seed, tag=tag, todo=todo, pairs=pairs, span_pre=span_pre, data_root=PROJECT_ROOT,
            mode=C.MODE_GPU, env_info=env_info, policy=policy, on_gpu=True)
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=list(C.SEEDS))
    ap.add_argument("--part", type=int, choices=sorted(C.PARTS), help="seed 42 だけ: 1〜4（各 75 対）")
    ap.add_argument("--pairs", default=None, help="seed 43 / 44 だけ: 対の番号をカンマで（既定は流せる対すべて）")
    ap.add_argument("--confirm_stopped", action="append",
                    help="最後まで終わっていない実行の置き場（squeue で動いていないことを確かめたもの）")
    ap.add_argument("--dry_run", action="store_true", help="CPU の試運転（モデルは読まない）")
    args = ap.parse_args(argv)
    if args.dry_run:
        dry_run(args)
    else:
        gpu_main(args)


if __name__ == "__main__":
    main()
