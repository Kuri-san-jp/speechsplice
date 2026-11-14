#!/usr/bin/env python3
"""t3fix (ii) GPU: Table 3 T5Gemma-TTS の掃引 EN 7 対と ZH 対 196 を、対ごと seed 43（だめなら 44）で作り直す.

著者の決定（2026-10-02、(3)）:
  * EN（Table 3 (b) の掃引 l*=1/8/15/22/26）: 対 112,118,128,138,141,158,238 を対ごと作り直す
    （BL・TGT と 5 点の INJ = 7 本。BL・TGT は 5 点で共通）。今の無効の理由: TGT 112・BL 138・BL 238 が無音
    （5 点すべてに入る同じファイル）、INJ 118 が l*=15、128・158 が l*=26、141 が l*=1 で無音。
  * ZH: 対 196 を対ごと作り直す（BL・TGT・(a) の INJ・5 点の INJ = 8 本。今は l*=8 の INJ が無音）。
  * seed は 43、まだ使えなければ 44。選ぶ基準は使えるかどうかだけ（PIB・MCD の値は見ない）。
  * 出力は新しい置き場 eval_results/_t3fix/regen/ だけ。本番への差し替えは t3fix_regen_apply.py（別の段階）。

使えるか（結果を見る前に固定。対の中のすべての音声とセルで）:
  1. 生成の失敗が無い（例外・音声を返さない）。実行時の範囲が本番の掃引の記録（layer_inject_meta.json）と同じ。
  2. どの音声も scripts/regen2/silence_criterion.py で無音でない（読める）。
  3. Table 3 の採点（speechsplice_release/src/phoneme_per.py。1 行も変えずに GPU で回す）が、その対が入る
     どのセルでも行を出す（位置合わせの失敗 align_failed・無音の飛ばし silent_audio が無い）。PIB の値は見ない。
  環境の失敗（GPU のメモリ不足など）は対の良し悪しではないので、ジョブを止め、同じコマンドで続きから流す
  （同じ seed）。CUDA の実行時の失敗は同じ対・同じ seed で 2 回目なら生成の失敗（regen2_common と同じ扱い）。
  採点器の例外: phoneme_per.py は対ごとに例外を捕まえて {"err": ...} の行を書く。この行は使えるかの判定に
  入れない。
    - 環境の失敗（regen2_common.is_infra_error。GPU のメモリ不足など）: ジョブを止め（終了 4）、同じコマンドで
      同じ seed のまま採点し直す。
    - CUDA の実行時の失敗（is_cuda_fault）: 1 回目は同じく終了 4。同じ seed・同じセルで 2 回目なら、どの対が
      原因か分けられないので、seed を進めずに著者の判断（終了 3）。
    - それ以外の例外: seed を進めずに著者の判断（終了 3）。著者が「位置合わせの失敗と同じに数える」と決めたら
      --count_scorer_err_as_invalid を付けて同じコマンドで流す（その対は使えないとして次の seed へ）。

生成の経路（本番と同じ関数・同じ引数。本番のファイルは 1 バイトも変えない）:
  scripts/layer_ablation/eval_en_crossculture_t5gemma_layer.py を（__main__ ではなく）モジュールとして読み、
  その synth_plain（BL・TGT。--replicate_bl_tgt の経路）と synth_inj_at_layer（INJ）を呼ぶ。seed はモジュール変数
  SEED を置き換える（どの合成も直前に seed_everything(SEED) で揃えるので、ほかの対の有無は結果に影響しない）。
    - 掃引 5 点: inject_layer 0/7/14/21/25、hook の経路（本番の 15 回の掃引はすべて path=hook_intermediate、
      L25 も final_via_hook）。
    - ZH の (a): inject_layer 25、hook なし（= eval_zh_crossculture_t5gemma_fixed.py の t5_synth_inj と同じ関数・引数。
      対 196 は 5/17 の古い範囲の関数でも今の関数でも範囲が同じ）。
  モデルの読み込みは本番の main() の行と同じ（bf16、device_map="auto"）。保存は main() の _sv と同じ
  （torchaudio.save、PCM_S 16 bit）。1 つのジョブでモデルを 1 回だけ読む。
  9/18 の確認（_layerabl/_md5_L25_n5_{A,B,C_hook}）で、この経路は seed 42 の BL・TGT・INJ を md5 まで再現した。
  今回も seed 43 の前に、対象外の対（EN 113、ZH 197）を seed 42 で作り直して本番と md5 を比べる。
  一致しなければ止める（--continue_on_check_mismatch は著者が認めたときだけ）。

pyopenjtalk: このスクリプトは読まない。sys.modules で塞いでから本番のモジュールを読む（同梱の長さ推定器の
  try/except で None になる）。作り直す EN・ZH の文は言語判定が ja にならないので、長さ推定器は pyopenjtalk を
  使わない（本番と同じ計算）。ja と判定された文が来たら、本番と計算が変わるので止める（--plan で事前に確かめる）。

途中で止まったとき: 同じコマンドで流し直す。終わった対（record.json がある試行）はそのまま使い、
  終わっていない試行は残したまま新しい試行（aNN）で作る。採点（done.json）と最後のまとめ（final_meta.json）も
  同じで、途中で止まった aNN は残し、新しい aNN で作り直す。JSON は一時ファイルに書いてから名前を付けるので、
  半端な JSON は残らない。何も消さない・上書きしない。

置き場（すべて新規）: eval_results/_t3fix/regen/
  prod_snapshot.json                 本番の対象のファイルの md5・inode・nlink と採点の行（最初の実行で書く。以後は照合）
  check42/<lang>/aNN/                seed 42 の再現確認の音声と record.json、check42/result.json
  seed<S>/gen/<lang>/pNNNN/aNN/      作った音声（bl, tgt, inj_<cell>）と record.json
  seed<S>/score/aNN/<lang>_<cell>/   採点の写し（対象の対 + 確認の対の本番の音声）と <lang>_<cell>_pib.json
  seed<S>/decision.json              その seed での対ごとの判定（使える・使えない理由）
  final/aNN/<lang>/pNNNN/*.wav       採用した seed の音声（まとめの試行ごと）
  final/final_meta.json              最後に書く。音声のある final/aNN と採点の行を指す（差し替えの段階が読む）
  runs/run_*.json、strikes.json

終了コード: 0 = 完了、3 = 著者の判断が要る（seed 44 でも使えない対・範囲の不一致・seed 42 の不一致・ja の判定など）、
  4 = 環境の失敗（同じコマンドで流し直す）、75 = gpu_guard（GPU が Slurm の外で使われている）。

GPU（プロジェクト直下から。GPU 1 枚。投入の前に squeue で利用者全体 2 枚の上限を確かめる）:
  SLURM_TIME=01:00:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py --min_free_gb 30 -- /mnt/scripts/spanfix/t3fix_regen_gpu.py
CPU の予行（モデルは読まない。範囲・言語判定・本番の状態・時間の見積もり。ホストで直接）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/t3fix_regen_gpu.py --plan
  式と表の値の自己確認は反映の段階を見る（B1・B2 の後は監査の値と照合する。t3fix_common.expected_table3）。
  作り直しを反映した後は「ジョブは要らない」と示して終わる。
偽の生成で配管を試す（ホスト、CPU。試験の写しの中だけ。t3fix_selftest.py が使う）:
  taskset -c 0-12 python3 .../t3fix_regen_gpu.py --fake --sandbox <eval_results/_t3fix/selftest/.../ev>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import t3fix_common as T  # noqa: E402

LAUNCH_ENV = T.setup_env()   # numpy・torch・HF を読む前に

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

LANGS = ("en", "zh")
EXIT_AUTHOR = 3
EXIT_ENV = 4
ALLOWED_SLURM = ("00:20:00", "00:40:00", "01:00:00", "02:00:00", "04:00:00")
SEC_PER_SYNTH = 8.0     # 本番の掃引: EN 38.22 分 / 300 本、ZH 37.5〜39.4 分 / 300 本
LOAD_MIN = 4.0          # T5Gemma-TTS 2b-2b と xcodec2 の読み込み（余裕を見て）
SCORE_MIN_PER_CELL = 0.5  # phoneme_per.py を 1 セル回す（wav2vec2 の読み込みを含む）
RULE = ("A pair is valid at a seed iff (1) every synthesis of the pair returned audio without an exception and with "
        "the runtime entity span equal to the production sweep meta; (2) no wav of the pair is silent or unreadable "
        "under scripts/regen2/silence_criterion.py; (3) speechsplice_release/src/phoneme_per.py gives an available "
        "row (inj_per_vs_tgt not None) for the pair in every Table 3 cell it enters (align_failed / silent_audio "
        "rows make the pair invalid). Seeds 43 then 44, chosen only by this rule; PIB/MCD values are never looked "
        "at. Environment failures retry the same seed, also inside the scorer (err rows with GPU memory or CUDA "
        "faults are re-scored with the same seed); any other scorer exception stops for the author's decision and "
        "counts as invalid only with --count_scorer_err_as_invalid.")


class Stop(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code
        self.msg = msg


class PyopenjtalkRequested(RuntimeError):
    pass


def regen_base(ev):
    return os.path.join(T.out_root(ev), "regen")


def cellmap(lang):
    return {c["cell"]: c for c in T.regen_cells(lang)}


def span_cell_layer(cell):
    """範囲を照合する本番の掃引の層（ZH の (a) は L25 の記録と照合する。範囲の関数は層に依らない）."""
    return cell["layer"]


# ---------------------------------------------------------------- 本番の状態
def prod_spans(ev, lang, pairs):
    """{L: {pair: (span_a, span_b)}}（本番の掃引の layer_inject_meta.json）."""
    out = {}
    for L in T.SWEEP_LAYERS:
        rows = {r["pair_idx"]: r for r in T.read_json(os.path.join(ev, T.sweep_meta(lang, L)))["rows"]}
        out[L] = {p: (rows[p].get("span_a"), rows[p].get("span_b"), rows[p].get("status")) for p in pairs}
    return out


def scan_invalid(ev, lang, pairs):
    """対象のセル（ZH は (a) と掃引、EN は掃引）で、無音・欠け・読めない音声を含む対."""
    cache, bad = {}, {}
    for p in pairs:
        for cell in T.regen_cells(lang):
            for c in ("bl", "tgt", "inj"):
                path = T.prod_wav(ev, cell, p, c)
                if os.path.exists(path):
                    st = os.stat(path)
                    key = (st.st_dev, st.st_ino)
                    if key not in cache:
                        cache[key] = T.wav_check(path)
                    w = cache[key]
                else:
                    w = {"bad": True, "why": "missing"}
                if w["bad"]:
                    bad.setdefault(p, []).append("%s/%s:%s" % (cell["cell"], c, w.get("why")))
    return bad


def scan_pairs(ev, lang):
    """本番では 300 対すべて、試験の写しでは写した対だけを調べる."""
    if T.is_sandbox(ev):
        return sorted(set(T.REGEN_TARGETS[lang]) | {T.CHECK_PAIRS[lang]})
    return list(range(300))


def prod_snapshot(ev):
    """対象・確認の対の本番の音声（md5・inode・nlink・無音か）と、採点の行。EN は凍結・対象外の同じ inode も."""
    snap = {"files": {}, "rows": {}, "protected": {}, "created": T.now()}
    for lang in LANGS:
        pairs = list(T.REGEN_TARGETS[lang]) + [T.CHECK_PAIRS[lang]]
        for p in pairs:
            for cell in T.regen_cells(lang):
                for c in ("bl", "tgt", "inj"):
                    path = T.prod_wav(ev, cell, p, c)
                    d = T.stat_info(path)
                    w = T.wav_check(path)
                    d.update({"silent": bool(w.get("silent")), "why": w.get("why")})
                    snap["files"][os.path.relpath(path, ev)] = d
        for cell in T.regen_cells(lang):
            doc = T.read_json(os.path.join(ev, cell["pib"]))
            rows = {r["pair_idx"]: r for r in doc["rows"]}
            snap["rows"][cell["pib"]] = {str(p): rows.get(p) for p in pairs}
            if cell.get("mcd"):
                mdoc = T.read_json(os.path.join(ev, cell["mcd"]))
                mrows = {r["pair_idx"]: r for r in mdoc["rows"]}
                snap["rows"][cell["mcd"]] = {str(p): mrows.get(p) for p in pairs}
    for p in T.REGEN_TARGETS["en"]:
        for c in ("bl", "tgt", "inj"):
            path = os.path.join(ev, T.EN_B_DIR, "s%04d_%s.wav" % (p, c))
            snap["protected"][os.path.relpath(path, ev)] = T.stat_info(path)
        for d in T.EN_PROG_DIRS:
            for c in ("bl", "tgt"):
                path = os.path.join(ev, d, "s%04d_%s.wav" % (p, c))
                if os.path.exists(path):
                    snap["protected"][os.path.relpath(path, ev)] = T.stat_info(path)
    snap["frozen_en_a"] = T.frozen_en_a_digest(ev)
    return snap


def snapshot_diff(a, b):
    """2 つの snapshot の違い（md5・inode と採点の行。nlink は見ない）."""
    diffs = []
    for k in ("files", "protected"):
        for rel in sorted(set(a[k]) | set(b[k])):
            x, y = a[k].get(rel), b[k].get(rel)
            if (x or {}).get("md5") != (y or {}).get("md5") or (x or {}).get("ino") != (y or {}).get("ino"):
                diffs.append("%s:%s" % (k, rel))
    for rel in sorted(set(a["rows"]) | set(b["rows"])):
        if a["rows"].get(rel) != b["rows"].get(rel):
            diffs.append("rows:%s" % rel)
    if a.get("frozen_en_a") != b.get("frozen_en_a"):
        diffs.append("frozen_en_a")
    return diffs


# ---------------------------------------------------------------- 生成の部品
class RealBackend:
    """本番の層ごとの注入スクリプトの関数をそのまま呼ぶ（GPU、コンテナの中だけ）."""

    mode = "gpu"

    def __init__(self, ev, pairs, log):
        if not T.IN_CONTAINER:
            raise Stop(EXIT_ENV, "GPU の実行はコンテナの中だけ（slurm_run.sh 経由）")
        self.pairs = pairs
        self.lang_log = []
        os.chdir(T.PROJECT_ROOT)   # 本番の main() の冒頭（335 行）と同じ
        # pyopenjtalk はこちらでは読まない。本番のモジュールを読む前に塞ぐ
        sys.modules["pyopenjtalk"] = None
        import spanfix_common as SFC
        W = SFC.load_shim_module()
        self.offline = SFC.setup_offline(W)   # HF_HUB_OFFLINE=1 で手元の snapshot を確かめ、tokenizer の読み替えを入れる
        spec = importlib.util.spec_from_file_location("t3fix_layer_prod", T.P(T.LAYER_REL))
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        self.M = M
        E = sys.modules["encoder_injection_t5gemma"]
        DE = sys.modules["duration_estimator"]
        if DE.pyopenjtalk is not None:
            raise Stop(EXIT_AUTHOR, "pyopenjtalk が読み込まれている（塞いだはず）")

        def _ja_guard(text):
            raise PyopenjtalkRequested("duration estimator asked for pyopenjtalk (ja) on: %r" % text)
        DE._phoneme_count_ja = _ja_guard
        orig_est = E.estimate_duration
        lang_log = self.lang_log

        def _logged(*a, **kw):
            tl = kw.get("target_lang")
            lang_log.append({"target_lang": tl, "text": kw.get("target_text")})
            if isinstance(tl, str) and tl.lower().startswith("ja"):
                raise PyopenjtalkRequested("language detected as %r for %r" % (tl, kw.get("target_text")))
            return orig_est(*a, **kw)
        E.estimate_duration = _logged
        import torch
        import torchaudio
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from data.tokenizer import AudioTokenizer
        self.torch, self.torchaudio = torch, torchaudio
        if not torch.cuda.is_available():
            raise Stop(EXIT_ENV, "CUDA が見えない")
        t0 = time.time()
        # ---- 本番の main() と同じ読み込み（eval_en_crossculture_t5gemma_layer.py 363-387 行）
        model = AutoModelForSeq2SeqLM.from_pretrained("Aratako/T5Gemma-TTS-2b-2b", trust_remote_code=True,
                                                      dtype=torch.bfloat16, device_map="auto")
        model.eval()
        cfg = model.config
        tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
        tok = AutoTokenizer.from_pretrained(tok_name)
        audio_tok = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
        self.sr = audio_tok.sample_rate
        self.codec_sr = getattr(cfg, "encodec_sr", 50)
        self.add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0
        n_layers = len(model.encoder_module.layers)
        cfg_n = None
        try:
            cfg_n = int(cfg.t5_config_dict["encoder"]["num_hidden_layers"])
        except Exception:  # noqa: BLE001
            pass
        if n_layers != 26 or (cfg_n is not None and cfg_n != n_layers):
            raise Stop(EXIT_AUTHOR, "encoder の層の数が 26 でない（%s / config %s）" % (n_layers, cfg_n))
        dm = getattr(model, "hf_device_map", None) or {}
        off = {k: str(v) for k, v in dm.items() if str(v) in ("cpu", "disk")}
        if off:
            raise Stop(EXIT_ENV, "T5Gemma の層が CPU/disk に置かれた: %s" % off)
        self.model, self.tok, self.audio_tok = model, tok, audio_tok
        self.info = {"load_sec": round(time.time() - t0, 1), "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0), "tok_name": tok_name, "sr": self.sr,
                     "codec_sr": self.codec_sr, "add_bos": self.add_bos, "n_layers": n_layers,
                     "hf_device_map": {k: str(v) for k, v in dm.items()}, "offline": self.offline,
                     "pyopenjtalk_blocked": sys.modules.get("pyopenjtalk", "absent") is None,
                     "layer_module_LANG": getattr(M, "LANG", None)}
        log("[t3fix] model loaded: %s" % json.dumps({k: self.info[k] for k in ("load_sec", "gpu", "sr", "add_bos")}))

    def synth_plain(self, lang, pair, which, seed):
        self.M.SEED = seed
        text = pair["text_a"] if which == "bl" else pair["text_b"]
        return self.M.synth_plain(self.model, self.tok, self.audio_tok, text, "cuda", self.sr, self.codec_sr), {}

    def synth_inj(self, lang, pair, cell, seed):
        self.M.SEED = seed
        return self.M.synth_inj_at_layer(self.model, self.tok, self.audio_tok, pair["text_a"], pair["text_b"],
                                         pair["word_a"], pair["word_b"], "cuda", self.add_bos, cell["layer"],
                                         cell["use_hook"], self.sr, self.codec_sr, selfcheck=False)

    def save(self, audio, path):
        """本番 main() の _sv と同じ保存（409-412 行）."""
        a = audio[0]
        a = a if a.dim() > 1 else a.unsqueeze(0)
        if os.path.exists(path):
            raise Stop(EXIT_ENV, "保存先が既にある: %s" % path)
        self.torchaudio.save(path, a.cpu(), self.sr, encoding="PCM_S", bits_per_sample=16)
        return round(audio[0].shape[-1] / self.sr, 3)

    def score(self, stg, lang, out_json, log_path, seed, cell):
        env = dict(os.environ, REPO_ROOT=T.PROJECT_ROOT)
        with open(log_path, "x") as lf:
            rc = subprocess.run([sys.executable, "-u", T.P(T.PER_REL), "--wav_dir", stg, "--lang", lang,
                                 "--n", "300", "--output_json", out_json], cwd=T.PROJECT_ROOT, env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not os.path.exists(out_json):
            raise Stop(EXIT_ENV, "phoneme_per.py が失敗（rc=%s、ログ %s）" % (rc, log_path))
        return T.read_json(out_json)


class FakeBackend:
    """配管の試験用（CPU、試験の写しの中だけ）。音声は決まった周波数の正弦波、採点は偽の行."""

    mode = "FAKE_selftest"
    sr = 16000

    # 採点器の例外の偽物（--fake_score_err）: (en, L0, 対 112, seed 43) の行に入れる
    SCORE_ERR_AT = ("en", "L0", 112, 43)
    SCORE_ERR_TEXT = {"infra": "CUDA out of memory. Tried to allocate 20.00 MiB (fake infra failure in the scorer)",
                      "cuda": "CUDA error: an illegal memory access was encountered (fake CUDA fault in the scorer)",
                      "other": "Calculated padded input size per channel: (1). Kernel size: (10). (fake scorer error)"}

    def __init__(self, ev, scenario, crash_after, log, score_err=None):
        if not T.is_sandbox(ev):
            raise Stop(EXIT_AUTHOR, "--fake は試験の写し（--sandbox）の中だけ")
        self.ev = ev
        self.n = 0
        self.crash_after = crash_after
        self.score_err = score_err
        self.lang_log = []
        self.silent, self.genfail, self.alignfail = set(), set(), set()
        if scenario == "resolved":
            self.silent = {("en", 138, "tgt", 43)}
            self.genfail = {("en", 158, "inj_L14", 43)}
            self.alignfail = {("en", "L7", 141, 43)}
        elif scenario == "unresolved":
            self.silent = {("en", 118, "inj_L0", 43), ("en", 118, "inj_L0", 44)}
        self.spans = {lang: prod_spans(ev, lang, list(T.REGEN_TARGETS[lang]) + [T.CHECK_PAIRS[lang]])
                      for lang in LANGS}
        self.info = {"scenario": scenario, "crash_after": crash_after, "score_err": score_err, "sr": self.sr}

    def _tick(self):
        self.n += 1
        if self.crash_after and self.n == self.crash_after:
            raise RuntimeError("CUDA out of memory. (fake infra failure for the selftest)")

    def _audio(self, lang, p, name, seed):
        import numpy as np
        if seed == T.CHECK_SEED:
            return ("copy", lang, p, name)
        if (lang, p, name, seed) in self.silent:
            return np.zeros(int(1.6 * self.sr), dtype=np.float32)
        h = int(hashlib.md5(("%s|%d|%s|%d" % (lang, p, name, seed)).encode()).hexdigest()[:8], 16)
        dur = 1.2 + (h % 800) / 1000.0
        f = 150 + (h % 400)
        t = np.arange(int(dur * self.sr)) / self.sr
        return (0.3 * np.sin(2 * np.pi * f * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

    def synth_plain(self, lang, pair, which, seed):
        self._tick()
        return self._audio(lang, pair["pair_idx"], which, seed), {}

    def synth_inj(self, lang, pair, cell, seed):
        self._tick()
        p = pair["pair_idx"]
        name = "inj_%s" % cell["cell"]
        if (lang, p, name, seed) in self.genfail:
            raise RuntimeError("fake generation failure (not an environment failure)")
        sa, sb, _ = self.spans[lang][span_cell_layer(cell)][p]
        meta = {"point": "fake", "route": "fake", "span_a": sa, "span_b": sb}
        return self._audio(lang, p, name, seed), meta

    def save(self, audio, path):
        import soundfile as sf
        if os.path.exists(path):
            raise Stop(EXIT_ENV, "保存先が既にある: %s" % path)
        if isinstance(audio, tuple) and audio[0] == "copy":
            _, lang, p, name = audio
            cm = cellmap(lang)
            cell = cm[name[4:]] if name.startswith("inj_") else T.regen_cells(lang)[0]
            T.copy_new(T.prod_wav(self.ev, cell, p, "inj" if name.startswith("inj_") else name), path)
            return None
        sf.write(path, audio, self.sr, subtype="PCM_16")
        return round(len(audio) / self.sr, 3)

    def score(self, stg, lang, out_json, log_path, seed, cell):
        rows = []
        prod = {r["pair_idx"]: r for r in T.read_json(os.path.join(self.ev, cell["pib"]))["rows"]}
        for f in sorted(os.listdir(stg)):
            if not f.endswith("_inj.wav"):
                continue
            p = int(f[1:5])
            if p == T.CHECK_PAIRS[lang]:
                rows.append(prod[p])
            elif self.score_err and (lang, cell["cell"], p, seed) == self.SCORE_ERR_AT:
                rows.append({"pair_idx": p, "err": self.SCORE_ERR_TEXT[self.score_err]})
            elif (lang, cell["cell"], p, seed) in self.alignfail:
                rows.append({"pair_idx": p, "skip": "align_failed"})
            else:
                h = int(hashlib.md5(("%s|%s|%d|%d" % (lang, cell["cell"], p, seed)).encode()).hexdigest()[:8], 16)
                bl, inj = round((h % 1000) / 1000.0, 4), round(((h >> 10) % 1000) / 1000.0, 4)
                rows.append({"pair_idx": p, "available": True, "bl_ent_ipa": "fake", "inj_ent_ipa": "fake",
                             "tgt_ent_ipa": "fake", "bl_per_vs_tgt": bl, "inj_per_vs_tgt": inj,
                             "delta_per": round(bl - inj, 4), "inj_better": bool(inj < bl)})
        doc = {"summary": {"fake": True}, "rows": rows}
        T.write_json_new(doc, out_json, indent=2)
        with open(log_path, "x") as lf:
            lf.write("fake scorer\n")
        return doc


# ---------------------------------------------------------------- 失敗の記録（環境の失敗と CUDA の失敗）
def read_strikes(base):
    p = os.path.join(base, "strikes.json")
    return T.read_json(p) if os.path.exists(p) else {}


def add_strike(base, key, rec):
    p = os.path.join(base, "strikes.json")
    doc = read_strikes(base)
    doc.setdefault(key, []).append(dict(rec, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))
    tmp = p + ".%d.tmp" % os.getpid()
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)   # _t3fix の中の帳面だけ
    return len([r for r in doc[key] if r.get("kind") == "cuda_fault"])


# ---------------------------------------------------------------- 1 対の生成
def latest_complete(base_dir):
    recs = [d for d in T.attempt_dirs(base_dir) if os.path.exists(os.path.join(d, "record.json"))]
    if len(recs) > 1:
        raise Stop(EXIT_AUTHOR, "完了した試行が 2 つある（同じ seed で作り直したことになる）: %s" % recs)
    return (T.read_json(os.path.join(recs[0], "record.json")), recs[0]) if recs else (None, None)


def gen_pair(B, ev, base, lang, pair, seed, base_dir, spans, log):
    p = pair["pair_idx"]
    d = T.next_attempt_dir(base_dir)
    T.write_json_new({"lang": lang, "pair_idx": p, "seed": seed, "started": T.now(), "pid": os.getpid(),
                      "slurm_job": os.environ.get("SLURM_JOB_ID")}, os.path.join(d, "started.json"))
    cm = cellmap(lang)
    rec = {"lang": lang, "pair_idx": p, "seed": seed, "attempt": os.path.relpath(d, ev), "mode": B.mode,
           "texts": {"text_a": pair["text_a"], "text_b": pair["text_b"], "word_a": pair["word_a"],
                     "word_b": pair["word_b"]},
           "gen": {}, "wavs": {}, "status": None, "reasons": [], "started": T.now()}
    t_pair = time.time()
    for name in T.wav_names(lang):
        t0 = time.time()
        n_lang = len(B.lang_log)
        try:
            if name in ("bl", "tgt"):
                audio, meta = B.synth_plain(lang, pair, name, seed)
                cell = None
            else:
                cell = cm[name[4:]]
                audio, meta = B.synth_inj(lang, pair, cell, seed)
        except PyopenjtalkRequested as e:
            raise Stop(EXIT_AUTHOR, "対 %s/%d の %s: %s（本番と計算が変わるので止める）" % (lang, p, name, e))
        except Stop:
            raise
        except Exception as e:  # noqa: BLE001
            msg = "%s: %s" % (type(e).__name__, str(e)[:500])
            key = "seed%d:%s:%d" % (seed, lang, p)
            if T.RC.is_infra_error(msg):
                add_strike(base, key, {"kind": "infra", "name": name, "err": msg, "attempt": rec["attempt"]})
                raise Stop(EXIT_ENV, "環境の失敗（対 %s/%d、%s、seed %d）: %s。同じコマンドで流し直す" % (lang, p, name, seed, msg))
            if T.RC.is_cuda_fault(msg):
                n = add_strike(base, key, {"kind": "cuda_fault", "name": name, "err": msg, "attempt": rec["attempt"]})
                if n < T.RC.CUDA_STRIKES_TO_FAIL:
                    raise Stop(EXIT_ENV, "CUDA の失敗（1 回目。同じ seed で流し直す）: %s" % msg)
                rec["gen"][name] = {"status": "error", "error": msg, "cuda_strikes": n,
                                    "traceback": traceback.format_exc()[-1500:]}
                rec["status"] = "generation_failed"
                rec["reasons"].append("%s:cuda_fault_x%d" % (name, n))
                rec.update(complete=True, finished=T.now(), sec=round(time.time() - t_pair, 1))
                T.write_json_new(rec, os.path.join(d, "record.json"))
                raise Stop(EXIT_ENV, "CUDA の失敗（2 回目。この対は seed %d で生成の失敗とした）。流し直す" % seed)
            rec["gen"][name] = {"status": "error", "error": msg, "traceback": traceback.format_exc()[-1500:]}
            rec["status"] = "generation_failed"
            rec["reasons"].append("%s:exception" % name)
            break
        rec["gen"][name] = {"lang_detect": B.lang_log[n_lang:]}
        if audio is None:
            rec["gen"][name].update(status="no_audio", meta=meta)
            rec["status"] = "generation_failed"
            rec["reasons"].append("%s:no_audio" % name)
            break
        if cell is not None:
            want = spans[span_cell_layer(cell)][p]
            got = (meta.get("span_a"), meta.get("span_b"))
            rec["gen"][name]["span_runtime"] = got
            rec["gen"][name]["span_prod_meta"] = want[:2]
            if list(got) != [want[0], want[1]]:
                rec["gen"][name]["status"] = "span_mismatch"
                rec["status"] = "span_mismatch"
                rec.update(complete=True, finished=T.now())
                T.write_json_new(rec, os.path.join(d, "record.json"))
                raise Stop(EXIT_AUTHOR, "対 %s/%d の %s: 実行時の範囲 %s が本番の記録 %s と違う" % (lang, p, name, got, want[:2]))
        path = os.path.join(d, "%s.wav" % name)
        dur = B.save(audio, path)
        w = T.wav_check(path)
        rec["wavs"][name] = {"file": os.path.relpath(path, ev), "md5": T.md5_file(path), "dur_sec": dur,
                             "silent": bool(w.get("silent")), "bad": bool(w.get("bad")), "why": w.get("why"),
                             "rms": w.get("rms"), "peak": w.get("peak"), "active_frac": w.get("active_frac")}
        rec["gen"][name].update(status="ok", meta={k: v for k, v in (meta or {}).items() if k != "selfcheck"},
                                sec=round(time.time() - t0, 1))
        if w.get("bad"):
            rec["reasons"].append("%s:%s" % (name, w.get("why")))
    if rec["status"] is None:
        rec["status"] = "silent" if rec["reasons"] else "generated_ok"
    inj = {k: v["md5"] for k, v in rec["wavs"].items() if k.startswith("inj_")}
    if "inj_A" in inj and "inj_L25" in inj:
        rec["inj_A_equals_inj_L25"] = inj["inj_A"] == inj["inj_L25"]   # 記録だけ（本番では 295/300 対で同じ）
    rec.update(complete=True, finished=T.now(), sec=round(time.time() - t_pair, 1))
    T.write_json_new(rec, os.path.join(d, "record.json"))
    log("[t3fix] seed %d %s p%04d -> %s %s (%.0f s)" % (seed, lang, p, rec["status"], rec["reasons"], rec["sec"]))
    return rec, d


def get_or_gen(B, ev, base, lang, pair, seed, base_dir, spans, log):
    rec, d = latest_complete(base_dir)
    if rec is not None:
        log("[t3fix] seed %d %s p%04d: 完了した試行 %s を使う（%s）" % (seed, lang, pair["pair_idx"],
                                                               os.path.relpath(d, ev), rec["status"]))
        if rec["status"] == "span_mismatch":
            raise Stop(EXIT_AUTHOR, "対 %s/%d は範囲の不一致で止まっている（著者の判断）" % (lang, pair["pair_idx"]))
        return rec, d
    if B is None:
        raise Stop(EXIT_ENV, "生成が要るのに生成器が無い")
    return gen_pair(B, ev, base, lang, pair, seed, base_dir, spans, log)


# ---------------------------------------------------------------- seed 42 の再現確認
def run_check42(B, ev, base, pairs_by_lang, spans, log):
    res_p = os.path.join(base, "check42", "result.json")
    if os.path.exists(res_p):
        return T.read_json(res_p)
    out = {"seed": T.CHECK_SEED, "pairs": dict(T.CHECK_PAIRS), "per_file": {}, "gen_status": {}}
    for lang in LANGS:
        p = T.CHECK_PAIRS[lang]
        rec, d = get_or_gen(B, ev, base, lang, pairs_by_lang[lang][p], T.CHECK_SEED,
                            os.path.join(base, "check42", lang), spans[lang], log)
        out["gen_status"][lang] = rec["status"]
        cm = cellmap(lang)
        for name in T.wav_names(lang):
            if name.startswith("inj_"):
                prod = T.prod_wav(ev, cm[name[4:]], p, "inj")
            else:
                prod = T.prod_wav(ev, T.regen_cells(lang)[0], p, name)
            got = (rec["wavs"].get(name) or {}).get("md5")
            out["per_file"]["%s/p%04d/%s" % (lang, p, name)] = {"regen": got, "prod": T.md5_file(prod),
                                                               "match": got == T.md5_file(prod)}
    out["match_all"] = all(v["match"] for v in out["per_file"].values())
    out["n_match"] = sum(v["match"] for v in out["per_file"].values())
    out["n_files"] = len(out["per_file"])
    out["finished"] = T.now()
    T.write_json_new(out, res_p)
    log("[t3fix] seed 42 の再現確認: %d/%d 一致" % (out["n_match"], out["n_files"]))
    return out


# ---------------------------------------------------------------- 採点
def row_err_kind(r):
    """phoneme_per.py の {"err": ...} の行の種類: None（例外の行でない）/ infra / cuda_fault / other."""
    e = (r or {}).get("err")
    if e is None:
        return None
    if T.RC.is_infra_error(e):
        return "infra"
    if T.RC.is_cuda_fault(e):
        return "cuda_fault"
    return "other"


def err_rows(rows):
    """{種類: [(pair_idx, err), ...]}（例外の行だけ）."""
    out = {}
    for r in rows:
        k = row_err_kind(r)
        if k is not None:
            out.setdefault(k, []).append((r.get("pair_idx"), str(r.get("err"))[:300]))
    return out


def done_has_env_errors(done):
    return any(row_err_kind(r) in ("infra", "cuda_fault")
               for lc in (done.get("rows") or {}).values() for cr in lc.values() for r in cr.values())


def check_score_errors(base, ev, seed, lang, cell, doc, attempt, opts):
    """採点の行の例外を、使えるかの判定に入れる前に分ける。環境の失敗は同じ seed で採点し直す（終了 4）。
    CUDA の失敗は同じ seed・同じセルで 2 回目なら著者の判断（終了 3）。ほかの例外は、著者が
    --count_scorer_err_as_invalid で認めていなければ著者の判断（終了 3）。どれも seed を進めない."""
    errs = err_rows(doc["rows"])
    if not errs:
        return
    key = "score:seed%d:%s:%s" % (seed, lang, cell["cell"])
    for kind in ("infra", "cuda_fault"):
        if errs.get(kind):
            n = add_strike(base, key, {"kind": kind, "rows": errs[kind][:10], "attempt": os.path.relpath(attempt, ev)})
            if kind == "cuda_fault" and n >= T.RC.CUDA_STRIKES_TO_FAIL:
                raise Stop(EXIT_AUTHOR, "採点（seed %d、%s %s）で CUDA の失敗が %d 回目: %s。どの対が原因か分けられないので"
                                        "seed は進めない（著者の判断）" % (seed, lang, cell["cell"], n, errs[kind][:3]))
            raise Stop(EXIT_ENV, "採点（seed %d、%s %s）で環境の失敗: %s。同じコマンドで流し直す（同じ seed で採点し直す）"
                                 % (seed, lang, cell["cell"], errs[kind][:3]))
    if errs.get("other") and not opts.get("count_scorer_err_as_invalid"):
        raise Stop(EXIT_AUTHOR, "採点器の例外（環境の失敗ではない。seed %d、%s %s）: %s。seed は進めない。"
                                "位置合わせの失敗と同じに数える（その対は使えない）と著者が決めたら、"
                                "--count_scorer_err_as_invalid を付けて同じコマンドで流す"
                                % (seed, lang, cell["cell"], errs["other"][:3]))


def score_seed(B, ev, base, seed, ok_pairs, gens, log, opts=None):
    opts = opts or {}
    sdir = os.path.join(base, "seed%d" % seed, "score")
    want = {lang: sorted(ok_pairs[lang]) for lang in LANGS}
    for a in reversed(T.attempt_dirs(sdir)):
        dp = os.path.join(a, "done.json")
        if os.path.exists(dp):
            done = T.read_json(dp)
            if done["pairs"] == want and not done_has_env_errors(done):
                log("[t3fix] seed %d: 採点 %s を使う" % (seed, os.path.relpath(a, ev)))
                return done
    if B is None:
        raise Stop(EXIT_ENV, "採点が要るのに生成器が無い")
    a = T.next_attempt_dir(sdir)
    rows, check = {}, {}
    for lang in LANGS:
        if not want[lang]:
            continue
        chk = T.CHECK_PAIRS[lang]
        rows[lang], check[lang] = {}, {}
        for cell in T.regen_cells(lang):
            stg = os.path.join(a, "%s_%s" % (lang, cell["cell"]))
            for p in want[lang]:
                g = gens[lang][p]
                for src_name, dst in (("bl", "bl"), ("tgt", "tgt"), ("inj_%s" % cell["cell"], "inj")):
                    T.copy_new(os.path.join(ev, g["wavs"][src_name]["file"]), os.path.join(stg, "s%04d_%s.wav" % (p, dst)))
            for c in ("bl", "tgt", "inj"):   # 採点器の確認: 確認の対の本番の音声も一緒に採点する
                T.copy_new(T.prod_wav(ev, cell, chk, c), os.path.join(stg, "s%04d_%s.wav" % (chk, c)))
            out = stg + "_pib.json"
            doc = B.score(stg, lang, out, stg + "_pib.log", seed, cell)
            check_score_errors(base, ev, seed, lang, cell, doc, a, opts)   # done.json を書く前に
            got = {r["pair_idx"]: r for r in doc["rows"]}
            rows[lang][cell["cell"]] = {str(p): got.get(p) for p in want[lang]}
            prod = {r["pair_idx"]: r for r in T.read_json(os.path.join(ev, cell["pib"]))["rows"]}
            check[lang][cell["cell"]] = {"pair": chk, "equal": got.get(chk) == prod.get(chk),
                                         "regen": got.get(chk), "prod": prod.get(chk)}
    done = {"seed": seed, "pairs": want, "rows": rows, "scorer_check": check,
            "scorer_check_all_equal": all(v["equal"] for lc in check.values() for v in lc.values()),
            "attempt": os.path.relpath(a, ev), "finished": T.now()}
    T.write_json_new(done, os.path.join(a, "done.json"))
    log("[t3fix] seed %d: 採点した（採点器の確認 %s）" % (seed, done["scorer_check_all_equal"]))
    return done


def decide(seed, todo, gens, scored, opts=None):
    opts = opts or {}
    out = {"seed": seed, "todo": todo, "pairs": {}, "rule": RULE, "decided": T.now(),
           "count_scorer_err_as_invalid": bool(opts.get("count_scorer_err_as_invalid"))}
    for lang in LANGS:
        out["pairs"][lang] = {}
        for p in todo[lang]:
            g = gens[lang][p]
            reasons = list(g["reasons"]) if g["status"] != "generated_ok" else []
            if g["status"] != "generated_ok":
                reasons.insert(0, g["status"])
            else:
                for cell in T.regen_cells(lang):
                    r = (((scored or {}).get("rows") or {}).get(lang) or {}).get(cell["cell"], {}).get(str(p))
                    ek = row_err_kind(r)
                    if ek in ("infra", "cuda_fault"):   # check_score_errors が先に止めるはず（念のため）
                        raise Stop(EXIT_ENV, "seed %d %s 対 %d %s: 採点の行が環境の失敗（%s）。同じ seed で採点し直す"
                                             % (seed, lang, p, cell["cell"], r.get("err")))
                    if ek == "other" and not opts.get("count_scorer_err_as_invalid"):
                        raise Stop(EXIT_AUTHOR, "seed %d %s 対 %d %s: 採点器の例外（%s）。著者の判断"
                                                "（認めるなら --count_scorer_err_as_invalid）" % (seed, lang, p, cell["cell"],
                                                                                       r.get("err")))
                    if not T.pib_row_ok(r):
                        why = (r or {}).get("skip") or (ek and "err") or "no_row"
                        reasons.append("pib_%s:%s" % (cell["cell"], why))
            out["pairs"][lang][str(p)] = {"valid": not reasons, "reasons": reasons, "status": g["status"],
                                          "attempt": g["attempt"]}
    return out


# ---------------------------------------------------------------- 時間の見積もり
def estimate(n_seed44_pairs=None):
    n_check = len(T.wav_names("en")) + len(T.wav_names("zh"))
    n_seed = len(T.REGEN_TARGETS["en"]) * len(T.wav_names("en")) + len(T.REGEN_TARGETS["zh"]) * len(T.wav_names("zh"))
    n_cells = len(T.regen_cells("en")) + len(T.regen_cells("zh"))
    typical = LOAD_MIN + (n_check + n_seed) * SEC_PER_SYNTH / 60 + n_cells * SCORE_MIN_PER_CELL
    worst = typical + n_seed * SEC_PER_SYNTH / 60 + n_cells * SCORE_MIN_PER_CELL
    pick = next((t for t in ALLOWED_SLURM if int(t[:2]) * 60 + int(t[3:5]) >= 1.5 * worst), None)
    return {"syntheses_check42": n_check, "syntheses_per_seed": n_seed, "score_cells_per_seed": n_cells,
            "typical_min": round(typical, 1), "worst_min_all_pairs_to_seed44": round(worst, 1),
            "slurm_time": pick, "basis": "%.1f s per synthesis (production sweeps), %.1f min model load, "
                                         "%.1f min per phoneme_per.py cell; SLURM_TIME >= 1.5 x worst" % (
                                             SEC_PER_SYNTH, LOAD_MIN, SCORE_MIN_PER_CELL)}


GPU_CMD = ("SLURM_TIME=%s slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py "
           "--min_free_gb 30 -- /mnt/scripts/spanfix/t3fix_regen_gpu.py")


# ---------------------------------------------------------------- --plan（CPU、ホスト）
def wiring_check():
    """RealBackend が前提にしている本番のスクリプトの形を、ソースの AST と文字列だけで確かめる."""
    import ast
    src = open(T.P(T.LAYER_REL), encoding="utf-8").read()
    tree = ast.parse(src)
    funcs = {n.name: [a.arg for a in n.args.args] for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns = {t.id: getattr(n.value, "value", None) for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)}
    inj = open(T.P("scripts/encoder_injection_t5gemma.py"), encoding="utf-8").read()
    de = open(os.path.join(T.T5GEMMA_ROOT, "duration_estimator.py"), encoding="utf-8").read()
    gen_fn = inj[inj.index("def generate_with_modified_encoder"):inj.index("def find_entity_token_range")]
    return {
        "synth_plain_signature": funcs.get("synth_plain") == ["model", "text_tokenizer", "audio_tok", "text", "device",
                                                              "codec_audio_sr", "codec_sr", "modified_hs", "attn",
                                                              "pos_ids"],
        "synth_inj_at_layer_signature": funcs.get("synth_inj_at_layer") == [
            "model", "text_tokenizer", "audio_tok", "text_a", "text_b", "word_a", "word_b", "device", "add_bos",
            "inject_layer", "use_hook", "codec_audio_sr", "codec_sr", "selfcheck"],
        "module_SEED_42": assigns.get("SEED") == 42,
        "module_LANG_en": assigns.get("LANG") == "en",
        "seed_everything_uses_module_SEED": src.count("seed_everything(SEED)") >= 2,
        "save_line": 'torchaudio.save(pth, a.cpu(), sr, encoding="PCM_S", bits_per_sample=16)' in src,
        "load_line": ('"Aratako/T5Gemma-TTS-2b-2b", trust_remote_code=True,\n        dtype=torch.bfloat16, '
                      'device_map="auto")') in src,
        "estimate_duration_called_by_name": "estimate_duration(target_text=target_text_norm, target_lang=lang_code)" in gen_fn,
        "estimate_duration_imported_by_name": "from duration_estimator import estimate_duration" in inj,
        "vendor_pyopenjtalk_optional": "try:\n    import pyopenjtalk\nexcept ImportError:\n    pyopenjtalk = None" in de,
        "vendor_phoneme_count_ja_by_name": "return _phoneme_count_ja(text)" in de,
    }


def write_plan(ev, out):
    pdir = os.path.join(regen_base(ev), "plan")
    os.makedirs(pdir, exist_ok=True)
    pp = os.path.join(pdir, "plan_%s_%d.json" % (T.stamp(), os.getpid()))
    T.write_json_new(out, pp)
    return pp


def plan_mode(ev, log):
    out = {"created": T.now(), "host": T.host_state(), "code_pin_errors": T.code_pin_errors(),
           "manuscript_check": T.manuscript_check(), "apply_state": T.apply_state(ev), "problems": []}
    if out["code_pin_errors"]:
        out["problems"] += out["code_pin_errors"]
    # 式と表の値の自己確認（反映の段階を見る: 反映の前は原稿の値、B1・B2 の後は監査の値、作り直しの後は記録の値）
    out["formula_selftest"] = T.formula_selftest(ev)
    if not out["formula_selftest"]["ok"]:
        out["problems"].append("式・表の値の自己確認に失敗: %s" % T.formula_fail_reasons(out["formula_selftest"]))
    log("[plan] apply state: %s; formula/table self-check ok=%s (expected values from: %s)" % (
        {k: ("applied" if v["applied"] else ("INTERRUPTED" if v["interrupted"] else "not applied"))
         for k, v in out["apply_state"].items()}, out["formula_selftest"]["ok"],
        {k: x["source"] for k, x in out["formula_selftest"]["expected"].items()}))
    if out["apply_state"]["regen_apply"]["applied"]:
        out["note"] = "regenerated pairs already applied; no GPU job is needed"
        pp = write_plan(ev, out)
        log("[plan] 作り直した対は反映済み（%s）。GPU のジョブは要らない。" % out["apply_state"]["regen_apply"]["record"])
        log("[plan] problems: %s" % (out["problems"] or "none"))
        log("[plan] -> %s" % pp)
        return 3 if out["problems"] else 0
    if os.path.exists(os.path.join(regen_base(ev), "final", "final_meta.json")):
        log("[plan] GPU の結果（final/final_meta.json）は既にある。ジョブを出し直しても何もしない。次は t3fix_regen_apply.py")
    pairs = {lang: T.load_pairs(lang) for lang in LANGS}
    # 1. 無音の走査（本番の 300 対）: 対象の対の集合が著者の挙げた対と同じか
    out["invalid_scan"] = {}
    for lang in LANGS:
        bad = scan_invalid(ev, lang, scan_pairs(ev, lang))
        out["invalid_scan"][lang] = {str(p): v for p, v in sorted(bad.items())}
        if sorted(bad) != sorted(T.REGEN_TARGETS[lang]):
            out["problems"].append("%s: 使えない対 %s が対象 %s と違う" % (lang, sorted(bad), list(T.REGEN_TARGETS[lang])))
    # 2. 範囲（tokenizer だけ）: 今の関数の範囲が本番の掃引の記録と同じか。ZH は 5/17 の古い関数とも同じか
    import spanfix_common as SFC
    W = SFC.load_shim_module()
    off = SFC.setup_offline(W)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.read_json(os.path.join(off["offline_preflight"]["Aratako/T5Gemma-TTS-2b-2b"]["snapshot"],
                                                                 "config.json")).get("text_tokenizer_name"))
    fixed = T.load_function_from_source(T.P("scripts/encoder_injection_t5gemma.py"), "find_entity_token_range")
    legacy = T.load_function_from_source(T.P(T.LEGACY_SPAN_REL), "find_entity_token_range")
    out["spans"] = {}
    for lang in LANGS:
        ps = list(T.REGEN_TARGETS[lang]) + [T.CHECK_PAIRS[lang]]
        sp = prod_spans(ev, lang, ps)
        for p in ps:
            q = pairs[lang][p]
            fa, fb = fixed(tok, q["text_a"], q["word_a"], add_bos=0), fixed(tok, q["text_b"], q["word_b"], add_bos=0)
            la, lb = legacy(tok, q["text_a"], q["word_a"], add_bos=0), legacy(tok, q["text_b"], q["word_b"], add_bos=0)
            fx = [list(fa) if fa else None, list(fb) if fb else None]
            lg = [list(la) if la else None, list(lb) if lb else None]
            meta_all = {L: [sp[L][p][0], sp[L][p][1]] for L in T.SWEEP_LAYERS}
            ok = all(v == fx for v in meta_all.values())
            out["spans"]["%s/%d" % (lang, p)] = {"fixed": fx, "legacy_0517": lg, "prod_meta_all_layers_equal": ok,
                                                "legacy_equals_fixed": lg == fx}
            if not ok:
                out["problems"].append("%s 対 %d: 今の関数の範囲 %s が本番の掃引の記録と違う %s" % (lang, p, fx, meta_all))
            if lang == "zh" and lg != fx:
                out["problems"].append("zh 対 %d: (a) を作った 5/17 の関数と今の関数で範囲が違う" % p)
    # 3. 長さ推定器の言語判定（vendor の関数をそのまま、langdetect で）: ja になる文が無いこと
    import re
    from typing import Optional
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
        DetectorFactory.seed = 0
    except ImportError:
        detect, LangDetectException = None, Exception
    det = T.load_function_from_source(os.path.join(T.T5GEMMA_ROOT, "duration_estimator.py"), "_safe_detect_language",
                                      {"re": re, "Optional": Optional, "detect": detect,
                                       "LangDetectException": LangDetectException})
    out["lang_detect"] = {}
    for lang in LANGS:
        for p in list(T.REGEN_TARGETS[lang]) + [T.CHECK_PAIRS[lang]]:
            q = pairs[lang][p]
            r = {"text_a": det(q["text_a"]), "text_b": det(q["text_b"])}
            out["lang_detect"]["%s/%d" % (lang, p)] = r
            if "ja" in r.values():
                out["problems"].append("%s 対 %d: 長さ推定器の言語判定が ja（pyopenjtalk の経路）" % (lang, p))
    out["langdetect_available"] = detect is not None
    # 4. 生成の経路の静的な確認（ファイルを import しない）: このラッパーが使う本番の関数・引数・保存・読み込みの行
    out["wiring"] = wiring_check()
    if not all(out["wiring"].values()):
        out["problems"].append("本番のスクリプトの形がラッパーの前提と違う: %s" % {k: v for k, v in out["wiring"].items() if not v})
    # 5. 本番の状態（式と表の値の自己確認は最初に行った）
    out["prod_snapshot"] = prod_snapshot(ev)
    est = estimate()
    out["estimate"] = est
    out["gpu_command"] = GPU_CMD % est["slurm_time"]
    out["existing_regen_state"] = {"prod_snapshot": os.path.exists(os.path.join(regen_base(ev), "prod_snapshot.json")),
                                   "final_meta": os.path.exists(os.path.join(regen_base(ev), "final", "final_meta.json"))}
    pp = write_plan(ev, out)
    log("[plan] invalid pairs: %s" % {k: sorted(int(x) for x in v) for k, v in out["invalid_scan"].items()})
    log("[plan] spans: %s" % {k: (v["fixed"], v["prod_meta_all_layers_equal"], v["legacy_equals_fixed"])
                              for k, v in out["spans"].items()})
    log("[plan] lang detect: %s" % out["lang_detect"])
    log("[plan] wiring (production functions the GPU wrapper calls): %s" % out["wiring"])
    log("[plan] estimate: %s" % json.dumps(est, ensure_ascii=False))
    log("[plan] GPU command: %s" % out["gpu_command"])
    log("[plan] problems: %s" % (out["problems"] or "none"))
    log("[plan] -> %s" % pp)
    return 3 if out["problems"] else 0


# ---------------------------------------------------------------- 本体
def main_run(args, ev, log):
    base = regen_base(ev)
    os.makedirs(base, exist_ok=True)
    errs = T.code_pin_errors()
    if errs and not args.fake:
        raise Stop(EXIT_AUTHOR, "本番のコードが固定の md5 と違う: %s" % errs)
    pairs = {lang: T.load_pairs(lang) for lang in LANGS}
    final_p = os.path.join(base, "final", "final_meta.json")
    if os.path.exists(final_p):   # 完了済み（差し替えの後は本番が変わっているので、照合より先に見る）
        fm = T.read_json(final_p)
        log("[t3fix] 完了済み（%s）。unresolved=%s" % (final_p, fm.get("unresolved")))
        return EXIT_AUTHOR if any(fm.get("unresolved", {}).values()) else 0
    # 本番の状態: 最初の実行で書き、以後は照合する（差し替えの段階もこれと照合する）
    snap_p = os.path.join(base, "prod_snapshot.json")
    snap = prod_snapshot(ev)
    if os.path.exists(snap_p):
        diff = snapshot_diff(T.read_json(snap_p), snap)
        if diff:
            raise Stop(EXIT_AUTHOR, "本番の対象のファイルが前の実行の後に変わった: %s" % diff[:10])
    else:
        for lang in LANGS:
            bad = scan_invalid(ev, lang, scan_pairs(ev, lang))
            snap.setdefault("invalid_scan", {})[lang] = {str(p): v for p, v in sorted(bad.items())}
            if sorted(bad) != sorted(T.REGEN_TARGETS[lang]):
                raise Stop(EXIT_AUTHOR, "%s: 使えない対 %s が対象 %s と違う" % (lang, sorted(bad), T.REGEN_TARGETS[lang]))
        T.write_json_new(snap, snap_p)
    spans = {lang: prod_spans(ev, lang, list(T.REGEN_TARGETS[lang]) + [T.CHECK_PAIRS[lang]]) for lang in LANGS}
    for lang in LANGS:
        for L, d in spans[lang].items():
            for p, (sa, sb, st) in d.items():
                if st != "ok" or sa is None or sb is None:
                    raise Stop(EXIT_AUTHOR, "本番の掃引の記録に対 %s/%d（L%d）の範囲が無い" % (lang, p, L))
    B = (FakeBackend(ev, args.fake_scenario, args.fake_crash_after, log, score_err=args.fake_score_err)
         if args.fake else RealBackend(ev, pairs, log))
    opts = {"count_scorer_err_as_invalid": bool(args.count_scorer_err_as_invalid)}
    run_info = {"backend": B.info, "opts": opts}
    # seed 42 の再現確認
    chk = run_check42(B, ev, base, pairs, spans, log)
    run_info["check42_match_all"] = chk["match_all"]
    if not chk["match_all"] and not args.continue_on_check_mismatch:
        raise Stop(EXIT_AUTHOR, "seed 42 の再現が本番と一致しない（%d/%d）。環境の違いを著者に報告する。"
                                "認められたら --continue_on_check_mismatch を付けて流す" % (chk["n_match"], chk["n_files"]))
    # seed 43 → 44
    todo = {lang: list(T.REGEN_TARGETS[lang]) for lang in LANGS}
    chosen, decisions, scored_by_seed = {lang: {} for lang in LANGS}, {}, {}
    for seed in T.SEEDS:
        if not any(todo.values()):
            break
        dec_p = os.path.join(base, "seed%d" % seed, "decision.json")
        gens = {lang: {} for lang in LANGS}
        for lang in LANGS:
            for p in todo[lang]:
                gens[lang][p], _ = get_or_gen(B, ev, base, lang, pairs[lang][p], seed,
                                              os.path.join(base, "seed%d" % seed, "gen", lang, "p%04d" % p),
                                              spans[lang], log)
        ok_pairs = {lang: [p for p in todo[lang] if gens[lang][p]["status"] == "generated_ok"] for lang in LANGS}
        scored = score_seed(B, ev, base, seed, ok_pairs, gens, log, opts) if any(ok_pairs.values()) else None
        scored_by_seed[seed] = scored
        if os.path.exists(dec_p):
            dec = T.read_json(dec_p)
            if dec["todo"] != {k: list(v) for k, v in todo.items()}:
                raise Stop(EXIT_AUTHOR, "seed %d の判定の対象 %s が今の対象 %s と違う" % (seed, dec["todo"], todo))
        else:
            dec = decide(seed, todo, gens, scored, opts)
            T.write_json_new(dec, dec_p)
        decisions[seed] = dec
        for lang in LANGS:
            for p in todo[lang]:
                if dec["pairs"][lang][str(p)]["valid"]:
                    chosen[lang][p] = seed
        log("[t3fix] seed %d: %s" % (seed, {lang: {p: (v["valid"], v["reasons"]) for p, v in dec["pairs"][lang].items()}
                                            for lang in LANGS}))
        todo = {lang: [p for p in todo[lang] if not dec["pairs"][lang][str(p)]["valid"]] for lang in LANGS}
    unresolved = {lang: list(v) for lang, v in todo.items()}
    # 採用した seed の音声と行を final にまとめる。音声は新しい final/aNN に写し（途中で止まったまとめは残す）、
    # final_meta.json は最後に書く（それまでは「完了」とみなさない）
    fin = T.next_attempt_dir(os.path.join(base, "final"))
    meta = {"created": T.now(), "mode": B.mode, "rule": RULE, "chosen_seed": {}, "validity": {}, "unresolved": unresolved,
            "wavs": {}, "rows": {}, "check42": chk, "scorer_check": {}, "code_md5": T.code_md5_record(),
            "prod_snapshot": os.path.relpath(snap_p, ev), "backend": B.info, "opts": opts,
            "final_dir": os.path.relpath(fin, ev), "cells": {lang: T.regen_cells(lang) for lang in LANGS}}
    crash_final = args.fake_crash_in_final if args.fake else 0
    n_copied = 0
    for seed, dec in decisions.items():
        for lang in LANGS:
            for p, v in dec["pairs"][lang].items():
                meta["validity"].setdefault(lang, {}).setdefault(p, {})[str(seed)] = {"valid": v["valid"],
                                                                                       "reasons": v["reasons"],
                                                                                       "attempt": v["attempt"]}
        sc = scored_by_seed.get(seed)
        if sc:
            meta["scorer_check"][str(seed)] = {"all_equal": sc["scorer_check_all_equal"],
                                               "cells": {lang: {c: v["equal"] for c, v in lc.items()}
                                                         for lang, lc in sc["scorer_check"].items()}}
    for lang in LANGS:
        meta["chosen_seed"][lang] = {str(p): s for p, s in chosen[lang].items()}
        for p, seed in chosen[lang].items():
            rec, d = latest_complete(os.path.join(base, "seed%d" % seed, "gen", lang, "p%04d" % p))
            wd = {}
            for name in T.wav_names(lang):
                dst = os.path.join(fin, lang, "p%04d" % p, "%s.wav" % name)
                T.copy_new(os.path.join(ev, rec["wavs"][name]["file"]), dst)
                n_copied += 1
                if crash_final and n_copied == crash_final:
                    raise RuntimeError("fake kill during the final assembly (selftest), after %d copies" % n_copied)
                wd[name] = {"file": os.path.relpath(dst, ev), "md5": T.md5_file(dst), "from": rec["wavs"][name]["file"],
                            "silent": rec["wavs"][name]["silent"]}
                if wd[name]["md5"] != rec["wavs"][name]["md5"]:
                    raise Stop(EXIT_ENV, "final に写した md5 が違う: %s" % dst)
            meta["wavs"].setdefault(lang, {})[str(p)] = {"seed": seed, "attempt": rec["attempt"], "files": wd,
                                                         "inj_A_equals_inj_L25": rec.get("inj_A_equals_inj_L25")}
            sc = scored_by_seed[seed]
            for cell in T.regen_cells(lang):
                meta["rows"].setdefault(lang, {}).setdefault(cell["cell"], {})[str(p)] = sc["rows"][lang][cell["cell"]][str(p)]
    T.write_json_new(meta, final_p)
    log("[t3fix] final -> %s; chosen=%s unresolved=%s" % (final_p, meta["chosen_seed"], unresolved))
    run_info.update(chosen=meta["chosen_seed"], unresolved=unresolved)
    if any(unresolved.values()):
        log("[t3fix] seed 44 でも使えない対がある。45 以上は決まっていないので著者の判断。差し替えの段階はこの対を扱えない。")
        return EXIT_AUTHOR
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="CPU の予行（ホスト）")
    ap.add_argument("--fake", action="store_true", help="偽の生成と採点（試験の写しの中だけ）")
    ap.add_argument("--sandbox", default=None)
    ap.add_argument("--fake_scenario", default="resolved", choices=["resolved", "unresolved", "all_valid"])
    ap.add_argument("--fake_crash_after", type=int, default=0)
    ap.add_argument("--fake_score_err", default=None, choices=["infra", "cuda", "other"],
                    help="試験用: 偽の採点の (en, L0, 対 112, seed 43) の行を例外の行にする")
    ap.add_argument("--fake_crash_in_final", type=int, default=0,
                    help="試験用: final のまとめで、この本数を写した後に止める")
    ap.add_argument("--continue_on_check_mismatch", action="store_true",
                    help="seed 42 の再現が一致しなくても続ける（著者が認めたときだけ）")
    ap.add_argument("--count_scorer_err_as_invalid", action="store_true",
                    help="環境の失敗でない採点器の例外の行を、位置合わせの失敗と同じに数える（著者が認めたときだけ）")
    args = ap.parse_args()
    ev = T.ev_root(args.sandbox)
    if args.fake and not args.sandbox:
        raise SystemExit("--fake は --sandbox と一緒に使う")
    if (args.fake_score_err or args.fake_crash_in_final) and not args.fake:
        raise SystemExit("--fake_score_err / --fake_crash_in_final は --fake と一緒に使う")
    base = regen_base(ev)
    runs = os.path.join(base, "runs")
    os.makedirs(runs, exist_ok=True)
    tag = "%s_%d" % (T.stamp(), os.getpid())
    logs = []

    def log(msg):
        print(msg, flush=True)
        logs.append([T.now(), msg])
    start = {"started": T.now(), "argv": sys.argv, "ev": ev, "mode": "plan" if args.plan else ("fake" if args.fake else "gpu"),
             "host": T.host_state(), "launch_env": LAUNCH_ENV, "slurm_job": os.environ.get("SLURM_JOB_ID"),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "code_md5": T.code_md5_record()}
    T.write_json_new(start, os.path.join(runs, "run_%s.json" % tag))
    t0 = time.time()
    code, msg = 0, "ok"
    try:
        code = plan_mode(ev, log) if args.plan else main_run(args, ev, log)
    except Stop as e:
        code, msg = e.code, e.msg
        log("[t3fix] STOP (exit %d): %s" % (code, msg))
    except Exception as e:  # noqa: BLE001
        code, msg = EXIT_ENV, "%s: %s" % (type(e).__name__, str(e)[:500])
        log("[t3fix] 予期しない例外（exit %d）: %s\n%s" % (code, msg, traceback.format_exc()))
    T.write_json_new({"finished": T.now(), "exit": code, "msg": msg, "elapsed_min": round((time.time() - t0) / 60, 2),
                      "log": logs}, os.path.join(runs, "run_end_%s.json" % tag))
    sys.exit(code)


if __name__ == "__main__":
    main()
