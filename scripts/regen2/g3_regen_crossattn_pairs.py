#!/usr/bin/env python3
"""付録 B（cross-attention 遮断, tab:crossattn_mask）の EN・ZH で欠けた対を、本番と同じコードで作り直す（G3, 2026-09-29 改訂）.

経緯:
  eval_results/crossattn_entity_dtw_encc_lp_n300 と crossattn_entity_dtw_zhcc_lp_n300 は n_aligned=299。
  EN の対 237（Tiffany→Hashimoto, frames 22/None/48/48/50）と ZH の対 42（陈→高桥, frames 27/69/None/None/57）で
  Whisper が実体語の位置を見つけられなかった。本番スクリプトは wav を保存しないので、既存データからは直せない。

seed の決まり（著者の決定 (1)。判定は g3_common.py にまとめてあり、このスクリプトは始める前に確かめて、外れていれば止まる）:
  * seed 42: 再現の確認だけ（本番と frames が一致するか）。取り込みには使わない。著者の決定に無い追加の条件なので、
    流すのは著者が承認したときだけ（任意。seed 43 はこれを待たない）。コードはどの対でも受け付ける。
  * seed 43: 本番（seed 42）で aligned=False だった対だけ。seed 42 の再実行の有無・結果には左右されない。
  * seed 44: seed 43 の最後の記録が「対として使えない」（unaligned / generation_failed / silent）の対だけ。
  * 45 以上は流さない（決まっていない）。選ぶ基準は有効性だけで、数値は見ない。
  * run_error / alignment_exception / capture_error / no_row_unexplained は「対の良し悪し」ではないので次の seed に
    進まない。原因を直してから同じ seed を --attempt k+1 で流す（前の回が消えずに残り、取り込みは最後の回を使う）。
    alignment_exception（Whisper の呼び出しの例外）が同じ seed で 2 回続いたら 3 回目は流さず著者の判断。

手順（regen_pairs_seed.py / regen_pairs_seed_nt.py と同じ型）:
  - 本番スクリプト scripts/projection/eval_crossattn_entity_dtw.py を改変せずに main() を呼ぶ（md5 を確かめる）。
    引数は本番と同じ（.perm_cmds.txt 16714 行: --lang L --cross_culture --inject_mode length_preserve --n 300）で、
    --output_dir（新しい置き場）だけを足す。
  - 1 回の起動で 1 対だけを扱う（EN と ZH は別のプロセス・別のジョブ）。対象以外の 299 対を「済み」（placeholder）と
    した checkpoint.json を新しい置き場に先に置くので、本番の resume の仕組みで対象の対だけが生成・採点される。
  - モジュール変数 SEED を --seed に置き換える（main 冒頭の seed_all と run_condition / synth_simple の
    seed_everything が参照する）。
  - encoder_injection_t5gemma.generate_with_modified_encoder を、呼び出しをそのまま通して出力だけを写し取る
    薄い包みに差し替える。5 本（BL, INJ, BL_mask, INJ_mask, TGT）の wav を float32 で保存する。
  - 本番の main() は対の処理中の例外を print して握りつぶすので、M.print を「そのまま出力し、err の行を記録する」
    包みに差し替えて、例外（run_error）と生成の失敗（generation_failed）を分ける。
  - 本番の forced_alignment_utils.WhisperAligner.transcribe_words も Whisper の例外を握りつぶし
    （"  [Whisper] transcribe failed: ..." を print して ([], None) を返す）、本番の行は aligned=False になる。
    同じ包みを forced_alignment_utils.print にも入れて（出力は変わらない）、この行を whisper_errors に記録し、
    「Whisper が実体語を見つけられなかった」（unaligned, 次の seed へ）と「Whisper が例外で落ちた」
    （alignment_exception, 同じ seed のやり直し）を分ける。
  - 本番のファイル 4 本（評価・注入・整列・offline の読み替え）の md5 は g3_common.PINNED_MD5 に固定。違えば
    モデルを読まずに止まる。
  - HF_HUB_OFFLINE=1 では、transformers 4.57.3 が T5Gemma の tokenizer を repo id で読むと Hub に問い合わせて落ちる。
    9/28 の前例（scripts/nontemplate/gen_t5gemma_pairs_module.py）の offline_preflight と
    install_offline_tokenizer_shim をそのまま使い（ファイルから読むだけで、sys.path は変えない）、
    repo id を手元の snapshot に読み替える。事前確認と本番の main() は同じ読み方をする。
  - 無音は scripts/regen2/silence_criterion.py（regen2 共通の基準）で、保存した wav から判定する。
  - 記録は <置き場>/regen_row.json（新規。既にあれば止まる）。本番の置き場には何も書かない。何も消さない。

置き場: eval_results/_regen2_crossattn_seed<seed>/<lang>_p<idx:04d>/（2 回目以降は ..._a<k>/）

環境変数（コマンドに env を付けない。このスクリプトが先頭で、numpy・torch・HF の library を読む前に自分で入れる）:
  * スレッド上限 5 本（OMP / MKL / OPENBLAS / NUMEXPR / VECLIB）= 13。GPU の処理では torch.set_num_threads(13) も呼ぶ。
  * HF とキャッシュの置き場。<根> はコンテナの中なら /mnt、ホストなら $REPO_ROOT:
      HF_HOME=<根>/.cache/huggingface  HF_HUB_CACHE=<根>/.cache/huggingface/hub  HF_HUB_OFFLINE=1
      TORCH_HOME=<根>/.cache/torch  XDG_CACHE_HOME=<根>/.cache
    setdefault ではなく上書きする。slurm_run.sh は HF_HOME=/tmp/.cache/huggingface（空のキャッシュ）と
    TORCH_HOME=/mnt/.torch_cache を渡すので、そのままでは決まった置き場にならない。
    起動したときの値は記録（regen_row.json の env.env_at_launch、試運転の報告の hf_env）に残す。

GPU（プロジェクト直下から。1 ジョブ = 1 言語 = 1 seed = GPU 1 枚。投入の前に squeue で合計 2 枚の上限を確かめる。
  ログは >> で追記する: 同じ seed・回を誤って 2 回投入しても、seed の門が止める前に前のログが消えない）:
  seed 43 を先に流す（EN と ZH は別のジョブ。seed 42 を待たない）:
  SLURM_TIME=00:30:00 slurm_run.sh python3 -u /mnt/scripts/regen2/g3_regen_crossattn_pairs.py \
    --targets en:237 --seed 43 >> eval_results/_regen2_crossattn_en237_seed43_a1.log 2>&1
  SLURM_TIME=00:30:00 slurm_run.sh python3 -u /mnt/scripts/regen2/g3_regen_crossattn_pairs.py \
    --targets zh:42 --seed 43 >> eval_results/_regen2_crossattn_zh42_seed43_a1.log 2>&1
  seed 44 と同じ seed のやり直し（--attempt k）は、g3_merge_crossattn.py --status がそう示したときだけ。
  seed 42（再現の確認。任意）は著者が承認したときだけ、seed 43 の後に、合計 2 枚の中で流す。取り込まない。

CPU の試運転（モデルは読まない。トークナイザだけ。ホストで直接。出力は毎回新しい
  eval_results/_regen2_dryrun/g3_crossattn/run_<日時>_seed<seed>/ に置き、何も消さない）:
  taskset -c 0-12 nice -n 10 python3 -B scripts/regen2/g3_regen_crossattn_pairs.py --targets en:237,zh:42 --seed 43 --dry_run

状態と取り込み（ホストで直接。CPU のみ）:
  taskset -c 0-12 nice -n 10 python3 -B scripts/regen2/g3_merge_crossattn.py --status
  taskset -c 0-12 nice -n 10 python3 -B scripts/regen2/g3_merge_crossattn.py --lang en   （または --lang zh）
"""
import os
import sys

THREADS = 13
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TORCH_HOME", "XDG_CACHE_HOME")
# 起動したときの値（上書きする前。記録に残す）
ENV_AT_LAUNCH = {k: os.environ.get(k) for k in THREAD_VARS + CACHE_VARS}
# スレッド上限（numpy / torch を読み込む前に入れる）
for _k in THREAD_VARS:
    os.environ[_k] = str(THREADS)
# HF とキャッシュの置き場（HF の library を読み込む前に入れる。ここより上で読むのは os と sys だけ）。
# setdefault ではなく上書き: slurm_run.sh は HF_HOME=/tmp/.cache/huggingface と TORCH_HOME=/mnt/.torch_cache を渡す。
_R = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
CACHE_ENV = {"HF_HOME": _R + "/.cache/huggingface", "HF_HUB_CACHE": _R + "/.cache/huggingface/hub",
             "HF_HUB_OFFLINE": "1", "TORCH_HOME": _R + "/.cache/torch", "XDG_CACHE_HOME": _R + "/.cache"}
for _k, _v in CACHE_ENV.items():
    os.environ[_k] = _v
sys.dont_write_bytecode = True  # 既存の scripts/ 以下に .pyc を新しく作らない

import argparse  # noqa: E402
import ast  # noqa: E402
import builtins  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import g3_common as G  # noqa: E402  同じ置き場（scripts/regen2）
import silence_criterion as SC  # noqa: E402  regen2 共通の無音の基準

PROJECT_ROOT = G.PROJECT_ROOT
assert _R == PROJECT_ROOT, (_R, PROJECT_ROOT)  # キャッシュの置き場の根と、記録・判定に使う根が同じ


def P(rel):
    return os.path.join(PROJECT_ROOT, rel)


EVAL_SCRIPT = P(G.EVAL_REL)
INJ_SCRIPT = P(G.INJ_REL)
LEGACY_SPAN_SRC = P(G.LEGACY_SPAN_REL)
SHIM_SCRIPT = P(G.SHIM_REL)
PAIRS_MODULE = {"en": "en_cross_culture_names", "zh": "zh_cross_culture_names"}
CONDS = G.CONDS
N_PAIRS = G.N_PAIRS
FA_SCRIPT = P(G.FA_REL)
PAIR_ERR_RE = re.compile(r"^\s*\[\d+\] err: ")  # 本番 main() の「print(f"  [{i}] err: ...")」
# 本番 forced_alignment_utils.WhisperAligner.transcribe_words の「print(f"  [Whisper] transcribe failed: ...")」
WHISPER_ERR_RE = re.compile(r"^\s*\[Whisper\] transcribe failed: ")


def parse_targets(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lang, idx = tok.split(":")
        lang = lang.strip().lower()
        assert lang in G.PROD_DIR, f"対象外の言語: {lang}（en / zh のみ）"
        out.append((lang, int(idx)))
    assert out, "--targets が空"
    assert len(set(out)) == len(out), f"--targets に重複: {out}"
    return out


def assert_not_production(path):
    rp = os.path.realpath(path)
    for d in G.PROD_DIR.values():
        pd = os.path.realpath(P(d))
        assert rp != pd and not rp.startswith(pd + os.sep), f"本番の置き場には書かない: {path}"
    assert "/crossattn_entity_dtw_" not in rp + "/", f"本番と紛らわしい置き場には書かない: {path}"


def load_pairs(lang):
    """eval_crossattn_entity_dtw.py の main() と同じ組み立て（cross_culture, [:n]）."""
    p = P("data")
    if p not in sys.path:
        sys.path.insert(0, p)
    mod = __import__(PAIRS_MODULE[lang])
    cc = mod.build_300_pairs()[:N_PAIRS]
    pairs = [{"text_a": q["text_a"], "text_b": q["text_b"], "word_a": q["word_a"], "word_b": q["word_b"],
              "pair_idx": q["pair_idx"]} for q in cc]
    assert len(pairs) == N_PAIRS and [q["pair_idx"] for q in pairs] == list(range(N_PAIRS))
    return pairs


def load_prod_rows(lang):
    res = G.read_json(P(os.path.join(G.PROD_DIR[lang], "results.json")))
    return {r["pair_idx"]: r for r in res["rows"]}


# ---------------------------------------------------------------- HF offline（9/28 の前例をそのまま使う）
def load_shim_module():
    """gen_t5gemma_pairs_module.py をファイルから読む（sys.path は変えない。上は標準ライブラリだけ）."""
    spec = importlib.util.spec_from_file_location("_g3_offline_shim", SHIM_SCRIPT)
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    return W


def setup_offline(W, require):
    """HF_HUB_OFFLINE なら、生成の経路で読む 5 つの repo が手元にあることを確かめ、tokenizer の読み替えを入れる."""
    info = {"hf_offline": bool(W.hf_offline()), "offline_preflight_ok": None, "offline_preflight": None,
            "offline_tokenizer_shim": []}
    if not info["hf_offline"]:
        if require:
            raise SystemExit("[g3] HF_HUB_OFFLINE=1 で流す（5 月以前のキャッシュに固定するため）")
        return info
    ok, pre = W.offline_preflight()
    info["offline_preflight_ok"] = bool(ok)
    info["offline_preflight"] = pre
    for r, v in pre.items():
        print("[offline] %-45s %s" % (r, v.get("snapshot") or v.get("error")), flush=True)
    if not ok:
        raise SystemExit("[g3] HF_HUB_OFFLINE=1 だが、生成の経路で読む repo が手元のキャッシュに無い")
    cfg = G.read_json(os.path.join(pre[G.T5GEMMA_TTS_ID]["snapshot"], "config.json"))
    tok_name = cfg.get("text_tokenizer_name") or cfg.get("t5gemma_model_name")
    info["text_tokenizer_name_in_config"] = tok_name
    if tok_name != G.TOKENIZER_ID:
        raise SystemExit(f"[g3] T5Gemma-TTS の config の tokenizer が {tok_name}（{G.TOKENIZER_ID} のはず）")
    W.install_offline_tokenizer_shim(info["offline_tokenizer_shim"])  # 記録はこのリストに追記される
    return info


# ---------------------------------------------------------------- span 事前確認
def load_function_from_source(path, name):
    """ファイルを import せず（副作用なし）、関数定義だけを取り出して exec する."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f"{name} が {path} に 1 つだけあるはず（{len(nodes)}）"
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns[name]


def span_preflight(pairs_by_target):
    """今の find_entity_token_range と 5/20 時点の旧ロジックの範囲が一致するかを調べる.

    tokenizer は本番の main() と同じく repo id で読む（HF_HUB_OFFLINE では上の読み替えが効く。
    読めなければ例外のまま止まる。黙って別の読み方に切り替えない）。
    EN/ZH は lang を渡すので normalize_text_with_lang は文字列を変えない（ja のときだけ正規化）。
    add_bos は範囲の両方に同じオフセットを足すだけなので、一致の判定には影響しない（0 で比べる）。
    """
    from transformers import AutoTokenizer
    cur_fn = load_function_from_source(INJ_SCRIPT, "find_entity_token_range")
    old_fn = load_function_from_source(LEGACY_SPAN_SRC, "find_entity_token_range")
    tok = AutoTokenizer.from_pretrained(G.TOKENIZER_ID)
    out = {}
    for (lang, idx), pair in pairs_by_target.items():
        rec = {"tokenizer_class": type(tok).__name__, "tokenizer_name_or_path": getattr(tok, "name_or_path", None)}
        for side, text, word in (("a", pair["text_a"], pair["word_a"]), ("b", pair["text_b"], pair["word_b"])):
            cur = cur_fn(tok, text, word, add_bos=0)
            old = old_fn(tok, text, word, add_bos=0)
            ids = tok.encode(text, add_special_tokens=False)
            pieces = tok.convert_ids_to_tokens(ids)
            rec[side] = {"text": text, "word": word, "current": list(cur) if cur else None,
                         "legacy_0520": list(old) if old else None,
                         "match": cur is not None and old is not None and tuple(cur) == tuple(old),
                         "tokens_current": pieces[cur[0]:cur[1]] if cur else None,
                         "tokens_legacy_0520": pieces[old[0]:old[1]] if old else None}
        rec["all_match"] = rec["a"]["match"] and rec["b"]["match"]
        out[f"{lang}:{idx}"] = rec
    return out


# ---------------------------------------------------------------- 生成の写し取り
class GenCapture:
    """generate_with_modified_encoder をそのまま呼び、出力の音声を写し取る.

    1 回の main() で扱うのは 1 対だけなので、呼び出しは main() の決まった順
    （run_condition の BL, INJ, BL_mask, INJ_mask、synth_simple の TGT）で 5 回来る。ラベルはこの順で付け、
    中身で次を確かめる（確かめに通らなければ wav は保存せず、capture の失敗として記録する）:
      - target_text が [text_a, text_a, text_a, text_a, text_b]
      - encoder_attention_mask に 0 がある呼び出しが [no, no, yes, yes, no]
      - memory の差（最大絶対差）: INJ は BL と大きく違い、BL_mask は BL と、INJ_mask は INJ とほぼ同じ
        （GPU の非決定性で完全一致しないことがあるので、INJ-BL の差の 1% 以下を「同じ」とみなす）
    generate が None を返したか（生成の失敗）は、写し取りの成否と関係なく必ず記録する。
    T5Gemma の hf_device_map も記録する（層が cpu / disk に置かれていたら環境の問題として run_error）。
    """

    def __init__(self, orig):
        self.orig = orig
        self.sig = inspect.signature(orig)
        self.pair = None
        self.calls = []
        self._mems = []
        self.device_map = None

    def reset(self, pair):
        self.pair = pair
        self.calls = []
        self._mems = []

    def __call__(self, *args, **kwargs):
        out = self.orig(*args, **kwargs)
        entry = {"returned_none": out is None}
        self.calls.append(entry)
        try:
            b = self.sig.bind(*args, **kwargs)
            b.apply_defaults()
            a = b.arguments
            if self.device_map is None:
                dm = getattr(a["model"], "hf_device_map", None)
                self.device_map = {str(k): str(v) for k, v in dm.items()} if isinstance(dm, dict) else {}
            text = a["target_text"]
            role = ("a" if text == self.pair["text_a"] else "b" if text == self.pair["text_b"] else "other") \
                if self.pair else "other"
            attn = a["encoder_attention_mask"]
            audio = None
            if out is not None:
                x = out[0]  # run_condition / synth_simple と同じく先頭のバッチを使う
                audio = x.detach().float().cpu().numpy().reshape(-1)
            entry.update({"text_role": role, "masked": bool((attn == 0).any().item()),
                          "n_masked_cols": int((attn == 0).sum().item()),
                          "enc_len": int(a["modified_memory"].shape[1]), "sr": int(a["codec_audio_sr"]),
                          "audio": audio})
            self._mems.append(a["modified_memory"].detach().float().cpu())
        except Exception as e:  # 写し取りの失敗で本番の処理を止めない
            print(f"[g3] capture failed: {type(e).__name__}: {e}", flush=True)
            entry.update({"text_role": "capture_error", "error": str(e)[:300]})
            if len(self._mems) < len(self.calls):
                self._mems.append(None)
        return out

    def device_offload(self):
        return any(v in ("cpu", "disk") for v in (self.device_map or {}).values())

    def finalize(self):
        """順にラベルを付け、中身の確かめの結果を返す（calls に label を書き込む）."""
        def maxdiff(i, j):
            x, y = self._mems[i], self._mems[j]
            if x is None or y is None or x.shape != y.shape:
                return None
            return float((x - y).abs().max().item())

        n = len(self.calls)
        chk = {"n_calls": n,
               "text_roles": [c.get("text_role") for c in self.calls],
               "masked": [c.get("masked") for c in self.calls],
               "returned_none": [c.get("returned_none") for c in self.calls]}
        ok = n == len(CONDS) and chk["text_roles"] == ["a", "a", "a", "a", "b"] \
            and chk["masked"] == [False, False, True, True, False]
        if n >= 4 and len(self._mems) >= 4:
            d_inj_bl, d_blm_bl, d_injm_inj = maxdiff(1, 0), maxdiff(2, 0), maxdiff(3, 1)
            chk.update({"maxdiff_inj_vs_bl": d_inj_bl, "maxdiff_blmask_vs_bl": d_blm_bl,
                        "maxdiff_injmask_vs_inj": d_injm_inj})
            ok = ok and None not in (d_inj_bl, d_blm_bl, d_injm_inj) and d_inj_bl > 0 \
                and d_blm_bl <= 0.01 * d_inj_bl and d_injm_inj <= 0.01 * d_inj_bl
        else:
            ok = False
        for k, c in enumerate(self.calls):
            c["label"] = CONDS[k] if (ok and k < len(CONDS)) else f"call{k}"
        chk["order_ok"] = bool(ok)
        return chk


class PrintTap:
    """本番の print をそのまま出し、握りつぶされた例外の行だけを記録する.

    pair_errors:    本番 main() の対の処理中の例外（"  [i] err: ..."）
    whisper_errors: forced_alignment_utils の Whisper の例外（"  [Whisper] transcribe failed: ..."）
    """

    def __init__(self):
        self.pair_errors = []
        self.whisper_errors = []

    def reset(self):
        self.pair_errors = []
        self.whisper_errors = []

    def __call__(self, *args, **kwargs):
        builtins.print(*args, **kwargs)
        try:
            s = " ".join(str(x) for x in args)
            if PAIR_ERR_RE.match(s):
                self.pair_errors.append(s.strip()[:400])
            elif WHISPER_ERR_RE.match(s):
                self.whisper_errors.append(s.strip()[:400])
        except Exception:  # noqa: BLE001
            pass


def install_print_taps(M, FA, tap):
    """本番 main() の module（M）と forced_alignment_utils（FA）の print を tap に差し替える.

    どちらの module も print を module の名前で引く（関数の中の print は module の globals を先に見る）ので、
    module に print を置けば出力は変わらずに行を記録できる。FA は本番のファイルそのものでなければならない。
    """
    fa_file = os.path.realpath(FA.__file__)
    assert fa_file == os.path.realpath(FA_SCRIPT), f"forced_alignment_utils が本番のファイルでない: {fa_file}"
    fa_md5 = G.md5_file(fa_file)
    assert fa_md5 == G.PINNED_MD5[G.FA_REL], f"forced_alignment_utils の md5 が固定の値と違う: {fa_md5}"
    M.print = tap
    FA.print = tap
    return {"eval_module_tapped": getattr(M, "print", None) is tap,
            "forced_alignment_utils_tapped": getattr(FA, "print", None) is tap,
            "forced_alignment_utils_file": fa_file, "forced_alignment_utils_md5": fa_md5,
            "whisper_error_regex": WHISPER_ERR_RE.pattern, "pair_error_regex": PAIR_ERR_RE.pattern}


def save_wavs(calls, data_root, wav_dir_rel, idx):
    """確かめに通ったラベルの wav だけを float32 で保存し、無音の判定は保存したファイルから行う."""
    import soundfile as sf
    labels = [c.get("label") for c in calls]
    info = {}
    for c in calls:
        lab = c.get("label")
        if lab not in CONDS or c.get("audio") is None:
            continue
        if labels.count(lab) != 1:
            continue  # 同じラベルが 2 回以上: 取り違えを避けて保存しない
        os.makedirs(os.path.join(data_root, wav_dir_rel), exist_ok=True)
        rel = os.path.join(wav_dir_rel, f"s{idx:04d}_{lab}.wav")
        path = os.path.join(data_root, rel)
        assert not os.path.exists(path), f"既にある wav は上書きしない: {rel}"
        sf.write(path, c["audio"], c["sr"], subtype="FLOAT")  # 評価に使った値をそのまま残す（無損失）
        info[lab] = {"path": rel, "sr": c["sr"], "md5": G.md5_file(path), **SC.assess_file(path)}
    return info


# ---------------------------------------------------------------- 1 対の処理
def write_seed_checkpoint(tdir, idx):
    """新しい置き場を作り、対象以外を placeholder とした checkpoint.json を置く（既にあれば止まる）."""
    os.makedirs(tdir, exist_ok=False)
    rows = [{"pair_idx": i, "placeholder": True} for i in range(N_PAIRS) if i != idx]
    ckpt = os.path.join(tdir, "checkpoint.json")
    with open(ckpt, "x") as f:
        json.dump({"rows": rows}, f)
    return ckpt


def would_run(pairs, ckpt):
    """main() の resume と同じ判定で、生成される対の pair_idx を返す."""
    done = {r["pair_idx"] for r in G.read_json(ckpt).get("rows", [])}
    return [p.get("pair_idx", i) for i, p in enumerate(pairs) if p.get("pair_idx", i) not in done]


def next_step_text(mode, seed, attempt, status, lang):
    if status == G.STATUS_ALIGN_EXC:
        return (f"Whisper の呼び出しが例外で落ちた（環境の問題。whisper_errors を見る）。原因を直してから、同じ seed {seed} を "
                f"--attempt {attempt + 1} で流す（次の seed には進まない。2 回続いたら著者の判断）")
    if status in G.RETRY_SAME_SEED:
        return f"原因を直してから、同じ seed {seed} を --attempt {attempt + 1} で流す（次の seed には進まない）"
    if mode != G.MODE_REGEN:
        return "再現の確認だけ（取り込まない）" if mode == G.MODE_REPRO else "FAKE（試運転）"
    if status == G.STATUS_VALID:
        return f"取り込み: g3_merge_crossattn.py --lang {lang}"
    if seed == 43:
        return "この対だけ seed 44 を流す"
    return "seed 44 でも使えない。45 以上は決まっていないので著者の判断を仰ぐ"


def vendor_md5():
    out = {}
    for name in ("data.tokenizer", "duration_estimator", "inference_tts_utils"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "__file__", None):
            out[name] = {"file": m.__file__, "md5": G.md5_file(m.__file__)}
    return out


def run_one(M, cap, tap, *, lang, idx, pair, prod_row, seed, attempt, data_root, mode, span_rec, env_info,
            policy, on_gpu=True):
    import torch
    rel_dir = G.pair_dir_rel(seed, lang, idx, attempt)
    tdir = os.path.join(data_root, rel_dir)
    assert_not_production(tdir)
    ckpt = write_seed_checkpoint(tdir, idx)
    pairs = load_pairs(lang)
    assert pairs[idx] == pair
    assert would_run(pairs, ckpt) == [idx], "対象の対だけが生成される設定になっていない"

    argv = G.prod_args(lang) + ["--output_dir", tdir]
    print(f"[g3] mode={mode} lang={lang} pair={idx} seed={seed} attempt={attempt} argv={argv}", flush=True)
    print(f"[g3]   text_a={pair['text_a']!r} text_b={pair['text_b']!r}", flush=True)
    M.SEED = seed
    cap.reset(pair)
    tap.reset()
    saved = sys.argv
    sys.argv = [EVAL_SCRIPT] + argv
    t0 = time.time()
    err = None
    try:
        M.main()
    except BaseException as ex:  # 例外でも、写し取った wav と checkpoint と記録は残す
        err = f"{type(ex).__name__}: {ex}"
        print(f"[g3] main() ended with {err}", flush=True)
        traceback.print_exc()
    finally:
        sys.argv = saved
    elapsed = round((time.time() - t0) / 60, 2)

    rows = G.read_json(ckpt).get("rows", [])
    got = [r for r in rows if r.get("pair_idx") == idx and not r.get("placeholder")]
    row = got[0] if got else None
    order_chk = cap.finalize()
    try:
        wavs = save_wavs(cap.calls, data_root, os.path.join(rel_dir, "wav"), idx)
    except Exception as ex:  # noqa: BLE001  保存の失敗は写し取りの失敗（capture_error）として記録を残す
        print(f"[g3] saving wavs failed: {type(ex).__name__}: {ex}", flush=True)
        traceback.print_exc()
        order_chk["save_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        wavs = {}
    any_silent = any(v["silent"] for v in wavs.values())
    any_none = any(c.get("returned_none") for c in cap.calls)
    capture_ok = bool(order_chk["order_ok"] and len(wavs) == len(CONDS))
    status = G.classify(row, err, tap.pair_errors, tap.whisper_errors, any_none, capture_ok, len(wavs), any_silent,
                        cap.device_offload())
    env = {"project_root": PROJECT_ROOT, "host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__,
           "cuda_device": torch.cuda.get_device_name(0) if on_gpu else None,  # ホストの試運転では GPU に触れない
           "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch_num_threads": torch.get_num_threads(),
           "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                                         "VECLIB_MAXIMUM_THREADS")},
           "hf_device_map": cap.device_map, "device_offload": cap.device_offload()}
    for k in CACHE_VARS:
        env[k] = os.environ.get(k)
    env["env_at_launch"] = dict(ENV_AT_LAUNCH)  # スクリプトが上書きする前の値（slurm_run.sh が渡したもの）
    env["env_set_by_script"] = dict(CACHE_ENV)
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        env["transformers"] = None
    env.update(env_info)
    rec = {
        "item": "G3 Appendix B tab:crossattn_mask",
        "producer": G.PRODUCER, "mode": mode,
        "lang": lang, "pair_idx": idx, "seed": seed, "attempt": attempt, "pair_dir": rel_dir,
        "status": status, "valid": status == G.STATUS_VALID,
        "pair_invalid": status in G.PAIR_INVALID, "retry_same_seed": status in G.RETRY_SAME_SEED,
        "merge_eligible": bool(mode == G.MODE_REGEN and seed in G.REGEN_SEEDS and status == G.STATUS_VALID),
        "next_step": next_step_text(mode, seed, attempt, status, lang),
        "valid_criterion": "row aligned (Whisper found the entity in all 5 wavs, with no Whisper exception), "
                           "all 5 frames present, all 5 wavs captured in the checked order, and none silent under "
                           + SC.CRITERION_TEXT,
        "silence_criterion_id": SC.CRITERION_ID, "silence_criterion": SC.CRITERION_TEXT,
        "silence_params": {k: getattr(SC, k) for k in G.SILENCE_PARAMS},
        "seed_policy": policy,
        "frames": G.frames_of(row), "frames_order": list(CONDS),
        "production_dir": G.PROD_DIR[lang],
        "production_frames": G.frames_of(prod_row),
        "production_aligned": prod_row.get("aligned") if prod_row else None,
        "frames_match_production": (G.frames_of(row) == G.frames_of(prod_row)) if (row and prod_row) else None,
        "row": row,  # 本番の results.json の rows と同じ形式
        "wavs": wavs, "any_silent": any_silent, "any_returned_none": any_none,
        "capture_ok": capture_ok, "capture_labels": [c.get("label") for c in cap.calls],
        "capture_checks": order_chk,
        "capture_meta": [{k: v for k, v in c.items() if k != "audio"} for c in cap.calls],
        "pair_loop_errors": list(tap.pair_errors),
        "whisper_errors": list(tap.whisper_errors),
        "span_preflight": span_rec,
        "main_argv": argv, "main_error": err, "elapsed_min": elapsed,
        "code_md5": G.current_code_md5(PROJECT_ROOT),
        "vendor_md5": vendor_md5() if mode != G.MODE_FAKE else None,
        "env": env,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if prod_row and row and prod_row.get("aligned") and row.get("aligned"):
        # 本番で整列していた対を seed 42 で流したとき（再現の確認用）: 数値の最大差
        diffs = [abs(row[k] - prod_row[k]) for k in row
                 if isinstance(row.get(k), (int, float)) and isinstance(prod_row.get(k), (int, float))
                 and not isinstance(row.get(k), bool)]
        rec["max_abs_diff_vs_production"] = max(diffs) if diffs else None
    G.write_json_new(rec, os.path.join(tdir, "regen_row.json"))
    print(f"[g3] RESULT lang={lang} pair={idx} seed={seed} attempt={attempt} mode={mode} status={status} "
          f"frames={rec['frames']} production_frames={rec['production_frames']} "
          f"match={rec['frames_match_production']} capture_ok={capture_ok} silent={any_silent} "
          f"whisper_errors={len(tap.whisper_errors)} pair_loop_errors={len(tap.pair_errors)} "
          f"elapsed={elapsed}min", flush=True)
    print(f"[g3] NEXT: {rec['next_step']}", flush=True)
    return rec


# ---------------------------------------------------------------- seed の決まり
def seed_policy(lang, idx, seed, attempt, data_root, code_root):
    """この (lang, idx, seed, attempt) を流してよいか。errs が空なら流してよい."""
    errs = []
    ev = {"seed": seed, "attempt": attempt}
    if seed not in G.ALL_SEEDS:
        return [f"seed {seed} は決まりの外（42 = 再現の確認、43、44 だけ）"], ev
    rel_dir = G.pair_dir_rel(seed, lang, idx, attempt)
    if os.path.exists(os.path.join(data_root, rel_dir)):
        errs.append(f"{rel_dir} は既にある（消さない。やり直すなら --attempt {attempt + 1}）")
    att, aerrs = G.attempts_of(data_root, seed, lang, idx)
    errs += aerrs
    if attempt == 1:
        if att:
            errs.append(f"seed {seed} の回が既にある: {[r for _, r in att]}")
    else:
        prev, p_prev, e_prev, chain = G.final_record(data_root, seed, lang, idx)
        errs += e_prev
        ev["previous_attempts"] = chain
        last_status = chain[-1]["status"] if chain else None
        if not att or att[-1][0] != attempt - 1:
            errs.append(f"前の回（attempt {attempt - 1}）が最後の回になっていない: {[k for k, _ in att]}")
        elif last_status not in G.RETRYABLE:
            errs.append(f"前の回の状態は {last_status}。同じ seed のやり直しは {list(G.RETRYABLE)} のときだけ")
        elif prev is not None and prev.get("status") != last_status:
            errs.append("前の回の記録が読めない")
        elif G.repeated_alignment_exception(chain):
            errs.append(f"seed {seed} で Whisper の例外（alignment_exception）が 2 回続いた。決まった例外がくり返している"
                        "恐れがあるので流さない（whisper_errors を見て著者の判断を仰ぐ）")
    if seed == 42:
        _, pev = G.seed43_allowed(code_root, lang, idx)  # 再現の確認はどの対でもよい（証拠だけ残す）
        ev.update(pev)
        ev["purpose"] = "reproduction check only; never merged"
    elif seed == 43:
        e, pev = G.seed43_allowed(code_root, lang, idx)
        errs += e
        ev.update(pev)
    else:
        e, pev = G.seed44_allowed(data_root, code_root, lang, idx)
        errs += e
        ev.update(pev)
    return errs, ev


# ---------------------------------------------------------------- 試運転（CPU, モデルなし）
def static_patch_checks():
    """本番スクリプトに、差し替えに使う箇所があるかを ast で確かめる（import しない）."""
    src = open(EVAL_SCRIPT, encoding="utf-8").read()
    tree = ast.parse(src)
    seed_assign = any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SEED" for t in n.targets)
                      for n in tree.body)
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    lazy_import = any(isinstance(n, ast.ImportFrom) and n.module == "encoder_injection_t5gemma"
                      and any(a.name == "generate_with_modified_encoder" for a in n.names)
                      for n in ast.walk(main))
    top_import = any(isinstance(n, ast.ImportFrom) and n.module == "encoder_injection_t5gemma" for n in tree.body)
    seed_uses = sum(1 for n in ast.walk(main) if isinstance(n, ast.Name) and n.id == "SEED")
    resume = "if pair_idx in done: continue" in src
    err_print = 'print(f"  [{i}] err: {str(e)[:200]}"); continue' in src
    print_shadowed = any(isinstance(n, (ast.FunctionDef, ast.Assign)) and (
        getattr(n, "name", None) == "print" or any(getattr(t, "id", None) == "print" for t in getattr(n, "targets", [])))
        for n in tree.body)
    tok_by_id = "t5_tok = AutoTokenizer.from_pretrained(tok_name)" in src
    fa_import = "from forced_alignment_utils import WhisperAligner, find_entity_chunks, slice_entity_audio" in src
    inj_tree = ast.parse(open(INJ_SCRIPT, encoding="utf-8").read())
    gen = [n for n in inj_tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_with_modified_encoder"]
    gen_params = [a.arg for a in gen[0].args.args] if gen else []
    need = ["model", "modified_memory", "encoder_attention_mask", "target_text", "codec_audio_sr"]
    # forced_alignment_utils: Whisper の例外を握りつぶす print がそのままあり、print が module で隠されていない
    fa_src = open(FA_SCRIPT, encoding="utf-8").read()
    fa_tree = ast.parse(fa_src)
    fa_print_lines = [ln.strip() for ln in fa_src.splitlines() if "print(" in ln]
    fa_print_shadowed = any(isinstance(n, (ast.FunctionDef, ast.Assign, ast.Import, ast.ImportFrom)) and (
        getattr(n, "name", None) == "print"
        or any(getattr(t, "id", None) == "print" for t in getattr(n, "targets", []))
        or any((a.asname or a.name) == "print" for a in getattr(n, "names", [])))
        for n in fa_tree.body)
    fa_cls = [n for n in fa_tree.body if isinstance(n, ast.ClassDef) and n.name == "WhisperAligner"]
    tw = [n for n in (fa_cls[0].body if fa_cls else []) if isinstance(n, ast.FunctionDef) and n.name == "transcribe_words"]
    tw_src = ast.get_source_segment(fa_src, tw[0]) if tw else ""
    pinned_errs = G.pinned_file_errors(PROJECT_ROOT)
    res = {"eval_script_md5": G.md5_file(EVAL_SCRIPT), "eval_script_md5_is_0520": G.md5_file(EVAL_SCRIPT) == G.EXPECTED_EVAL_MD5,
           "pinned_md5": G.PINNED_MD5, "pinned_md5_errors": pinned_errs,
           "module_SEED_assign": seed_assign, "main_lazy_imports_generate": lazy_import,
           "no_top_level_import_of_generate": not top_import, "SEED_uses_in_main": seed_uses,
           "resume_skips_done_pairs": resume, "pair_loop_err_print_present": err_print,
           "module_print_not_shadowed": not print_shadowed, "tokenizer_loaded_by_repo_id_in_main": tok_by_id,
           "eval_imports_forced_alignment_utils": fa_import,
           "fa_whisper_fail_print_present": G.FA_WHISPER_FAIL_PRINT in tw_src,
           "fa_print_lines": fa_print_lines,
           "fa_only_print_is_whisper_fail": fa_print_lines == [G.FA_WHISPER_FAIL_PRINT],
           "fa_module_print_not_shadowed": not fa_print_shadowed,
           "fa_whisper_fail_matches_tap_regex": bool(WHISPER_ERR_RE.match("  " + G.WHISPER_FAIL_PREFIX + "x")),
           "generate_params": gen_params, "generate_params_ok": all(p in gen_params for p in need)}
    assert (res["eval_script_md5_is_0520"] and not pinned_errs and seed_assign and lazy_import and not top_import
            and seed_uses >= 4 and resume and err_print and not print_shadowed and res["generate_params_ok"]
            and fa_import and res["fa_whisper_fail_print_present"] and res["fa_only_print_is_whisper_fail"]
            and not fa_print_shadowed and res["fa_whisper_fail_matches_tap_regex"]), res
    return res


def _fake_generate_factory(np, torch, amp_fn, none_for=None):
    @torch.no_grad()  # 本物と同じ decorator（signature が引数名で bind できることも確かめる）
    def fake_generate(model, text_tokenizer, audio_tokenizer, modified_memory, encoder_attention_mask,
                      encoder_position_ids, target_text, device, codec_audio_sr, codec_sr):
        if none_for is not None and none_for(target_text, encoder_attention_mask):
            return None
        n = int(0.6 * codec_audio_sr)
        t = torch.arange(n, dtype=torch.float32) / codec_audio_sr
        return (amp_fn(target_text) * torch.sin(2 * np.pi * 220 * t)).view(1, 1, -1)
    return fake_generate


def capture_selftest(data_root, wav_rel):
    """偽の生成関数で、ラベル付け・保存・無音判定・None の記録が意図どおりかを確かめる."""
    import numpy as np
    import torch
    import soundfile as sf

    fake_generate = _fake_generate_factory(np, torch, lambda s: 0.0 if s.endswith("SILENT.") else 0.3)
    pair = {"text_a": "A x b.", "text_b": "A y b.", "word_a": "x", "word_b": "y"}
    src = torch.randn(1, 6, 4)
    inj = src.clone()
    inj[0, 1:2] = torch.randn(1, 4)
    ones = torch.ones(1, 6, dtype=torch.long)
    msk = ones.clone()
    msk[0, 1:2] = 0
    sr = 16000
    jitter = 1e-6  # GPU の非決定性に見立てた小さな揺れ（1% 以下なら同じとみなす）

    def feed(cap, seq):
        for mem, attn, text in seq:
            cap(None, None, None, mem, attn, None, text, "cpu", sr, 50)

    good_seq = ((src, ones, pair["text_a"]), (inj, ones, pair["text_a"]), (src + jitter, msk, pair["text_a"]),
                (inj - jitter, msk, pair["text_a"]), (torch.randn(1, 6, 4), ones, pair["text_b"]))
    cap = GenCapture(fake_generate)
    cap.reset(pair)
    feed(cap, good_seq)
    chk = cap.finalize()
    labels = [c["label"] for c in cap.calls]
    assert chk["order_ok"] and labels == list(CONDS) and cap.device_map == {} and not cap.device_offload(), (labels, chk)
    # 崩れた順（INJ と BL_mask を入れ替え）は検出して保存しないこと
    bad = GenCapture(fake_generate)
    bad.reset(pair)
    feed(bad, ((src, ones, pair["text_a"]), (src, msk, pair["text_a"]), (inj, ones, pair["text_a"]),
               (inj, msk, pair["text_a"]), (torch.randn(1, 6, 4), ones, pair["text_b"])))
    bchk = bad.finalize()
    assert not bchk["order_ok"] and not save_wavs(bad.calls, data_root, os.path.join(wav_rel, "bad_order"), 0), bchk
    info = save_wavs(cap.calls, data_root, wav_rel, 0)
    assert sorted(info) == sorted(CONDS) and not any(v["silent"] for v in info.values()), info
    x, fs = sf.read(os.path.join(data_root, info["bl"]["path"]), dtype="float32")
    assert fs == sr and np.array_equal(x, cap.calls[0]["audio"].astype(np.float32)), "保存した wav が元と一致しない"
    # None を返した呼び出しは、写し取りの成否と関係なく記録される
    none_gen = _fake_generate_factory(np, torch, lambda s: 0.3, none_for=lambda s, a: bool((a == 0).any().item()))
    ncap = GenCapture(none_gen)
    ncap.reset(pair)
    feed(ncap, good_seq)
    nchk = ncap.finalize()
    assert nchk["returned_none"] == [False, False, True, True, False], nchk
    # 無音の wav は共通の基準で silent になる
    sil = GenCapture(_fake_generate_factory(np, torch, lambda s: 0.0))
    sil.reset(pair)
    feed(sil, good_seq)
    sil.finalize()
    sinfo = save_wavs(sil.calls, data_root, os.path.join(wav_rel, "silent"), 0)
    assert sorted(sinfo) == sorted(CONDS) and all(v["silent"] for v in sinfo.values()), sinfo
    # hf_device_map に cpu があれば device_offload
    class _M:  # noqa: N801
        hf_device_map = {"encoder": 0, "decoder.layers.30": "cpu"}
    ocap = GenCapture(fake_generate)
    ocap.reset(pair)
    ocap(_M(), None, None, src, ones, None, pair["text_a"], "cpu", sr, 50)
    assert ocap.device_offload(), ocap.device_map
    return {"labels": labels, "checks": chk, "bad_order_detected": True, "wav_lossless_roundtrip": True,
            "returned_none_recorded": nchk["returned_none"], "silent_detected_by_shared_criterion": True,
            "device_offload_detected": True, "saved": {k: v["path"] for k, v in info.items()}}


def classify_selftest():
    row_ok = {"aligned": True, **{f"frames_{c}": 10 for c in CONDS}}
    row_un = {"aligned": False, **{f"frames_{c}": 10 for c in CONDS}}
    wf = ["[Whisper] transcribe failed: CUDA out of memory"]
    cases = [
        # (row, main_error, pair_loop_errors, whisper_errors, any_none, capture_ok, n_wavs, any_silent, offload)
        ((row_ok, None, [], [], False, True, 5, False, False), "valid"),
        ((row_ok, None, [], [], False, True, 5, True, False), "silent"),
        ((row_un, None, [], [], False, True, 5, False, False), "unaligned"),
        ((row_un, None, [], [], False, False, 3, False, False), "unaligned"),  # 整列の失敗は写し取りと関係ない
        # Whisper の例外で aligned=False になった行は unaligned（PAIR_INVALID）にしない
        ((row_un, None, [], wf, False, True, 5, False, False), "alignment_exception"),
        ((row_un, None, [], wf, False, True, 5, True, False), "alignment_exception"),
        ((row_ok, None, [], wf, False, True, 5, False, False), "alignment_exception"),
        ((row_un, "RuntimeError: x", [], wf, False, True, 5, False, False), "run_error"),
        ((None, None, [], [], True, True, 4, False, False), "generation_failed"),
        ((None, None, ["[237] err: CUDA out of memory"], [], False, False, 2, False, False), "run_error"),
        ((None, None, [], [], False, False, 0, False, False), "no_row_unexplained"),
        ((row_ok, None, [], [], False, False, 5, False, False), "capture_error"),
        ((row_ok, None, [], [], False, True, 4, False, False), "capture_error"),
        ((row_ok, "RuntimeError: x", [], [], False, True, 5, False, False), "run_error"),
        ((row_ok, None, [], [], False, True, 5, False, True), "run_error"),
    ]
    out = []
    for args, want in cases:
        got = G.classify(*args)
        assert got == want, (args, got, want)
        assert (got in G.PAIR_INVALID) is (want in ("unaligned", "generation_failed", "silent")), got
        out.append(want)
    assert G.STATUS_ALIGN_EXC in G.RETRY_SAME_SEED and G.STATUS_ALIGN_EXC not in G.PAIR_INVALID
    chains = {"ae": [{"status": "alignment_exception"}],
              "ae,ae": [{"status": "alignment_exception"}, {"status": "alignment_exception"}],
              "ae,run_error": [{"status": "alignment_exception"}, {"status": "run_error"}],
              "ae,run_error,ae": [{"status": "alignment_exception"}, {"status": "run_error"},
                                  {"status": "alignment_exception"}]}
    rep = {k: G.repeated_alignment_exception(v) for k, v in chains.items()}
    assert rep == {"ae": False, "ae,ae": True, "ae,run_error": False, "ae,run_error,ae": False}, rep
    return {"n_cases": len(cases), "statuses": out, "repeated_alignment_exception": rep}


def plumbing_selftest(seed, run_root):
    """偽の main()（本番と同じ resume 判定と呼び出し順）で run_one → regen_row.json を通す.

    行の数値は本番の隣の対（idx-1）を写しただけの偽物。出力は今回の試運転の置き場（FAKE）だけ。
    偽の記録は、取り込みの出どころの確かめ（g3_common.provenance_errors）で必ず拒まれることも確かめる。
    """
    import copy
    import numpy as np
    import torch

    lang, idx = "en", 237
    pair = load_pairs(lang)[idx]
    prod_rows = load_prod_rows(lang)
    donor = prod_rows[idx - 1]
    assert donor.get("aligned") is True
    assert prod_rows[idx].get("aligned") is False
    # 本番の forced_alignment_utils.py をそのまま（別名で）読み、Whisper の部分だけ偽物にする
    spec = importlib.util.spec_from_file_location("_g3_fa_selftest", FA_SCRIPT)
    FA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FA)

    def fake_aligner(pipe):
        al = object.__new__(FA.WhisperAligner)  # __init__（Whisper の読み込み）は通さない
        al.target_sr = 16000
        al.pipe = pipe
        return al

    def pipe_raises(*a, **k):
        raise RuntimeError("FAKE CUDA out of memory (selftest)")

    def pipe_no_entity(*a, **k):
        return {"chunks": [{"text": " something", "timestamp": (0.0, 0.4)}, {"text": " else", "timestamp": (0.4, 0.8)}]}

    results = {}
    for case in ("valid", "generation_failed", "run_error", "unaligned", "alignment_exception"):
        data_root = os.path.join(run_root, f"plumbing_FAKE_{case}")
        assert "/_regen2_dryrun/g3_crossattn/" in data_root
        none_for = (lambda s, a: bool((a == 0).any().item())) if case == "generation_failed" else None
        cap = GenCapture(_fake_generate_factory(np, torch, lambda s: 0.3, none_for=none_for))
        tap = PrintTap()
        whisper_calls = []

        class FakeM:
            SEED = 42

            @staticmethod
            def main():
                argv = sys.argv
                out = argv[argv.index("--output_dir") + 1]
                ck = os.path.join(out, "checkpoint.json")
                rows = G.read_json(ck)["rows"]
                done = {r["pair_idx"] for r in rows}
                todo = [p["pair_idx"] for p in load_pairs(lang) if p["pair_idx"] not in done]
                assert todo == [idx], todo
                src = torch.randn(1, 11, 8)
                inj = src.clone()
                inj[0, 0] += 1.0
                ones = torch.ones(1, 11, dtype=torch.long)
                msk = ones.clone()
                msk[0, 0] = 0
                outs = []
                for mem, attn, text in ((src, ones, pair["text_a"]), (inj, ones, pair["text_a"]),
                                        (src.clone(), msk, pair["text_a"]), (inj.clone(), msk, pair["text_a"]),
                                        (torch.randn(1, 13, 8), ones, pair["text_b"])):
                    outs.append(cap(None, None, None, mem, attn, None, text, "cpu", 16000, 50))
                if case == "run_error":
                    tap(f"  [{idx}] err: FAKE whisper crash")  # 本番 main() の握りつぶし
                elif case in ("unaligned", "alignment_exception"):
                    # 本番の entity_three_features と同じく、本物の transcribe_words を呼ぶ（Whisper の pipe だけ偽物）
                    al = fake_aligner(pipe_raises if case == "alignment_exception" else pipe_no_entity)
                    chunks, a_np = al.transcribe_words(np.full(16000, 0.1, dtype=np.float32), 16000, language=lang)
                    whisper_calls.append({"n_chunks": len(chunks), "audio_none": a_np is None})
                    row = copy.deepcopy(prod_rows[idx])  # 本番の aligned=False の行と同じ形
                    assert row.get("aligned") is False
                    rows.append(row)
                elif not any(o is None for o in outs):
                    row = copy.deepcopy(donor)
                    row.update({k: pair[k] for k in ("pair_idx", "text_a", "text_b", "word_a", "word_b")})
                    rows.append(row)
                with open(ck, "w") as f:  # 本番 main() と同じく checkpoint を書き直す（今回作った置き場の中だけ）
                    json.dump({"rows": rows}, f)

        taps = install_print_taps(FakeM, FA, tap)  # GPU の main() と同じ差し替え（FA は本番のファイル）
        assert taps["eval_module_tapped"] and taps["forced_alignment_utils_tapped"], taps
        rec = run_one(FakeM, cap, tap, lang=lang, idx=idx, pair=pair, prod_row=prod_rows.get(idx), seed=seed,
                      attempt=1, data_root=data_root, mode=G.MODE_FAKE, span_rec={"FAKE": True},
                      env_info={"offline_tokenizer_shim": [], "print_taps": taps}, policy={"FAKE": True},
                      on_gpu=False)
        assert FakeM.SEED == seed and rec["status"] == case, (case, rec["status"])
        rel_path = os.path.join(rec["pair_dir"], "regen_row.json")
        perrs = G.provenance_errors(rec, rel_path, data_root, PROJECT_ROOT, lang, idx, seed, G.MODE_REGEN)
        must = ("mode が", "コンテナ", "GPU の記録", "span_preflight", "main_argv")
        assert all(any(m in e for e in perrs) for m in must), perrs
        extra = {}
        if case == "alignment_exception":
            # 本物の transcribe_words の握りつぶしの print が、tap で whisper_errors に入った
            assert whisper_calls == [{"n_chunks": 0, "audio_none": True}], whisper_calls
            assert len(rec["whisper_errors"]) == 1 and rec["whisper_errors"][0].startswith(G.WHISPER_FAIL_PREFIX), rec
            assert rec["status"] not in G.PAIR_INVALID and rec["pair_invalid"] is False and rec["retry_same_seed"]
            assert any("Whisper の例外" in e for e in perrs), perrs
            # tap が無ければ（whisper_errors が空なら）同じ行は unaligned（次の seed へ）と数えられていた
            without_tap = G.classify(rec["row"], None, [], [], False, rec["capture_ok"], len(rec["wavs"]),
                                     rec["any_silent"], False)
            assert without_tap == "unaligned", without_tap
            extra = {"whisper_errors": rec["whisper_errors"], "status_without_fa_tap": without_tap}
        elif case == "unaligned":
            assert whisper_calls == [{"n_chunks": 2, "audio_none": False}] and rec["whisper_errors"] == [], rec
            assert rec["pair_invalid"] is True
            extra = {"whisper_errors": rec["whisper_errors"]}
        else:
            assert rec["whisper_errors"] == [], rec["whisper_errors"]
        results[case] = {"regen_row_json": os.path.relpath(os.path.join(data_root, rel_path), PROJECT_ROOT),
                         "status": rec["status"], "capture_ok": rec["capture_ok"], "wavs": sorted(rec["wavs"]),
                         "pair_loop_errors": rec["pair_loop_errors"], "next_step": rec["next_step"],
                         "seed_override_applied": FakeM.SEED == seed,
                         "provenance_refuses_fake": True, "n_provenance_errors": len(perrs), **extra}
    return results


def offline_dry(W):
    """HF_HUB_OFFLINE の下で、読み替えなしの repo id 読みが落ちること・読み替えありで読めることを確かめる."""
    rep = {"hf_offline": bool(W.hf_offline())}
    if not rep["hf_offline"]:
        rep["note"] = "HF_HUB_OFFLINE が立っていないので、落ちる経路は確かめていない"
        rep.update(setup_offline(W, require=False))
        return rep
    from transformers import AutoTokenizer
    import transformers
    rep["transformers"] = transformers.__version__
    try:
        AutoTokenizer.from_pretrained(G.TOKENIZER_ID)
        rep["without_shim_by_repo_id"] = "loaded (no crash on this host)"
    except Exception as e:  # noqa: BLE001
        rep["without_shim_by_repo_id"] = f"{type(e).__name__}: {str(e)[:200]}"
    info = setup_offline(W, require=True)
    tok = AutoTokenizer.from_pretrained(G.TOKENIZER_ID)
    rep.update(info)
    rep["with_shim_by_repo_id"] = {"class": type(tok).__name__, "name_or_path": tok.name_or_path,
                                   "vocab_size": len(tok)}
    assert info["offline_tokenizer_shim"] and info["offline_tokenizer_shim"][-1].get("snapshot"), info
    return rep


def dry_run(targets, seed, attempt):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_rel = os.path.join(G.DRYRUN_ROOT_REL, f"run_{ts}_seed{seed}")
    run_root = P(run_rel)
    os.makedirs(run_root, exist_ok=False)  # 毎回新しい置き場（前の試運転は消さない）
    report = {"seed": seed, "attempt": attempt, "targets": [f"{l}:{i}" for l, i in targets],
              "date": time.strftime("%Y-%m-%d %H:%M:%S"), "run_dir": run_rel,
              "hf_env": {k: os.environ.get(k) for k in CACHE_VARS},
              "env_at_launch": dict(ENV_AT_LAUNCH), "env_set_by_script": dict(CACHE_ENV)}
    assert report["hf_env"] == CACHE_ENV, report["hf_env"]
    report["static_patch_checks"] = static_patch_checks()
    report["gpu_entry_refusals"] = refusal_selftest()  # torch を読む前に流す
    report["silence_criterion"] = {"id": SC.CRITERION_ID, "text": SC.CRITERION_TEXT,
                                   "params_errors": G.silence_param_errors(), "selftest": SC.selftest()}
    assert not report["silence_criterion"]["params_errors"], report["silence_criterion"]
    report["classify_selftest"] = classify_selftest()
    # seed の決まり（本物の置き場を読むだけ）
    report["seed_policy"] = {}
    for lang, idx in targets:
        errs, ev = seed_policy(lang, idx, seed, attempt, PROJECT_ROOT, PROJECT_ROOT)
        report["seed_policy"][f"{lang}:{idx}"] = {"allowed": not errs, "errors": errs, "evidence": ev}
    report["next_action_now"] = {f"{l}:{i}": G.next_action(PROJECT_ROOT, PROJECT_ROOT, l, i) for l, i in targets}
    # HF offline と span（本番 main() と同じ repo id の読み方で）
    W = load_shim_module()
    report["offline"] = offline_dry(W)
    pairs_by_target = {(l, i): load_pairs(l)[i] for l, i in targets}
    report["span_preflight"] = span_preflight(pairs_by_target)
    report["span_all_match"] = all(v["all_match"] for v in report["span_preflight"].values())
    per = {}
    prod = {(l, i): load_prod_rows(l).get(i) for l, i in targets}
    for lang, idx in targets:
        tdir = os.path.join(run_root, "resume_check", G.pair_dir_rel(seed, lang, idx, attempt))
        ck = write_seed_checkpoint(tdir, idx)
        runs = would_run(load_pairs(lang), ck)
        pr = prod[(lang, idx)]
        per[f"{lang}:{idx}"] = {"pair": pairs_by_target[(lang, idx)], "would_run": runs,
                                "only_target_runs": runs == [idx],
                                "main_argv_in_container": G.prod_args(lang) + [
                                    "--output_dir", f"{G.CONTAINER_ROOT}/{G.pair_dir_rel(seed, lang, idx, attempt)}"],
                                "production_row_aligned": pr.get("aligned") if pr else None,
                                "production_frames": G.frames_of(pr)}
        assert runs == [idx]
    report["per_target"] = per
    report["capture_selftest"] = capture_selftest(PROJECT_ROOT, os.path.join(run_rel, "capture_selftest_wav"))
    report["plumbing_selftest_FAKE"] = plumbing_selftest(seed, run_root)
    out = os.path.join(run_root, "dryrun_report.json")
    G.write_json_new(report, out)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[g3] dry run OK (span_all_match={report['span_all_match']}); report: {out}")


# ---------------------------------------------------------------- main
def gpu_guards(targets):
    """GPU の処理の前の確かめ（torch を読む前・何も書く前）。外れていれば SystemExit."""
    if len(targets) != 1:
        raise SystemExit("[g3] GPU では 1 回の起動で 1 対だけ（EN と ZH は別のプロセス・別のジョブで流す）")
    if PROJECT_ROOT != G.CONTAINER_ROOT:
        raise SystemExit("[g3] GPU の処理は slurm_run.sh のコンテナの中（/mnt）で流す")
    if os.environ.get("HF_HOME") != G.CONTAINER_HF_HOME:
        raise SystemExit(f"[g3] HF_HOME={os.environ.get('HF_HOME')!r}（{G.CONTAINER_HF_HOME} で流す）")
    pe = G.pinned_file_errors(PROJECT_ROOT)
    if pe:
        raise SystemExit("[g3] 本番のファイルが固定の md5 と違うので流さない:\n  - " + "\n  - ".join(pe))
    se = G.silence_param_errors()
    if se:
        raise SystemExit("[g3] 共通の無音の基準の数値が決まりと違うので流さない:\n  - " + "\n  - ".join(se))
    return targets[0]


def refusal_selftest():
    """GPU の処理の入口の拒否を、ホストで GPU に触れずに確かめる（main をそのまま呼ぶ）.

    ホスト（/mnt/scripts が無い）でだけ流す。コンテナの中で流すと、1 対・seed 43 の場合は本当に GPU の処理に進むので。
    """
    if PROJECT_ROOT == G.CONTAINER_ROOT:
        return {"skipped": "コンテナの中では流さない"}
    out = {}
    cases = {
        "gpu_mode_two_targets": ["--targets", "en:237,zh:42", "--seed", "43"],
        "gpu_mode_on_host": ["--targets", "en:237", "--seed", "43"],
        "seed45_rejected_by_argparse": ["--targets", "en:237", "--seed", "45"],
        "seed45_rejected_even_in_dry_run": ["--targets", "en:237", "--seed", "45", "--dry_run"],
    }
    for name, argv in cases.items():
        mods_before = set(sys.modules)
        try:
            main(argv)
        except SystemExit as e:
            out[name] = {"argv": argv, "refused": True, "exit": str(e.code)[:200]}
        else:
            raise AssertionError(f"{name}: 拒否されるはずが通った")
        new = set(sys.modules) - mods_before
        assert "torch" not in new and "torch.cuda" not in new, f"{name}: torch を新しく読んだ"
        out[name]["torch_loaded_before"] = "torch" in mods_before
    assert "1 対だけ" in out["gpu_mode_two_targets"]["exit"], out
    assert "コンテナ" in out["gpu_mode_on_host"]["exit"], out
    assert out["seed45_rejected_by_argparse"]["exit"] == "2", out
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True,
                    help="lang:pair_idx。GPU では 1 つだけ（EN と ZH は別のプロセス・別のジョブ）。試運転は複数可")
    ap.add_argument("--seed", type=int, required=True, choices=list(G.ALL_SEEDS))
    ap.add_argument("--attempt", type=int, default=1,
                    help="同じ seed のやり直しの回（前の回が run_error / alignment_exception / capture_error / "
                         "no_row_unexplained のときだけ）")
    ap.add_argument("--dry_run", action="store_true", help="CPU の試運転（モデルは読まない）")
    args = ap.parse_args(argv)
    targets = parse_targets(args.targets)
    assert args.attempt >= 1

    if args.dry_run:
        dry_run(targets, args.seed, args.attempt)
        return

    lang, idx = gpu_guards(targets)
    import torch
    assert torch.cuda.is_available(), "GPU で実行する（slurm_run.sh 経由）"
    torch.set_num_threads(THREADS)

    errs, policy = seed_policy(lang, idx, args.seed, args.attempt, PROJECT_ROOT, PROJECT_ROOT)
    print("[g3] seed policy:", json.dumps({"errors": errs, "evidence": policy}, ensure_ascii=False), flush=True)
    if errs:
        raise SystemExit("[g3] seed の決まりに合わないので流さない:\n  - " + "\n  - ".join(errs))
    mode = G.MODE_REPRO if args.seed == G.REPRO_SEED else G.MODE_REGEN

    # HF offline: 5 つの repo の確認と tokenizer の読み替え（事前確認と main() の両方に効く）
    W = load_shim_module()
    env_info = setup_offline(W, require=True)

    pair = load_pairs(lang)[idx]
    span = span_preflight({(lang, idx): pair})
    print("[g3] span preflight:", json.dumps(span, ensure_ascii=False), flush=True)
    if not span[f"{lang}:{idx}"]["all_match"]:
        raise SystemExit(f"[g3] 実体語の範囲が 5/20 の旧ロジックと一致しない: {lang}:{idx}。著者の判断を仰ぐ。")

    # 本番スクリプトを module として読む（本番と同じ sys.path の並びがここで入る）。
    # 本番はスクリプトの置き場（scripts/projection）が sys.path[0]。このラッパーの置き場（scripts/regen2、
    # 他の作業のファイルも置かれる）は外し、名前の衝突で本番と違うモジュールを読まないようにする。
    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.getcwd()) != here]
    sys.path.insert(0, os.path.dirname(EVAL_SCRIPT))
    import eval_crossattn_entity_dtw as M
    import encoder_injection_t5gemma as EI  # main() が呼び出し時に from-import するので、先に差し替える
    assert os.path.realpath(M.__file__) == os.path.realpath(EVAL_SCRIPT), M.__file__
    assert os.path.realpath(EI.__file__) == os.path.realpath(INJ_SCRIPT), EI.__file__
    vendor = os.path.realpath(M.T5GEMMA_ROOT)
    for name in ("data.tokenizer", "duration_estimator", "inference_tts_utils"):
        f = os.path.realpath(sys.modules[name].__file__)
        assert f.startswith(vendor + os.sep), f"{name} が vendor 以外から読まれた: {f}"
    FA = sys.modules["forced_alignment_utils"]  # 本番スクリプトが import した整列の module
    assert M.WhisperAligner is FA.WhisperAligner, "本番 main() の WhisperAligner が forced_alignment_utils のものでない"
    print(f"[g3] imports: M={M.__file__} EI={EI.__file__} FA={FA.__file__} vendor={vendor}", flush=True)
    print(f"[g3] module SEED {M.SEED} -> {args.seed}", flush=True)
    cap = GenCapture(EI.generate_with_modified_encoder)
    EI.generate_with_modified_encoder = cap
    tap = PrintTap()
    # main() と forced_alignment_utils の print を tap に（出力は変わらない。握りつぶされた例外の行を記録する）
    env_info["print_taps"] = install_print_taps(M, FA, tap)
    print(f"[g3] print taps: {env_info['print_taps']}", flush=True)

    run_one(M, cap, tap, lang=lang, idx=idx, pair=pair, prod_row=load_prod_rows(lang).get(idx), seed=args.seed,
            attempt=args.attempt, data_root=PROJECT_ROOT, mode=mode, span_rec=span, env_info=env_info,
            policy=policy, on_gpu=True)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
