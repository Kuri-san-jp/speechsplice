#!/usr/bin/env python3
"""付録 G の G2P・ROME/MEMIT・SpeechSplice を、Baseline / Text Hint と同じ声まね条件で作る。
無効な単位（Baseline / Text Hint の既存分を含む）は seed 43 → 44 で単位ごと作り直す（2026-09-29 作成）.

共通の定義は appg_vc_common.py（声まね条件・単位・置き場・判定）。本番のスクリプトは 1 バイトも変えず、import して使う:
  声まね合成   scripts/projection/eval_utmos_speaker_fixed.voice_clone_synth
               （seed_everything(SEED) → estimate_duration（参照の話速）→ inference_one_sample。復号の定数は関数の中）
               seed は E.SEED を置き換える（scripts/spkvar/regen_pairs_seed.py と同じやり方）。
               見本の文は '太平洋は世界で一番大きな太陽です' をそのまま渡す（空だと Whisper で書き起こし直すので渡す）。
  SpeechSplice E.get_encoder_output_with_prefix（見本の文 + x_sep + 文 + EOS を符号化）、E.find_entity_token_range、
               scripts/projection/splice_clone.prepare_clone_splice（系列ごと差し替え、長さが違えば長さのそろった
               ダミー文、長さの推定は text_a）→ voice_clone_synth の hook で注入。E.main() の 1 対ぶんの手順（L416-565）と
               同じ順。E.main() そのものは呼ばない（UTMOS / SQUIM を読んで測る余分な処理があり、25 対ごとにしか記録しないため）。
               JA の 300 対では語が文にそのまま入っているので、main の _resolve_surface は語をそのまま返す（空運転で全対を確認）。
  G2P          scripts/projection/eval_g2p_baseline_jacc_n300.g2p_katakana（pykakasi）で text_a の word_a を置き換えた文。
               5 月の g2p_transcripts.json の modified_text と 300 対すべて一致することを確かめてから作る。
  ROME/MEMIT   scripts/projection/eval_rome_jacc_n300_text_a.apply_rome_edit_jacc（層 2,5,8、memit、eps=1.0）。
               編集の入力（30 組、text_ref = 最初の text_b、text_test = 最初の text_a、共分散の文 600 本の先頭 200）は
               同じスクリプトの main() L120-136 と同じ作り方。編集は変えず、合成だけを声まねにする（text_a を合成）。
  Text Hint など（textedit、作り直しのときだけ）  既定（--textedit_unit b7、著者の決定 2026-09-29）は
               b7 = text_a.replace(word_a, 'word_a（word_b）', 1) だけを声まね合成する（本番の bl・b1 は変えない）。
               --textedit_unit bl_b1_b7 なら bl = text_a、b1 = text_b、b7 の順に 3 本（5/15 の回と同じ順）。
               文は本番の results.json の text_a / text_b / text_b7 と一致を確かめる。
               b7 だけで作ってよい理由: 5/15 の回（eval_baseline_text_edit.py）は合成のたびに voice_clone の最初で
               seed_everything(SEED) を入れ直す（L133-134。呼び出しは bl L233・b1 L235・b7 L238）。ここで使う
               voice_clone_synth も最初に入れ直す（eval_utmos_speaker_fixed.py L211）。空運転（--dry_run）で、
               この 2 か所を構文から確かめ、本物の voice_clone_synth（重い処理だけ乱数を引く偽物に替える）で
               「bl → b1 → b7 の b7」と「b7 だけ」が同じ乱数になることを確かめる（reseed_check）。
               check42 は記録だけ（選ぶことには使わない）。

pyopenjtalk（判断 D4）: この台本は自分では import しない。vendor の長さ推定（duration_estimator.py L23-26, L94-98）が
  コンテナの pyopenjtalk を使う経路は残す（塞ぐと音素数が len(text) になり、Baseline を再現しなくなる）。
  その経路が有効か（duration_estimator.pyopenjtalk が None でないか）を確かめ、無効なら止まる。空運転（--dry_run）だけは塞ぐ。
  長さの推定が例外を出すと voice_clone_synth は黙って別の長さを使うので、estimate_duration を包んで値と例外を記録し、
  例外があれば止まる（環境の側の失敗）。参照つきの枝（同じ言語の枝）を通ったことも 1 回ごとに確かめる。

使い方（プロジェクト直下から。GPU 1 枚。同時は 2 つまで）:
  新しい 3 行の seed 42（単位ごとに記録。止まったら同じ命令で続きから）:
    SLURM_TIME=02:00:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_gen.py --method splice --seed 42 --pairs 0-149
    SLURM_TIME=02:00:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_gen.py --method splice --seed 42 --pairs 150-299
    SLURM_TIME=02:00:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_gen.py --method g2p --seed 42
    SLURM_TIME=02:00:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_gen.py --method rome --seed 42
  無効な単位の作り直し（1 つの命令で check42（seed 42 の再現の確かめ。記録だけ）→ seed 43 → seed 44）:
    Text Hint の無音 13 本（b7 だけ。合成は多くて 39 回）:
    SLURM_TIME=00:20:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_gen.py --method textedit --regen --textedit_unit b7 --stop_after_min 15
  確かめの小さな回（別の置き場。Baseline との一致を先に見る）:
    SLURM_TIME=00:20:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_gen.py --method splice --seed 42 --pairs 0-3 --output_root eval_results/_appg_vc_smoke --stop_after_min 15
CPU の空運転（モデルを読まない。文・トークン・置き場・キャッシュ・作る対を確かめる）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_gen.py --method splice --seed 42 --dry_run
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_gen.py --method textedit --regen --textedit_unit b7 --dry_run
終了コード: 0 = 完了、2 = seed 44 でも無効な単位が残る、3 = 環境の側の失敗で止まった、4 = 時間の上限で止まった（同じ命令で続き）。
"""
import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules:
    raise SystemExit("[threads] torch was imported before the thread cap")
THREADS = 13
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = str(THREADS)
if os.path.isdir("/mnt/scripts"):
    # slurm_run.sh は HF_HOME=/tmp/.cache/huggingface を渡すので代入で上書きする
    os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/mnt/.cache/huggingface/hub"
    os.environ["TORCH_HOME"] = "/mnt/.cache/torch"
    os.environ["XDG_CACHE_HOME"] = "/mnt/.cache"
    os.environ["NUMBA_CACHE_DIR"] = "/mnt/.cache/numba"
    os.makedirs("/mnt/.cache/numba", exist_ok=True)
else:  # ホストの空運転（同じ実体のキャッシュ）
    os.environ["HF_HOME"] = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface/hub")
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/numba")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "--dry_run" in sys.argv[1:]:
    sys.modules["pyopenjtalk"] = None  # 空運転だけ。生成（GPU）では vendor の長さ推定の経路を残す（判断 D4）

import argparse  # noqa: E402
import errno  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402
AC = V.AC

STOP_ENV = 3
STOP_TIME = 4
UNRESOLVED = 2
PROJ_DIR = os.path.join(V.PROJ, "scripts", "projection")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ 本番のモジュールを読む
def import_E():
    """eval_utmos_speaker_fixed を、直接実行したときと同じ並び（scripts/projection が先頭）で読む。"""
    if PROJ_DIR not in sys.path:
        sys.path.insert(0, PROJ_DIR)
    import torch
    torch.set_num_threads(THREADS)
    import eval_utmos_speaker_fixed as E  # noqa: E402  改変なし
    import splice_clone as SPL  # noqa: E402  改変なし
    return torch, E, SPL


def import_G2():
    import eval_g2p_baseline_jacc_n300 as G2  # noqa: E402  改変なし（読むと pykakasi を用意する）
    return G2


def import_RJ():
    import eval_rome_jacc_n300_text_a as RJ  # noqa: E402  改変なし
    return RJ


def md5_src(fn):
    return hashlib.md5(inspect.getsource(fn).encode()).hexdigest()


# ------------------------------------------------------------------ 文（5 月の記録と照合する）
def texts_for(method, pairs, G2=None):
    """{i: {key: 合成する文}}。splice は bl/inj の元の文（inj は注入の土台 text_a）。"""
    out = {}
    for p in pairs:
        i = p["pair_idx"]
        if method == "textedit":
            out[i] = {"bl": p["text_a"], "b1": p["text_b"], "b7": V.hint_ja(p["text_a"], p["word_a"], p["word_b"])}
        elif method == "g2p":
            g = G2.g2p_katakana(p["word_a"])
            out[i] = {"g2p": p["text_a"].replace(p["word_a"], g, 1), "_g2p_pred": g}
        elif method == "rome":
            out[i] = {"rome": p["text_a"]}
        else:
            out[i] = {"bl": p["text_a"], "inj_source": p["text_a"], "inj_reference": p["text_b"]}
    return out


def verify_texts(method, pairs, texts):
    """どの方法でも: 300 対（text_a / text_b / word_a / word_b）が本番の Baseline の回（results.json）と同じか。
    textedit はさらに b7 の文（text_b7）、g2p は 5 月の modified_text / g2p_pred_for_word_a と 300 対すべて一致するか。"""
    bad = []
    rows = {r["pair_idx"]: r for r in json.load(open(V.proj_path(V.BASELINE_RESULTS_REL)))["rows"]}
    for p in pairs:
        i = p["pair_idx"]
        r = rows.get(i) or {}
        if any(r.get(k) != p[k] for k in ("text_a", "text_b", "word_a", "word_b")):
            bad.append(i)
        elif method == "textedit" and texts[i]["b7"] != r.get("text_b7"):
            bad.append(i)
    src = [V.BASELINE_RESULTS_REL]
    if method == "g2p":
        g = {r["pair_idx"]: r for r in json.load(open(V.proj_path(V.OLD_G2P_JSON_REL)))["rows"]}
        for p in pairs:
            i = p["pair_idx"]
            r = g.get(i) or {}
            if texts[i]["g2p"] != r.get("modified_text") or texts[i]["_g2p_pred"] != r.get("g2p_pred_for_word_a"):
                bad.append(i)
        src.append(V.OLD_G2P_JSON_REL)
    bad = sorted(set(bad))
    return {"checked": True, "against": src, "n_checked": len(pairs), "n_mismatch": len(bad), "mismatch": bad[:20]}


# ------------------------------------------------------------------ SpeechSplice のトークンの計画（モデルなしで本番の関数を通す）
class _StubOut:
    def __init__(self, h):
        self.last_hidden_state = h


class _StubEncoder:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        T = input_ids.shape[1]
        return _StubOut(self.torch.arange(T, dtype=self.torch.float32).view(1, T, 1))


class _StubModel:
    """get_encoder_output_with_prefix が使う部分だけを持つ代わりのモデル（hidden = 位置の番号）。"""
    def __init__(self, torch, args):
        self.args = args
        self.encoder_module = _StubEncoder(torch)

    def _build_position_ids(self, x_lens, T, device):
        return None


def splice_plan(torch, E, SPL, text_tok, model_args, pairs, idxs):
    """各対の span・Δtok・注入の文を、本番の関数（get_encoder_output_with_prefix / find_entity_token_range /
    prepare_clone_splice）で求める。注入の文を符号化した長さが splice 後の長さと一致すること（hook の assert）も確かめる。"""
    stub = _StubModel(torch, model_args)
    per, bad = {}, []
    for i in idxs:
        p = pairs[i]
        try:
            src = E.get_encoder_output_with_prefix(stub, text_tok, V.PROMPT_TEXT, p["text_a"], "cpu", lang=V.LANG)
            ref = E.get_encoder_output_with_prefix(stub, text_tok, V.PROMPT_TEXT, p["text_b"], "cpu", lang=V.LANG)
            info = splice_ranges(E, text_tok, p, src, ref)
            modified, inj_t, inj_dur, dtok = SPL.prepare_clone_splice(
                text_tok, src, ref, info["s_range"], info["r_range"], V.LANG,
                duration_source_text=p["text_a"], natural_candidates=(p["text_b"],))
            inj_target = inj_t if inj_t is not None else p["text_a"]
            chk = E.get_encoder_output_with_prefix(stub, text_tok, V.PROMPT_TEXT, inj_target, "cpu", lang=V.LANG)
            h = modified[0, :, 0].tolist()
            s_s, s_e = info["s_range"]
            r_s, r_e = info["r_range"]
            want = list(range(0, s_s)) + list(range(r_s, r_e)) + list(range(s_e, src["last_hidden_state"].shape[1]))
            ok_len = len(chk["tokens"]) == modified.shape[1]
            ok_rows = [int(x) for x in h] == want
            per[i] = {**info, "delta_tok": int(dtok), "T_src": int(src["last_hidden_state"].shape[1]),
                      "T_out": int(modified.shape[1]), "inj_target_text": inj_target,
                      "inj_target_kind": ("text_a" if inj_t is None else ("text_b" if inj_t == p["text_b"] else "filler")),
                      "inj_duration_text": inj_dur, "hook_length_ok": ok_len, "rows_ok": ok_rows,
                      "n_prefix_tokens": len(src["prefix_tokens"]), "target_start_in_full": src["target_start_in_full"]}
            if not (ok_len and ok_rows):
                bad.append(i)
        except Exception as e:  # noqa: BLE001
            per[i] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            bad.append(i)
    dist = {}
    for v in per.values():
        if "delta_tok" in v:
            dist[str(v["delta_tok"])] = dist.get(str(v["delta_tok"]), 0) + 1
    kinds = {}
    for v in per.values():
        if "inj_target_kind" in v:
            kinds[v["inj_target_kind"]] = kinds.get(v["inj_target_kind"], 0) + 1
    tsf = sorted({v["target_start_in_full"] for v in per.values() if "target_start_in_full" in v})
    npt = sorted({v["n_prefix_tokens"] for v in per.values() if "n_prefix_tokens" in v})
    asym = [i for i, v in per.items() if "s_range" in v and
            (v["s_range"][1] - v["s_range"][0]) != (v["r_range"][1] - v["r_range"][0])]
    return {"n": len(idxs), "n_bad": len(bad), "bad": bad[:20], "delta_tok_counts": dist, "inj_target_kinds": kinds,
            "target_start_in_full_values": tsf, "n_prefix_tokens_values": npt,
            "unequal_span_pairs": asym, "filler_pairs": [i for i, v in per.items() if v.get("inj_target_kind") == "filler"],
            "examples": {str(i): per[i] for i in list(per)[:2]}, "_per": per}


def splice_ranges(E, text_tok, p, src_enc, ref_enc):
    """E.main() L478-498 と同じ。JA の 300 対は語がそのまま文にあるので _resolve_surface は語を返す（無ければ対の失敗）。"""
    for w, enc in ((p["word_a"], src_enc), (p["word_b"], ref_enc)):
        if w not in enc["target_text"]:
            raise V.PairFailure(f"entity {w!r} not in normalized target text {enc['target_text']!r}")
    s_sub = E.find_entity_token_range(text_tok, src_enc["target_text"], p["word_a"], add_bos=0)
    r_sub = E.find_entity_token_range(text_tok, ref_enc["target_text"], p["word_b"], add_bos=0)
    if s_sub is None or r_sub is None:
        raise V.PairFailure(f"entity token range None: s={s_sub} r={r_sub}")
    s_off = src_enc["target_start_in_full"]
    r_off = ref_enc["target_start_in_full"]
    return {"s_sub": list(s_sub), "r_sub": list(r_sub),
            "s_range": (s_sub[0] + s_off, s_sub[1] + s_off), "r_range": (r_sub[0] + r_off, r_sub[1] + r_off)}


def rome_inputs(pairs):
    """eval_rome_jacc_n300_text_a.main() L120-136 と同じ作り方。"""
    from collections import defaultdict
    by_pair = defaultdict(lambda: {"a_ctx": [], "b_ctx": [], "word_a": None, "word_b": None, "pair_idx_list": []})
    for p_ in pairs:
        key = (p_["word_a"], p_["word_b"])
        by_pair[key]["a_ctx"].append(p_["text_a"])
        by_pair[key]["b_ctx"].append(p_["text_b"])
        by_pair[key]["word_a"] = p_["word_a"]
        by_pair[key]["word_b"] = p_["word_b"]
        by_pair[key]["pair_idx_list"].append(p_["pair_idx"])
    groups = list(by_pair.values())
    target_entities = [{"word_a": g["word_a"], "word_b": g["word_b"],
                        "text_ref": g["b_ctx"][0], "text_test": g["a_ctx"][0]} for g in groups]
    corpus_texts = [c for g in groups for c in g["a_ctx"] + g["b_ctx"]]
    return groups, target_entities, corpus_texts


# ------------------------------------------------------------------ 条件の記録
def local_snapshot(repo):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
    d = os.path.join(hub, f"models--{repo.replace('/', '--')}", "snapshots")
    if os.path.isdir(d):
        s = sorted(os.listdir(d))
        if s:
            return os.path.join(d, s[-1])
    return repo


def base_condition(torch, E, SPL, dur_mod, itu_mod):
    import torchaudio
    import transformers
    cfg = json.load(open(os.path.join(local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), "config.json")))
    tok_repo = cfg.get("text_tokenizer_name") or cfg.get("t5gemma_model_name")
    return {
        "ref_wav_rel": V.REF_WAV_REL, "ref_wav_md5": V.md5_file(V.proj_path(V.REF_WAV_REL)),
        "prompt_text": V.PROMPT_TEXT, "prompt_sec": V.PROMPT_SEC, "lang": V.LANG, "decode": V.DECODE_DOC,
        "voice_clone_synth_source_md5": md5_src(E.voice_clone_synth),
        "eval_utmos_speaker_fixed_md5": V.md5_file(E.__file__), "splice_clone_md5": V.md5_file(SPL.__file__),
        "encoder_injection_md5": V.md5_file(sys.modules["encoder_injection_t5gemma"].__file__),
        "vendor_duration_estimator_md5": V.md5_file(dur_mod.__file__),
        "vendor_inference_tts_utils_md5": V.md5_file(itu_mod.__file__),
        "vendor_duration_uses_pyopenjtalk": getattr(dur_mod, "pyopenjtalk", None) is not None,
        "vendor_pyopenjtalk_version": AC.dist_version("pyopenjtalk"),
        "torch": torch.__version__, "torchaudio": torchaudio.__version__, "transformers": transformers.__version__,
        "model_snapshot": os.path.basename(local_snapshot("Aratako/T5Gemma-TTS-2b-2b")),
        "tokenizer_snapshot": os.path.basename(local_snapshot(tok_repo)),
    }


def method_condition(method, G2=None, RJ=None, g2p_texts_md5=None):
    d = {"method": method}
    if method == "textedit":  # textedit だけに足す（ほかの方法の gen_config.json と比べる中身は変えない）
        d.update(textedit_unit=V.textedit_unit(), textedit_regen_keys=V.regen_keys("textedit"))
    if method == "g2p":
        d.update(g2p_script_md5=V.md5_file(G2.__file__), g2p_texts_md5=g2p_texts_md5,
                 g2p_katakana_source_md5=md5_src(G2.g2p_katakana))
    if method == "rome":
        d.update(rome_script_md5=V.md5_file(RJ.__file__), rome_encoder_md5=V.md5_file(sys.modules["rome_encoder_t5gemma"].__file__),
                 apply_rome_edit_jacc_source_md5=md5_src(RJ.apply_rome_edit_jacc),
                 rome_layers=list(V.ROME_LAYERS), rome_mode=V.ROME_MODE, rome_eps=V.ROME_EPS)
    return d


# ------------------------------------------------------------------ 生成の文脈（モデル）
class Ctx(types.SimpleNamespace):
    pass


def load_model(ctx):
    torch, E = ctx.torch, ctx.E
    log("[Info] Loading T5Gemma-TTS ...")
    model_id = local_snapshot("Aratako/T5Gemma-TTS-2b-2b")
    model = E.AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16,
                                                    device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tkn = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    ctx.text_tok = E.AutoTokenizer.from_pretrained(local_snapshot(tkn), local_files_only=True)
    ctx.audio_tok = E.AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    ctx.codec_audio_sr = ctx.audio_tok.sample_rate
    ctx.codec_sr = getattr(cfg, "encodec_sr", 50)
    ctx.model = model
    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.model_info = {"model_id": model_id, "tokenizer": local_snapshot(tkn), "codec_audio_sr": ctx.codec_audio_sr,
                      "codec_sr": ctx.codec_sr, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                      "x_sep_token": getattr(model.args, "x_sep_token", None),
                      "add_bos_to_text": getattr(model.args, "add_bos_to_text", None),
                      "add_eos_to_text": getattr(model.args, "add_eos_to_text", None)}


def apply_rome(ctx):
    """編集は eval_rome_jacc_n300_text_a.apply_rome_edit_jacc をそのまま使う（層 2 → 5 → 8 の順に積み重ねる）。"""
    torch, RJ = ctx.torch, ctx.RJ
    _, target_entities, corpus_texts = rome_inputs(ctx.pairs)
    log(f"[ROME] layers={list(V.ROME_LAYERS)} mode={V.ROME_MODE} groups={len(target_entities)} corpus={len(corpus_texts)}")
    cov, saved, info = {}, {}, []
    t0 = time.time()
    for li in V.ROME_LAYERS:
        w = ctx.model.encoder_module.layers[li].mlp.down_proj.weight
        saved[li] = w.detach().clone()
        dW = RJ.apply_rome_edit_jacc(ctx.model, ctx.text_tok, li, target_entities, corpus_texts, ctx.device, cov,
                                     edit_mode=V.ROME_MODE, eps=V.ROME_EPS)
        if dW is None:
            raise SystemExit(f"[ROME] layer {li}: 編集が作れなかった（apply_rome_edit_jacc が None）")
        w2 = ctx.model.encoder_module.layers[li].mlp.down_proj.weight.detach()
        sha = hashlib.sha256(w2.to("cpu").contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
        info.append({"layer": li, "dW_norm": float(dW.norm().item()), "dW_max_abs": float(dW.abs().max().item()),
                     "edited_weight_sha256": sha})
    ctx.rome_saved = saved
    ctx.rome_edit = {"layers": info, "elapsed_min": round((time.time() - t0) / 60, 2),
                     "n_groups": len(target_entities), "n_corpus": len(corpus_texts)}
    log(f"[ROME] edit done: {json.dumps(info)}")


def synth(ctx, target_text, modified_hidden=None, duration_text=None):
    """声まね合成 1 回（E.voice_clone_synth、seed は E.SEED）。長さの推定の値を記録し、代わりの長さなら止める。"""
    ctx.cap.clear()
    audio = ctx.E.voice_clone_synth(
        ctx.model, ctx.text_tok, ctx.audio_tok, target_text=target_text, lang=V.LANG, device=ctx.device,
        codec_audio_sr=ctx.codec_audio_sr, codec_sr=ctx.codec_sr, ref_wav=ctx.ref_wav, ref_text=V.PROMPT_TEXT,
        ref_prompt_sec=V.PROMPT_SEC, modified_hidden=modified_hidden, duration_text=duration_text)
    cap = list(ctx.cap)
    if len(cap) != 1 or cap[0].get("error") or not cap[0].get("with_reference"):
        raise V.DurationPathError(f"duration estimator path not as expected: {cap}")
    return audio, cap[0]["sec"]


def save_tmp(ctx, d, method, key, i, audio, run_id):
    a2 = audio if audio.dim() > 1 else audio.unsqueeze(0)
    name = V.file_name(method, key, i)
    tmp = os.path.join(d, f".{name[:-4]}.tmp_{run_id}.wav")
    ctx.E.save_audio(tmp, a2, ctx.codec_audio_sr)  # 本番と同じ保存の関数（vendor の save_audio）
    return tmp


def gen_unit(ctx, method, tag, i, d, run_id):
    """1 単位を作って一時ファイルに書く。戻り値 (tmp_files, item, per_key)。"""
    p = ctx.pairs[i]
    tx = ctx.texts[i]
    tmp, per_key = {}, {}
    item = {"seed": V.TAG_SEED[tag], "run_id": run_id, "word_a": p["word_a"], "word_b": p["word_b"],
            "text_a": p["text_a"], "text_b": p["text_b"], "prompt_text": V.PROMPT_TEXT}
    if method == "splice":
        ctx.E.seed_everything(ctx.E.SEED)  # E.main() L419 と同じ（各合成の直前にも入れ直される）
        src = ctx.E.get_encoder_output_with_prefix(ctx.model, ctx.text_tok, V.PROMPT_TEXT, p["text_a"], ctx.device, lang=V.LANG)
        ref = ctx.E.get_encoder_output_with_prefix(ctx.model, ctx.text_tok, V.PROMPT_TEXT, p["text_b"], ctx.device, lang=V.LANG)
        info = splice_ranges(ctx.E, ctx.text_tok, p, src, ref)
        modified, inj_t, inj_dur, dtok = ctx.SPL.prepare_clone_splice(
            ctx.text_tok, src, ref, info["s_range"], info["r_range"], V.LANG,
            duration_source_text=p["text_a"], natural_candidates=(p["text_b"],))
        inj_target = inj_t if inj_t is not None else p["text_a"]
        plan = [("bl", p["text_a"], None, None), ("inj", inj_target, modified, inj_dur)]
        item["splice"] = {"s_range": list(info["s_range"]), "r_range": list(info["r_range"]), "delta_tok": int(dtok),
                          "T_src": int(src["last_hidden_state"].shape[1]), "T_out": int(modified.shape[1]),
                          "target_start_in_full": int(src["target_start_in_full"]),
                          "n_prefix_tokens": len(src["prefix_tokens"]), "inj_target_text": inj_target,
                          "inj_duration_text": inj_dur}
    elif method == "textedit":
        # b7 だけ（既定）か bl・b1・b7。どちらでも合成ごとに voice_clone_synth の最初で seed_everything(E.SEED)
        keys = V.stage_keys(method, tag)
        plan = [(k, tx[k], None, None) for k in ("bl", "b1", "b7") if k in keys]
        item["textedit_unit"] = V.textedit_unit()
        item["unit_keys"] = [k for k, *_ in plan]
    elif method == "g2p":
        plan = [("g2p", tx["g2p"], None, None)]
        item["g2p_pred_for_word_a"] = tx["_g2p_pred"]
    else:
        plan = [("rome", tx["rome"], None, None)]
    for key, text, mod, dur_text in plan:
        audio, sec = synth(ctx, text, modified_hidden=mod, duration_text=dur_text)
        tmp[key] = save_tmp(ctx, d, method, key, i, audio, run_id)
        per_key[key] = {"target_text": text, "target_gen_sec": sec,
                        "duration_text": dur_text if dur_text is not None else text,
                        "injected": mod is not None}
    return tmp, item, per_key


# ------------------------------------------------------------------ 1 つの段（置き場）を作る
class ItemLocks:
    """対ごとの排他（1 つのファイルのバイト範囲のロック）。2 つの分担が同じ対を作らないための保険。"""
    def __init__(self, d):
        self.path = os.path.join(d, ".item_locks")
        self.f = open(self.path, "a+")
        self.ok = True

    def try_lock(self, i):
        if not self.ok:
            return True
        try:
            fcntl.lockf(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, i)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):  # 別の回が持っている
                return False
            # ロックが使えないファイルシステム（分担の範囲が重ならないことに頼る）
            log(f"[warn] 対ごとのロックが使えない（{e}）。分担の範囲を重ねないこと")
            self.ok = False
            return True

    def unlock(self, i):
        if self.ok:
            try:
                fcntl.lockf(self.f, fcntl.LOCK_UN, 1, i)
            except OSError:
                pass


def run_stage(ctx, method, tag, targets, args, run_id, t_start):
    """targets のうち記録の無い対を順に作る。戻り値: "done" / "time"。環境の側の失敗は SystemExit(3)。"""
    root = args.output_root
    d = V.method_dir(method, tag, root)
    V.assert_writable_dir(d, root)
    os.makedirs(os.path.join(d, "items"), exist_ok=True)
    cond = {**ctx.cond, **ctx.mcond, "tag": tag, "seed": V.TAG_SEED[tag], "written": V.now(), "run_id": run_id}
    ok, diff = V.guard_dir_config(d, cond, root)
    if not ok:
        raise SystemExit(f"[guard] {V.rel_to_proj(d)} の条件が前の回と違う: {diff}。別の置き場に出す")
    pending = [i for i in targets if not os.path.isfile(V.item_path(method, tag, i, root))]
    log(f"[stage] {method}/{tag} seed={V.TAG_SEED[tag]} targets={len(targets)} pending={len(pending)} dir={V.rel_to_proj(d)}")
    if not pending:
        return "done"
    ensure_model(ctx)
    ctx.E.SEED = V.TAG_SEED[tag]  # voice_clone_synth / main と同じく seed_everything(SEED) が実行時に読む
    locks = ItemLocks(d)
    streak = []
    n_done = 0
    for k, i in enumerate(pending):
        if (time.time() - t_start) / 60 > args.stop_after_min:
            log(f"[time] {args.stop_after_min} 分を超えたので止める（同じ命令で続きから）。この段の済み {n_done}/{len(pending)}")
            return "time"
        ip = V.item_path(method, tag, i, root)
        if not locks.try_lock(i):
            log(f"  [{i}] 別の回が作っている。飛ばす")
            continue
        try:
            if os.path.isfile(ip):
                continue  # 別の回が作り終えた
            moved = V.set_aside_unrecorded(method, tag, i, root, int(time.time()))
            if moved:
                log(f"  [{i}] 記録の無い wav を名前を変えて残した: {moved}")
            try:
                tmp, item, per_key = gen_unit(ctx, method, tag, i, d, run_id)
            except Exception as e:  # noqa: BLE001
                env = V.is_environment_failure(e)
                streak.append(i)
                if env or (args.max_consecutive_errors > 0 and len(streak) >= args.max_consecutive_errors):
                    stopped = []
                    for j in streak[:-1]:  # 同じ回の続いた失敗の記録は外して残す（同じ seed でやり直せるように）
                        jp = V.item_path(method, tag, j, root)
                        if os.path.isfile(jp):
                            os.replace(jp, f"{jp}.stopped_{run_id}")
                            stopped.append(j)
                    why = "environment failure" if env else f"{len(streak)} consecutive pair failures"
                    msg = (f"[stop] {why} at pair {i}: {type(e).__name__}: {str(e)[:300]}; "
                           f"set-aside error records (not counted, redo with the same seed): {stopped}. "
                           "原因を確かめてから同じ命令でもう一度流す")
                    log(msg)
                    raise SystemExit(STOP_ENV) from RuntimeError(msg)
                V.record_error(method, tag, i, root, {"seed": V.TAG_SEED[tag], "run_id": run_id}, e)
                log(f"  [{i}] pair failure recorded: {type(e).__name__}: {str(e)[:200]}")
            else:
                streak = []
                rec = V.commit_unit(method, tag, i, root, tmp, item, per_key)
                n_done += 1
                if rec["silent_keys_at_generation"]:
                    log(f"  [{i}] silent at generation: {rec['silent_keys_at_generation']}")
        finally:
            locks.unlock(i)
        if (k + 1) % 10 == 0 or k + 1 == len(pending):
            el = (time.time() - t_start) / 60
            log(f"  [{method}/{tag} {k + 1}/{len(pending)}] elapsed={el:.1f}m")
            if (k + 1) % 50 == 0:
                import gc
                gc.collect()
                if ctx.torch.cuda.is_available():
                    ctx.torch.cuda.empty_cache()
    return "done"


def ensure_model(ctx):
    if getattr(ctx, "model", None) is not None:
        return
    load_model(ctx)
    ctx.run_record["model"] = ctx.model_info
    if ctx.method == "splice":
        pl = splice_plan(ctx.torch, ctx.E, ctx.SPL, ctx.text_tok, ctx.model.args, ctx.pairs, list(range(V.N_PAIRS)))
        ctx.run_record["splice_token_plan_container"] = {k: v for k, v in pl.items() if k != "_per"}
        log(f"[splice] token plan (container tokenizer): Δtok={pl['delta_tok_counts']} kinds={pl['inj_target_kinds']} "
            f"target_start_in_full={pl['target_start_in_full_values']} bad={pl['bad']}")
        if pl["n_bad"]:
            raise SystemExit(f"[guard] splice の準備が通らない対がある: {pl['bad']}")
    if ctx.method == "rome":
        apply_rome(ctx)
        ctx.run_record["rome_edit"] = ctx.rome_edit
        ref = ctx.cond_rome_ref
        if ref:
            same = [a["edited_weight_sha256"] == b["edited_weight_sha256"] for a, b in zip(ctx.rome_edit["layers"], ref)]
            ctx.run_record["rome_edit_same_as_seed42_run"] = same
            log(f"[ROME] edited weights identical to the seed42 run: {same}")
    save_run(ctx)


def save_run(ctx):
    d = os.path.join(V.root_path(ctx.args.output_root), ctx.method, "runs")
    os.makedirs(d, exist_ok=True)
    V.atomic_write_json(os.path.join(d, f"{ctx.run_id}.json"), ctx.run_record)


# ------------------------------------------------------------------ b7 だけで作ってよいかの確かめ（CPU、モデルなし）
def _first_stmt(fn_node):
    """関数の最初の文（docstring は飛ばす）。"""
    import ast
    body = list(fn_node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body[0] if body else None


def _is_seed_call(stmt):
    import ast
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "seed_everything"
            and len(stmt.value.args) == 1 and isinstance(stmt.value.args[0], ast.Name)
            and stmt.value.args[0].id == "SEED")


def reseed_source_lines(E):
    """構文から確かめる: (1) 5/15 の回 eval_baseline_text_edit.py の voice_clone の最初の文が seed_everything(SEED)、
    bl / b1 / b7 がそれぞれ別の voice_clone(...) の呼び出し。(2) ここで使う E.voice_clone_synth の最初の文も
    seed_everything(SEED)。(3) seed_everything が入れ直す乱数。行番号を返す（ファイルの今の中身から数える）。"""
    import ast
    import textwrap
    out = {"ok": False}
    p = os.path.join(PROJ_DIR, "eval_baseline_text_edit.py")
    src = open(p, encoding="utf-8").read()
    tree = ast.parse(src)
    out["eval_baseline_text_edit"] = {"path": V.rel_to_proj(p), "md5": V.md5_file(p)}
    d = out["eval_baseline_text_edit"]
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SEED" for t in n.targets) \
                and n.col_offset == 0:
            d["SEED_line"] = n.lineno
            d["SEED_value"] = getattr(n.value, "value", None)
        if isinstance(n, ast.ImportFrom) and any(a.name == "seed_everything" for a in n.names):
            d["seed_everything_import_line"] = n.lineno
            d["seed_everything_from"] = n.module
    vc = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "voice_clone"]
    if len(vc) == 1:
        st = _first_stmt(vc[0])
        d["voice_clone_def_line"] = vc[0].lineno
        d["voice_clone_first_stmt_line"] = st.lineno if st else None
        d["voice_clone_first_stmt_is_seed_everything_SEED"] = bool(st and _is_seed_call(st))
        inner = {}
        for n in ast.walk(vc[0]):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                inner.setdefault(n.func.id, []).append(n.lineno)
        d["voice_clone_calls_after_seed"] = {k: v for k, v in inner.items()
                                             if k in ("seed_everything", "estimate_duration", "inference_one_sample")}
    calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) \
                and n.value.func.id == "voice_clone" and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            calls[n.targets[0].id] = n.lineno
    d["synthesis_calls"] = calls  # bl_audio / b1_audio / b7_audio -> 行
    # (2) 生成で使う関数
    lines, start = inspect.getsourcelines(E.voice_clone_synth)
    fn = ast.parse(textwrap.dedent("".join(lines))).body[0]
    st = _first_stmt(fn)
    out["voice_clone_synth"] = {"path": V.rel_to_proj(inspect.getsourcefile(E.voice_clone_synth)),
                                "def_line": start, "first_stmt_line": (start + st.lineno - 1) if st else None,
                                "first_stmt_is_seed_everything_SEED": bool(st and _is_seed_call(st)),
                                "seed_everything_module": getattr(E.seed_everything, "__module__", None)}
    # (3) seed_everything の中身
    sl, ss = inspect.getsourcelines(E.seed_everything)
    out["seed_everything"] = {"path": V.rel_to_proj(inspect.getsourcefile(E.seed_everything)),
                              "lines": [ss, ss + len(sl) - 1],
                              "body": [x.strip() for x in sl[1:] if x.strip()]}
    out["ok"] = bool(d.get("voice_clone_first_stmt_is_seed_everything_SEED")
                     and out["voice_clone_synth"]["first_stmt_is_seed_everything_SEED"]
                     and sorted(calls) == ["b1_audio", "b7_audio", "bl_audio"] and d.get("SEED_value") == 42
                     and d.get("seed_everything_from") == "encoder_injection_t5gemma"
                     and out["voice_clone_synth"]["seed_everything_module"] == "encoder_injection_t5gemma")
    return out


def reseed_check(torch, E, texts, idxs, seeds=(42, 43, 44)):
    """本物の E.voice_clone_synth を CPU で呼び、重い 2 つ（inference_one_sample / estimate_duration）だけを、
    torch・numpy・random の乱数を引く偽物に一時的に替える（モデルは読まない。hook も付けない）。
    同じ seed で「bl → b1 → b7 と続けた b7」と「関係のない乱数を引いたあとの b7 だけ」の引いた乱数が同じなら、
    合成の最初で乱数が入れ直されている = b7 だけで作っても単位の中の b7 と同じ乱数で作られる。"""
    import random as _random
    import numpy as _np
    orig = (E.inference_one_sample, E.estimate_duration, E.SEED)
    got = []

    def fake_est(*a, **kw):
        got.append(("est", kw.get("reference_speech") is not None))
        return 3.0

    def fake_inf(**kw):
        x = torch.cat([torch.rand(6), torch.tensor([float(_np.random.rand()), _random.random()])])
        return None, x.unsqueeze(0)

    res = {"pairs": list(idxs), "seeds": list(seeds), "per": {}}
    ok = True
    try:
        E.inference_one_sample = fake_inf
        E.estimate_duration = fake_est
        ref = V.proj_path(V.REF_WAV_REL)
        stand_in = types.SimpleNamespace(args=types.SimpleNamespace())  # model.args を読むだけ（偽の合成に渡る）

        def one(text):
            return E.voice_clone_synth(stand_in, None, None, target_text=text, lang=V.LANG, device="cpu",
                                       codec_audio_sr=16000, codec_sr=50, ref_wav=ref, ref_text=V.PROMPT_TEXT,
                                       ref_prompt_sec=V.PROMPT_SEC)
        for s in seeds:
            for i in idxs:
                E.SEED = s
                unit = {k: one(texts[i][k]) for k in ("bl", "b1", "b7")}
                torch.rand(11)
                _np.random.rand(5)
                _random.random()  # 関係のない乱数を引いて状態を変えてから
                E.SEED = s
                alone = one(texts[i]["b7"])
                same = bool(torch.equal(unit["b7"], alone))
                same_all = bool(torch.equal(unit["bl"], unit["b7"]) and torch.equal(unit["b1"], unit["b7"]))
                res["per"][f"{s}:{i}"] = {"b7_in_unit_equals_b7_alone": same,
                                          "bl_b1_b7_start_from_same_state": same_all}
                ok &= same and same_all
        draws = []
        for s in seeds[:2]:  # 偽物が本当に seed で変わる乱数を引いているか（引いていなければ上の一致は意味がない）
            E.SEED = s
            draws.append(one(texts[idxs[0]]["b7"]))
        diff_seed = not torch.equal(draws[0], draws[1])
        res["different_seeds_give_different_draws"] = diff_seed
        ok &= diff_seed
        est = [w for t, w in got if t == "est"]
        res["n_estimate_duration_calls"] = len(est)
        res["estimate_duration_with_reference_branch"] = bool(est) and all(est)
    finally:
        E.inference_one_sample, E.estimate_duration, E.SEED = orig
    res["ok"] = bool(ok and res["estimate_duration_with_reference_branch"])
    return res


# ------------------------------------------------------------------ 空運転
def dry_run(args, method, pairs, texts, vt, stages_plan):
    import torch
    torch.set_num_threads(THREADS)
    rep = {"mode": "dry_run", "method": method, "output_root": V.rel_to_proj(V.root_path(args.output_root)),
           "seed": args.seed, "regen": args.regen, "texts_check": vt, "stages": stages_plan,
           "pyopenjtalk_blocked_in_dry_run": sys.modules.get("pyopenjtalk", "absent") is None}
    if method == "textedit":
        rep["textedit_unit"] = V.textedit_unit()
        rep["textedit_regen_keys"] = V.regen_keys("textedit")
        rep["textedit_stage_dirs"] = {t: V.rel_to_proj(V.method_dir("textedit", t, args.output_root))
                                      for t in ("check42", "seed43", "seed44")}
    # 参照音声と見本の文（5/15 のログの prefix 行とバイト単位で同じか）
    ref = V.proj_path(V.REF_WAV_REL)
    lg = V.parse_baseline_log()
    rep["reference"] = {"path": V.REF_WAV_REL, "md5": V.md5_file(ref), "md5_expected": V.REF_WAV_MD5,
                        "prompt_text": V.PROMPT_TEXT, "prefix_in_may_log": lg["prefix_line"],
                        "prompt_equals_may_log": lg["prefix_line"] == V.PROMPT_TEXT,
                        "may_log_speed_lines": lg["n_speed_lines"], "may_mean_synth_sec": lg["mean_synth_sec"]}
    # 本番のモジュールが読めるか（pyopenjtalk は塞いだまま）
    torch_, E, SPL = import_E()
    import duration_estimator as dur_mod
    import inference_tts_utils as itu
    rep["duration_estimator_pyopenjtalk_in_this_process"] = getattr(dur_mod, "pyopenjtalk", None) is not None
    rep["prompt_detect_language"] = dur_mod.detect_language(V.PROMPT_TEXT)
    rep["voice_clone_synth_source_md5"] = md5_src(E.voice_clone_synth)
    rep["E_SEED_default"] = E.SEED
    cond = base_condition(torch_, E, SPL, dur_mod, itu)
    cond["vendor_duration_uses_pyopenjtalk"] = "(not checked in dry run: blocked here; checked in the GPU run)"
    rep["condition_now"] = cond
    rep["method_condition_now"] = method_condition(method)
    if method == "textedit":
        rep["reseed_source_lines"] = reseed_source_lines(E)
        t43 = (stages_plan.get("seed43") or {}).get("targets") or []
        rep["reseed_check"] = reseed_check(torch_, E, texts, (t43 or [27, 46])[:3])
    if method == "rome":
        RJ = import_RJ()
        groups, te, corpus = rome_inputs(pairs)
        rep["rome"] = {"apply_rome_edit_jacc": md5_src(RJ.apply_rome_edit_jacc), "n_groups": len(groups),
                       "n_target_entities": len(te), "n_corpus": len(corpus), "cov_uses_first": 200,
                       "first_entity": te[0]}
    if method in ("splice",):
        cfg = json.load(open(os.path.join(local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), "config.json")))
        tk = V.AC.local_snapshot(cfg["text_tokenizer_name"])
        text_tok = E.AutoTokenizer.from_pretrained(tk, local_files_only=True)
        pl = splice_plan(torch_, E, SPL, text_tok, types.SimpleNamespace(**cfg), pairs, list(range(V.N_PAIRS)))
        rep["splice_token_plan_host"] = {k: v for k, v in pl.items() if k != "_per"}
        rep["splice_token_plan_host"]["per_pair_head"] = {str(i): pl["_per"][i] for i in (0, 59, 99)}
    os.environ["HF_HUB_OFFLINE"] = "1"
    pre_ok, pre = AC.offline_preflight(AC.GEN_REPOS)
    rep["offline_preflight_ok"] = pre_ok
    rep["offline_preflight"] = pre
    # 時間の見積もり（5/15 の合成 1 回の平均時間から）
    n_synth = {"textedit": len(V.regen_keys("textedit")), "g2p": 1, "rome": 1, "splice": 2}[method]
    todo = sum(len(s.get("pending") or []) for s in stages_plan.values() if isinstance(s, dict))
    ms = lg["mean_synth_sec"] or 10.0
    rep["estimate_min"] = {"units_pending_now": todo, "synth_per_unit": n_synth, "mean_synth_sec_may": round(ms, 2),
                           "synth_min": round(todo * n_synth * ms / 60, 1),
                           "note": "plus ~3-5 min model load; ROME adds the MEMIT edit (~5-10 min on 13 CPU threads)"}
    problems = []
    if rep["reference"]["md5"] != V.REF_WAV_MD5:
        problems.append("reference md5 differs")
    if not rep["reference"]["prompt_equals_may_log"]:
        problems.append("prompt text differs from the May log")
    if rep["prompt_detect_language"] != "ja":
        problems.append("prompt language is not ja (cross-lingual branch would be taken)")
    if vt.get("checked") and vt["n_mismatch"]:
        problems.append(f"texts differ from the May record: {vt['mismatch']}")
    if method == "splice" and rep["splice_token_plan_host"]["n_bad"]:
        problems.append(f"splice plan failed: {rep['splice_token_plan_host']['bad']}")
    if method == "textedit":
        if not rep["reseed_source_lines"]["ok"]:
            problems.append("re-seed before every synthesis not confirmed from the source (reseed_source_lines)")
        if not rep["reseed_check"]["ok"]:
            problems.append("b7 alone != b7 inside the unit in the CPU re-seed check (reseed_check)")
    if isinstance(stages_plan, dict) and stages_plan.get("problems"):
        problems.append(f"scan problems: {stages_plan['problems']}")
    if not pre_ok:
        problems.append("HF cache missing")
    rep["problems"] = problems
    op = V.proj_path(args.dryrun_out or f"{V.DRYRUN_REL}/gen_{run_label(method, args)}_dryrun.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    V.atomic_write_json(op, rep)
    short = {k: v for k, v in rep.items() if k not in ("offline_preflight", "condition_now", "stages")}
    log(json.dumps(short, ensure_ascii=False, indent=1)[:6000])
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("pending",)} if isinstance(v, dict) else v
                    for k, v in stages_plan.items()}, ensure_ascii=False)[:3000])
    log(f"[dry_run] saved {op}")
    if problems:
        raise SystemExit(f"[dry_run] problems: {problems}")


# ------------------------------------------------------------------ main
def run_label(method, args):
    """run_id と空運転の出力の名前。textedit の b7 だけは textedit_b7only_regen（前の 3 本の名前と分ける）。"""
    lab = "regen" if args.regen else "seed42"
    if method == "textedit" and V.textedit_unit() != "bl_b1_b7":
        lab = f"{V.textedit_unit()}only_{lab}"
    return f"{method}_{lab}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(V.METHODS))
    ap.add_argument("--seed", type=int, default=None, help="42 だけ（新しい 3 行の本番）。作り直しは --regen")
    ap.add_argument("--regen", action="store_true", help="無効な単位を check42 → seed43 → seed44 の順に作る")
    ap.add_argument("--pairs", default=None, help="seed 42 の分担（例 0-149）。--regen では使わない")
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL)
    ap.add_argument("--max_consecutive_errors", type=int, default=3)
    ap.add_argument("--stop_after_min", type=float, default=110.0, help="この分数を超えたら新しい単位を始めない")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dryrun_out", default=None)
    ap.add_argument("--textedit_unit", default=None, choices=sorted(V.TEXTEDIT_UNITS),
                    help=f"textedit の作り直しで作る wav（既定 {V.TEXTEDIT_UNIT_DEFAULT} = b7 だけ。著者の決定 2026-09-29）")
    args = ap.parse_args()
    method = args.method
    if args.textedit_unit is not None and method != "textedit":
        raise SystemExit("--textedit_unit は --method textedit だけで使う")
    V.set_textedit_unit(args.textedit_unit or V.TEXTEDIT_UNIT_DEFAULT)
    root = V.root_path(args.output_root)
    if args.regen == (args.seed is not None):
        raise SystemExit("--seed 42 か --regen のどちらか一方を指定する")
    if args.seed is not None:
        if args.seed != V.MAIN_SEED:
            raise SystemExit("--seed は 42 だけ（43 / 44 は --regen が無効な単位だけに使う）")
        if method == "textedit":
            raise SystemExit("Baseline / Text Hint の seed 42 は本番（5/15）をそのまま使う。作らない（無効な単位は --regen）")
    if args.regen and args.pairs:
        raise SystemExit("--regen の対は判定で決まる（--pairs は使わない）")

    pairs = V.load_pairs()
    G2 = None
    if method == "g2p":
        _, _E, _S = import_E()  # sys.path をそろえてから G2 を読む
        G2 = import_G2()
    texts = texts_for(method, pairs, G2)
    vt = verify_texts(method, pairs, texts)
    if vt.get("checked") and vt["n_mismatch"] and not args.dry_run:
        raise SystemExit(f"[guard] 合成する文が 5 月の記録と違う: {vt}")

    # 段の計画
    if args.seed is not None:
        targets = V.parse_pairs(args.pairs) if args.pairs else list(range(V.N_PAIRS))
        stages_plan = {"seed42": {"targets_n": len(targets),
                                  "pending": [i for i in targets if not os.path.isfile(V.item_path(method, "seed42", i, root))]}}
        stages_plan["seed42"]["pending_n"] = len(stages_plan["seed42"]["pending"])
    else:
        s = V.scan_method(method, root)
        stages_plan = {"scan_next_action": s["next_action"], "problems": s["problems"],
                       "seed42": {k: s["stages"]["seed42"][k] for k in ("complete", "n_valid", "n_invalid", "invalid",
                                                                         "n_not_attempted", "n_inconsistent")}}
        for tag in ("check42", "seed43", "seed44"):
            t = s["targets"][tag]
            stages_plan[tag] = {"targets": t, "pending": None if t is None else
                                [i for i in t if not os.path.isfile(V.item_path(method, tag, i, root))]}
        stages_plan["unresolved"] = s["unresolved"]

    if args.dry_run:
        dry_run(args, method, pairs, texts, vt, stages_plan)
        return 0

    # ---- ここから GPU（空運転ではない） ----
    torch, E, SPL = import_E()
    import duration_estimator as dur_mod  # E が既に読んだ vendor のモジュール（属性を見るだけ。pyopenjtalk は import しない）
    import inference_tts_utils as itu
    if getattr(dur_mod, "pyopenjtalk", None) is None:
        raise SystemExit("[guard] vendor の長さ推定が pyopenjtalk なし（len(text)）になっている。5/15 の条件とずれるので止める")
    if dur_mod.detect_language(V.PROMPT_TEXT) != V.LANG:
        raise SystemExit("[guard] 見本の文の言語判定が ja でない（別の長さ推定の枝に入る）")
    ref_wav = V.proj_path(V.REF_WAV_REL)
    if V.md5_file(ref_wav) != V.REF_WAV_MD5:
        raise SystemExit("[guard] 参照音声の md5 が 5/15 の回と違う")
    lg = V.parse_baseline_log()
    if lg["prefix_line"] != V.PROMPT_TEXT:
        raise SystemExit(f"[guard] 見本の文が 5/15 のログと違う: {lg['prefix_line']!r}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    pre_ok, pre = AC.offline_preflight(AC.GEN_REPOS)
    if not pre_ok:
        raise SystemExit(f"[offline] 生成に使う repo が手元のキャッシュに無い: {pre}")

    # 長さの推定を包む（値と例外を記録するだけ。呼び出しと戻り値は変えない）
    cap = []
    _orig_est = E.estimate_duration

    def _est(*a, **kw):
        try:
            v = _orig_est(*a, **kw)
        except Exception as ex:  # noqa: BLE001
            cap.append({"error": repr(ex)[:300]})
            raise
        cap.append({"sec": float(v), "with_reference": kw.get("reference_speech") is not None,
                    "target_lang": kw.get("target_lang")})
        return v
    E.estimate_duration = _est

    run_id = V.new_run_id(run_label(method, args))
    RJ = import_RJ() if method == "rome" else None
    g2p_md5 = hashlib.md5(json.dumps([texts[i]["g2p"] for i in range(V.N_PAIRS)], ensure_ascii=False).encode()).hexdigest() \
        if method == "g2p" else None
    cond = base_condition(torch, E, SPL, dur_mod, itu)
    mcond = method_condition(method, G2, RJ, g2p_md5)
    bad_common = V.guard_root_common(cond, args.output_root)
    if bad_common:
        raise SystemExit(f"[guard] 同じ出力の置き場のほかの回と声まねの条件が違う: {bad_common}")
    ctx = Ctx(torch=torch, E=E, SPL=SPL, G2=G2, RJ=RJ, method=method, pairs=pairs, texts=texts, cap=cap,
              ref_wav=ref_wav, cond=cond, mcond=mcond, args=args, run_id=run_id, model=None, cond_rome_ref=None)
    if method == "rome" and args.regen:  # seed 42 の回の編集と同じか（記録だけ）
        rr = sorted(os.listdir(os.path.join(root, "rome", "runs"))) if os.path.isdir(os.path.join(root, "rome", "runs")) else []
        for n in rr:
            r = V.load_json(os.path.join(root, "rome", "runs", n)) or {}
            if not r.get("regen") and r.get("rome_edit"):
                ctx.cond_rome_ref = r["rome_edit"]["layers"]
                break
    ctx.run_record = {"run_id": run_id, "started": V.now(), "method": method, "regen": args.regen, "seed": args.seed,
                      "pairs_arg": args.pairs, "script": os.path.abspath(__file__), "script_md5": V.md5_file(os.path.abspath(__file__)),
                      "appg_vc_common_md5": V.md5_file(os.path.join(_HERE, "appg_vc_common.py")),
                      "silence_criterion": AC.criterion_doc(), "condition": cond, "method_condition": mcond,
                      "texts_check": vt, "threads": {"torch": torch.get_num_threads(), "OMP": os.environ["OMP_NUM_THREADS"]},
                      "hf_env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "NUMBA_CACHE_DIR")},
                      "offline_preflight": pre, "hostname": os.uname().nodename, "stages": {}}
    if method == "textedit":
        ctx.run_record["textedit_unit"] = V.textedit_unit()
        ctx.run_record["textedit_regen_keys"] = V.regen_keys("textedit")
        ctx.run_record["check42_note"] = "check42 is a record of seed-42 reproduction only; never used to choose"
    save_run(ctx)
    t_start = time.time()
    rc = 0
    try:
        if args.seed is not None:
            st = run_stage(ctx, method, "seed42", targets, args, run_id, t_start)
            ctx.run_record["stages"]["seed42"] = st
            rc = STOP_TIME if st == "time" else 0
        else:
            s = V.scan_method(method, root)
            if s["problems"] or not s["stages"]["seed42"]["complete"]:
                raise SystemExit(f"[guard] 作り直しの前に: {s['next_action']} {s['problems']}")
            t43 = s["targets"]["seed43"]
            log(f"[regen] {method}: seed 42 で無効な単位 {len(t43)}: {t43}")
            if method == "textedit":
                log(f"[regen] textedit: 作る wav {V.regen_keys('textedit')}（--textedit_unit {V.textedit_unit()}）"
                    f" 置き場 {V.rel_to_proj(V.method_dir('textedit', 'seed43', root))} ほか。check42 は記録だけ")
            if not t43:
                log("[regen] 無効な単位なし。何もしない")
            else:
                for tag in ("check42", "seed43"):
                    st = run_stage(ctx, method, tag, t43, args, run_id, t_start)
                    ctx.run_record["stages"][tag] = st
                    save_run(ctx)
                    if st == "time":
                        rc = STOP_TIME
                        break
                if rc == 0:
                    s = V.scan_method(method, root)
                    if s["problems"]:
                        raise SystemExit(f"[guard] {s['problems']}")
                    t44 = s["targets"]["seed44"] or []
                    log(f"[regen] seed 43 でも無効な単位 {len(t44)}: {t44}")
                    st = run_stage(ctx, method, "seed44", t44, args, run_id, t_start)
                    ctx.run_record["stages"]["seed44"] = st
                    if st == "time":
                        rc = STOP_TIME
                    else:
                        s = V.scan_method(method, root)
                        ctx.run_record["final_scan"] = {k: s[k] for k in ("next_action", "seed_counts", "unresolved", "problems")}
                        ctx.run_record["check42_reproduction"] = {
                            k: v for k, v in s["stages"]["check42"].get("reproduction", {}).items() if k != "per_pair"}
                        if s["unresolved"]:
                            log(f"[regen] seed 44 でも無効: {s['unresolved']}（著者に聞く）")
                            rc = UNRESOLVED
    except SystemExit as e:
        ctx.run_record["stopped"] = V.now()
        ctx.run_record["stop"] = str(e.__cause__ or e)[:600]
        save_run(ctx)
        log(str(e.__cause__ or e))
        raise
    finally:
        if method == "rome" and getattr(ctx, "rome_saved", None):
            for li, w in ctx.rome_saved.items():
                ctx.model.encoder_module.layers[li].mlp.down_proj.weight.data.copy_(w)
    ctx.run_record["finished"] = V.now()
    ctx.run_record["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    ctx.run_record["exit_code"] = rc
    save_run(ctx)
    s = V.scan_method(method, root)
    log(f"[done] {method}: {s['next_action']} seed_counts={s['seed_counts']} rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
