#!/usr/bin/env python3
"""CSM-1B に SpeechSplice を適用して cross-culture 300 対を生成する (GPU).

生成するもの (既存の Qwen3-TTS / Llasa-3B / XTTS-v2 行と同じ命名規約):
    {wav_dir}/s{pair_idx:04d}_bl.wav  = TTS(text_a)                  実体語は word_a を読む
    {wav_dir}/s{pair_idx:04d}_inj.wav = SpeechSplice(text_a)          word_b を読ませたい
    {wav_dir}/s{pair_idx:04d}_tgt.wav = TTS(text_b)                  正解の読み
3 本そろっていない対は採点器側で捨てられるので、TGT の生成を省略してはならない
(scripts/per_fix/phoneme_per_v3.py:101-102)。

採点は既存の scripts/per_fix/phoneme_per_v3.py をそのまま使う (新しい採点器は作らない)。
本スクリプトが書く results.json は {"rows": [{"pair_idx", "word_a", "word_b", ...}]} の形で、
phoneme_per_v3.py の --pairs_json にそのまま渡せる。

--------------------------------------------------------------------------
黙った失敗を起こさないための 3 つの決め事 (2026-09-17 / ラウンド 2)
--------------------------------------------------------------------------
(1) **率の分母は「試行した対の総数」(n_attempted) で固定する。**
    生き残った対だけで割ると、落ちた対がまるごと消えて率が勝手に上がる。
    旧版 PIB はこれで 300 対中 182 対を黙って捨てたまま「有効 300 対」として
    通していた。n_attempted / n_ok / n_failed を必ず出し、除外率が
    --max_drop_rate を超えたら summary["warnings"] に文字列で残す。
    生き残り分母の率は *_rate_survivors、条件が実際に生成できた対だけを
    分母にした率は rates_given_generated に分けて併記する (どちらも参考値)。

(2) **3 条件は互いに独立に記録する。**
    BL / TGT / INJ をそれぞれ別の try で回し、成功した条件はその場で wav を保存し
    row に統計を書く。旧版は 1 対 = 1 つの try だったため、INJ が落ちると
    生成し終えている BL と TGT の記録まで丸ごと消え、source_in_bl_rate が
    INJ の失敗で押し下げられて「BL が喋れていない」と誤診していた。
    条件別の母数は n_ok_by_cond / fail_by_cond に出す。

(3) **音声の出所 (provenance) は wav_dir 側に記録する。**
    採点されるのは results.json ではなく wav_dir の中身なので、指紋を out_dir に
    置くだけでは足りない。{wav_dir}/_wav_manifest.json に
      * その wav_dir を作った実行設定の指紋 (layer / seed / 復号設定 / 参照音声 /
        model revision など、音のビットを決める鍵だけ)
      * wav 1 本ごとの sha256・バイト数・どの out_dir のどの run が書いたか
    を書く。起動時に
      a. 台帳の指紋が今回の設定と食い違えば **停止する**
      b. 1 本ずつ sha256 を照合し、今回の指紋で作られたと確かめられない wav は
         `_stale_<run_id>_` を付けて退避する (削除はしない)
    ので、「前の層で作った音声が新しい層の結果として採点される」経路が塞がる。
    2026-09-17 の LongCat 統制実験 (--splice_part hL_only の付け忘れで、本条件と
    bit 単位で同一の音声が統制条件として記録された) と同型の事故を防ぐ。

(4) **「設定」と「実装」と「入力」を別々に照合する (2026-09-17 / ラウンド 3)。**
    ラウンド 2 の指紋は旗で渡した設定しか見ていなかったので、csm_splice.py の
    splice 実装を書き換えても、あるいは実行中に splice_synthesize を別の関数に
    差し替えても、ハッシュが 1 ビットも動かなかった。同じコマンドで再開すると
    前の実装で作った音声をそのまま流用し、results.json だけ今回の run_id で
    書き直される。LongCat の事故を「旗の付け忘れ」ではなく「コード編集」の側から
    再現できる状態だった。照合を 3 層に分ける:

      a. 設定 (audio_fp_static_hash) … 旗で渡した値。食い違えば **停止**。
         別条件のディレクトリを指してしまった可能性が高く、直すべきは人間の
         コマンドだから。
      b. 実装 (code_fp_hash) … csm_splice.py の AST ハッシュ + 音を作る呼び出し
         (synthesize / splice_synthesize / save_wav / build_ref / load_csm /
         PrefillSplice.__call__ など) の **実行時の実体**のバイトコードハッシュ。
         食い違っても停止しない。食い違った指紋で書かれた wav を信用しない
         = 退避して **作り直す**。実装が変わったなら音は作り直すのが正しく、
         止めても仕方がないから。必ず warnings に残る。
      c. 入力 (input_fp) … その wav を作った文そのもの (BL=text_a、TGT=text_b、
         INJ=text_a/word_a/text_b/word_b + layer) と話者・参照音声のハッシュ。
         wav 1 本ごとに台帳へ書き、照合する。データ側が差し替わった、対の順番が
         変わった、driver が別の文を渡すようになった、のいずれでも発火する。

(5) **実行の最後にディスクと記録を突き合わせる (2026-09-17 / ラウンド 3)。**
    不変条件: **実行が終わった時点で wav_dir に残る s####_{c}.wav は、
    results.json がその条件を ok と記録した対のものだけ。**
    ラウンド 2 の退避はディスク上の wav が「今回の指紋か」しか見ていなかったので、
    層も seed も同じまま INJ だけが落ちた再実行では、前回の INJ wav が残ったまま
    採点され、results.json は同じ条件を「失敗」と記録していた。採点器
    (phoneme_per_v3.py) はファイル名しか見ないので、記録と採点が食い違う。
    実行の最後に scope の全対 x 3 条件を走査し、
      * ok と記録していないのに wav がある -> 退避する (削除しない)
      * ok と記録したのに wav が無い -> その条件を error に落とす
    そのうえで n_scorable_pairs (ディスク上で 3 本そろう対) を数え、
    3 条件とも ok の対の数と食い違えば warnings に残す。

**source_in_bl_rate / target_in_inj_rate を必ず記録する。**
source_in_bl_rate が 0 なら、そもそも BL が text_a を喋れていない = 生成が成立して
いない。CosyVoice2 の旧実験はここで破綻していた。音素誤り率だけを見ると気付けない。

統制:
  * seed を BL / TGT / INJ の generate 直前に必ずリセットする (csm_splice.seed_all)。
    CSM は backbone と depth decoder の両方が既定でサンプリングする。
  * 3 条件で復号設定 (do_sample / temperature / depth_* / max_new_tokens) と
    prompt 構築を完全に共有する。BL と INJ は prompt が 1 トークンも違わない。
    差は prefill 中の splice だけ (実行時に assert する)。
  * 参照音声ターンは全対で固定。CSM は文脈が無いと話者が毎回変わるので、
    BL / TGT / INJ で声が揺れると音素誤り率の比較に話者要因が混ざる。

再開:
  中断しても **同じコマンド**で再開できる。{out_dir}/progress.json に済んだ行を、
  {wav_dir}/_wav_manifest.json に音声の出所を書き、両方そろっている対だけ飛ばす。
  設定が食い違えば停止する。承知のうえで続けるなら --force_resume_mismatch、
  作り直すなら --overwrite。

実行例 (GPU):
  SLURM_TIME=06:00:00 slurm_run.sh \\
    python3 scripts/csm/run_csm_eval.py --lang en --n 300 --layer 14 \\
      --wav_dir eval_results/csm_en_n300_L14/wav \\
      --out_dir eval_results/csm_en_n300_L14 --allow_download
"""

# 2026-09-27: CPU スレッド上限（1 本あたり 13）。torch などを読む前に設定する。
import os as _os_cap
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os_cap.environ[_v] = "13"
try:
    import torch as _torch_cap
    _torch_cap.set_num_threads(13)
except Exception:
    pass
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
import uuid

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "csm"))

import csm_splice as C  # noqa: E402  (HF_HOME はこの import で代入上書きされる)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Table-3 の全アーキテクチャで共有している自然音声の参照
# (speechsplice_release/configs/voice_prompts.yaml と同じもの)。
# CSM は 24 kHz なので csm_splice.load_ref_audio が読み直す (元は 44.1 kHz)。
DEFAULT_REFS = {
    "en": (os.path.join(PROJECT_ROOT, "eval_results", "_natural_refs_norm", "en_natural_norm.wav"),
           "Let me see, I'll give them a new pair of boots every Christmas."),
}

COND = ("bl", "inj", "tgt")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pairs(lang, n):
    """Table 3 (cross-architecture) で既存モデルが使っているデータそのもの."""
    if lang != "en":
        raise SystemExit(
            f"--lang {lang} は対象外。sesame/csm-1b のモデルカードの言語タグは en のみ。\n"
            f"  ja / zh を回しても読みが崩れて音素誤り率の比較が成立しない可能性が高い。\n"
            f"  どうしても回すなら、先に小規模で成立するか確かめること。")
    from en_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()[:n]
    for i, p in enumerate(pairs):
        p.setdefault("pair_idx", i)
    return pairs


def wav_paths(wav_dir, pidx):
    return {c: os.path.join(wav_dir, f"s{pidx:04d}_{c}.wav") for c in COND}


def wavs_complete(wav_dir, pidx):
    """ファイルが 3 本あるかどうかだけ。**これを再開の判定に使ってはならない。**

    ファイルの有無は「どの設定で作られたか」を何も言わない。判定には
    wav_dir の台帳 (_wav_manifest.json) と照合した trusted 集合を使うこと。
    """
    return all(os.path.isfile(p) for p in wav_paths(wav_dir, pidx).values())


_WAV_RE = re.compile(r"^s(\d{4})_(bl|inj|tgt)\.wav$")


def scan_wav_dir(wav_dir):
    """[(pair_idx, cond, path), ...]。退避済み (_stale_*) は拾わない."""
    out = []
    if not os.path.isdir(wav_dir):
        return out
    for name in sorted(os.listdir(wav_dir)):
        m = _WAV_RE.match(name)
        if m:
            out.append((int(m.group(1)), m.group(2), os.path.join(wav_dir, name)))
    return out


# --------------------------------------------------------------------------
# ASR (source_in_bl / target_in_inj)
# --------------------------------------------------------------------------
class Asr:
    """Whisper で書き起こして実体語が入っているかを判定する.

    eval_qwen3_crossculture_acoustic_fixed.py と同じ WhisperAligner を読むので、
    既存の decoder-only 3 例と同じ ASR モデル・同じ判定になる。
    """

    def __init__(self, device="cuda", model_id="openai/whisper-large-v3"):
        from forced_alignment_utils import WhisperAligner
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.aligner = WhisperAligner(device=device, model_id=model_id, dtype=dtype)
        self.model_id = model_id

    def transcribe(self, audio, sr, lang="en"):
        chunks, _ = self.aligner.transcribe_words(audio, sr, language=lang)
        return " ".join(c.get("word", "") for c in chunks).strip()


def transcribe_cond(asr, cond, audio, sr, lang, pair=None):
    """1 条件ぶんの書き起こし。**書き起こしの入口をここ 1 箇所にまとめてある。**

    生成経路と --asr_only 経路で必ずこの関数を通す。条件名を引数に取るのは、
    どの条件の音を書き起こしているかがログと検証で追えるようにするため。
    """
    return asr.transcribe(audio, sr, lang=lang)


def in_text(name, transcript, lang="en"):
    """実体語が書き起こしに入っているか (eval_qwen3_crossculture_acoustic_fixed.py:215-218 と同じ)."""
    if not transcript:
        return False
    if lang in ("ja", "zh"):
        return name in transcript.replace(" ", "")
    return name.lower() in transcript.lower()


# --------------------------------------------------------------------------
# 集計
#
# **率の分母は必ず「試行した対の総数」(n_attempted)**。生き残った対だけで割ると、
# 生成が落ちた対がまるごと消えて率が勝手に上がる。旧版 PIB はこれで 300 対のうち
# 182 対を黙って捨てたまま「有効 300 対」として通していた。
#
# 率は「行が ok の対」ではなく「その条件の生成が成立した対」から数える。
# INJ が落ちても BL が成立していれば source_in_bl には数える (ラウンド 2 の修正)。
# --------------------------------------------------------------------------
# 除外率がこれを超えたら summary["warnings"] に文字列を残す (既定 5%)。
DEFAULT_MAX_DROP_RATE = 0.05


def _rate(rows, key, denom):
    """True の個数 / denom。denom は「試行した対の総数」を渡すこと.

    denom が 0 なら None。rows に値が無い行があっても分母は縮まない。

    **1 対も測れていないときは 0.0 ではなく None を返す (ラウンド 3 の修正)。**
    --no_asr や ASR が全滅した実行で source_in_bl_rate=0.0 と書かれると、
    「BL が word_a を読めていない」と読めてしまう。実際には測っていないだけ。
    測っていないことは None と n_transcribed_by_cond で表し、warnings にも残す。
    """
    d = int(denom)
    if d <= 0:
        return None
    if not any(r.get(key) is not None for r in rows):
        return None
    return round(sum(1 for r in rows if r.get(key) is True) / float(d), 4)


def _rate_survivors(rows, key):
    """生き残りだけを分母にした率 (参考値。主たる率ではない)."""
    vs = [r for r in rows if r.get(key) is not None]
    if not vs:
        return None
    return round(sum(1 for r in vs if r[key]) / float(len(vs)), 4)


def _dedupe(seq):
    """順番を保って重複を落とす (同じ照合を 2 回走らせるため)."""
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# その条件の wav が無ければ意味を失う判定 (採点器が採点できない音の判定を
# 率に混ぜない)。ラウンド 2 は prompt_len 検査で落ちた対 (wav を 1 本も保存
# しない) でも source_in_bl=True が残り、source_in_bl_rate=1.0 になっていた。
COND_JUDGEMENTS = {"bl": ("source_in_bl",), "tgt": ("target_in_tgt",),
                   "inj": ("target_in_inj", "source_in_inj")}


def drop_cond_judgements(row, cond):
    """その条件の書き起こしと判定を row から落とす (値を残さない)."""
    for k in COND_JUDGEMENTS[cond]:
        row.pop(k, None)
    row.pop(f"{cond}_transcript", None)


def _cond_ok(r, c):
    """その条件の生成が成立したか。条件別の記録が無い古い行も扱える."""
    v = r.get(f"{c}_status")
    if v is not None:
        return v == "ok"
    if r.get(f"{c}_codes_fp") is not None or r.get(f"{c}_sec") is not None:
        return True
    return r.get("status") == "ok"


def summarize(rows, args, extra=None, attempted=None):
    """attempted = 試行した対の pair_idx のリスト (= 率の分母)."""
    if attempted is None:
        # 呼び忘れ防止。rows しか無いときは rows を試行総数とみなす。
        attempted = [int(r["pair_idx"]) for r in rows if "pair_idx" in r]
    attempted = sorted({int(i) for i in attempted})
    n_att = len(attempted)
    att_set = set(attempted)

    in_scope = [r for r in rows if int(r.get("pair_idx", -1)) in att_set]
    have_row = {int(r["pair_idx"]) for r in in_scope if "pair_idx" in r}
    ok = [r for r in in_scope if r.get("status") == "ok"]
    err = [r for r in in_scope if r.get("status") == "error"]
    n_missing = n_att - len(have_row)        # 行すら無い (中断で未着手)
    n_failed = n_att - len(ok)               # 落ちた + 未着手
    drop_rate = round(n_failed / float(n_att), 4) if n_att else None

    # ---- 条件別の母数 ----------------------------------------------------
    # 「その条件の生成が成立した対の数」。行が error でも BL だけは成立している、
    # という状態を正しく数える。未着手の対は 3 条件とも失敗として数える。
    n_ok_by_cond = {c: sum(1 for r in in_scope if _cond_ok(r, c)) for c in COND}
    fail_by_cond = {c: n_att - n_ok_by_cond[c] for c in COND}
    # 3 条件とも成立した対 = 採点器が採点してよい対。ディスク上の本数
    # (n_scorable_pairs) と食い違ったら、記録と採点が別のものを見ている。
    n_all_cond_ok = sum(1 for r in in_scope if all(_cond_ok(r, c) for c in COND))

    # どの run が書いた行なのか。再開で前の run の音声を使い回している対がある
    # ことは必ず見えるようにする (ラウンド 2 は summary.run_id と rows[*].run_id の
    # 食い違いだけが痕跡で、n_skip はどこにも書かれていなかった)。
    rows_by_run = {}
    for r in in_scope:
        k = str(r.get("run_id"))
        rows_by_run[k] = rows_by_run.get(k, 0) + 1

    stages = {}
    for r in err:
        k = str(r.get("fail_stage") or "unknown")
        stages[k] = stages.get(k, 0) + 1
    if n_missing:
        stages["not_attempted"] = n_missing

    n_transcribed = {c: sum(1 for r in in_scope if r.get(f"{c}_transcript") is not None)
                     for c in COND}
    n_asr_failed = sum(1 for r in in_scope if r.get("asr_fail_conds"))

    def fstat(key):
        vs = [r[key] for r in in_scope if r.get(key) is not None]
        if not vs:
            return None
        return {"mean": round(float(np.mean(vs)), 3), "min": round(float(np.min(vs)), 3),
                "max": round(float(np.max(vs)), 3), "n": len(vs)}

    def cond_count(suffix, want=True):
        return {c: sum(1 for r in in_scope if r.get(f"{c}_{suffix}") is want) for c in COND}

    def stop_reasons():
        out = {}
        for c in COND:
            d = {}
            for r in in_scope:
                v = r.get(f"{c}_stop_reason")
                if v is not None:
                    d[str(v)] = d.get(str(v), 0) + 1
            out[c] = d
        return out

    s = {
        "model": "sesame/csm-1b",
        "lang": args.lang, "layer": args.layer, "seed": args.seed,
        # ---- 母数 (率の分母はここ) ----
        "n_requested": args.n,
        "n_attempted": n_att,
        "n_ok": len(ok),
        "n_failed": n_failed,
        "n_error": len(err),
        "n_not_attempted": n_missing,
        "n_rows": len(rows),
        "n_asr_scored": n_transcribed["bl"],
        "n_transcribed_by_cond": n_transcribed,
        "n_asr_failed": n_asr_failed,
        "drop_rate": drop_rate,
        "rate_denominator": "n_attempted",
        "fail_by_stage": stages,
        "fail_by_cond": fail_by_cond,
        "n_ok_by_cond": n_ok_by_cond,
        "n_pairs_all_cond_ok": n_all_cond_ok,
        "n_rows_by_run_id": rows_by_run,
        # ---- 生成が成立しているかの検査 (0 なら音素誤り率を見る意味がない) ----
        "source_in_bl_rate": _rate(in_scope, "source_in_bl", n_att),
        "target_in_tgt_rate": _rate(in_scope, "target_in_tgt", n_att),
        "target_in_inj_rate": _rate(in_scope, "target_in_inj", n_att),
        "source_in_inj_rate": _rate(in_scope, "source_in_inj", n_att),
        # ---- 参考 1: 生き残りだけを分母にした率 ----
        "source_in_bl_rate_survivors": _rate_survivors(in_scope, "source_in_bl"),
        "target_in_tgt_rate_survivors": _rate_survivors(in_scope, "target_in_tgt"),
        "target_in_inj_rate_survivors": _rate_survivors(in_scope, "target_in_inj"),
        "source_in_inj_rate_survivors": _rate_survivors(in_scope, "source_in_inj"),
        # ---- 参考 2: その条件の生成が成立した対だけを分母にした率 ----
        "rates_given_generated": {
            "source_in_bl": _rate(in_scope, "source_in_bl", n_ok_by_cond["bl"]),
            "target_in_tgt": _rate(in_scope, "target_in_tgt", n_ok_by_cond["tgt"]),
            "target_in_inj": _rate(in_scope, "target_in_inj", n_ok_by_cond["inj"]),
            "source_in_inj": _rate(in_scope, "source_in_inj", n_ok_by_cond["inj"]),
        },
        # ---- splice が本当に掛かったか ----
        "n_fired_1": sum(1 for r in in_scope if r.get("fired") == 1),
        "n_inj_differs_from_bl": sum(1 for r in in_scope if r.get("inj_differs_from_bl")),
        "n_equal_len": sum(1 for r in in_scope if r.get("equal_len")),
        "n_mean_pool": sum(1 for r in in_scope if r.get("splice_mode") == "mean_pool_broadcast"),
        # ---- 生成長 (3 条件とも同じ形で出す) ----
        "sec_bl": fstat("bl_sec"), "sec_inj": fstat("inj_sec"), "sec_tgt": fstat("tgt_sec"),
        "frames_bl": fstat("bl_frames"), "frames_inj": fstat("inj_frames"),
        "frames_tgt": fstat("tgt_frames"),
        "n_clipped_wavs": {c: sum(1 for r in in_scope
                                  if (r.get(f"{c}_n_clipped") or 0) > 0) for c in COND},
        "n_hit_max": cond_count("hit_max"),
        "n_truncated_at_cap": cond_count("truncated_at_cap"),
        "n_no_eos": cond_count("eos_found", want=False),
        "stop_reason": stop_reasons(),
        "kappa_at_splice_mean": None,
    }
    s["n_hit_max_total"] = sum(s["n_hit_max"].values())
    s["n_truncated_at_cap_total"] = sum(s["n_truncated_at_cap"].values())
    ks = [r["kappa_at_splice"] for r in in_scope if r.get("kappa_at_splice") is not None]
    if ks:
        s["kappa_at_splice_mean"] = round(float(np.mean(ks)), 4)
        s["kappa_at_splice_std"] = round(float(np.std(ks)), 4)
    if extra:
        s.update(extra)

    # 「3 本そろった対の数」は監査スクリプトが n_scorable_triplets という名前で
    # 探しているので、同じ数を両方の名前で出す (1 対 = wav 3 本 = triplet)。
    if s.get("n_scorable_pairs") is not None:
        s["n_scorable_triplets"] = s["n_scorable_pairs"]

    # ---- 前の run から使い回している対 ----
    cur_run = str(s.get("run_id"))
    s["n_rows_from_other_runs"] = sum(v for k, v in rows_by_run.items() if k != cur_run)

    # ---- warnings: 数字で追えるよう文字列で summary に残す ----
    w = list(s.get("warnings", []))
    thr = float(getattr(args, "max_drop_rate", DEFAULT_MAX_DROP_RATE))
    # 1 対も測れていない率を挙げる (None は「0」ではなく「測っていない」)
    unmeasured = [k for k in ("source_in_bl_rate", "target_in_tgt_rate",
                              "target_in_inj_rate", "source_in_inj_rate")
                  if s.get(k) is None]
    if unmeasured and n_att:
        w.append(f"{unmeasured} は 1 対も測れていないので None にしてある "
                 f"(書き起こしの本数 {n_transcribed}、run_asr="
                 f"{getattr(args, 'run_asr', True)})。0.0 と書くと『読めていない』と"
                 f"読めてしまうので区別している。")
    # rows に居るが今回の --n の外にある対 (--n を減らして再開したとき)
    n_out = len(rows) - len(in_scope)
    s["n_rows_out_of_scope"] = n_out
    if n_out:
        w.append(f"results.json には今回の --n={args.n} の外の対が {n_out} 行 残っている "
                 f"(前回より --n を減らした)。summary の母数には入れていないが、"
                 f"wav_dir には その対の wav も残っているので、採点は必ず "
                 f"--n {args.n} を付けて掛けること。")
    if s["n_rows_from_other_runs"]:
        w.append(
            f"{s['n_rows_from_other_runs']}/{n_att} 対は今回の run ({cur_run}) では"
            f"生成しておらず、前の run の音声をそのまま使っている "
            f"(run_id 別の行数 {rows_by_run})。設定・実装・入力は照合済みだが、"
            f"『この実行が何対を実際に作ったか』は n_generated_this_run を見ること。")
    spread = (s.get("wav_provenance") or {}).get("wavs_with_other_fingerprint") or {}
    if spread:
        w.append(
            f"この wav_dir の台帳には、今回と違う指紋で書かれた記録が残っている "
            f"{ {k: len(v) for k, v in spread.items()} }。"
            f"台帳の見出しは今回の指紋なので、見出しだけ見ると全部同じ設定に見える。"
            f"1 本ごとの照合は効いているが、この wav_dir を別の --n で採点すると"
            f"設定が混ざる。")
    if s.get("n_scorable_pairs") is not None and s["n_scorable_pairs"] != n_all_cond_ok:
        w.append(
            f"採点器が採点できる対 (wav が 3 本そろう対) が {s['n_scorable_pairs']} で、"
            f"3 条件とも成立したと記録した対 {n_all_cond_ok} と 食い違う。"
            f"results.json と wav_dir が別のものを指している。")
    # 閾値ちょうど (15/300 = 0.05) でも出す。「超えたら」を厳密な > にすると
    # 境界がすり抜ける。drop_rate が 0 のときは --max_drop_rate 0 でも出さない。
    if n_att and drop_rate is not None and drop_rate > 0 and drop_rate >= thr:
        w.append(
            f"除外率 {drop_rate:.1%} ({n_failed}/{n_att} 対) が閾値 {thr:.1%} に達した。"
            f"内訳 {stages}。率は n_attempted={n_att} を分母にしてある。"
            f"条件ごとに除外率が違うと比較が歪むので fail_by_cond を確認すること。")
    if n_missing:
        w.append(f"{n_missing}/{n_att} 対は行すら無い (未着手または中断)。"
                 f"率の分母には数えてある。")
    if getattr(args, "run_asr", True) and n_att and n_transcribed["bl"] < n_att:
        w.append(f"BL の書き起こしが付いたのは {n_transcribed['bl']}/{n_att} 対 "
                 f"(条件別 {n_transcribed})。*_rate は n_attempted 分母。")
    if n_asr_failed:
        w.append(f"{n_asr_failed}/{n_att} 対で ASR 自体が落ちた。生成は成立していても "
                 f"判定が付いていないので率が下振れする。")
    nz = [v for v in fail_by_cond.values() if v]
    if nz and len(set(fail_by_cond.values())) > 1:
        w.append(f"条件ごとの失敗数がそろっていない: {fail_by_cond} "
                 f"(生成が成立した対 {n_ok_by_cond})。条件間で分母が違うまま比較しないこと。")
    n_clip = sum(s["n_clipped_wavs"].values())
    if n_clip:
        w.append(f"保存時に振幅が [-1, 1] に丸められた wav が {s['n_clipped_wavs']} 件。"
                 f"ASR は丸める前の配列を、採点器は丸めたあとの wav を読むので、"
                 f"その対は判定と採点で聞いている音が違う。")
    if s["n_truncated_at_cap_total"]:
        w.append(f"max_new_tokens={getattr(args, 'max_new_tokens', None)} に到達して "
                 f"EOS が出なかった生成が {s['n_truncated_at_cap']} 件。"
                 f"文の途中で切れた音声が混ざっている。")
    if n_ok_by_cond["inj"] and s["n_fired_1"] != n_ok_by_cond["inj"]:
        w.append(f"splice hook の fired==1 が {s['n_fired_1']}/{n_ok_by_cond['inj']} "
                 f"(INJ の生成が成立した対を分母にした)。")
    if n_ok_by_cond["inj"] and n_ok_by_cond["bl"]:
        n_cmp = sum(1 for r in in_scope if r.get("inj_differs_from_bl") is not None)
        if n_cmp and s["n_inj_differs_from_bl"] != n_cmp:
            w.append(f"{n_cmp - s['n_inj_differs_from_bl']}/{n_cmp} 対で "
                     f"INJ の生成コードが BL と完全に同じ。hook が効いていない疑い。")
    sb = s.get("source_in_bl_rate")
    if sb is not None and sb < 0.5:
        n_true = sum(1 for r in in_scope if r.get("source_in_bl") is True)
        w.append(
            f"source_in_bl_rate={sb} (n_attempted={n_att} 分母)。"
            f"内訳: BL の生成が成立したのは {n_ok_by_cond['bl']}/{n_att} 対、"
            f"うち書き起こしに word_a があったのは {n_true} 対 "
            f"(生成が成立した対だけを分母にすると "
            f"{s['rates_given_generated']['source_in_bl']})。"
            f"『BL が喋れていない』のか『BL の生成自体が落ちている』のかは "
            f"fail_by_cond / n_ok_by_cond で切り分けること。")
    # TGT にも BL と同じ健全性検査を掛ける。TGT が word_b を読めていなければ
    # target_in_inj の比較そのものが成立しない (ラウンド 2 は BL にしか検査が無かった)。
    tt = s.get("target_in_tgt_rate")
    if tt is not None and tt < 0.5:
        n_true = sum(1 for r in in_scope if r.get("target_in_tgt") is True)
        w.append(
            f"target_in_tgt_rate={tt} (n_attempted={n_att} 分母)。"
            f"内訳: TGT の生成が成立したのは {n_ok_by_cond['tgt']}/{n_att} 対、"
            f"うち書き起こしに word_b があったのは {n_true} 対 "
            f"(生成が成立した対だけを分母にすると "
            f"{s['rates_given_generated']['target_in_tgt']})。"
            f"TGT が目標語を読めていないなら target_in_inj の比較は成立しない。")
    if s.get("target_in_inj_rate") == 0.0 and n_ok_by_cond["inj"]:
        w.append("target_in_inj_rate=0。splice が音として効いていない可能性。")
    s["warnings"] = w
    return s


def report(summary):
    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"[母数] n_attempted={summary.get('n_attempted')} "
          f"n_ok={summary.get('n_ok')} n_failed={summary.get('n_failed')} "
          f"(error={summary.get('n_error')} 未着手={summary.get('n_not_attempted')}) "
          f"drop_rate={summary.get('drop_rate')}", flush=True)
    print(f"[母数] 条件別に生成が成立した対 {summary.get('n_ok_by_cond')} "
          f"/ 失敗 {summary.get('fail_by_cond')} / 段階別 {summary.get('fail_by_stage')}",
          flush=True)
    print(f"[母数] 率の分母 = {summary.get('rate_denominator')} "
          f"(生き残り分母は *_rate_survivors、条件別分母は rates_given_generated)",
          flush=True)
    print(f"[実行] この run で生成した対={summary.get('n_generated_this_run')} "
          f"再開で飛ばした対={summary.get('n_resumed_skipped')} "
          f"前の run の音声を使っている対={summary.get('n_rows_from_other_runs')} "
          f"(run_id 別 {summary.get('n_rows_by_run_id')})", flush=True)
    print(f"[採点] wav が 3 本そろう対={summary.get('n_scorable_pairs')} "
          f"/ 3 条件とも ok と記録した対={summary.get('n_pairs_all_cond_ok')}",
          flush=True)
    prov = summary.get("wav_provenance") or {}
    if prov:
        print(f"[音声] wav_dir={prov.get('wav_dir')} "
              f"設定={prov.get('audio_fp_static_hash')}/{prov.get('audio_fp_full_hash')} "
              f"実装={prov.get('code_fp_hash')} "
              f"(照合={prov.get('code_fp_checked')}) "
              f"照合済み={prov.get('n_trusted')} 退避={prov.get('n_quarantined')}",
              flush=True)
    for msg in summary.get("warnings", []):
        print(f"[warn] {msg}", flush=True)


# --------------------------------------------------------------------------
# 実行設定の指紋 (再開時の照合)
#
# 2026-09-17 の LongCat 統制実験で、--splice_part hL_only を付け忘れたまま
# 出力先の名前だけ統制用にしたため、本条件と bit 単位で同一の音声が
# 「統制条件の結果」として記録された (md5 照合で発覚。ΔPER が小数点以下まで
# 一致したのが手がかり)。同じ事故が層掃引でも起きる: out_dir を分け忘れて
# --layer だけ変えると、wav は前の層のまま results.json だけ新しい層番号になる。
#
# 対策は 2 段構え:
#   * out_dir/progress.json  … 再開する行が今回の設定で作られたかを照合する
#   * wav_dir/_wav_manifest.json … **採点される音声そのもの**の出所を照合する
# どちらも食い違えば停止する。黙って無視しない。
# --------------------------------------------------------------------------
# 静的な鍵 (モデルを読む前に決まる)。ここが欠けている progress.json は
# 指紋を持たない旧版の産物なので、そのままでは再開させない。
CORE_FP_KEYS = (
    "lang", "layer", "seed", "do_sample", "temperature", "depth_temperature",
    "depth_top_k", "max_new_tokens", "min_sec", "fail_on_cap",
    "model_id", "dtype", "attn_implementation",
    "speaker", "ref_mode", "ref_sha256", "ref_text_sha256", "ref_max_sec",
    "splice_target", "splice_len_policy", "asr_model", "run_asr",
    "wav_dir", "code_fp_hash",
)

# 版の違いだけでは停止しない鍵 (warnings には残す)。
#   * transformers_version  … 版の差
#   * device_type           … CPU で採点し直すことがある (生成時は台帳側で照合する)
#   * driver_ast            … run_csm_eval.py 自身。集計コードを直しただけで
#                             300 対 x 3 条件を作り直すのは割に合わない。
#                             driver が別の文を渡すようになった場合は、文そのものの
#                             ハッシュ (input_fp) が wav 1 本ごとに効くので捕まる。
SOFT_FP_KEYS = ("transformers_version", "device_type", "driver_ast")

# 実装の指紋。**食い違っても停止しない**が、その指紋で書かれていない wav は
# 信用せず退避して作り直す (下の audit_wav_dir)。停止しないのは、実装が変わった
# なら音は作り直すのが正しく、人間に直させることが何も無いから。必ず warnings に残る。
CODE_FP_KEYS = ("code_fp_hash", "code_fp", "splice_impl_sha256")

# 音声のビットを決める鍵。ASR 関係と out_dir は入れない (書き起こしは音を変えない)。
# **実装の指紋 (code_fp_hash) はここに入れない。** ここに入れると
# check_manifest_header が停止してしまい、「作り直す」ができなくなる。
AUDIO_FP_STATIC_KEYS = (
    "lang", "layer", "seed", "do_sample", "temperature", "depth_temperature",
    "depth_top_k", "max_new_tokens", "min_sec", "fail_on_cap",
    "model_id", "dtype", "attn_implementation",
    "speaker", "ref_mode", "ref_sha256", "ref_text_sha256", "ref_max_sec",
    "splice_target", "splice_len_policy",
)
# モデルを読んで初めて分かる鍵。--asr_only では照合できない。
AUDIO_FP_DYNAMIC_KEYS = ("model_revision", "num_layers", "sampling_rate",
                         "num_codebooks", "device_type")


def _sha256_file(path, limit_mb=64):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if n >= limit_mb * (1 << 20):
                break
    return h.hexdigest()[:16]


def _sha256_text(s):
    return hashlib.sha256(("" if s is None else str(s)).encode("utf-8")).hexdigest()[:16]


def _sha256_obj(d):
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _tf_version():
    try:
        import transformers
        return str(transformers.__version__)
    except Exception:
        return None


# --------------------------------------------------------------------------
# 実装の指紋 (code fingerprint)
#
# 旗で渡した設定だけを指紋にすると、**コードを書き換えても指紋が動かない**。
# csm_splice.py の splice 実装を差し替えても、実行中に C.splice_synthesize を
# 別の関数に置き換えても、ラウンド 2 の指紋は 1 ビットも変わらなかった。
# 同じコマンドで再開すると前の実装で作った wav をそのまま流用できてしまう
# (2026-09-17 の LongCat 事故を、旗ではなくコード編集の側から再現できる)。
#
# 2 通りの書き換えを両方とも捕まえる:
#   * ファイルを書き換えた      -> csm_splice.py の AST ハッシュが動く
#   * 実行時に関数を差し替えた  -> 呼び出し実体のバイトコードハッシュが動く
# AST は注釈 (comment) と docstring を落としてから取るので、コメントを直した
# だけで 900 本の wav が作り直しになることはない。実行される中身が変われば動く。
# --------------------------------------------------------------------------
# 音を作る呼び出し。実行時に csm_splice から引き直す (差し替えを見るため)。
AUDIO_IMPL_FUNCS = (
    "synthesize", "splice_synthesize", "save_wav", "build_ref", "load_csm",
    "load_ref_audio", "expected_audio_frames", "seed_all", "codes_fingerprint",
    "assert_splice_target",
)
AUDIO_IMPL_METHODS = (
    ("PrefillSplice", "__init__"), ("PrefillSplice", "__call__"),
    ("GenConfig", "generate_kwargs"),
)


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _ast_sha256(path):
    """注釈と docstring を落とした AST のハッシュ。整形やコメントでは動かない."""
    if not path or not os.path.isfile(path):
        return f"<missing:{path}>"
    try:
        with open(path, "rb") as f:
            tree = _strip_docstrings(ast.parse(f.read()))
        return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()[:16]
    except Exception:
        # parse できないものは生のバイト列で見る (無言で諦めない)
        return "raw:" + _sha256_file(path)


def _code_bytes(code, depth=0):
    parts = [code.co_code, repr(code.co_names).encode("utf-8"),
             repr(code.co_varnames).encode("utf-8")]
    for c in code.co_consts:
        if hasattr(c, "co_code") and depth < 6:
            parts.append(_code_bytes(c, depth + 1))       # 内包表記・入れ子の関数
        else:
            parts.append(repr(c).encode("utf-8"))
    return b"|".join(parts)


def _callable_id(fn):
    """呼び出し実体の同一性。差し替えると qualname かバイトコードが変わる."""
    if fn is None:
        return "<none>"
    f = getattr(fn, "__func__", fn)
    name = (f"{getattr(f, '__module__', '?')}."
            f"{getattr(f, '__qualname__', getattr(f, '__name__', type(f).__name__))}")
    code = getattr(f, "__code__", None)
    if code is None:                                   # C 実装 / functools など
        return name + "|<nocode>"
    return f"{name}|{hashlib.sha256(_code_bytes(code)).hexdigest()[:12]}"


def code_fingerprint():
    """音のビットを決めるコードの指紋 (ファイルの AST + 呼び出し実体)."""
    d = {
        "csm_splice_ast": _ast_sha256(getattr(C, "__file__", None)),
        "pairs_data_ast": None,
    }
    try:
        import en_cross_culture_names as _D
        d["pairs_data_ast"] = _ast_sha256(getattr(_D, "__file__", None))
    except Exception as e:
        d["pairs_data_ast"] = f"<import失敗:{type(e).__name__}>"
    for nm in AUDIO_IMPL_FUNCS:
        d[f"fn:{nm}"] = _callable_id(getattr(C, nm, None))
    for cls, meth in AUDIO_IMPL_METHODS:
        d[f"fn:{cls}.{meth}"] = _callable_id(getattr(getattr(C, cls, None), meth, None))
    return d


def driver_ast():
    """run_csm_eval.py 自身の AST ハッシュ (記録と警告のみ。停止も退避もしない)."""
    return _ast_sha256(os.path.abspath(__file__))


def cond_input_fp(pair, cond, fp):
    """その wav を作った入力そのもののハッシュ (wav 1 本ごとに台帳へ書く).

    driver が別の文を渡すようになった / データ側が差し替わった / 対の順番が
    変わった、のいずれでも動く。実装の指紋とは独立に効く。
    """
    if cond == "bl":
        payload = ["bl", pair["text_a"]]
    elif cond == "tgt":
        payload = ["tgt", pair["text_b"]]
    else:
        payload = ["inj", pair["text_a"], pair["word_a"],
                   pair["text_b"], pair["word_b"], int(fp.get("layer", -1))]
    return _sha256_obj({
        "pair_idx": int(pair["pair_idx"]), "cond": cond, "payload": payload,
        "speaker": fp.get("speaker"), "ref_mode": fp.get("ref_mode"),
        "ref_sha256": fp.get("ref_sha256"), "ref_text_sha256": fp.get("ref_text_sha256"),
        "ref_max_sec": fp.get("ref_max_sec"),
    })


def want_input_fps(pairs, fp):
    return {(int(p["pair_idx"]), c): cond_input_fp(p, c, fp)
            for p in pairs for c in COND}


def static_fingerprint(args):
    """モデルを読む前に決まる分の指紋。GPU を掴む前に照合できる."""
    ref_sha = None
    ref_text = None
    if args.ref_mode == "clone":
        rw, rt = DEFAULT_REFS[args.lang]
        rw = args.ref_wav or rw
        ref_text = args.ref_text if args.ref_text is not None else rt
        ref_sha = _sha256_file(rw) if os.path.isfile(rw) else f"<missing:{rw}>"
    code_fp = code_fingerprint()
    return {
        "lang": args.lang,
        "layer": int(args.layer),
        "seed": int(args.seed),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "depth_temperature": float(args.depth_temperature),
        "depth_top_k": int(args.depth_top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "min_sec": float(args.min_sec),
        "fail_on_cap": bool(args.fail_on_cap),
        "model_id": str(args.model_id),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        # CPU と GPU では同じ seed でもビットが変わる。生成時だけ照合する
        # (--asr_only は生成しないので照合の対象外。SOFT_FP_KEYS にも入れてある)。
        "device_type": str(args.device).split(":")[0],
        "speaker": str(args.speaker),
        "ref_mode": str(args.ref_mode),
        "ref_sha256": ref_sha,
        "ref_text_sha256": _sha256_text(ref_text),
        "ref_max_sec": float(args.ref_max_sec),
        # splice の種類。差し替え先と長さ不一致時の扱いまで含める。
        "splice_target": f"backbone_model.layers[{int(args.layer)}].output",
        "splice_len_policy": "token_wise|mean_pool_broadcast",
        # **splice の種類**。ラウンド 2 の指紋には splice_target (layer から作り直す
        # だけ) と splice_len_policy (書き換わらない定数) しか無く、splice の実装を
        # 変えても 1 ビットも動かなかった。splice を作っている実体そのものを見る。
        "splice_impl_sha256": _sha256_obj(
            {k: v for k, v in code_fp.items()
             if "splice" in k or k == "csm_splice_ast"}),
        # 実装の指紋 (音を作る呼び出し全体)。
        "code_fp_hash": _sha256_obj(code_fp),
        "code_fp": code_fp,
        "driver_ast": driver_ast(),
        "asr_model": str(args.asr_model),
        "run_asr": bool(args.run_asr),
        # 出力先。rows と wav の対応が変わるので照合対象に入れる
        # (参考情報として読み飛ばしていたのがラウンド 1 の穴)。
        "wav_dir": os.path.abspath(args.wav_dir) if args.wav_dir else None,
        # 版は参考 (差があっても停止はしないが warnings に残す)
        "transformers_version": _tf_version(),
    }


def audio_fp_view(fp, dynamic=False):
    keys = AUDIO_FP_STATIC_KEYS + (AUDIO_FP_DYNAMIC_KEYS if dynamic else ())
    return {k: fp.get(k) for k in keys}


def audio_fp_hashes(fp):
    """(静的な指紋のハッシュ, モデル込みの指紋のハッシュ or None)."""
    st = _sha256_obj(audio_fp_view(fp, dynamic=False))
    full = None
    if all(fp.get(k) is not None for k in AUDIO_FP_DYNAMIC_KEYS):
        full = _sha256_obj(audio_fp_view(fp, dynamic=True))
    return st, full


def audio_fp_diff(old_fp, new_fp, dynamic):
    keys = AUDIO_FP_STATIC_KEYS + (AUDIO_FP_DYNAMIC_KEYS if dynamic else ())
    return [(k, old_fp.get(k, "<欠落>"), new_fp.get(k, "<欠落>"))
            for k in keys if old_fp.get(k, "<欠落>") != new_fp.get(k, "<欠落>")]


def compare_fingerprint(old, new):
    """(停止すべき食い違い, 停止しない食い違い, 実装の食い違い) を返す.

    * 双方が値を持つ鍵はすべて照合する (照合対象を core に絞らない)。
    * core の鍵が old に無ければ「欠落」として食い違い扱いにする
      (指紋を持たない旧版の progress.json をそのまま再開させないため)。
    * model_revision のように片側にしか無い動的な鍵は、old に無いだけなら
      食い違いにしない (--asr_only はモデルを読まないため)。
    * wav_dir は **停止すべき食い違い**。ラウンド 1 はここを参考情報として
      読み飛ばしていたため、out_dir を据え置いたまま --wav_dir だけ別条件の
      完成品に向けると全対 skip で通ってしまった。
    * code_fp_hash / code_fp (実装) は **停止しない**。代わりに、その指紋で
      書かれていない wav を信用せず作り直す (audit_wav_dir)。ただし
      「前回と同じ実装だった」と黙って扱うことはせず、必ず warnings に残す。
      code_fp_hash が old に無い (= ラウンド 2 以前の progress.json) 場合だけは
      実装の記録がまったく無いので停止する。
    """
    hard, soft, code = [], [], []
    if not isinstance(old, dict):
        return [("<fingerprint>", "<無し>", "<有り>")], [], []
    for k in CORE_FP_KEYS:
        if k not in old:
            hard.append((k, "<欠落>", new.get(k)))
    for k in sorted(set(old) | set(new)):
        if k in old and k in new and old[k] != new[k]:
            if k in CODE_FP_KEYS:
                code.append((k, old[k], new[k]))
            elif k in SOFT_FP_KEYS:
                soft.append((k, old[k], new[k]))
            else:
                hard.append((k, old[k], new[k]))
    return hard, soft, code


def code_fp_diff(old_fp, new_fp):
    """実装の指紋のうち、どの項目が変わったか (何を書き換えたかが分かるように)."""
    o = (old_fp or {}).get("code_fp") or {}
    n = (new_fp or {}).get("code_fp") or {}
    return [(k, o.get(k, "<欠落>"), n.get(k, "<欠落>"))
            for k in sorted(set(o) | set(n)) if o.get(k, "<欠落>") != n.get(k, "<欠落>")]


def check_resume_fingerprint(prev, fp, n_prev_rows, force):
    """食い違えば SystemExit。force=True なら警告文字列を返して続行."""
    if n_prev_rows == 0:
        return []
    hard, soft, code = compare_fingerprint(prev.get("fingerprint"), fp)
    out = []
    for k, o, n in soft:
        out.append(f"再開時に {k} が変わっている (前回={o!r} 今回={n!r})。停止はしていない。")
        print(f"[resume][warn] {out[-1]}", flush=True)
    if code:
        det = code_fp_diff(prev.get("fingerprint"), fp)
        out.append(
            f"再開しようとしている progress.json は **別の実装**で作られている "
            f"(code_fp_hash 前回={prev.get('fingerprint', {}).get('code_fp_hash')!r} "
            f"今回={fp.get('code_fp_hash')!r})。変わった項目 {[k for k, _o, _n in det]}。"
            f"停止はしないが、前の実装で作った wav は信用せず作り直す。")
        print("[resume][warn] " + out[-1], flush=True)
        for k, o, n in det[:8]:
            print(f"    {k}: 前回={o!r}  今回={n!r}", flush=True)
    if not hard:
        return out
    lines = [f"    {k}: 前回={o!r}  今回={n!r}" for k, o, n in hard]
    body = ("再開しようとしている progress.json の実行設定が今回と食い違う。\n"
            + "\n".join(lines) + "\n"
            f"  前回の行数 = {n_prev_rows}。このまま続けると、前の設定で作った wav を\n"
            "  使い回したまま results.json には今回の設定が書かれる\n"
            "  (2026-09-17 の LongCat 統制実験と同じ事故)。")
    if not force:
        raise SystemExit(
            "[resume] " + body + "\n"
            "  対処: 出力先を分ける (--out_dir と --wav_dir を設定ごとに変える) か、\n"
            "        --overwrite で作り直すか、承知のうえなら\n"
            "        --force_resume_mismatch を付けること。")
    print("[resume][warn] " + body, flush=True)
    out.append(f"再開時に設定が食い違ったまま --force_resume_mismatch で続行した: "
               f"{[(k, o, n) for k, o, n in hard]}")
    return out


# --------------------------------------------------------------------------
# wav_dir の provenance 台帳
#
# **採点されるのは results.json ではなく wav_dir の中身。**
# 指紋を out_dir にしか置かないと、
#   (a) --overwrite で作り直しても、落ちた条件の wav は前の設定のまま残る
#   (b) out_dir だけ層ごとに分けて wav_dir を共有すると照合が一度も走らない
#   (c) out_dir を据え置いたまま --wav_dir を別条件の完成品に向けると全対 skip
# の 3 経路が残る。台帳を wav_dir 側に置き、wav 1 本ごとに
# 「どの指紋のどの run が書いたか」と sha256 を記録して照合する。
#
# 不変条件: **実行が終わった時点で wav_dir にある s####_{bl,inj,tgt}.wav は
# すべて、その実行の指紋で書かれたものとして台帳に載っている。**
# 載っていないものは `_stale_<run_id>_` を付けて退避する (削除はしない)。
# --------------------------------------------------------------------------
WAV_MANIFEST_NAME = "_wav_manifest.json"
MANIFEST_SCHEMA = "csm_wav_manifest_v1"


def wav_key(pidx, cond):
    return f"s{int(pidx):04d}_{cond}"


def manifest_path(wav_dir):
    return os.path.join(wav_dir, WAV_MANIFEST_NAME)


def load_manifest(wav_dir):
    p = manifest_path(wav_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            man = json.load(f)
    except Exception as e:
        raise SystemExit(
            f"[wav_dir] {p} を読めなかった: {e}\n"
            f"  音声の出所が確かめられないので続行しない。\n"
            f"  作り直すなら --overwrite、承知のうえなら台帳を退避してから回すこと。")
    if not isinstance(man, dict) or "wavs" not in man:
        raise SystemExit(f"[wav_dir] {p} の形が台帳ではない。")
    return man


def new_manifest(wav_dir, fp, st_hash, full_hash):
    return {"schema": MANIFEST_SCHEMA, "wav_dir": os.path.abspath(wav_dir),
            "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
            "audio_fingerprint": audio_fp_view(fp, dynamic=True),
            "code_fp_hash": fp.get("code_fp_hash"), "code_fp": fp.get("code_fp"),
            "driver_ast": fp.get("driver_ast"),
            "producers": [], "wavs": {}}


def save_manifest(man, wav_dir):
    p = manifest_path(wav_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, p)


def _code_fp_note(man, fp, args, final):
    """台帳の実装の指紋と今回の実装を突き合わせる。**停止しない**。

    生成するときは、食い違う指紋で書かれた wav を audit_wav_dir が信用せずに
    退避して作り直すので、ここは記録と警告だけでよい。
    --asr_only は生成しないので実装を照合する意味が無い (どの実装で作るべきだった
    かを asr_only は知らない)。ただし「照合していない」ことは必ず書き残す。
    """
    cur = fp.get("code_fp_hash")
    old = man.get("code_fp_hash")
    if old == cur:
        return []
    det = code_fp_diff({"code_fp": man.get("code_fp")}, fp)
    if args.asr_only:
        if not final:
            return []
        return [f"この wav_dir の音声は **別の実装**で作られている "
                f"(台帳の code_fp_hash={old!r} 今回={cur!r}、変わった項目 "
                f"{[k for k, _o, _n in det][:8]})。--asr_only は生成しないので "
                f"実装は照合していない。results.json の fingerprint には今回の "
                f"実装が書かれるので、音声と食い違う点に注意すること。"]
    msg = (f"wav_dir の音声は別の実装で作られている (台帳の code_fp_hash={old!r} "
           f"今回={cur!r}、変わった項目 {[k for k, _o, _n in det][:8]})。"
           f"停止はせず、その指紋で書かれた wav は信用せずに作り直す。")
    if not final:
        return []
    print("[wav_dir][warn] " + msg, flush=True)
    for k, o, n in det[:8]:
        print(f"    {k}: 台帳={o!r}  今回={n!r}", flush=True)
    return [msg]


def check_manifest_header(man, fp, st_hash, full_hash, args, n_wavs, phase):
    """wav_dir に既にある音声が今回の設定のものか。食い違えば SystemExit.

    --overwrite / --force_resume_mismatch のときだけ警告して続行する。
    返り値は summary["warnings"] に足す文字列のリスト。
    phase="static" はモデルを読む前の照合 (GPU を掴む前に止めるため)、
    phase="final" はその実行での最後の照合。参考の注意書きは final でだけ出す。
    """
    warns = []
    forced = bool(args.force_resume_mismatch)
    final = (phase == "final")
    if man is None:
        if n_wavs == 0:
            return warns
        msg = (f"wav_dir に {n_wavs} 本の wav があるのに出所の台帳 "
               f"({WAV_MANIFEST_NAME}) が無い。どの設定で作られたか確かめられない。")
        if args.asr_only:
            if forced or args.adopt_unmanifested_wavs:
                warns.append(msg + " 指示により照合せずに採点した。層や seed の"
                                   "食い違いは検出できていない。")
                return warns
            raise SystemExit(
                "[wav_dir] " + msg + "\n"
                "  --asr_only は生成せずに採点だけやり直すので、出所が不明な音声を\n"
                "  任意の --layer の名前で採点し直せてしまう (2026-09-17 の LongCat\n"
                "  統制実験と同じ事故)。\n"
                "  対処: 生成し直す (--asr_only を外す) か、出所を承知のうえで\n"
                "        --adopt_unmanifested_wavs を付けること。")
        if not final:
            return warns
        if args.adopt_unmanifested_wavs:
            warns.append(msg + " --adopt_unmanifested_wavs により今回の設定で"
                               "作られたものとして台帳に取り込んだ。")
        else:
            warns.append(msg + " 今回の設定で作り直す (既存の wav は _stale_ に退避)。")
        return warns

    old_fp = man.get("audio_fingerprint") or {}
    bad = []
    if man.get("schema") != MANIFEST_SCHEMA:
        bad.append(("schema", man.get("schema"), MANIFEST_SCHEMA))
    if man.get("audio_fp_static_hash") != st_hash:
        bad.append(("audio_fp_static_hash", man.get("audio_fp_static_hash"), st_hash))
        bad += audio_fp_diff(old_fp, fp, dynamic=False)
    elif full_hash is not None and man.get("audio_fp_full_hash") not in (None, full_hash):
        bad.append(("audio_fp_full_hash", man.get("audio_fp_full_hash"), full_hash))
        bad += audio_fp_diff(old_fp, fp, dynamic=True)

    if not bad:
        if not final:
            return warns + _code_fp_note(man, fp, args, final=False)
        warns += _code_fp_note(man, fp, args, final=True)
        if args.asr_only and full_hash is None and man.get("audio_fp_full_hash") is not None:
            warns.append("--asr_only ではモデルを読まないので、台帳のうち "
                         f"{list(AUDIO_FP_DYNAMIC_KEYS)} は照合できていない "
                         "(静的な設定 = layer / seed / 復号設定 / 参照音声 は一致)。")
        # 同じ wav_dir を別の out_dir が書いていないか
        others = sorted({str(p.get("out_dir")) for p in man.get("producers", [])
                         if p.get("out_dir") and
                         os.path.abspath(str(p.get("out_dir"))) != os.path.abspath(args.out_dir)})
        if others:
            warns.append(f"この wav_dir は別の out_dir からも書かれている: {others}。"
                         f"設定は一致しているが、results.json が 2 つ同じ音声を指している。"
                         f"層や条件ごとに wav_dir を分けること。")
        return warns

    lines = [f"    {k}: 台帳={o!r}  今回={n!r}" for k, o, n in bad]
    body = (f"wav_dir ({args.wav_dir}) にある音声は今回と違う設定で作られている。\n"
            + "\n".join(lines) + "\n"
            f"  台帳の wav 本数 = {len(man.get('wavs', {}))}、"
            f"ディスク上の wav 本数 = {n_wavs}。\n"
            "  このまま続けると、前の設定で作った音声が今回の設定の結果として\n"
            "  採点される (2026-09-17 の LongCat 統制実験と同じ事故)。")
    if args.asr_only:
        if not forced:
            raise SystemExit(
                "[wav_dir] " + body + "\n"
                "  対処: --layer などを台帳に合わせるか、生成し直すこと。\n"
                "        承知のうえなら --force_resume_mismatch。")
        warns.append(f"--asr_only で台帳と設定が食い違ったまま採点した: "
                     f"{[(k, o, n) for k, o, n in bad]}")
        print("[wav_dir][warn] " + body, flush=True)
        return warns
    if args.overwrite or forced:
        print("[wav_dir][warn] " + body, flush=True)
        if final:
            warns.append(f"wav_dir の台帳と設定が食い違ったまま "
                         f"{'--overwrite' if args.overwrite else '--force_resume_mismatch'} "
                         f"で続行した: {[(k, o, n) for k, o, n in bad]}。"
                         f"合わない wav は _stale_ に退避する。")
        return warns
    raise SystemExit(
        "[wav_dir] " + body + "\n"
        "  対処: 設定ごとに --wav_dir を分ける (層掃引なら層ごとに別ディレクトリ) か、\n"
        "        --overwrite で作り直すか、承知のうえなら --force_resume_mismatch。")


def quarantine_wav(path, run_id):
    """削除はしない。`_stale_<run_id>_` を付けて同じディレクトリに残す."""
    d, b = os.path.split(path)
    dst = os.path.join(d, f"_stale_{run_id}_{b}")
    i = 1
    while os.path.exists(dst):
        dst = os.path.join(d, f"_stale_{run_id}_{i}_{b}")
        i += 1
    os.replace(path, dst)
    return dst


def audit_wav_dir(wav_dir, man, st_hash, full_hash, scope, args, run_id, quarantine=True,
                  code_hash=None, want_input=None):
    """wav を 1 本ずつ台帳と照合する.

    照合するもの (どれか 1 つでも合わなければ「今回の音声ではない」):
      * 設定    audio_fp_static_hash / audio_fp_full_hash
      * 実装    code_fp_hash   … code_hash=None を渡すと照合しない (--asr_only)
      * 入力    input_fp       … want_input[(pidx, cond)] と突き合わせる
      * 中身    bytes / sha256

    返り値 (trusted, quarantined, warnings)
      trusted     = {(pair_idx, cond)}  今回の指紋で作られたと確かめられた wav
      quarantined = [(path, dst, 理由), ...]
    """
    entries = (man or {}).get("wavs", {})
    want_input = want_input or {}
    trusted, quarantined, warns = set(), [], []
    n_out_of_scope = 0
    out_of_scope_files = []
    reasons = {}
    for pidx, cond, path in scan_wav_dir(wav_dir):
        e = entries.get(wav_key(pidx, cond))
        why = None
        if not isinstance(e, dict):
            why = "台帳に記録が無い"
        elif e.get("audio_fp_static_hash") != st_hash:
            why = (f"別の設定で作られている (台帳={e.get('audio_fp_static_hash')} "
                   f"今回={st_hash})")
        elif (full_hash is not None and e.get("audio_fp_full_hash") is not None
              and e.get("audio_fp_full_hash") != full_hash):
            why = (f"モデルが違う (台帳={e.get('audio_fp_full_hash')} 今回={full_hash})")
        elif code_hash is not None and e.get("code_fp_hash") != code_hash:
            # **ラウンド 3 の追加**。csm_splice.py を書き換えた / 実行時に
            # splice_synthesize を差し替えた場合にここで落ちる。
            why = (f"別の実装で作られている (台帳={e.get('code_fp_hash')} "
                   f"今回={code_hash})")
        elif ((pidx, cond) in want_input
              and e.get("input_fp") != want_input[(pidx, cond)]):
            why = (f"別の入力で作られている (台帳={e.get('input_fp')} "
                   f"今回={want_input[(pidx, cond)]})")
        else:
            try:
                size = os.path.getsize(path)
                sha = _sha256_file(path)
            except Exception as ex:
                why = f"読めなかった: {ex}"
                size = sha = None
            if why is None and e.get("bytes") is not None and int(e["bytes"]) != int(size):
                why = f"バイト数が台帳と違う ({e.get('bytes')} -> {size})"
            elif why is None and e.get("sha256") and e["sha256"] != sha:
                why = f"中身が台帳と違う (sha256 {e.get('sha256')} -> {sha})"

        if why is None and not args.overwrite:
            trusted.add((pidx, cond))
            continue
        if why is None and args.overwrite:
            # --overwrite でも、今回の指紋で作られた wav は上書きされるだけなので
            # 退避しない (同じ設定の作り直しは汚染ではない)。
            continue
        if pidx not in scope:
            n_out_of_scope += 1
            out_of_scope_files.append(os.path.basename(path))
            continue
        if not quarantine:
            reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
            continue
        if man is None and args.adopt_unmanifested_wavs:
            # 台帳が **まったく無い** wav_dir のときだけ、明示指示があれば
            # 今回の指紋で取り込む。台帳に記録があって食い違うものは取り込まない。
            continue
        dst = quarantine_wav(path, run_id)
        quarantined.append((path, dst, why))
        reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1

    if quarantined:
        print(f"[wav_dir] 今回の設定で作られたと確かめられない wav {len(quarantined)} 本を "
              f"_stale_{run_id}_* に退避した (削除していない)。内訳 {reasons}", flush=True)
        for src, dst, why in quarantined[:5]:
            print(f"    {os.path.basename(src)} -> {os.path.basename(dst)} ({why})",
                  flush=True)
        warns.append(f"今回の設定で作られたと確かめられない wav {len(quarantined)} 本を "
                     f"_stale_{run_id}_* に退避した。内訳 {reasons}。"
                     f"採点器はファイル名だけを見るので、退避しないと前の設定の音声が "
                     f"今回の結果として採点される。")
    elif reasons:
        warns.append(f"照合できない wav が {sum(reasons.values())} 本ある "
                     f"(内訳 {reasons})。退避していないので採点の対象に入る。")
    if n_out_of_scope:
        # **--n の外は触らない** (--n を伸ばして再開する運用を壊さないため)。
        # ただし「別の設定の音声が同じ wav_dir に残っている」ことは必ず書き残す。
        # ここを黙ると、あとから大きい --n で採点したときに層が混ざる。
        warns.append(
            f"今回の --n={getattr(args, 'n', '?')} の範囲外に、今回の設定では"
            f"作られていない wav が {n_out_of_scope} 本ある "
            f"({sorted(out_of_scope_files)[:12]}{' ...' if n_out_of_scope > 12 else ''})。"
            f"--n の外は触らないのでディスクに残してある。"
            f"**この wav_dir を今回より大きい --n で採点すると、別の設定の音声が"
            f"混ざる。** 採点は必ず --n {getattr(args, 'n', '?')} を付けて掛けるか、"
            f"設定ごとに wav_dir を分けること。")
    return trusted, quarantined, warns


def manifest_fp_spread(man, st_hash, code_hash):
    """台帳の中に、今回と違う指紋で書かれた記録がどれだけ残っているか.

    --n を減らして層を変えた再実行のあと、台帳の見出しは今回の指紋なのに
    記録ごとの指紋は前の層のまま、という状態が起きる (範囲外は触らないため)。
    これ自体は per-wav の照合が効いているので正しい状態だが、
    「見出しだけ見て全部同じ設定だと思う」ことがないよう数えて出す。
    """
    other = {}
    for k, e in ((man or {}).get("wavs") or {}).items():
        if not isinstance(e, dict):
            continue
        if e.get("audio_fp_static_hash") != st_hash:
            other.setdefault("設定", []).append(k)
        elif code_hash is not None and e.get("code_fp_hash") != code_hash:
            other.setdefault("実装", []).append(k)
    return {k: sorted(v) for k, v in other.items()}


def register_wav(man, pidx, cond, path, st_hash, full_hash, out_dir, run_id, g,
                 code_hash=None, input_fp=None):
    sha = _sha256_file(path)
    man["wavs"][wav_key(pidx, cond)] = {
        "pair_idx": int(pidx), "cond": cond, "file": os.path.basename(path),
        "sha256": sha, "bytes": os.path.getsize(path),
        "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
        # 実装と入力も 1 本ごとに記録する。台帳ヘッダだけだと、
        # 途中で実装を差し替えた実行が混ざった wav_dir を見分けられない。
        "code_fp_hash": code_hash, "input_fp": input_fp,
        "out_dir": os.path.abspath(out_dir), "run_id": run_id,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "codes_fp": g.get("codes_fp"), "sec": g.get("sec"),
    }
    return sha


def scorable_pairs(wav_dir, scope):
    """phoneme_per_v3.py が採点できる対 = s####_{bl,inj,tgt}.wav が 3 本そろう対."""
    return sorted(i for i in scope
                  if all(os.path.isfile(p) for p in wav_paths(wav_dir, i).values()))


def reconcile_wav_dir(wav_dir, man, rows_by_idx, scope, args, run_id, trusted,
                      interrupted=False):
    """実行の最後に、ディスク上の wav と results.json の記録を突き合わせる.

    **ラウンド 3 の追加。** ラウンド 2 の退避は「今回の指紋か」しか見ていなかった
    ので、層も seed も実装も同じまま **その条件だけ生成に失敗した再実行**では、
    前回の wav がディスクに残ったまま採点された。results.json は同じ条件を
    「失敗」と記録するのに、採点器 (phoneme_per_v3.py) はファイル名しか見ない。
    実測: run1 で pair0 の INJ が失敗 -> run2 で pair0 の BL が失敗、のとき
    fail_by_cond={'bl':1} と記録されながら s0000_bl.wav は run1 のまま残り、
    採点できる対は 2/2 のままだった。

    不変条件: **実行が終わった時点で wav_dir に残る s####_{c}.wav は、
    results.json がその条件を ok と記録した対のものだけ。**

      * ok と記録していないのに wav がある -> 退避する (`_stale_`。削除しない)
      * ok と記録したのに wav が無い       -> その条件を error に落とす

    返り値 (quarantined, warnings)
    """
    quarantined, warns = [], []
    missing, orphan = [], []
    for pidx in sorted(scope):
        row = rows_by_idx.get(pidx)
        paths = wav_paths(wav_dir, pidx)
        for c in COND:
            exists = os.path.isfile(paths[c])
            if row is None:
                # 行が無い = この実行では一度も扱っていない対。
                # 中断していないなら、本来ありえない wav なので退避する。
                if exists:
                    orphan.append((pidx, c))
                    if not interrupted:
                        dst = quarantine_wav(paths[c], run_id)
                        quarantined.append((paths[c], dst,
                                            "results.json に行が無いのに wav がある"))
                        (man or {}).get("wavs", {}).pop(wav_key(pidx, c), None)
                        trusted.discard((pidx, c))
                continue
            ok = _cond_ok(row, c)
            if exists and not ok:
                dst = quarantine_wav(paths[c], run_id)
                quarantined.append((paths[c], dst,
                                    f"この実行は {c} を失敗と記録した"))
                (man or {}).get("wavs", {}).pop(wav_key(pidx, c), None)
                trusted.discard((pidx, c))
                row[f"{c}_stale_quarantined"] = os.path.basename(dst)
                drop_cond_judgements(row, c)
            elif ok and not exists:
                missing.append((pidx, c))
                row[f"{c}_status"] = "error"
                row[f"{c}_err"] = "ok と記録されているのに wav がディスクに無い"
                row["status"] = "error"
                row["fail_conds"] = sorted(set(list(row.get("fail_conds") or []) + [c]))
                row.setdefault("fail_stage", c)
                (man or {}).get("wavs", {}).pop(wav_key(pidx, c), None)
                trusted.discard((pidx, c))
                drop_cond_judgements(row, c)
            elif not ok:
                # wav も無く ok でもない条件。判定が残っていたら落とす
                # (prompt_len 検査で落ちた対は wav を 1 本も保存しないのに、
                #  ASR がメモリ上の音声を書き起こして判定だけ残っていた)。
                drop_cond_judgements(row, c)

    if quarantined:
        why = {}
        for _s, _d, w in quarantined:
            why[w.split(" (")[0]] = why.get(w.split(" (")[0], 0) + 1
        print(f"[wav_dir] 記録と食い違う wav {len(quarantined)} 本を "
              f"_stale_{run_id}_* に退避した (削除していない)。内訳 {why}", flush=True)
        warns.append(
            f"この実行が ok と記録していない条件の wav が {len(quarantined)} 本 "
            f"ディスクに残っていたので _stale_{run_id}_* に退避した (内訳 {why})。"
            f"採点器はファイル名しか見ないので、退避しないと『失敗』と記録した条件が "
            f"前回の音声で採点される。")
    if missing:
        warns.append(f"ok と記録されているのに wav が無い条件が {len(missing)} 件 "
                     f"({missing[:10]})。error に落として母数に残した。")
    if orphan and interrupted:
        warns.append(f"実行が中断したため、results.json に行の無い wav {len(orphan)} 本を "
                     f"そのまま残した。次の実行で照合される。")
    return quarantined, warns


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="CSM-1B SpeechSplice cross-culture eval")
    p.add_argument("--lang", default="en", choices=["en"],
                   help="CSM-1B は英語のみ (モデルカードの言語タグが en だけ)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--layer", type=int, default=C.DEFAULT_LAYER,
                   help=f"backbone_model.layers[L] の出力を差し替える。既定 {C.DEFAULT_LAYER} "
                        f"(= L-2)。層掃引の格子は {C.SWEEP_LAYERS}")
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--allow_download", action="store_true",
                   help="ローカルに重みが無いとき HF から取得する (transformers 経路で 7.13 GB)")
    p.add_argument("--seed", type=int, default=C.SEED)

    p.add_argument("--model_id", default=C.DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--hf_home", default=None)
    p.add_argument("--no_strict_weights", dest="strict_weights", action="store_false",
                   help="missing_keys / transformers_weights の検査を外す (推奨しない)")

    p.add_argument("--speaker", default=C.DEFAULT_SPEAKER,
                   help="話者 ID。整数の文字列のみ。2 桁にすると本文の開始行がずれる")
    p.add_argument("--ref_mode", default="clone", choices=["clone", "none"],
                   help="clone = Table-3 共通の自然音声を参照ターンとして前置する (既定)。"
                        "none = 参照なし (CSM は話者が毎回変わるので比較が壊れやすい)")
    p.add_argument("--ref_wav", default=None)
    p.add_argument("--ref_text", default=None)
    p.add_argument("--ref_max_sec", type=float, default=10.0)

    p.add_argument("--temperature", type=float, default=C.DEFAULT_TEMPERATURE)
    p.add_argument("--depth_temperature", type=float, default=C.DEFAULT_DEPTH_TEMPERATURE)
    p.add_argument("--depth_top_k", type=int, default=C.DEFAULT_DEPTH_TOP_K)
    p.add_argument("--max_new_tokens", type=int, default=C.DEFAULT_MAX_NEW_TOKENS,
                   help="音声フレーム数。12.5 Hz なので 256 = 20.5 秒")
    p.add_argument("--greedy", dest="do_sample", action="store_false",
                   help="公式既定は do_sample=True。他モデルと条件を揃えるため既定のまま推奨")
    p.set_defaults(do_sample=True)
    p.add_argument("--min_sec", type=float, default=0.5,
                   help="これより短い生成は壊れたものとして error にする")
    p.add_argument("--fail_on_cap", action="store_true",
                   help="EOS が出ないまま max_new_tokens まで走り切った生成を error にする。"
                        "既定は記録 + 警告のみ (summary の n_truncated_at_cap)")

    p.add_argument("--no_asr", dest="run_asr", action="store_false",
                   help="source_in_bl / target_in_inj の判定を省く (推奨しない)")
    p.set_defaults(run_asr=True)
    p.add_argument("--asr_model", default="openai/whisper-large-v3")
    p.add_argument("--asr_only", action="store_true",
                   help="生成せず、既にある wav を書き起こして率だけ出し直す。"
                        "wav_dir の台帳と設定が食い違えば停止する")
    p.add_argument("--no_kappa", dest="collect_kappa", action="store_false")
    p.set_defaults(collect_kappa=True)

    p.add_argument("--overwrite", action="store_true", help="再開せず最初からやり直す")
    p.add_argument("--force_resume_mismatch", action="store_true",
                   help="progress.json / wav_dir の台帳が今回の設定と食い違っていても"
                        "続行する。既定では停止する。付けた場合は warnings に残る")
    p.add_argument("--adopt_unmanifested_wavs", action="store_true",
                   help="台帳の無い wav_dir の既存 wav を、今回の設定で作られたものとして"
                        "取り込む (旧版で作った wav_dir 用)。台帳に記録があって"
                        "食い違うものは取り込まない。warnings に残る")
    p.add_argument("--max_drop_rate", type=float, default=DEFAULT_MAX_DROP_RATE,
                   help="除外率 (n_failed / n_attempted) がこれを超えたら "
                        "summary の warnings に文字列を残す")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    return p


def _record_gen(row, cond, g):
    """1 条件ぶんの生成統計を row に書く (3 条件とも同じ形)."""
    row[f"{cond}_status"] = "ok"
    row[f"{cond}_sec"] = g["sec"]
    row[f"{cond}_frames"] = g["n_frames"]
    row[f"{cond}_cutoff_frames"] = g["cutoff_frames"]
    row[f"{cond}_eos_found"] = g["eos_found"]
    row[f"{cond}_hit_max"] = g["hit_max_new_tokens"]
    row[f"{cond}_truncated_at_cap"] = g["truncated_at_cap"]
    row[f"{cond}_stop_reason"] = g["stop_reason"]
    row[f"{cond}_codes_fp"] = g["codes_fp"]


def main():
    args = build_argparser().parse_args()
    if args.hf_home:
        C.setup_hf_home(args.hf_home)
    os.chdir(PROJECT_ROOT)

    if args.out_dir is None:
        args.out_dir = os.path.join("eval_results", f"csm_{args.lang}_n{args.n}_L{args.layer}")
    if args.wav_dir is None:
        args.wav_dir = os.path.join(args.out_dir, "wav")
    args.out_dir = os.path.abspath(args.out_dir)
    args.wav_dir = os.path.abspath(args.wav_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    progress_path = os.path.join(args.out_dir, "progress.json")
    results_path = os.path.join(args.out_dir, "results.json")

    pairs = load_pairs(args.lang, args.n)
    print(f"[info] {args.lang} cross-culture: {len(pairs)} pairs", flush=True)
    print(f"[info] run_id={run_id}", flush=True)
    print(f"[info] out_dir={args.out_dir}", flush=True)
    print(f"[info] wav_dir={args.wav_dir}", flush=True)
    print(f"[info] HF_HOME={os.environ['HF_HOME']}", flush=True)

    # ---------------- 実行設定の指紋 ----------------
    # モデルを読む前に照合する。GPU を確保してから止めるのは無駄。
    fp = static_fingerprint(args)
    st_hash, full_hash = audio_fp_hashes(fp)          # full_hash はまだ None
    code_hash = fp["code_fp_hash"]
    print(f"[info] 設定の指紋={st_hash} 実装の指紋={code_hash} "
          f"driver={fp['driver_ast']}", flush=True)
    resume_warnings = []
    prov_warnings = []

    # ---------------- 再開 (out_dir 側) ----------------
    rows_by_idx = {}
    if os.path.isfile(progress_path) and not args.overwrite:
        try:
            with open(progress_path) as f:
                prev = json.load(f)
        except Exception as e:
            # 壊れた progress.json を「空の再開」として黙って通さない。
            raise SystemExit(f"[resume] {progress_path} を読めなかった: {e}\n"
                             f"  作り直すなら --overwrite を付けること。")
        prev_rows = prev.get("rows", [])
        resume_warnings = check_resume_fingerprint(
            prev, fp, len(prev_rows), args.force_resume_mismatch)
        for r in prev_rows:
            rows_by_idx[int(r["pair_idx"])] = r
        print(f"[resume] {progress_path} から {len(rows_by_idx)} 行を読んだ", flush=True)
    elif os.path.isfile(progress_path) and args.overwrite:
        print(f"[info] --overwrite。{progress_path} を無視して作り直す", flush=True)

    attempted = [int(p["pair_idx"]) for p in pairs]
    scope = set(attempted)
    pair_by_idx = {int(p["pair_idx"]): p for p in pairs}
    # wav 1 本ごとの「入力の指紋」。driver が別の文を渡すようになった / データが
    # 差し替わった / 対の順番が変わった場合に、その wav だけを信用しなくなる。
    want_input = want_input_fps(pairs, fp)

    # ---------------- wav_dir 側の台帳 (静的な照合) ----------------
    # **採点されるのはここの中身**。モデルを読む前に、層 / seed / 復号設定 /
    # 参照音声が台帳と一致するかを確かめる。
    man = load_manifest(args.wav_dir)
    n_wavs_on_disk = len(scan_wav_dir(args.wav_dir))
    prov_warnings += check_manifest_header(
        man, fp, st_hash, None, args, n_wavs_on_disk,
        phase="final" if args.asr_only else "static")

    def provenance_block(trusted, quarantined):
        return {"wav_provenance": {
            "wav_dir": args.wav_dir,
            "manifest": manifest_path(args.wav_dir),
            "audio_fp_static_hash": st_hash,
            "audio_fp_full_hash": full_hash,
            # 実装の指紋。--asr_only は生成しないので照合していない (checked=False)。
            "code_fp_hash": code_hash,
            "code_fp_hash_in_manifest": (man or {}).get("code_fp_hash"),
            "code_fp_checked": (not args.asr_only),
            "driver_ast": fp.get("driver_ast"),
            "run_id": run_id,
            "n_wavs_on_disk": len(scan_wav_dir(args.wav_dir)),
            "n_trusted": len(trusted),
            "n_quarantined": len(quarantined),
            "quarantined": [os.path.basename(d) for _, d, _ in quarantined][:50],
            # 台帳の見出しは今回の指紋でも、記録ごとの指紋が違うものが残ることがある
            # (--n の外は触らないため)。見出しだけ見て安心しないように数えて出す。
            "wavs_with_other_fingerprint": {
                k: v[:20] for k, v in
                manifest_fp_spread(man, st_hash,
                                   None if args.asr_only else code_hash).items()},
        }}

    # ---------------- ASR だけ回す ----------------
    if args.asr_only:
        import soundfile as sf
        # code_hash=None: --asr_only は生成しないので、どの実装で作られるべき
        # だったかを知らない。実装は照合せず「照合していない」と書き残す
        # (_code_fp_note が warnings に入れる)。入力 (input_fp) は照合する。
        trusted, quarantined, w = audit_wav_dir(
            args.wav_dir, man, st_hash, None, scope, args, run_id, quarantine=False,
            code_hash=None, want_input=want_input)
        if args.adopt_unmanifested_wavs and man is None:
            trusted = {(p, c) for p, c, _ in scan_wav_dir(args.wav_dir)}
        prov_warnings += w
        asr = Asr(device=args.device, model_id=args.asr_model)
        meta = {"asr_model": args.asr_model, "mode": "asr_only", "run_id": run_id}
        # 前回の行から引き継がない鍵 (今回の書き起こしで必ず作り直す)。
        # 引き継ぐと、出所が確かめられない対の古い判定が率に混ざる。
        judged = ({"status", "fail_stage", "fail_conds", "err", "trace",
                   "asr_fail_conds", "source_in_bl", "target_in_tgt",
                   "target_in_inj", "source_in_inj"}
                  | {f"{c}_transcript" for c in COND}
                  | {f"{c}_asr_err" for c in COND}
                  | {f"{c}_status" for c in COND})
        for i, pair in enumerate(pairs):
            pidx = int(pair["pair_idx"])
            prev_row = rows_by_idx.get(pidx) or {}
            row = {"pair_idx": pidx, "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "text_a": pair["text_a"], "text_b": pair["text_b"],
                   "layer": args.layer, "seed": args.seed, "run_id": run_id}
            # ---------------------------------------------------------------
            # **条件ごとに独立に扱う (ラウンド 3 の修正)。**
            # ラウンド 2 は「1 条件でも出所を確かめられなければ、その対の
            # 3 条件すべてを捨てる」作りだった。生成 run が blocking-3 として
            # 直した『条件間の巻き添え』が --asr_only 経路に丸ごと残っていた。
            # 実測: pair1 の INJ だけ欠けた wav_dir を採点すると
            #   生成 run   n_ok_by_cond={'bl':3,'inj':2,'tgt':3} source_in_bl_rate=1.0
            #   --asr_only n_ok_by_cond={'bl':2,'inj':2,'tgt':2} source_in_bl_rate=0.667
            # と、同じ wav・同じ設定なのに BL の母数と率が変わっていた。
            # ---------------------------------------------------------------
            untrusted = [c for c in COND if (pidx, c) not in trusted]
            for k, v in prev_row.items():           # 生成統計だけ引き継ぐ
                # 出所を確かめられない条件の統計は引き継がない (古い判定が混ざる)
                if k not in judged and k.split("_")[0] not in untrusted:
                    row.setdefault(k, v)
            paths = wav_paths(args.wav_dir, pidx)
            tr, failed = {}, []
            for c in COND:
                if c in untrusted:
                    # この条件だけ母数から落とす。他の条件は巻き添えにしない。
                    row[f"{c}_status"] = "error"
                    row[f"{c}_err"] = "wav の出所を台帳で確かめられない"
                    continue
                # 台帳と照合できた = その条件の生成は成立している。
                # 以降の失敗は ASR の失敗であって生成の失敗ではない。
                row[f"{c}_status"] = "ok"
                try:
                    a, sr = sf.read(paths[c], dtype="float32")
                    tr[c] = transcribe_cond(asr, c, a, sr, args.lang, pair=pair)
                    row[f"{c}_transcript"] = tr[c]
                except Exception as e:
                    failed.append(c)
                    row[f"{c}_asr_err"] = f"{type(e).__name__}: {e}"[:200]
            if failed:
                row["asr_fail_conds"] = failed
            if "bl" in tr:
                row["source_in_bl"] = in_text(pair["word_a"], tr["bl"], args.lang)
            if "tgt" in tr:
                row["target_in_tgt"] = in_text(pair["word_b"], tr["tgt"], args.lang)
            if "inj" in tr:
                row["target_in_inj"] = in_text(pair["word_b"], tr["inj"], args.lang)
                row["source_in_inj"] = in_text(pair["word_a"], tr["inj"], args.lang)
            if untrusted:
                row["status"] = "error"
                row["fail_stage"] = "untrusted_wav"
                row["fail_conds"] = sorted(untrusted)
                row["err"] = f"出所を確かめられない条件がある: {untrusted}"
            elif failed:
                row["status"] = "error"
                row["fail_stage"] = "asr"
            else:
                row["status"] = "ok"
            rows_by_idx[pidx] = row
            if (i + 1) % args.log_every == 0:
                print(f"  [asr {i + 1}/{len(pairs)}]", flush=True)

        rows = [rows_by_idx[k] for k in sorted(rows_by_idx)]
        ex = dict(meta)
        ex["fingerprint"] = fp
        ex.update(provenance_block(trusted, quarantined))
        # --asr_only は wav_dir を書き換えない (退避もしない) ので、出所を
        # 確かめられない対も採点器からは見えたままになる。黙って通さない。
        sp = scorable_pairs(args.wav_dir, scope)
        n_all_trusted = sum(1 for i in scope if all((i, c) in trusted for c in COND))
        ex["n_scorable_pairs"] = len(sp)
        ex["n_pairs_all_trusted"] = n_all_trusted
        ex["n_generated_this_run"] = 0
        ex["n_resumed_skipped"] = 0
        pw = list(prov_warnings)
        if len(sp) > n_all_trusted:
            pw.append(
                f"--asr_only は wav_dir を書き換えないので、出所を確かめられない wav も "
                f"ディスクに残っている。採点器が採点できる対 {len(sp)} に対し、"
                f"3 条件とも照合できた対は {n_all_trusted}。"
                f"phoneme_per_v3.py をこの wav_dir に掛けると、この summary が母数から "
                f"落とした対まで採点される。作り直すなら --asr_only を外すこと。")
        ex["warnings"] = _dedupe(list(resume_warnings) + pw)
        summary = summarize(rows, args, extra=ex, attempted=attempted)
        tmp = results_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, results_path)
        print(f"\n[info] saved {results_path}", flush=True)
        report(summary)
        return

    # ---------------- モデル読み込み ----------------
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_id} (device={args.device} dtype={args.dtype}) ...",
          flush=True)
    t0 = time.time()
    bundle = C.load_csm(args.model_id, device=args.device, dtype=dtype,
                        allow_download=args.allow_download,
                        attn_implementation=args.attn_implementation,
                        strict_weights=args.strict_weights)
    print(f"[info] loaded in {time.time() - t0:.1f}s from {bundle.source}", flush=True)
    print(f"[info] backbone layers={bundle.num_layers} sr={bundle.sampling_rate} "
          f"frame_rate={bundle.frame_rate} codebooks={bundle.num_codebooks}", flush=True)
    C.assert_splice_target(bundle, args.layer)
    print(f"[info] splice target = backbone_model.layers[{args.layer}] "
          f"(grid {C.SWEEP_LAYERS}, deepest = L-2 = {bundle.num_layers - 2})", flush=True)

    # モデルを読んで初めて分かる分を指紋に足し、もう一度照合する。
    # snapshot ディレクトリ名 = HF の commit hash なので revision の代わりになる。
    fp["model_revision"] = os.path.basename(str(bundle.source).rstrip("/"))
    fp["num_layers"] = int(bundle.num_layers)
    fp["sampling_rate"] = int(bundle.sampling_rate)
    fp["num_codebooks"] = int(bundle.num_codebooks)
    st_hash2, full_hash = audio_fp_hashes(fp)
    assert st_hash2 == st_hash
    if os.path.isfile(progress_path) and not args.overwrite and rows_by_idx:
        try:
            with open(progress_path) as f:
                _prev = json.load(f)
        except Exception:
            _prev = {}
        resume_warnings += check_resume_fingerprint(
            _prev, fp, len(rows_by_idx), args.force_resume_mismatch)
    prov_warnings += check_manifest_header(man, fp, st_hash, full_hash, args,
                                           n_wavs_on_disk, phase="final")

    # ---------------- wav_dir の監査と退避 ----------------
    # ここを通ったあと、wav_dir に残る s####_*.wav は必ず今回の指紋のもの。
    trusted, quarantined, w = audit_wav_dir(
        args.wav_dir, man, st_hash, full_hash, scope, args, run_id, quarantine=True,
        code_hash=code_hash, want_input=want_input)
    prov_warnings += w
    if man is None:
        man = new_manifest(args.wav_dir, fp, st_hash, full_hash)
        if args.adopt_unmanifested_wavs:
            for pidx, cond, path in scan_wav_dir(args.wav_dir):
                register_wav(man, pidx, cond, path, st_hash, full_hash,
                             args.out_dir, run_id, {}, code_hash=code_hash,
                             input_fp=want_input.get((pidx, cond)))
                trusted.add((pidx, cond))
    man["schema"] = MANIFEST_SCHEMA
    man["wav_dir"] = args.wav_dir
    man["audio_fp_static_hash"] = st_hash
    man["audio_fp_full_hash"] = full_hash
    man["audio_fingerprint"] = audio_fp_view(fp, dynamic=True)
    man["code_fp_hash"] = code_hash
    man["code_fp"] = fp.get("code_fp")
    man["driver_ast"] = fp.get("driver_ast")
    # 退避した wav の記録は落とす (ディスク上に無いものを台帳に残さない)
    man.setdefault("wavs", {})
    for src, _dst, _why in quarantined:
        man["wavs"].pop(os.path.basename(src)[:-4], None)
    for k, e in list(man["wavs"].items()):
        fn = e.get("file", k + ".wav") if isinstance(e, dict) else k + ".wav"
        if not os.path.isfile(os.path.join(args.wav_dir, fn)):
            man["wavs"].pop(k, None)
    man.setdefault("producers", []).append(
        {"run_id": run_id, "out_dir": args.out_dir,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
         "code_fp_hash": code_hash, "driver_ast": fp.get("driver_ast"),
         "n": args.n, "layer": args.layer, "mode": "generate"})
    man["producers"] = man["producers"][-20:]
    save_manifest(man, args.wav_dir)
    print(f"[wav_dir] 台帳 {manifest_path(args.wav_dir)} / 設定 {st_hash}/{full_hash} / "
          f"実装 {code_hash} / 照合済み {len(trusted)} 本 / 退避 {len(quarantined)} 本",
          flush=True)

    # ---------------- 参照音声 ----------------
    meta = {}
    ref = None
    if args.ref_mode == "clone":
        rw, rt = DEFAULT_REFS[args.lang]
        rw = args.ref_wav or rw
        rt = args.ref_text if args.ref_text is not None else rt
        if not os.path.isfile(rw):
            raise SystemExit(f"参照 wav が無い: {rw}")
        ref = C.build_ref(bundle, rw, rt, speaker=args.speaker, max_sec=args.ref_max_sec)
        print(f"[info] ref={rw} ({ref.orig_sr} Hz -> {bundle.sampling_rate} Hz, "
              f"{ref.n_samples} samples, {ref.n_frames} audio frames)", flush=True)
        meta["ref_wav"] = rw
        meta["ref_text"] = rt
        meta["ref_n_frames"] = ref.n_frames
    else:
        print("[warn] --ref_mode none。CSM は文脈が無いと話者が毎回変わるので、"
              "BL / TGT / INJ で声が揺れて音素誤り率の比較に話者要因が混ざる。", flush=True)
    meta["ref_mode"] = args.ref_mode
    meta["run_id"] = run_id

    asr = None
    if args.run_asr:
        print(f"[info] loading ASR {args.asr_model} ...", flush=True)
        asr = Asr(device=args.device, model_id=args.asr_model)
        meta["asr_model"] = args.asr_model

    gen = C.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      temperature=args.temperature,
                      depth_temperature=args.depth_temperature,
                      depth_top_k=args.depth_top_k,
                      max_new_tokens=args.max_new_tokens, min_sec=args.min_sec,
                      fail_on_cap=args.fail_on_cap)
    meta["gen"] = gen.as_dict()
    meta["model_source"] = bundle.source
    meta["num_layers"] = bundle.num_layers
    meta["sweep_layers"] = list(C.SWEEP_LAYERS)

    def flush(final=False):
        rows = [rows_by_idx[k] for k in sorted(rows_by_idx)]
        tmp = progress_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"rows": rows, "args": vars(args), "fingerprint": fp,
                       "run_id": run_id}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, progress_path)
        save_manifest(man, args.wav_dir)
        if final:
            ex = dict(meta)
            ex["fingerprint"] = fp
            ex.update(provenance_block(trusted, quarantined))
            ws = _dedupe(list(resume_warnings) + list(prov_warnings))
            if ws:
                ex["warnings"] = ws
            summary = summarize(rows, args, extra=ex, attempted=attempted)
            tmp = results_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
            os.replace(tmp, results_path)
            return summary
        return None

    # ---------------- 本体 ----------------
    #
    # **3 条件はそれぞれ独立に回し、成功した条件はその場で保存して記録する。**
    # 1 対 = 1 つの try にすると、INJ が落ちただけで生成し終えている BL と TGT の
    # 記録まで消え、source_in_bl_rate が INJ の失敗で押し下げられる。
    n_done = n_skip = n_err = 0
    t_start = time.time()
    # 中断 (例外 / Ctrl-C / --stop_on_error) でも、必ず下の
    # reconcile_wav_dir を通してから results.json を書く。
    # そうしないと『失敗と記録した条件の wav が残ったまま』で終わる。
    interrupted = None
    try:
        for i, pair in enumerate(pairs):
            pidx = int(pair["pair_idx"])
            # 再開の判定は「行が ok」+「3 本とも台帳と照合済み」。
            # ファイルの有無だけで飛ばさない (どの設定で作られたか分からないため)。
            if (not args.overwrite and pidx in rows_by_idx
                    and rows_by_idx[pidx].get("status") == "ok"
                    and all((pidx, c) in trusted for c in COND)):
                n_skip += 1
                continue
            paths = wav_paths(args.wav_dir, pidx)
            row = {"pair_idx": pidx, "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "text_a": pair["text_a"], "text_b": pair["text_b"],
                   "layer": args.layer, "seed": args.seed, "run_id": run_id}

            gens = {}
            fail_conds = []

            def _try(cond, fn):
                try:
                    g = fn()
                except Exception as e:
                    import traceback
                    row[f"{cond}_status"] = "error"
                    row[f"{cond}_err"] = f"{type(e).__name__}: {e}"[:300]
                    row[f"{cond}_trace"] = traceback.format_exc()[-400:]
                    fail_conds.append(cond)
                    print(f"  [{pidx}] {cond} error: {row[f'{cond}_err']}", flush=True)
                    if args.stop_on_error:
                        # 途中まで書いた行と台帳を残してから落とす
                        row["status"] = "error"
                        row["fail_conds"] = sorted(set(fail_conds))
                        row["fail_stage"] = cond
                        rows_by_idx[pidx] = row
                        flush()
                        raise
                    return None
                gens[cond] = g
                _record_gen(row, cond, g)
                return g

            _try("bl", lambda: C.synthesize(bundle, pair["text_a"], ref=ref,
                                            speaker=args.speaker, gen=gen))
            _try("tgt", lambda: C.synthesize(bundle, pair["text_b"], ref=ref,
                                             speaker=args.speaker, gen=gen))
            _try("inj", lambda: C.splice_synthesize(
                bundle, pair["text_a"], pair["word_a"], pair["text_b"], pair["word_b"],
                layer=args.layer, ref=ref, speaker=args.speaker, gen=gen,
                collect_kappa=args.collect_kappa))

            if "inj" in gens:
                for k in ("fired", "decode_calls", "foreign_calls", "splice_mode",
                          "equal_len", "m_src_entity_tokens", "n_tgt_entity_tokens",
                          "s_span", "t_span", "s_tokens", "t_tokens",
                          "prompt_len", "hidden_index",
                          "kappa_at_splice", "kappa_last_layer", "kappa_by_layer"):
                    if k in gens["inj"]:
                        row[k] = gens["inj"][k]

            # BL と INJ の prompt が 1 トークンも違わないこと (差は splice だけ)。
            # 両方そろったときだけ検査する。
            check_failed = None
            if "bl" in gens and "inj" in gens:
                if int(gens["bl"]["prompt_len"]) != int(gens["inj"]["prompt_len"]):
                    check_failed = "prompt_len_check"
                    row["err"] = (f"BL と INJ の prompt 長が違う "
                                  f"({gens['bl']['prompt_len']} vs {gens['inj']['prompt_len']})")
                    print(f"  [{pidx}] {row['err']}", flush=True)
                # splice が生成を 1 ビットも変えていなければ hook が効いていない。
                row["inj_differs_from_bl"] = bool(
                    gens["inj"]["codes_fp"] != gens["bl"]["codes_fp"])

            # prompt_len 検査に落ちた対は wav を 1 本も保存しない。3 条件とも
            # 「生成は走ったが使えない」ので、条件別の記録も error に揃える
            # (揃えておかないと reconcile が『ok なのに wav が無い』と言う)。
            if check_failed is not None:
                for c in COND:
                    row[f"{c}_status"] = "error"
                    row.setdefault(f"{c}_err", f"{check_failed}: wav を保存していない")

            # ---- 保存。成功した条件だけをその場で書き、台帳にも載せる ----
            if check_failed is None:
                for c in list(gens):
                    try:
                        _, dur, qc = C.save_wav(paths[c], gens[c]["audio"], bundle.sampling_rate)
                        row[f"{c}_peak"] = qc["peak"]
                        # save_wav は [-1, 1] に丸めてから PCM_16 で書く。ASR は丸める前の
                        # 配列を読み、採点器は丸めたあとの wav を読むので、丸めが起きた対は
                        # 判定と採点で聞いている音が違う。数えて warnings に出す。
                        row[f"{c}_n_clipped"] = qc["n_clipped"]
                        row[f"{c}_wav"] = os.path.basename(paths[c])
                        row[f"{c}_wav_sha256"] = register_wav(
                            man, pidx, c, paths[c], st_hash, full_hash,
                            args.out_dir, run_id, gens[c], code_hash=code_hash,
                            input_fp=want_input.get((pidx, c)))
                        trusted.add((pidx, c))
                    except Exception as e:
                        row[f"{c}_status"] = "error"
                        row[f"{c}_err"] = f"save_wav: {type(e).__name__}: {e}"[:300]
                        fail_conds.append(c)
                        trusted.discard((pidx, c))
                        gens.pop(c, None)
                        print(f"  [{pidx}] {c} save_wav error: {row[f'{c}_err']}", flush=True)
                        if args.stop_on_error:
                            rows_by_idx[pidx] = row
                            flush()
                            raise

            # ---- ASR。条件ごとに独立。ASR の失敗は生成の母数を汚さない ----
            asr_failed = []
            if asr is not None:
                tr = {}
                for c in COND:
                    # **wav として保存できた条件だけ書き起こす。**
                    # 保存できていない音の判定を率に入れると、採点器が採点でき
                    # ない対まで率に混ざる (prompt_len 検査で落ちた対など)。
                    if c not in gens or row.get(f"{c}_status") != "ok":
                        continue
                    try:
                        tr[c] = transcribe_cond(asr, c, gens[c]["audio"],
                                                bundle.sampling_rate, args.lang, pair=pair)
                        row[f"{c}_transcript"] = tr[c]
                    except Exception as e:
                        asr_failed.append(c)
                        row[f"{c}_asr_err"] = f"{type(e).__name__}: {e}"[:200]
                if "bl" in tr:
                    row["source_in_bl"] = in_text(pair["word_a"], tr["bl"], args.lang)
                if "tgt" in tr:
                    row["target_in_tgt"] = in_text(pair["word_b"], tr["tgt"], args.lang)
                if "inj" in tr:
                    row["target_in_inj"] = in_text(pair["word_b"], tr["inj"], args.lang)
                    row["source_in_inj"] = in_text(pair["word_a"], tr["inj"], args.lang)
                if asr_failed:
                    row["asr_fail_conds"] = asr_failed

            # ---- 対としての成否。status は **生成** の成否を表す ----
            if fail_conds or check_failed:
                row["status"] = "error"
                row["fail_conds"] = sorted(set(fail_conds))
                row["fail_stage"] = (fail_conds[0] if fail_conds else check_failed)
                n_err += 1
            else:
                row["status"] = "ok"
                n_done += 1
            rows_by_idx[pidx] = row

            if (i + 1) % args.log_every == 0:
                seen = set(attempted[: i + 1])
                cur = [r for r in rows_by_idx.values() if int(r.get("pair_idx", -1)) in seen]
                n_ok_now = sum(1 for r in cur if r.get("status") == "ok")
                # 途中経過の率も分母は「ここまでに試行した対の総数」。
                sb = _rate(cur, "source_in_bl", len(seen))
                ti = _rate(cur, "target_in_inj", len(seen))
                el = time.time() - t_start
                per = el / max(1, n_done)
                print(f"  [{i + 1}/{len(pairs)}] ok={n_ok_now}/{len(seen)} err={n_err} "
                      f"skip={n_skip} source_in_bl={sb} target_in_inj={ti} "
                      f"(分母={len(seen)}, {per:.1f}s/pair, "
                      f"eta {per * (len(pairs) - i - 1) / 60:.0f}min)",
                      flush=True)
            if (i + 1) % args.save_every == 0:
                flush()
    except BaseException as _e:                     # KeyboardInterrupt も拾う
        interrupted = _e
        print(f"[warn] 実行が中断した: {type(_e).__name__}: {_e}", flush=True)

    # ---------------- ディスクと記録の突き合わせ ----------------
    # ここを通ったあと、wav_dir に残る s####_{c}.wav は「その条件を ok と記録した
    # 対のもの」だけになる。中断したときも必ず通す。
    recon_q, recon_w = reconcile_wav_dir(
        args.wav_dir, man, rows_by_idx, scope, args, run_id, trusted,
        interrupted=(interrupted is not None))
    quarantined += recon_q
    prov_warnings += recon_w
    if interrupted is not None:
        prov_warnings.append(
            f"実行は最後まで走っていない ({type(interrupted).__name__}: "
            f"{str(interrupted)[:150]})。results.json は途中までの記録。"
            f"率の分母は --n のままなので、未着手の対は失敗として数えてある。")
    run_stats = {"n_generated_this_run": n_done, "n_resumed_skipped": n_skip,
                 "n_error_this_run": n_err,
                 "n_scorable_pairs": len(scorable_pairs(args.wav_dir, scope)),
                 "interrupted": (None if interrupted is None
                                 else f"{type(interrupted).__name__}: "
                                      f"{str(interrupted)[:150]}")}
    meta.update(run_stats)

    summary = flush(final=True)
    print(f"\n[info] saved {results_path}", flush=True)
    print(f"[info] attempted={len(attempted)} generated={n_done} resumed={n_skip} "
          f"error={n_err} elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
    report(summary)
    if quarantined:
        print(f"\n[注意] 記録と合わない wav {len(quarantined)} 本を "
              f"_stale_{run_id}_* に退避した。削除はしていないので、中身を確かめてから "
              f"手で処分すること。", flush=True)
    print("\n次に採点する (既存の採点器をそのまま使う):", flush=True)
    print(f"  SLURM_TIME=02:00:00 slurm_run.sh \\\n"
          f"    python3 scripts/per_fix/phoneme_per_v3.py \\\n"
          f"      --wav_dir {args.wav_dir} --lang {args.lang} --n {args.n} \\\n"
          f"      --output_json {os.path.join(args.out_dir, 'per_v3.json')} \\\n"
          f"      --pairs_json {results_path}", flush=True)
    if interrupted is not None:
        # 記録と台帳を残したうえで、中断は中断として落とす (成功で終わらせない)。
        raise interrupted


if __name__ == "__main__":
    main()
