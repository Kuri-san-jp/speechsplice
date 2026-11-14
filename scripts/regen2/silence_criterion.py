#!/usr/bin/env python3
"""regen2 の無音の基準（1 か所だけで定義する共通の部品, 2026-09-29）.

基準（著者の決定。regen2 のすべての作業で同じものを使う）:
  wav が「無音」とは、次のどちらかが成り立つこと。
    (a) RMS < 0.005 かつ peak < 0.05
    (b) 20 ms の区間（重ならない区間。最後の半端は捨てる）のうち、区間 RMS > 0.01 の割合 < 0.05

計算のしかたは appk_silent_scan.py の measure / silent_reasons と同じ
（float32 で読み、多チャンネルは平均、RMS は float64 で計算、区間長 = round(sr * 0.02)、
 区間が 1 つも取れない短い wav は、全体の RMS > 0.01 なら active_frac = 1、そうでなければ 0）。

numpy と soundfile だけを使う（torch は読まない）。GPU 側・CPU 側のどちらからでも読める。
使い方:
  from silence_criterion import assess_file, assess_array, CRITERION_ID
  a = assess_file("x.wav")   # {"sr","dur","rms","peak","active_frac","silent_reasons","silent", "criterion_id"}
試験:
  python3 scripts/regen2/silence_criterion.py --selftest
"""
import sys

RMS_TH = 0.005
PEAK_TH = 0.05
FRAME_MS = 20
FRAME_RMS_TH = 0.01
ACTIVE_FRAC_TH = 0.05
CRITERION_ID = "regen2-silence-2026-09-29"
CRITERION_TEXT = ("silent iff (RMS < 0.005 and peak < 0.05) or "
                  "(fraction of 20 ms frames with frame RMS > 0.01) < 0.05")


def measure_array(x, sr):
    """RMS / peak / 長さ / 声のある 20 ms 区間の割合."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.reshape(-1)
    sr = int(sr)
    n = len(x)
    if n == 0:
        return {"sr": sr, "dur": 0.0, "rms": 0.0, "peak": 0.0, "active_frac": 0.0}
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x)))
    flen = max(1, int(round(sr * FRAME_MS / 1000.0)))
    nf = n // flen
    if nf == 0:
        act = 1.0 if rms > FRAME_RMS_TH else 0.0
    else:
        fr = x[: nf * flen].astype(np.float64).reshape(nf, flen)
        frms = np.sqrt(np.mean(fr ** 2, axis=1))
        act = float(np.mean(frms > FRAME_RMS_TH))
    return {"sr": sr, "dur": round(n / sr, 4), "rms": rms, "peak": peak, "active_frac": act}


def measure_file(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return measure_array(x, sr)


def silent_reasons(m):
    r = []
    if m["rms"] < RMS_TH and m["peak"] < PEAK_TH:
        r.append("rms_and_peak")
    if m["active_frac"] < ACTIVE_FRAC_TH:
        r.append("active_frac")
    return r


def _assess(m):
    r = silent_reasons(m)
    return dict(m, silent_reasons=r, silent=bool(r), criterion_id=CRITERION_ID)


def assess_array(x, sr):
    return _assess(measure_array(x, sr))


def assess_file(path):
    return _assess(measure_file(path))


def selftest():
    import numpy as np
    sr = 16000
    t = np.arange(sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    # 小さいが声のある wav: 50 区間のうち 3 区間（6%）だけ区間 RMS 約 0.015、1 標本だけ 0.06。
    # 全体の RMS は約 0.0037 < 0.005 だが peak >= 0.05 かつ active_frac = 0.06 >= 0.05 なので無音ではない
    quiet = np.zeros(sr)
    for k in (10, 25, 40):
        s = k * 320
        quiet[s:s + 320] = 0.015 * np.sqrt(2) * np.sin(2 * np.pi * 300 * t[s:s + 320])
    quiet[25 * 320 + 7] = 0.06
    cases = {
        "tone": (tone, False, []),
        "zeros": (np.zeros(sr), True, ["rms_and_peak", "active_frac"]),
        "quiet_but_voiced": (quiet, False, []),
        # 大きい click が 1 つだけ: (a) には掛からないが (b) で無音
        "single_click": (np.concatenate([np.zeros(sr - 10), np.full(10, 0.9)]), True, ["active_frac"]),
        # 小さい一様な雑音: (a) と (b) の両方
        "low_noise": (0.001 * np.random.RandomState(0).randn(sr), True, ["rms_and_peak", "active_frac"]),
        "too_short_loud": (np.full(100, 0.5), False, []),
        "empty": (np.zeros(0), True, ["rms_and_peak", "active_frac"]),
    }
    out = {}
    for name, (x, want, reasons) in cases.items():
        a = assess_array(x, sr)
        assert a["silent"] is want and a["silent_reasons"] == reasons, (name, a)
        out[name] = {k: a[k] for k in ("rms", "peak", "active_frac", "silent", "silent_reasons")}
    return out


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        import json
        print(json.dumps(selftest(), indent=1))
        print("[silence_criterion] selftest OK:", CRITERION_TEXT)
    else:
        print(__doc__)
