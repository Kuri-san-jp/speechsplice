"""Forced alignment utilities using Whisper word-level timestamps.

Provides:
  - WhisperAligner: pipeline-based wrapper for word-level Whisper transcription
  - find_entity_chunks: locate entity in transcription chunks
  - slice_entity_audio: cut audio np array to chunk span

Designed for entity-only PER computation in SpeechSplice evaluations.
"""
import re
import numpy as np
import torch
import torchaudio


WHISPER_LANG = {"en": "english", "ja": "japanese", "zh": "chinese"}


class WhisperAligner:
    def __init__(self, device="cuda", model_id="openai/whisper-large-v3", dtype=torch.float16):
        from transformers import pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=30,
            return_timestamps="word",
        )
        self.target_sr = 16000

    def _prepare_audio(self, audio, sr_in):
        """Return float32 numpy 1-D at 16 kHz."""
        if isinstance(audio, torch.Tensor):
            a = audio.detach().cpu().float()
            if a.dim() > 1:
                a = a.squeeze()
            audio = a.numpy()
        elif isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
        else:
            audio = np.asarray(audio, dtype=np.float32)

        if sr_in != self.target_sr:
            t = torch.from_numpy(audio).float()
            t = torchaudio.functional.resample(t, sr_in, self.target_sr)
            audio = t.numpy()
        return audio

    def transcribe_words(self, audio, sr_in, language="en"):
        """Return (chunks, audio_np_at_16k).

        chunks: list of {"word": str, "start": float, "end": float}
        On failure returns ([], None).

        For non-space languages (JA/ZH), Whisper often returns a single phrase
        as one chunk; we subdivide such chunks character-by-character with
        proportional timing so entity localization is character-resolved.
        """
        try:
            audio_np = self._prepare_audio(audio, sr_in)
            if audio_np.size < int(0.1 * self.target_sr):
                return [], audio_np
            lang_kw = WHISPER_LANG.get(language, language)
            out = self.pipe(
                audio_np,
                return_timestamps="word",
                generate_kwargs={"language": lang_kw, "task": "transcribe"},
            )
            chunks = []
            for c in out.get("chunks", []):
                ts = c.get("timestamp", (None, None))
                if ts is None or ts[0] is None or ts[1] is None:
                    continue
                chunks.append({"word": c.get("text", ""), "start": float(ts[0]), "end": float(ts[1])})

            if language in ("ja", "zh"):
                chunks = self._subdivide_by_char(chunks)

            return chunks, audio_np
        except Exception as e:
            print(f"  [Whisper] transcribe failed: {str(e)[:160]}")
            return [], None

    @staticmethod
    def _subdivide_by_char(chunks):
        """Split each chunk into character-level subchunks with proportional timing."""
        out = []
        for c in chunks:
            text = c.get("word", "").strip()
            t0, t1 = c["start"], c["end"]
            if not text:
                continue
            # remove punctuation
            chars = [ch for ch in text if ch not in " \u3000\u3001\u3002\uff0c\uff0e\uff1f\uff01.,?!"]
            if not chars:
                continue
            if len(chars) == 1:
                out.append({"word": chars[0], "start": t0, "end": t1})
                continue
            duration = t1 - t0
            per_char = duration / len(chars)
            for i, ch in enumerate(chars):
                out.append({
                    "word": ch,
                    "start": t0 + i * per_char,
                    "end": t0 + (i + 1) * per_char,
                })
        return out


def normalize_for_match(s, lang):
    """Lowercase + strip punctuation/whitespace for fuzzy matching."""
    if not s:
        return ""
    s = s.strip()
    if lang == "en":
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s']", "", s)
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = re.sub(r"[\s\u3000\u3001\u3002\uff0c\uff0e\uff1f\uff01.,?!]", "", s)
    return s


def find_entity_chunks_by_char_position(chunks, entity, lang, input_text):
    """Fallback: locate entity by char position ratio in input_text.

    Useful when Whisper transcription doesn't contain entity exactly
    (e.g., kanji entity transcribed as kana, or mixed scripts).
    Assumes char position in input_text is roughly proportional to
    chunk index (true when len(text) ≈ len(chunks)).
    """
    if not chunks or not input_text:
        return None, None
    norm_text = normalize_for_match(input_text, lang)
    norm_entity = normalize_for_match(entity, lang)
    if not norm_text or not norm_entity:
        return None, None
    char_pos = norm_text.find(norm_entity)
    if char_pos < 0:
        return None, None
    char_end = char_pos + len(norm_entity)
    n_chars = len(norm_text)
    n_chunks = len(chunks)
    s_idx = max(0, min(int((char_pos / n_chars) * n_chunks), n_chunks - 1))
    e_idx = max(0, min(int((char_end / n_chars) * n_chunks), n_chunks - 1))
    if e_idx <= s_idx:
        e_idx = s_idx
    return s_idx, e_idx


def find_entity_chunks(chunks, entity, lang, input_text=None):
    """Find chunk indices that cover `entity` substring across word chunks.

    Strategy:
      1. Concatenate normalized chunk texts.
      2. Locate normalized entity substring by char index.
      3. Map char range back to chunk indices.
      4. If exact match fails and input_text is given, fall back to
         char-position-ratio mapping using input_text.

    Returns (start_idx, end_idx) inclusive or (None, None) on failure.
    """
    if not chunks:
        return None, None
    norm_entity = normalize_for_match(entity, lang)
    if not norm_entity:
        return None, None

    norm_chunks = [normalize_for_match(c["word"], lang) for c in chunks]
    char_offsets = []
    cum = 0
    for nc in norm_chunks:
        char_offsets.append(cum)
        cum += len(nc)
    concat = "".join(norm_chunks)
    if not concat:
        return None, None

    pos = concat.find(norm_entity)
    if pos == -1:
        # partial match for non-space langs
        if lang in ("ja", "zh") and len(norm_entity) >= 2:
            for L in range(len(norm_entity) - 1, 0, -1):
                pos = concat.find(norm_entity[:L])
                if pos != -1:
                    break
        if pos == -1:
            # Final fallback: char-position ratio in input_text
            if input_text is not None:
                return find_entity_chunks_by_char_position(chunks, entity, lang, input_text)
            return None, None
    pos_end = pos + len(norm_entity) - 1

    start_idx = end_idx = None
    for i, off in enumerate(char_offsets):
        end_off = off + len(norm_chunks[i]) - 1
        if start_idx is None and off <= pos <= max(end_off, off):
            start_idx = i
        if off <= pos_end <= max(end_off, off):
            end_idx = i
            break
    if start_idx is None:
        if input_text is not None:
            return find_entity_chunks_by_char_position(chunks, entity, lang, input_text)
        return None, None
    if end_idx is None:
        end_idx = len(chunks) - 1
    return start_idx, end_idx


def slice_entity_audio(audio_np, sr, chunks, start_idx, end_idx, pad_sec=0.05):
    """Slice audio_np to cover chunks[start_idx..end_idx] with small padding."""
    if start_idx is None or end_idx is None:
        return None
    t0 = max(0.0, chunks[start_idx]["start"] - pad_sec)
    t1 = min(len(audio_np) / sr, chunks[end_idx]["end"] + pad_sec)
    s0, s1 = int(t0 * sr), int(t1 * sr)
    if s1 <= s0 + int(0.05 * sr):
        return None
    return audio_np[s0:s1]
