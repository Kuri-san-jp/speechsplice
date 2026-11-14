# Audio licensing, provenance and attribution

This file covers every audio file under `docs/audio/`. The same statements appear at
the end of `docs/index.html`.

## 1. What this audio is

**Everything under `audio/en/`, `audio/ja/`, `audio/zh/` and `audio/ja_het/` is
synthetic speech.** Each clip is the output of a frozen T5Gemma-TTS model performing
zero-shot voice cloning from a prompt recording. No clip is a recording of a real
person saying the sentence it is paired with, and no speaker uttered, consented to or
is aware of these particular sentences.

The single file under `audio/prompt/` is the exception: it is a **real recording of a
human speaker**, taken from the JSUT corpus, and it is the voice-cloning prompt for
every Japanese clip on the page. It is kept in a separate directory, and in a separate
section of the page, because the listeners in the paper's listening test heard only
synthetic clips — never this recording.

## 2. Terms under which this audio is offered

The synthetic clips are offered **for non-commercial research use**. That is the
strictest of the conditions attached to the resources they derive from.

This page does **not** place the synthetic clips under CC BY-SA 4.0. The reason is the
JSUT audio used as the Japanese voice prompt: JSUT's *audio* is not distributed under a
Creative Commons licence at all (its terms are quoted in section 4; only JSUT's *text*
is CC BY-SA 4.0), so no blanket copyleft grant can be made over speech that clones that
speaker.

The Japanese same-surface **sentence texts** are a separate matter. They come from
`ja-tts-g2p-bench`, which is released under CC BY-SA 4.0, so that share-alike condition
continues to apply to those sentence texts and to adaptations of them.

The JSUT prompt recording remains under the JSUT terms of use quoted in section 4:
academic research, non-commercial research, or personal use, with no redistribution of
the corpus.

For terms beyond non-commercial research use, contact the rights holders of the
underlying corpora directly. The JSUT page lists a contact address for commercial use.

## 3. File-by-file provenance

All files are WAV, 16-bit PCM, mono. Nothing was re-encoded, resampled or trimmed for
this page; each file is a byte-for-byte copy of the file it came from.

| Directory | Files | Rate | What it is | Voice prompt speaker |
|---|---|---|---|---|
| `audio/en/` | 20 (10 pairs) | 44.1 kHz | Synthetic. English proper-name stimuli from the AB selection test. | LibriTTS `test-clean` utterance `260_123440_000002_000004` (not redistributed) |
| `audio/ja/` | 20 (10 pairs) | 44.1 kHz | Synthetic. Japanese proper-name stimuli from the AB selection test. | JSUT `BASIC5000_0011` |
| `audio/zh/` | 20 (10 pairs) | 44.1 kHz | Synthetic. Chinese proper-name stimuli from the AB selection test. | BZNSYP utterance `000801`, resampled to 44.1 kHz mono (not redistributed) |
| `audio/ja_het/` | 20 (10 pairs) | 44.1 kHz | Synthetic. Japanese same-surface correction stimuli from the AB selection test. | JSUT `BASIC5000_0011` |
| `audio/prompt/` | 1 | 48 kHz | **Real human recording.** The JSUT voice-cloning prompt. | — (this *is* the prompt) |

Filenames follow `{set}-{item id}-{condition}.wav`, where `condition` is `baseline`
(unmodified synthesis) or `speechsplice` (encoder-output splice applied). Within a
pair the two clips share input text, speaker prompt, decoding configuration and random
seed, and have identical duration; they differ only in the splice.

Item ids included:

- `en` — 0001, 0002, 0009, 0013, 0016, 0017, 0018, 0023, 0035, 0044
- `ja` — 0005, 0006, 0012, 0023, 0026, 0031, 0037, 0043, 0053, 0054
- `zh` — 0005, 0006, 0012, 0023, 0026, 0031, 0037, 0043, 0053, 0054
- `ja_het` — 1013, 1020, 2001, 2007, 2052, 2060, 2063, 2072, 3001, 3008

The prompt file:

| File | Corpus item | Transcript | Rate | Duration | Processing |
|---|---|---|---|---|---|
| `audio/prompt/jsut-basic5000-0011.wav` | JSUT `basic5000` / `BASIC5000_0011` | 木曽川は、しばしば日本のライン川と呼ばれている。 | 48 kHz | 3.58 s | None. Byte-for-byte identical to the corpus file: no cut, no concatenation, no loudness change, no resampling. |

## 4. Attribution for third-party resources

### JSUT corpus — Japanese voice-cloning prompt, and the prompt recording published here

- **Authors:** Ryosuke Sonobe, Shinnosuke Takamichi and Hiroshi Saruwatari
  (Saruwatari Lab, The University of Tokyo). Shinnosuke Takamichi is the main
  contributor.
- **Source:** https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- **Licence, as stated by the distributor:** the corpus has no single licence.
  - *Text*: CC BY-SA 4.0 and comparable terms. For `basic5000` specifically, the
    distributed `LICENCE.txt` lists Wikipedia (CC BY-SA 3.0), the Tanaka Corpus
    (CC BY 2.0), and the authors' own sentences (CC BY-SA 4.0).
  - *Audio*: **not** under a Creative Commons licence. The distributed terms of use
    read that the audio data "may be used for — Research by academic institutions;
    Non-commercial research, including research conducted within commercial
    organizations; Personal use, including blog posts", and that
    "Re-distribution is not permited, but you can upload a part of this corpus
    (e.g., ~100 audio files) in your webpage or blog."
  - The single prompt file published here falls inside that allowance. The corpus
    itself is not redistributed. Commercial use requires contacting the authors.
- **Citation:** Ryosuke Sonobe, Shinnosuke Takamichi and Hiroshi Saruwatari,
  "JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech
  synthesis," arXiv preprint 1711.00354, 2017.

### ja-tts-g2p-bench — source of the Japanese same-surface items

- **Author:** kawajiri-tellernovel (filmapp)
- **Source:** https://huggingface.co/datasets/kawajiri-tellernovel/ja-tts-g2p-bench
- **Licence:** CC BY-SA 4.0, as declared on the dataset card.
- **Suggested credit:** "ja-tts-g2p-bench: Japanese TTS Kanji Reading Benchmark" by
  filmapp (2025).
- The same-surface sentences come from this benchmark. Items that were not uniquely
  answerable were replaced, and the set was filled to 50 items with hard-to-read
  Japanese place names and surnames drawn from Wikipedia.

### LibriTTS — English voice-cloning prompt (not redistributed here)

- **Authors:** Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia,
  Zhifeng Chen and Yonghui Wu
- **Source:** https://www.openslr.org/60/
- **Licence:** CC BY 4.0
- The 3.25-second `test-clean` utterance `260_123440_000002_000004` served as the English speaker prompt. The
  recording itself is not published here.

### BZNSYP / Chinese Standard Mandarin Speech Corpus — Chinese voice-cloning prompt (not redistributed here)

- **Distributor:** Beijing DataBaker Technology
- **Source:** https://www.data-baker.com/en/#/data/index/source
- **Licence:** released by the distributor for non-commercial use; not a Creative
  Commons corpus.
- One excerpt, resampled to 44.1 kHz mono, served as the Chinese speaker prompt. The
  excerpt itself is not published here.

### T5Gemma-TTS — the frozen model that produced every synthetic clip

- **Author:** Aratako
- **Source:** https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b
- Used without modification and without retraining. SpeechSplice changes nothing in
  the model weights. See the model card for its own licence terms.

### Wikipedia — source of the additional place names and surnames

- **Source:** https://www.wikipedia.org/
- **Licence:** CC BY-SA 4.0
- Articles on hard-to-read Japanese place names and surnames supplied the items that
  filled the same-surface set to 50.
