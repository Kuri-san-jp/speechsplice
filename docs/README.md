# SpeechSplice audio sample page

This directory is a self-contained static site that lets a reader listen to the
speech samples reported in the SpeechSplice paper.

## Contents

| Path | What it is |
|---|---|
| `index.html` | The whole page. Single file, no external CSS, JS, fonts, CDN or analytics. |
| `audio/en/`, `audio/ja/`, `audio/zh/` | Synthetic baseline / SpeechSplice pairs from the proper-name AB selection test (10 items each). |
| `audio/ja_het/` | Synthetic baseline / SpeechSplice pairs from the Japanese same-surface correction set (10 items, one per misreading category). |
| `audio/prompt/` | The JSUT voice-cloning prompt. A real human recording, not synthetic — kept apart from the stimuli on purpose. |
| `LICENSE_AUDIO.md` | Licensing, provenance and attribution for every audio file here. |

Every clip in the four stimulus directories is one of the files the listeners in the
paper's AB selection test actually heard, copied unchanged. Within a pair, the baseline
and the SpeechSplice clip share input text, speaker prompt, decoding configuration and
random seed, so they differ only in the encoder-output splice — and they have identical
duration.

## What the page shows

1. The paper title, authors and affiliations.
2. The paper's Abstract, quoted verbatim.
3. The reported AB-selection and naturalness-MOS numbers, transcribed from the paper.
4. The Japanese same-surface correction samples, in their own section: the written
   sentence is unchanged, the baseline misreads the target word, SpeechSplice supplies
   the intended reading. Each row also shows the kana ASR transcript of both clips.
5. The English, Japanese and Chinese proper-name samples.
6. The voice prompt recording, with its processing history.
7. Licensing, provenance and attribution.

Each sample row states whether the majority of that item's ten listener judgments went
to the clip realizing the target.

## Enabling GitHub Pages

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Under **Source**, choose **Deploy from a branch**.
4. Set the branch to `main` and the folder to **`/docs`**, then **Save**.
5. Wait for the build to finish. The page is served at
   `https://<owner>.github.io/<repository>/`.

No build step, no Jekyll configuration and no dependencies are needed: GitHub Pages
serves `index.html` and the WAV files as they are.

## Notes for maintainers

- Total size of `audio/` is about 20 MB; the largest single file is about 0.44 MB, well
  under GitHub's 100 MB per-file limit. Git LFS is not required.
- Audio is referenced with relative paths (`audio/...`), so the page works from any
  repository name or subpath.
- Do not re-encode or normalise the WAV files. They are the evaluated stimuli; changing
  them would break the correspondence with the reported results.
- Read `LICENSE_AUDIO.md` before adding or replacing any audio. In particular, the JSUT
  audio terms permit publishing only a limited portion of that corpus (about 100 files)
  on a web page, and do not permit redistributing the corpus.
