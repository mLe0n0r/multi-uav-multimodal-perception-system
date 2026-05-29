# Audio perception

The audio stream records **communications between emergency teams** at an incident. Audio perception **transcribes and structures** those exchanges: what each unit reports, **who** is speaking, and inferred cues (mentioned civilians and vehicles, units on scene, and similar quantities) for fusion with the visual stream.

## Repository structure

```text
perception/audio/
  ├─ code/transcribe_run.py       → transcript.json per run
  ├─ data_preparation/            → build synthetic dataset (notebooks)
  │     ├─ descriptions/
  │     └─ dialogues/
  ├─ input/                       → synthetic WAV per scenario
  ├─ transcriptions/              → WhisperX JSON per audio stem
  └─ models/                      → Whisper / pyannote weights (local, not in git)
```

## Processing

| Step | Implementation | Function |
|------|----------------|----------|
| Dataset prep | Notebooks | descriptions → dialogues → WAV |
| Transcription | **WhisperX** | → `transcriptions/<stem>.json` |
| Post-processing | `transcribe_run.py` | → `perception/transcript.json` |

## Execution flow (batch)

```text
      input/<audio>.wav
              │
              ▼
  transcribe_run.py  (WhisperX)
              │
              ▼
   transcriptions/<stem>.json
              │
              ▼
transcribe_run.py  (post-processing)
              │
              ▼
output/.../perception/transcript.json
```

Use **`--skip-whisperx`** when `transcriptions/<stem>.json` already exists.
