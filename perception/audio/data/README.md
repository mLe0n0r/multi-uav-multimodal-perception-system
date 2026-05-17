# Audio data

| Directory | Description |
|-----------|-------------|
| `descriptions/` | Scenario descriptions used as input to dialogue generation |
| `dialogues/` | Generated dialogues (JSON) |
| `generated_audios/` | Synthetic radio-style audio files (WAV) |
| `transcriptions/` | Transcription and diarization outputs (JSON) |

**Processing order:** descriptions → dialogues → generated_audios → transcriptions.

## Locally created directories (not version-controlled)

The following directories are not included in the repository. Create them under `perception/audio/data/` when required.

**`temp_audio/`** — The generation notebook writes intermediate audio clips to this location (e.g. `perception/audio/data/temp_audio/scenario3_a1/`). These files may be removed after the final WAV has been produced.

**`models/`** — Whisper and pyannote model weights (e.g. `faster-whisper-medium/`, `pyannote-speaker-diarization-3.1/`). Obtain from Hugging Face or copy from a local development environment; they exceed repository size limits.
