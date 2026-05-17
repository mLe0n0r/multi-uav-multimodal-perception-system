# Multi-Agent Perception for Flying Networks

Repository for the perception components of a multi-agent aerial system: vision processing on the drone and audio processing on the edge node.

## Repository structure

```
perception/
  vision/              # Drone: object detection, localization, fire detection
    code/
    notebooks/
    data/
    fire-detection/
  audio/               # Edge: audio generation and transcription
    code/
    notebooks/
    data/
datasets/              # YOLO training dataset (object detection)
```

**Vision:** Fire-detection labels are stored under `data/fire_labels/`. Model weights are under `data/weights/` (`yolov5s.pt` is version-controlled; `best.pt` must be supplied locally).

**Audio:** Pipeline artefacts follow `data/descriptions` → `data/dialogues` → `data/generated_audios` → `data/transcriptions`. Copy `perception/audio/.env.example` to `perception/audio/.env` and configure the ElevenLabs API key before running the generation notebook.

## Vision pipeline

Execute from the repository root:

```bash
python perception/vision/code/integration_pipeline.py \
  --image perception/vision/data/daylight_images/00000.png \
  --telemetry perception/vision/data/telemetryData/00000.txt \
  --mode day
```

Place `best.pt` in `perception/vision/data/weights/` prior to execution. See `perception/vision/data/weights/README.md` for details.

## Notebooks

- **Vision:** `perception/vision/notebooks/` — input paths are relative to `../data/`.
- **Audio:** `perception/audio/notebooks/` — input paths are relative to `../data/`. Requires `perception/audio/.env` for audio generation.
