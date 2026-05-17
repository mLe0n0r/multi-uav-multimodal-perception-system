# multi-agent-perception-flying-networks

## Structure

```
perception/
  vision/              # runs on drone
    code/              # integrated detection + localization pipeline
    notebooks/         # development notebooks
    data/              # sample images, telemetry, fire_labels, weights
    fire-detection/    # fire/smoke detection project (training & detect.py)
audioPerception/       # (to be reorganized)
datasets/              # YOLO training dataset (objects)
```

`data/fire_labels/` are labels produced with the fire-detection model. Runtime weights live in `data/weights/` (`yolov5s.pt` for fire, `best.pt` for objects — copy `best.pt` manually).

## Vision pipeline

From repo root:

```bash
python perception/vision/code/integration_pipeline.py \
  --image perception/vision/data/daylight_images/00000.png \
  --telemetry perception/vision/data/telemetryData/00000.txt \
  --mode day
```

Place `best.pt` in `perception/vision/data/weights/` before running (see `perception/vision/data/weights/README.md`).

## Notebooks

Open and run from `perception/vision/notebooks/`. Paths point to `../data/`.
