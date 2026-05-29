# Vision perception

Each UAV view shows the **incident scene from above**. Vision perception **detects and localizes** what is visible: people and vehicles (distinguishing normal from emergency vehicles), whether **fire** is present, and where each entity lies in the scene, including how reliable that localization is, for fusion with the audio stream.

## Repository structure

```text
perception/vision/
  ├─ code/integration_pipeline.py   → visual.json (live YOLO + MobileNet + fire)
  ├─ input/
  │     ├─ images/                  → frame PNGs
  │     ├─ telemetry/               → camera pose per frame
  │     └─ fire_labels/             → fire-detection training labels
  ├─ weights/                       → best.pt, mobilenet_best.pth, yolov5s.pt
  ├─ results/
  │     ├─ annotated_imgs/<scenario>/  → frames with boxes drawn
  │     ├─ labels/                  → YOLO predictions
  │     └─ data.yaml                → YOLO dataset config
  └─ fire-detection/                → YOLOv5 fire inference backend
```

## Models and roles

| Component | Model | Role |
|-----------|--------|------|
| Object detection | **YOLO** (`best.pt`, Ultralytics) | Detects **persons** and **vehicles** (inference at 640 and 1280 px) |
| Vehicle Distinction | **MobileNet V3 Small** (`mobilenet_best.pth`) | Classifies each vehicle crop as **normal** or **emergency** |
| Fire detection | **YOLOv5** (`yolov5s.pt`, `fire-detection/`) | State-of-Art model |
| Localization | Camera telemetry + geometry | 3D position, `localization_confidence`, `distance_to_fire` |

## Day vs night processing

| Mode | Flag | Processing |
|------|------|------------|
| Daylight | `--mode day` | Image passed directly to YOLO; persons from 1280 px pass, vehicles from 640 px pass + MobileNet |
| Night | `--mode night` | **Gamma correction** on the image before YOLO; same YOLO + MobileNet pipeline on the enhanced frame |

Batch runs may cache per-frame JSON under `output/.cache/vision/` before copying into each run’s `perception/` folder.

## Weights

Place `best.pt` and `mobilenet_best.pth` in `weights/`. Fire weights: `yolov5s.pt` in `weights/` or `fire-detection/`.
