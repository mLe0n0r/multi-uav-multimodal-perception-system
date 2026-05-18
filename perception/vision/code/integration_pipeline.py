from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VISION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VISION_DIR / "data"
WEIGHTS_DIR = DATA_DIR / "weights"
FIRE_DETECTION_DIR = VISION_DIR / "fire-detection"

CLASS_NAME_TO_ID: Dict[str, int] = {
    "person": 0,
    "normal_vehicle": 1,
    "emergency_vehicle": 2,
    "fire": 3,
}


def format_meters(value: float) -> str:
    return f"{float(value):.2f} m"


def format_confidence(value: float) -> str:
    return f"{float(value):.2f}"


def format_degrees(value: float) -> str:
    return f"{float(value):.2f} deg"


def load_yolo_model():
    weights = WEIGHTS_DIR / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. "
            "Place best.pt in perception/vision/data/weights/ (not version-controlled)."
        )
    return YOLO(str(weights))


def _ensure_fire_detection_path() -> None:
    root = str(FIRE_DETECTION_DIR.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class FireYOLOv5Model:
    """Wrapper for YOLOv5 fire weights (not compatible with ultralytics YOLOv8)."""

    backend: Any
    device: torch.device
    stride: int
    names: Any
    fire_class_ids: Set[int]


def _resolve_fire_weights(fire_weights: str | Path | None) -> Path:
    if fire_weights:
        weights_path = Path(fire_weights)
        if weights_path.exists():
            return weights_path
        raise FileNotFoundError(f"Fire weights not found: {weights_path}")

    for candidate in (WEIGHTS_DIR / "yolov5s.pt", FIRE_DETECTION_DIR / "yolov5s.pt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Fire weights not found. Place yolov5s.pt in {WEIGHTS_DIR} or {FIRE_DETECTION_DIR}."
    )


def _fire_class_indices(names: Any) -> Set[int]:
    if isinstance(names, dict):
        pairs = ((int(k), v) for k, v in names.items())
    else:
        pairs = enumerate(names)
    indices = {int(i) for i, n in pairs if "fire" in str(n).lower()}
    return indices if indices else {1}


def load_fire_model(fire_weights: str | Path | None = None) -> FireYOLOv5Model:
    weights_path = _resolve_fire_weights(fire_weights)
    _ensure_fire_detection_path()

    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    device = select_device(DEVICE if DEVICE == "cuda" else "cpu")

    # PyTorch >= 2.6 defaults to weights_only=True; YOLOv5 checkpoints need the full pickle.
    _orig_torch_load = torch.load

    def _torch_load_yolov5(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_yolov5  # type: ignore[assignment]
    try:
        backend = DetectMultiBackend(str(weights_path), device=device, dnn=False, data=None)
    finally:
        torch.load = _orig_torch_load
    stride = int(backend.stride.max().item()) if hasattr(backend.stride, "max") else int(backend.stride)
    names = backend.names
    return FireYOLOv5Model(
        backend=backend,
        device=device,
        stride=stride,
        names=names,
        fire_class_ids=_fire_class_indices(names),
    )


def load_mobilenet_model(device=DEVICE):
    weights = WEIGHTS_DIR / "mobilenet_best.pth"
    if not weights.exists():
        raise FileNotFoundError(f"MobileNet weights not found: {weights}")

    mobilenet = models.mobilenet_v3_small(weights=None)
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)
    mobilenet.load_state_dict(torch.load(weights, map_location=device))
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    return mobilenet


def get_transform():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def process_daylight_img(img, yolo_model, mobilenet, transform, conf_threshold=0.1):
    results_640 = yolo_model(img, conf=conf_threshold, imgsz=640, verbose=False)
    results_1280 = yolo_model(img, conf=conf_threshold, imgsz=1280, verbose=False)

    detections = []

    for r in results_1280:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name != "person":
                continue
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, "person", conf, (255, 0, 0)))

    for r in results_640:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name == "person":
                continue

            yolo_conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            input_tensor = transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = mobilenet(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            p_emergency = probs[0].item()
            p_normal = probs[1].item()
            if p_emergency > p_normal:
                class_name = "emergency_vehicle"
                cls_conf = p_emergency
            else:
                class_name = "normal_vehicle"
                cls_conf = p_normal

            final_conf = round(yolo_conf, 2)
            cls_conf = round(cls_conf, 2)

            if class_name == "emergency_vehicle":
                if cls_conf < 0.9:
                    if cls_conf <= 0.75:
                        continue
                    if final_conf < 0.4:
                        continue
            else:
                if final_conf < 0.4:
                    continue

            color = (0, 0, 255) if class_name == "emergency_vehicle" else (0, 255, 0)
            detections.append((x1, y1, x2, y2, class_name, final_conf, color))

    return detections, img


def process_night_img(img, yolo_model, mobilenet, transform):
    img_yolo = adjust_gamma(img, gamma=1.5)

    results_640 = yolo_model(img_yolo, conf=0.4, imgsz=640, verbose=False)
    results_1280 = yolo_model(img_yolo, conf=0.4, imgsz=1280, verbose=False)

    detections = []

    for r in results_1280:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name != "person":
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, "person", conf, (255, 0, 0)))

    for r in results_640:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name == "person":
                continue

            yolo_conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img_yolo[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            input_tensor = transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = mobilenet(input_tensor)
                temperature = 2
                probs = torch.softmax(outputs / temperature, dim=1)[0]

            p_emergency = probs[0].item()
            p_normal = probs[1].item()
            if p_emergency > p_normal:
                class_name = "emergency_vehicle"
            else:
                class_name = "normal_vehicle"

            final_conf = yolo_conf
            if final_conf < 0.4:
                continue

            color = (0, 0, 255) if class_name == "emergency_vehicle" else (0, 255, 0)
            detections.append((x1, y1, x2, y2, class_name, final_conf, color))

    return detections, img_yolo


def process_fire_img(
    img: np.ndarray,
    fire_model: FireYOLOv5Model,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> List[Tuple[int, int, int, int, str, float, Tuple[int, int, int]]]:
    _ensure_fire_detection_path()
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords

    im0 = img.copy()
    im, _, _ = letterbox(im0, new_shape=(imgsz, imgsz), stride=fire_model.stride, auto=True)
    im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])

    tensor = torch.from_numpy(im).to(fire_model.device).float() / 255.0
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    pred = fire_model.backend(tensor)
    pred = non_max_suppression(pred, conf_threshold, 0.45)

    detections: List[Tuple[int, int, int, int, str, float, Tuple[int, int, int]]] = []
    for det in pred:
        if det is None or len(det) == 0:
            continue
        det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det:
            if int(cls) not in fire_model.fire_class_ids:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append((x1, y1, x2, y2, "fire", float(conf), (255, 0, 255)))

    return detections


def intrinsics_from_fov(W, H, fov_x_deg):
    fov_x = math.radians(fov_x_deg)
    fx = (W / 2.0) / math.tan(fov_x / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def rot_from_unreal(pitch_deg, yaw_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


def ray_from_pixel_unreal(u, v, K, R_wc):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy

    ray_cam = np.array([1.0, -x, y], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)

    ray_world = R_wc @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    return ray_world


def read_telemetry(telemetry_data):
    with open(telemetry_data, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    values = list(map(float, line.split()))
    if len(values) != 6:
        raise ValueError(f"Telemetry invalid: {telemetry_data}")
    return tuple(values)  # (x, y, z, pitch, yaw, roll)


def localization_confidence(u, v, C, K, R_wc, bbox_w_px, bbox_h_px, class_name):
    def _project(uu, vv):
        d = ray_from_pixel_unreal(uu, vv, K, R_wc)
        return C + (-C[2] / d[2]) * d

    eps = 0.5
    P0 = _project(u, v)
    dPdu = (_project(u + eps, v) - P0) / eps
    dPdv = (_project(u, v + eps) - P0) / eps

    sens_x = float(np.hypot(dPdu[0], dPdv[0]))
    sens_y = float(np.hypot(dPdu[1], dPdv[1]))

    ref = abs(float(C[2])) / float(K[0, 0])
    conf_geom_x = float(min(1.0, ref / max(sens_x, 1e-12)))
    conf_geom_y = float(min(1.0, ref / max(sens_y, 1e-12)))

    sigma_u_det = max(1.0, 0.02 * bbox_w_px)
    sigma_v_det = max(2.0, 0.06 * bbox_h_px)

    if class_name in ("normal_vehicle", "emergency_vehicle"):
        sigma_u_pick = max(5.0, 0.10 * bbox_w_px)
        sigma_v_pick = max(6.0, 0.16 * bbox_h_px)
    elif class_name == "person":
        sigma_u_pick = max(1.5, 0.03 * bbox_w_px)
        sigma_v_pick = max(3.0, 0.10 * bbox_h_px)
    else:
        sigma_u_pick = max(2.0, 0.05 * bbox_w_px)
        sigma_v_pick = max(4.0, 0.12 * bbox_h_px)

    sigma_u = float(math.hypot(sigma_u_det, sigma_u_pick))
    sigma_v = float(math.hypot(sigma_v_det, sigma_v_pick))

    q_u = 1.0 / (1.0 + sigma_u / 8.0)
    q_v = 1.0 / (1.0 + sigma_v / 10.0)

    return {
        "confidence_x": float(conf_geom_x * q_u),
        "confidence_y": float(conf_geom_y * q_v),
    }


def localize_objects_with_confidence_from_labels(labels, telemetry_data, W=1280, H=720, fov_x=90):
    results = []

    x, y, z, pitch, yaw, roll = read_telemetry(telemetry_data)
    C = np.array([x, y, z], dtype=float)
    R_wc = rot_from_unreal(pitch, -yaw, roll)
    K = intrinsics_from_fov(W, H, fov_x)

    for obj_id, (cls, xc, yc, w, h) in enumerate(labels):
        cls = int(cls)
        if cls not in [0, 1, 2, 3]:
            continue

        u = xc * W
        v = (yc + h / 2) * H

        d = ray_from_pixel_unreal(u, v, K, R_wc)
        if abs(d[2]) < 1e-6:
            continue

        t = -C[2] / d[2]
        P = C + t * d

        if cls == 0:
            name = "person"
        elif cls == 1:
            name = "normal_vehicle"
        elif cls == 2:
            name = "emergency_vehicle"
        else:
            name = "fire"

        q = localization_confidence(
            u,
            v,
            C,
            K,
            R_wc,
            bbox_w_px=w * W,
            bbox_h_px=h * H,
            class_name=name,
        )

        results.append(
            {
                "id": obj_id,
                "class": name,
                "position": P,
                "confidence_x": q["confidence_x"],
                "confidence_y": q["confidence_y"],
            }
        )

    return results


def detections_to_labels(detections, W, H):
    labels = []
    detection_confidences = []

    for x1, y1, x2, y2, class_name, det_conf, _ in detections:
        cls = CLASS_NAME_TO_ID.get(class_name)
        if cls is None:
            continue

        x1 = min(max(float(x1), 0.0), float(W))
        y1 = min(max(float(y1), 0.0), float(H))
        x2 = min(max(float(x2), 0.0), float(W))
        y2 = min(max(float(y2), 0.0), float(H))
        if x2 <= x1 or y2 <= y1:
            continue

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        labels.append((cls, xc / W, yc / H, bbox_w / W, bbox_h / H))
        detection_confidences.append(float(det_conf))

    return labels, detection_confidences


def run_integrated_pipeline(
    image_path,
    telemetry_path,
    mode="day",
    fov_x=90,
    fire_weights: str | Path | None = None,
    fire_conf=0.25,
    fire_imgsz=640,
):
    image_path = Path(image_path)
    if not image_path.exists():
        candidate_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        suggestions = [
            str(image_path.with_suffix(ext))
            for ext in candidate_exts
            if image_path.with_suffix(ext).exists()
        ]
        suggestion_msg = f" Did you mean: {suggestions[0]}?" if suggestions else ""
        raise FileNotFoundError(f"Image path not found: {image_path}.{suggestion_msg}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    H, W = img.shape[:2]
    yolo_model = load_yolo_model()
    mobilenet = load_mobilenet_model(DEVICE)
    fire_model = load_fire_model(fire_weights)
    transform = get_transform()

    if mode == "night":
        detections, _ = process_night_img(img, yolo_model, mobilenet, transform)
    else:
        detections, _ = process_daylight_img(img, yolo_model, mobilenet, transform)

    fire_detections = process_fire_img(
        img=img,
        fire_model=fire_model,
        conf_threshold=fire_conf,
        imgsz=fire_imgsz,
    )
    detections.extend(fire_detections)

    labels, detection_confidences = detections_to_labels(detections, W=W, H=H)
    localized = localize_objects_with_confidence_from_labels(
        labels=labels,
        telemetry_data=str(telemetry_path),
        W=W,
        H=H,
        fov_x=fov_x,
    )

    x, y, z, pitch, yaw, roll = read_telemetry(str(telemetry_path))
    fire_positions = [obj["position"] for obj in localized if obj["class"] == "fire"]
    has_fire = len(fire_positions) > 0

    def distance_to_fire_str(position: np.ndarray):
        if not fire_positions:
            return None
        min_dist = min(float(np.linalg.norm(position - fire_pos)) for fire_pos in fire_positions)
        return format_meters(min_dist)

    objects = []
    for obj in localized:
        if obj["class"] == "fire":
            continue

        obj_id = int(obj["id"])
        pos = obj["position"]
        det_conf = detection_confidences[obj_id] if obj_id < len(detection_confidences) else 0.0

        objects.append(
            {
                "id": obj_id,
                "class": obj["class"],
                "detection_confidence": format_confidence(det_conf),
                "position": {
                    "x": format_meters(pos[0]),
                    "y": format_meters(pos[1]),
                    "z": format_meters(pos[2]),
                },
                "localization_confidence": {
                    "x": format_confidence(obj["confidence_x"]),
                    "y": format_confidence(obj["confidence_y"]),
                    "z": format_confidence(1.0),
                },
                "distance_to_fire": distance_to_fire_str(pos),
            }
        )

    counts = Counter(o["class"] for o in objects if o["class"] in ("person", "normal_vehicle", "emergency_vehicle"))
    return {
        "has_fire": has_fire,
        "camera": {
            "position": {"x": format_meters(x), "y": format_meters(y), "z": format_meters(z)},
            "orientation": {
                "pitch": format_degrees(pitch),
                "yaw": format_degrees(yaw),
                "roll": format_degrees(roll),
            },
        },
        "counts_by_class": {
            "person": int(counts.get("person", 0)),
            "normal_vehicle": int(counts.get("normal_vehicle", 0)),
            "emergency_vehicle": int(counts.get("emergency_vehicle", 0)),
        },
        "objects": objects,
    }


def dumps_output(output, indent=2):
    return json.dumps(output, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated object detection + localization pipeline")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry txt file")
    parser.add_argument("--mode", choices=["day", "night"], default="day")
    parser.add_argument("--fov-x", type=float, default=90.0)
    parser.add_argument(
        "--fire-weights",
        default=str(WEIGHTS_DIR / "yolov5s.pt"),
        help="Path to fire model weights",
    )
    parser.add_argument("--fire-conf", type=float, default=0.25, help="Confidence threshold for fire detector")
    parser.add_argument("--fire-imgsz", type=int, default=640, help="Inference size for fire detector")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    output = run_integrated_pipeline(
        image_path=Path(args.image),
        telemetry_path=Path(args.telemetry),
        mode=args.mode,
        fov_x=args.fov_x,
        fire_weights=args.fire_weights,
        fire_conf=args.fire_conf,
        fire_imgsz=args.fire_imgsz,
    )

    payload = dumps_output(output)
    print(payload)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Saved to {out_path}")
