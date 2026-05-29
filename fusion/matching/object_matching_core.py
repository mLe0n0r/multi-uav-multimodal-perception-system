"""Core matching logic ported from objectMatching.ipynb (cells 2-10)."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# --- notebook code cell 1 ---
FOV_X_DEG = 90.0
GROUND_Z = 0.0

BIG_COST = 1e9

REPROJ_THRESHOLDS_NORM = {
    0: 3.0,   # person
    1: 2.0,   # normal_vehicle
    2: 2.0,   # emergency_vehicle
}

ANGLE_MIN_RAD = np.deg2rad(3.0)
ANGLE_CLIP_RAD = np.deg2rad(25.0)

DIST_REF = 20.0
DIST_SCALE = 30.0

MAX_UNCERTAINTY = 3.0

MAX_MULTIPLIER = {
    0: 1.25,   # person: pode relaxar mais
    1: 0.9,    # vehicle: relaxar pouco
    2: 1.25,   # emergency_vehicle
}

# --- notebook code cell 3 ---
def intrinsics_from_fov(W, H, fov_x_deg):
    fx = W / (2.0 * np.tan(np.deg2rad(fov_x_deg) / 2.0))
    return np.array(
        [[fx, 0.0, W / 2.0], [0.0, fx, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

# --- notebook code cell 4 ---
def rotation_matrix(axis, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    mats = {
        "x": np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float),
        "y": np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float),
        "z": np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    }
    if axis not in mats:
        raise ValueError(f"Axis inválido: {axis}. Usa 'x', 'y' ou 'z'.")
    return mats[axis]


def rot_from_unreal(pitch_deg, yaw_deg, roll_deg):
    pitch, yaw, roll = np.deg2rad([pitch_deg, yaw_deg, roll_deg])
    return rotation_matrix("z", yaw) @ rotation_matrix("y", pitch) @ rotation_matrix("x", roll)


# --- notebook: data loading (objectMatching.ipynb cell 3) ---
def load_yolo_labels(path: Union[str, Path]) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 5:
                continue
            cls, xc, yc, w, h = int(float(vals[0])), *map(float, vals[1:])
            labels.append({"class": cls, "xc": xc, "yc": yc, "w": w, "h": h})
    return labels


def load_telemetry(path: Union[str, Path]) -> Tuple[np.ndarray, float, float, float]:
    with open(path, "r", encoding="utf-8") as f:
        vals = list(map(float, f.readline().strip().split()))
    if len(vals) != 6:
        raise ValueError(f"Telemetry inválida em {path}: {vals}")
    x, y, z, pitch, yaw, roll = vals
    return np.array([x, y, z], dtype=float), pitch, yaw, roll


def _parse_metric_value(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    match = re.match(r"([\d.+-]+)", text)
    if match:
        return float(match.group(1))
    return float(text)


def telemetry_from_visual(visual: Dict[str, Any]) -> Tuple[np.ndarray, float, float, float]:
    """Same convention as load_telemetry + integration_pipeline camera block."""
    cam = visual.get("camera") or {}
    pos = cam.get("position") or {}
    ori = cam.get("orientation") or {}
    C_w = np.array(
        [
            _parse_metric_value(pos.get("x")),
            _parse_metric_value(pos.get("y")),
            _parse_metric_value(pos.get("z")),
        ],
        dtype=float,
    )
    pitch = _parse_metric_value(ori.get("pitch", "0"))
    yaw = _parse_metric_value(ori.get("yaw", "0"))
    roll = _parse_metric_value(ori.get("roll", "0"))
    return C_w, pitch, yaw, roll


VISUAL_CLASS_TO_YOLO = {
    "person": 0,
    "normal_vehicle": 1,
    "emergency_vehicle": 2,
}


def labels_and_ids_from_visual(
    visual: Dict[str, Any],
    class_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Labels in JSON array order (same order as integration_pipeline objects list)."""
    cmap = class_map or VISUAL_CLASS_TO_YOLO
    labels: List[Dict[str, Any]] = []
    object_ids: List[int] = []
    for obj in visual.get("objects", []):
        bbox = obj.get("bbox")
        if not bbox:
            continue
        cls_name = obj.get("class")
        cls = cmap.get(cls_name)
        if cls is None:
            continue
        labels.append(
            {
                "class": cls,
                "xc": float(bbox["xc"]),
                "yc": float(bbox["yc"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            }
        )
        oid = obj.get("id")
        object_ids.append(int(oid) if oid is not None else len(object_ids))
    return labels, object_ids


def load_frame(
    img_path: Union[str, Path],
    label_path: Union[str, Path],
    telemetry_path: Union[str, Path],
    fov_x_deg: float = FOV_X_DEG,
) -> Dict[str, Any]:
    """Notebook load_frame: H,W from img.shape[:2], R_wc = rot_from_unreal(-pitch, yaw, roll)."""
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Não consegui carregar imagem: {img_path}")

    labels = load_yolo_labels(label_path)
    C_w, pitch, yaw, roll = load_telemetry(telemetry_path)

    H, W = img.shape[:2]
    K = intrinsics_from_fov(W, H, fov_x_deg)
    R_wc = rot_from_unreal(-pitch, yaw, roll)

    return {
        "img_path": str(img_path),
        "img": img,
        "labels": labels,
        "C_w": C_w,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "K": K,
        "R_wc": R_wc,
        "object_ids": list(range(len(labels))),
    }


def load_frame_from_visual(
    visual: Dict[str, Any],
    *,
    img_path: Optional[Union[str, Path]] = None,
    telemetry_path: Optional[Union[str, Path]] = None,
    fov_x_deg: float = FOV_X_DEG,
    class_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Build a notebook-compatible frame dict from perception visual JSON.

    Prefer img_path + telemetry_path when available (identical to load_frame).
    Otherwise uses image_size from JSON and camera block / telemetry file.
    """
    if img_path is not None:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Não consegui carregar imagem: {img_path}")
        H, W = img.shape[:2]
    else:
        size = visual.get("image_size") or {}
        W = int(size.get("width") or 1280)
        H = int(size.get("height") or 720)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img_path = None

    if telemetry_path is not None and Path(telemetry_path).is_file():
        C_w, pitch, yaw, roll = load_telemetry(telemetry_path)
    else:
        C_w, pitch, yaw, roll = telemetry_from_visual(visual)

    labels, object_ids = labels_and_ids_from_visual(visual, class_map=class_map)
    K = intrinsics_from_fov(W, H, fov_x_deg)
    R_wc = rot_from_unreal(-pitch, yaw, roll)

    return {
        "img_path": str(img_path) if img_path is not None else None,
        "img": img,
        "labels": labels,
        "C_w": C_w,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "K": K,
        "R_wc": R_wc,
        "object_ids": object_ids,
    }


# --- notebook code cell 5 ---
def yolo_to_xyxy(label, img_shape):
    """Converte bbox YOLO para formato xyxy em píxeis."""
    H, W = img_shape[:2]
    xc, yc, bw, bh = label["xc"], label["yc"], label["w"], label["h"]
    return label["class"], np.array(
        [
            (xc - bw / 2.0) * W,
            (yc - bh / 2.0) * H,
            (xc + bw / 2.0) * W,
            (yc + bh / 2.0) * H,
        ],
        dtype=float,
    )


def bbox_scale_single(det):
    """Calcula escala de uma deteção pela diagonal da sua bbox."""
    x1, y1, x2, y2 = det["bbox"]
    return max(np.hypot(max(x2 - x1, 1.0), max(y2 - y1, 1.0)), 1.0)

# --- notebook code cell 6 ---
def ray_from_pixel_unreal(u, v, K, R_wc):
    """
    Pixel -> raio no mundo.

    Convenção Unreal:
      camera +X = forward
      camera +Y = right
      camera +Z = up
    """

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = 1.0
    y = (u - cx) / fx
    z = -(v - cy) / fy

    ray_cam = np.array([x, y, z], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)

    ray_world = R_wc @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    return ray_world


def project_pixel_to_ground(u, v, frame, ground_z=0.0):
    """
    Projeta pixel para o plano z=ground_z.
    """

    K = frame["K"]
    R_wc = frame["R_wc"]
    C_w = frame["C_w"]

    ray_world = ray_from_pixel_unreal(u, v, K, R_wc)

    if abs(ray_world[2]) < 1e-9:
        return None, ray_world, None

    t = (ground_z - C_w[2]) / ray_world[2]

    if t <= 0:
        return None, ray_world, t

    P_w = C_w + t * ray_world

    return P_w, ray_world, t


def world_to_pixel_unreal(P_w, frame):
    """
    Projeta ponto 3D do mundo para pixel.
    """

    K = frame["K"]
    R_wc = frame["R_wc"]
    C_w = frame["C_w"]

    P_w = np.asarray(P_w, dtype=float).reshape(3)
    C_w = np.asarray(C_w, dtype=float).reshape(3)

    R_cw = R_wc.T
    P_c = R_cw @ (P_w - C_w)

    X = P_c[0]
    Y = P_c[1]
    Z = P_c[2]

    if X <= 1e-9:
        return None, P_c

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (Y / X) + cx
    v = fy * (-Z / X) + cy

    pt = np.array([u, v], dtype=float)

    return pt, P_c


def triangulate_rays_midpoint(C_a, ray_a, C_b, ray_b, eps=1e-9):
    """
    Triangula um ponto 3D a partir de dois raios no mundo.
    Retorna o ponto médio entre os pontos de menor distância entre as retas.
    """
    C_a = np.asarray(C_a, dtype=float).reshape(3)
    C_b = np.asarray(C_b, dtype=float).reshape(3)
    d1 = np.asarray(ray_a, dtype=float).reshape(3)
    d2 = np.asarray(ray_b, dtype=float).reshape(3)

    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < eps or n2 < eps:
        return None, {"reason": "invalid_ray", "ray_gap_m": np.inf, "triangulation_angle_deg": 0.0}

    d1 = d1 / n1
    d2 = d2 / n2

    A = np.column_stack((d1, -d2))
    b = C_b - C_a

    st, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    s, t = float(st[0]), float(st[1])

    P_a = C_a + s * d1
    P_b = C_b + t * d2

    P_mid = 0.5 * (P_a + P_b)
    ray_gap_m = float(np.linalg.norm(P_a - P_b))

    cosang = np.clip(np.dot(d1, d2), -1.0, 1.0)
    triangulation_angle_deg = float(np.rad2deg(np.arccos(abs(cosang))))

    return P_mid, {
        "reason": "ok",
        "ray_gap_m": ray_gap_m,
        "triangulation_angle_deg": triangulation_angle_deg,
        "s": s,
        "t": t,
    }


def enrich_matches_with_triangulation(matches, dets_a, frame_a, dets_b, frame_b):
    """Anexa posição 3D triangulada a cada match, sem alterar a seleção de matches."""
    C_a = frame_a["C_w"]
    C_b = frame_b["C_w"]

    for m in matches:
        i = m["A"]
        j = m["B"]

        ray_a = dets_a[i].get("ray_world")
        ray_b = dets_b[j].get("ray_world")

        P_tri, tri_info = triangulate_rays_midpoint(C_a, ray_a, C_b, ray_b)

        m["tri_P_w"] = P_tri
        m["tri_ray_gap_m"] = tri_info.get("ray_gap_m")
        m["triangulation_angle_deg"] = tri_info.get("triangulation_angle_deg")
        m["triangulation_ok"] = P_tri is not None

    return matches

# --- notebook code cell 7 ---
def get_detection_geometry(frame):
    """
    Pré-calcula:
    - classe
    - bbox xyxy
    - bottom-center
    - ponto 3D no chão
    """

    dets = []

    for i, label in enumerate(frame["labels"]):
        cls, bbox = yolo_to_xyxy(label, frame["img"].shape)
        pt = np.array([0.5 * (bbox[0] + bbox[2]), bbox[3]], dtype=float)

        P_w, ray_world, t = project_pixel_to_ground(
            pt[0],
            pt[1],
            frame,
            ground_z=GROUND_Z
        )

        dets.append({
            "idx": i,
            "class": cls,
            "bbox": bbox,
            "pt": pt,
            "P_w": P_w,
            "ray_world": ray_world,
            "ray_z": ray_world[2] if ray_world is not None else None,
            "t": t,
            "valid_ground": P_w is not None
        })

    return dets

# --- notebook code cell 8 ---
def reprojection_distance_to_bbox(det_src, frame_dst, det_dst):
    """
    Projeta det_src para frame_dst e mede distância à bbox det_dst.
    """

    if not det_src["valid_ground"]:
        return np.inf, None, None

    pt_proj, P_c = world_to_pixel_unreal(det_src["P_w"], frame_dst)

    if pt_proj is None:
        return np.inf, None, P_c

    u, v = pt_proj
    x1, y1, x2, y2 = det_dst["bbox"]
    dx = max(x1 - u, 0.0, u - x2)
    dy = max(y1 - v, 0.0, v - y2)

    return np.hypot(dx, dy), pt_proj, P_c


def directional_uncertainty_multiplier(det_src, frame_src, frame_dst, cls):
    """
    Multiplicador de incerteza por direção (src -> dst).
    Mantido fora de outras funções para leitura e reutilização mais simples.
    """
    ray_z = abs(det_src.get("ray_z") or 0.0)
    theta_min = np.arcsin(np.clip(ray_z, 1e-6, 1.0))
    theta_eff = np.clip(theta_min, ANGLE_MIN_RAD, ANGLE_CLIP_RAD)
    u_ray = np.sin(ANGLE_CLIP_RAD) / np.sin(theta_eff)

    t = det_src.get("t")
    t_val = float(t) if (t is not None and np.isfinite(t) and t > 0) else 0.0
    ref_log = np.log1p(DIST_REF / DIST_SCALE)
    raw_log = np.log1p(t_val / DIST_SCALE)
    u_dist = max(1.0, raw_log / max(ref_log, 1e-9))

    # Diferença de perspetiva entre câmaras para esta direção.
    fwd_src = frame_src["R_wc"] @ np.array([1.0, 0.0, 0.0])
    fwd_dst = frame_dst["R_wc"] @ np.array([1.0, 0.0, 0.0])
    fwd_src /= np.linalg.norm(fwd_src)
    fwd_dst /= np.linalg.norm(fwd_dst)

    cos_ang = np.clip(np.dot(fwd_src, fwd_dst), -1.0, 1.0)
    delta_deg = np.rad2deg(np.arccos(cos_ang))

    view_ref = 15.0
    view_max = 70.0
    t_view = np.clip((delta_deg - view_ref) / (view_max - view_ref), 0.0, 1.0)

    if cls == 0:
        max_factor = 1.3
    elif cls in (1, 2):
        max_factor = 2.0
    else:
        max_factor = 1.5

    u_view = 1.0 + t_view * (max_factor - 1.0)

    return np.clip(u_ray * u_dist * u_view, 1.0, MAX_UNCERTAINTY)


def symmetric_reprojection_cost_norm(det_a, frame_a, det_b, frame_b):
    """Calcula custo simétrico normalizado pela bbox de destino em cada direção."""

    d_ab, pt_ab, _ = reprojection_distance_to_bbox(det_a, frame_b, det_b)
    d_ba, pt_ba, _ = reprojection_distance_to_bbox(det_b, frame_a, det_a)

    scale_a = bbox_scale_single(det_a)
    scale_b = bbox_scale_single(det_b)

    c_ab = None
    c_ba = None
    w_ab = 0.0
    w_ba = 0.0

    if np.isfinite(d_ab):
        c_ab = d_ab / scale_b
        u_ab = directional_uncertainty_multiplier(det_a, frame_a, frame_b, det_a["class"])
        w_ab = 1.0 / max(u_ab, 1e-9)

    if np.isfinite(d_ba):
        c_ba = d_ba / scale_a
        u_ba = directional_uncertainty_multiplier(det_b, frame_b, frame_a, det_b["class"])
        w_ba = 1.0 / max(u_ba, 1e-9)

    if c_ab is None and c_ba is None:
        return np.inf, np.inf, pt_ab, pt_ba, scale_a, scale_b

    if c_ab is not None and c_ba is not None:
        w_sum = w_ab + w_ba
        cost_norm = float((w_ab * c_ab + w_ba * c_ba) / max(w_sum, 1e-9))
    elif c_ab is not None:
        cost_norm = float(c_ab)
    else:
        cost_norm = float(c_ba)

    if np.isfinite(d_ab) and np.isfinite(d_ba):
        d_reproj_px = 0.5 * (d_ab + d_ba)
    elif np.isfinite(d_ab):
        d_reproj_px = d_ab
    else:
        d_reproj_px = d_ba

    return cost_norm, d_reproj_px, pt_ab, pt_ba, scale_a, scale_b


def build_reprojection_cost_matrix(frame_a, frame_b):
    """
    Constrói matriz de custo normalizado.
    """

    dets_a = get_detection_geometry(frame_a)
    dets_b = get_detection_geometry(frame_b)

    n = len(dets_a)
    m = len(dets_b)

    C = np.full((n, m), BIG_COST, dtype=float)
    debug = {}

    for i, det_a in enumerate(dets_a):
        for j, det_b in enumerate(dets_b):

            # hard semantic gate
            if det_a["class"] != det_b["class"]:
                continue

            cost_norm, d_reproj, pt_ab, pt_ba, scale_a, scale_b = symmetric_reprojection_cost_norm(
                det_a,
                frame_a,
                det_b,
                frame_b
            )

            if not np.isfinite(cost_norm):
                continue

            C[i, j] = cost_norm

            debug[(i, j)] = {
                "class": det_a["class"],
                "d_reproj_px": d_reproj,
                "scale_A": scale_a,
                "scale_B": scale_b,
                "cost_norm": cost_norm,
            }

    return C, dets_a, dets_b, debug

# --- notebook code cell 9 ---
def _angle_factor(det_a, det_b):
    """Estima incerteza por geometria do raio (vista rasante vs estável)."""
    ray_z_a = abs(det_a.get("ray_z") or 0.0)
    ray_z_b = abs(det_b.get("ray_z") or 0.0)

    min_ray_z = min(ray_z_a, ray_z_b)
    theta_min = np.arcsin(np.clip(min_ray_z, 1e-6, 1.0))
    theta_eff = np.clip(theta_min, ANGLE_MIN_RAD, ANGLE_CLIP_RAD)

    return np.sin(ANGLE_CLIP_RAD) / np.sin(theta_eff)


def _distance_factor(det_a, det_b):
    """Estima incerteza por distância do objeto às câmaras."""

    def valid_t(det):
        t = det.get("t")
        if t is None or not np.isfinite(t) or t <= 0:
            return 0.0
        return float(t)

    max_t = max(valid_t(det_a), valid_t(det_b))

    ref_log = np.log1p(DIST_REF / DIST_SCALE)
    raw_log = np.log1p(max_t / DIST_SCALE)

    return max(1.0, raw_log / max(ref_log, 1e-9))


def _viewpoint_factor(frame_a, frame_b, cls):
    """Estima incerteza por mudança de perspetiva entre câmaras."""
    fwd_a = frame_a["R_wc"] @ np.array([1.0, 0.0, 0.0])
    fwd_b = frame_b["R_wc"] @ np.array([1.0, 0.0, 0.0])

    fwd_a /= np.linalg.norm(fwd_a)
    fwd_b /= np.linalg.norm(fwd_b)

    cos_ang = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
    delta_deg = np.rad2deg(np.arccos(cos_ang))

    view_ref = 15.0
    view_max = 70.0
    t = np.clip((delta_deg - view_ref) / (view_max - view_ref), 0.0, 1.0)

    if cls == 0:
        max_factor = 1.3
    elif cls in (1, 2):
        max_factor = 2.0
    else:
        max_factor = 1.5

    return 1.0 + t * (max_factor - 1.0)

def dynamic_threshold(det_a, frame_a, det_b, frame_b, base_thresholds=REPROJ_THRESHOLDS_NORM):
    """Calcula threshold dinâmico a partir de ângulo, distância e viewpoint."""
    cls = det_a["class"]
    base = base_thresholds.get(cls, 2.5)

    u_ray = _angle_factor(det_a, det_b)
    u_dist = _distance_factor(det_a, det_b)
    u_view = _viewpoint_factor(frame_a, frame_b, cls)

    u_total = np.clip(u_ray * u_dist * u_view, 1.0, MAX_UNCERTAINTY)

    raw_threshold = base * u_total

    ceiling = base * MAX_MULTIPLIER.get(cls, 2.0)

    threshold = min(raw_threshold, ceiling)

    return threshold


def match_frames_from_loaded(frame_a, frame_b, reproj_thresholds_norm=None):
    """
    Faz matching entre dois frames já carregados.
    """
    if reproj_thresholds_norm is None:
        reproj_thresholds_norm = REPROJ_THRESHOLDS_NORM

    C, dets_a, dets_b, debug = build_reprojection_cost_matrix(
        frame_a,
        frame_b
    )

    if C.size == 0:
        return [], list(range(len(dets_a))), list(range(len(dets_b))), C, debug, dets_a, dets_b

    row_ind, col_ind = linear_sum_assignment(C)

    matches = []
    matched_a = set()
    matched_b = set()

    for i, j in zip(row_ind, col_ind):
        cost = C[i, j]

        if cost >= BIG_COST:
            continue

        cls = dets_a[i]["class"]
        threshold = dynamic_threshold(
            dets_a[i],
            frame_a,
            dets_b[j],
            frame_b,
            base_thresholds=reproj_thresholds_norm,
        )

        if cost > threshold:
            continue

        info = debug.get((i, j), {})

        matches.append({
            "A": i,
            "B": j,
            "class": cls,
            "cost_norm": cost,
            "d_reproj_px": info.get("d_reproj_px", None),
            "scale_A": info.get("scale_A", None),
            "scale_B": info.get("scale_B", None),
        })

        matched_a.add(i)
        matched_b.add(j)

    unmatched_a = [i for i in range(len(dets_a)) if i not in matched_a]
    unmatched_b = [j for j in range(len(dets_b)) if j not in matched_b]

    matches = enrich_matches_with_triangulation(matches, dets_a, frame_a, dets_b, frame_b)

    return matches, unmatched_a, unmatched_b, C, debug, dets_a, dets_b


def match_two_frames(
    img_a_path,
    label_a_path,
    telemetry_a_path,
    img_b_path,
    label_b_path,
    telemetry_b_path,
    fov_x_deg=90.0,
    reproj_thresholds_norm=None,
):
    frame_a = load_frame(
        img_a_path,
        label_a_path,
        telemetry_a_path,
        fov_x_deg=fov_x_deg,
    )

    frame_b = load_frame(
        img_b_path,
        label_b_path,
        telemetry_b_path,
        fov_x_deg=fov_x_deg,
    )

    return match_frames_from_loaded(
        frame_a,
        frame_b,
        reproj_thresholds_norm=reproj_thresholds_norm,
    )


def count_by_object_type_after_matching(matches, unmatched_a, unmatched_b, dets_a, dets_b):
    counts = {
        "person": 0,
        "vehicle": 0,
        "emergency_vehicle": 0,
    }

    cls_to_name = {
        0: "person",
        1: "vehicle",
        2: "emergency_vehicle",
    }

    def add_class(cls):
        name = cls_to_name.get(cls)
        if name is not None:
            counts[name] += 1

    for m in matches:
        add_class(m["class"])

    for i in unmatched_a:
        add_class(dets_a[i]["class"])

    for j in unmatched_b:
        add_class(dets_b[j]["class"])

    return counts
