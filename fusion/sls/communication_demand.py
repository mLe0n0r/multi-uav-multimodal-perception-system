"""
Communication demand policy (thesis-oriented).

    Distance determines risk (see sls_builder.infer_risk_level).
    Role determines the minimum communication class (throughput_need).
    Requested or inferred services determine traffic demand (Mbps).

Fixed Mbps table — orders of magnitude for voice vs visual services.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Person near fire for operational risk on throughput (civilians at risk, no rich service).
PERSON_AT_RISK_METERS = 5.0
# Normal vehicle considered operationally relevant near the incident.
VEHICLE_OPERATIONAL_METERS = 3.0

RESPONDER_ROLES = frozenset({"possible_responder", "firefighter", "responder"})

# Qualitative class -> default Mbps when only role applies (no rich service on object).
ROLE_BASE_MBPS = {
    "low": 0.3,
    "medium": 1.0,
    "emergency_vehicle_medium": 2.0,
    "at_risk_no_service": 0.5,
}

SERVICE_MBPS = {
    "voice": 0.3,
    "point_visual": 3.0,  # single image or thermal frame transfer
    "stream_visual": 5.0,  # video / continuous thermal stream
}


def parse_distance_meters(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    match = re.match(r"([\d.]+)\s*m", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return None


def _scene_services(scene_service_types: List[str]) -> Set[str]:
    return set(scene_service_types or ["voice"])


def _has_stream_visual(services: Set[str]) -> bool:
    return bool(services & {"video", "image_or_video"})


def _has_point_visual(services: Set[str]) -> bool:
    return bool(services & {"thermal_image", "image_transfer"})


def infer_throughput_need(
    obj: Dict[str, Any],
    scene_service_types: List[str],
    *,
    thermal_consumer: bool = False,
    has_fire: bool = True,
) -> str:
    """
    Qualitative communication need from role + per-object service use.

    Not driven by proximity alone (proximity feeds risk_level instead).
    """
    need, _ = communication_demand_for_object(
        obj,
        scene_service_types,
        thermal_consumer=thermal_consumer,
        has_fire=has_fire,
    )
    return need


def communication_demand_for_object(
    obj: Dict[str, Any],
    scene_service_types: List[str],
    *,
    thermal_consumer: bool = False,
    has_fire: bool = True,
) -> Tuple[str, float]:
    """
    Return (throughput_need, traffic_demand_mbps) for one fused entity.
    """
    if obj.get("audio_only"):
        return "low", SERVICE_MBPS["voice"]

    cls = str(obj.get("class") or "")
    role = str(obj.get("inferred_role") or "").lower()
    services = _scene_services(scene_service_types)

    # Rich visual / thermal on this specific object (e.g. engine unit receiving thermal feed).
    if thermal_consumer:
        if _has_stream_visual(services):
            return "high", SERVICE_MBPS["stream_visual"]
        return "high", SERVICE_MBPS["point_visual"]

    if cls == "emergency_vehicle":
        # Operational node: always at least medium; never below 2.0 Mbps.
        return "medium", ROLE_BASE_MBPS["emergency_vehicle_medium"]

    if cls == "person":
        if role in RESPONDER_ROLES:
            return "medium", ROLE_BASE_MBPS["medium"]
        if has_fire:
            dist = parse_distance_meters(obj.get("distance_to_fire"))
            if dist is not None and dist <= PERSON_AT_RISK_METERS:
                return "medium", ROLE_BASE_MBPS["at_risk_no_service"]
        return "low", ROLE_BASE_MBPS["low"]

    if cls == "normal_vehicle":
        dist = parse_distance_meters(obj.get("distance_to_fire"))
        near_operation = dist is not None and dist <= VEHICLE_OPERATIONAL_METERS
        if near_operation:
            return "medium", ROLE_BASE_MBPS["medium"]
        return "low", ROLE_BASE_MBPS["low"]

    return "low", ROLE_BASE_MBPS["low"]
