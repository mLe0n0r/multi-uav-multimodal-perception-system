"""Fused per-class counts and entity groups across multi-view visual JSON."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

Key = Tuple[str, int]

VALID_CLASSES = ("person", "normal_vehicle", "emergency_vehicle")


def _match_list(matching: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not matching:
        return []
    return list(matching.get("matches") or matching.get("pairs") or [])


def visual_object_lookup(views: List[Dict[str, Any]]) -> Dict[Key, Dict[str, Any]]:
    lookup: Dict[Key, Dict[str, Any]] = {}
    for v in views:
        view_id = str(v.get("_view_id", "mono"))
        for obj in v.get("objects", []):
            oid = obj.get("id")
            if oid is not None:
                lookup[(view_id, int(oid))] = obj
    return lookup


def _union_find(keys: Set[Key]) -> Dict[Key, Key]:
    parent: Dict[Key, Key] = {k: k for k in keys}

    def find(x: Key) -> Key:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Key, b: Key) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def apply_cross_view_unions(
    lookup: Dict[Key, Dict[str, Any]],
    matching: Dict[str, Any],
) -> Dict[Key, Key]:
    keys = set(lookup.keys())
    parent, _, union = _union_find(keys)
    for m in _match_list(matching):
        a: Key = (str(m.get("view_a")), int(m["id_a"]))
        b: Key = (str(m.get("view_b")), int(m["id_b"]))
        if a not in lookup or b not in lookup:
            continue
        if lookup[a].get("class") == lookup[b].get("class"):
            union(a, b)
    return parent


def fused_entity_groups(
    views: List[Dict[str, Any]],
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[Key, List[Key]]:
    """Map union-find root -> member (view_id, object_id) keys."""
    lookup = visual_object_lookup(views)
    if not lookup:
        return {}

    if not matching or len(views) < 2 or not matching.get("same_incident", True):
        return {(vid, int(oid)): [(vid, int(oid))] for (vid, oid) in lookup}

    parent = apply_cross_view_unions(lookup, matching)

    def find(x: Key) -> Key:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    groups: Dict[Key, List[Key]] = {}
    for key in lookup:
        root = find(key)
        groups.setdefault(root, []).append(key)
    return groups


def deduped_visual_counts(
    views: List[Dict[str, Any]],
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Unique entities per class after cross-view matching.

    Each match between same-class detections merges two nodes (union-find).
    """
    if not views:
        return {cls: 0 for cls in VALID_CLASSES}

    if not matching or len(views) < 2 or not matching.get("same_incident", True):
        total = {cls: 0 for cls in VALID_CLASSES}
        for v in views:
            for cls, n in (v.get("counts_by_class") or {}).items():
                if cls in total:
                    total[cls] += int(n or 0)
        return total

    groups = fused_entity_groups(views, matching)
    lookup = visual_object_lookup(views)
    roots_by_class: Dict[str, Set[Key]] = {cls: set() for cls in VALID_CLASSES}
    for root, members in groups.items():
        cls = lookup[members[0]].get("class")
        if cls in roots_by_class:
            roots_by_class[cls].add(root)
    return {cls: len(roots_by_class[cls]) for cls in VALID_CLASSES}


def reference_view_id(matching: Optional[Dict[str, Any]], views: List[Dict[str, Any]]) -> str:
    if matching:
        ref = matching.get("reference_view")
        if ref:
            return str(ref)
        view_ids = matching.get("views")
        if view_ids:
            return str(view_ids[0])
    return str(views[0].get("_view_id", "mono"))


def fused_position_for_cluster(
    member_keys: List[Key],
    matching: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not matching:
        return None
    keyset = set(member_keys)
    for m in _match_list(matching):
        a: Key = (str(m.get("view_a")), int(m["id_a"]))
        b: Key = (str(m.get("view_b")), int(m["id_b"]))
        if a in keyset and b in keyset and m.get("position"):
            return copy.deepcopy(m["position"])
    return None


def pick_representative_visual(
    member_keys: List[Key],
    lookup: Dict[Key, Dict[str, Any]],
    matching: Optional[Dict[str, Any]],
    ref_view: str,
) -> Dict[str, Any]:
    ordered = sorted(
        member_keys,
        key=lambda k: (0 if k[0] == ref_view else 1, k[0], k[1]),
    )
    key = ordered[0]
    obj = copy.deepcopy(lookup[key])
    fused_pos = fused_position_for_cluster(member_keys, matching)
    if fused_pos:
        obj["position"] = fused_pos
    return obj


def build_fused_object_list(
    views: List[Dict[str, Any]],
    matching: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Key, int]]:
    """
    One row per fused entity (union-find cluster). Sequential ids 0..N-1.
    Returns (objects for LLM/SLS, map (view_id, local_id) -> fused_id).
    """
    lookup = visual_object_lookup(views)
    if not lookup:
        return [], {}

    if not matching or len(views) < 2 or not matching.get("same_incident", True):
        objects: List[Dict[str, Any]] = []
        key_map: Dict[Key, int] = {}
        fid = 0
        for v in views:
            for obj in v.get("objects", []):
                oid = obj.get("id")
                if oid is None:
                    continue
                vid = str(v.get("_view_id", "mono"))
                key_map[(vid, int(oid))] = fid
                objects.append(
                    {
                        "id": fid,
                        "class": obj.get("class"),
                        "detection_confidence": obj.get("detection_confidence"),
                        "position": copy.deepcopy(obj.get("position")),
                        "localization_confidence": copy.deepcopy(
                            obj.get("localization_confidence")
                        ),
                        "distance_to_fire": obj.get("distance_to_fire"),
                    }
                )
                fid += 1
        return objects, key_map

    ref_view = reference_view_id(matching, views)
    groups = fused_entity_groups(views, matching)
    sorted_roots = sorted(
        groups.keys(),
        key=lambda r: (
            lookup[groups[r][0]].get("class", ""),
            r[0],
            r[1],
        ),
    )
    objects = []
    key_map: Dict[Key, int] = {}
    for fused_id, root in enumerate(sorted_roots):
        members = groups[root]
        for key in members:
            key_map[key] = fused_id
        vis = pick_representative_visual(members, lookup, matching, ref_view)
        objects.append(
            {
                "id": fused_id,
                "class": vis.get("class"),
                "detection_confidence": vis.get("detection_confidence"),
                "position": copy.deepcopy(vis.get("position")),
                "localization_confidence": copy.deepcopy(
                    vis.get("localization_confidence")
                ),
                "distance_to_fire": vis.get("distance_to_fire"),
            }
        )
    return objects, key_map
