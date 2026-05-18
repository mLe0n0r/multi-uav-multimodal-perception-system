"""
Add a compact analytics summary to a WhisperX-style transcript JSON.

Estimates how many people / vehicles are referred to in speech when an explicit
quantity is spoken (e.g. "two civilians" -> 2). Bare words without a number
("civilian", "vehicle") are not counted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set

WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# Only digits and spoken numerals count as explicit quantities.
QTY_CAPTURE = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"

PEOPLE_PATTERN = re.compile(
    rf"\b{QTY_CAPTURE}?\s*"
    r"(?:people|persons|person|individuals|victims|responders|"
    r"firefighters|firefighter|crew|occupants?|civilians?)\b",
    re.IGNORECASE,
)

EMERGENCY_VEHICLE_PATTERN = re.compile(
    rf"\b{QTY_CAPTURE}?\s*"
    r"(?:emergency\s+vehicles?|fire\s+trucks?|fire\s+engines?|"
    r"ambulances?|rescue\s+vehicles?)\b",
    re.IGNORECASE,
)

NORMAL_VEHICLE_PATTERN = re.compile(
    rf"\b{QTY_CAPTURE}?\s*"
    r"(?<!emergency\s)(?<!fire\s)(?<!rescue\s)"
    r"(?:vehicles?|trucks?|cars?)\b",
    re.IGNORECASE,
)

CIVILIAN_PATTERN = re.compile(
    rf"\b{QTY_CAPTURE}\s+civilians?\b",
    re.IGNORECASE,
)

FIREFIGHTER_PATTERN = re.compile(
    rf"\b{QTY_CAPTURE}\s+"
    r"(?:firefighters?|fire\s+fighters?)\b",
    re.IGNORECASE,
)

CIVILIANS_AT_SAFE_DISTANCE_PATTERN = re.compile(
    r"civilians?.*\bsafe\s+distance\b|\bsafe\s+distance\b.*civilians?",
    re.IGNORECASE,
)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def iter_segments(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "segments" in transcript and isinstance(transcript["segments"], list):
        return transcript["segments"]
    return []


def collect_speaker_count(transcript: Dict[str, Any]) -> int:
    speakers: Set[str] = set()
    for segment in iter_segments(transcript):
        speaker = segment.get("speaker")
        if speaker:
            speakers.add(str(speaker))
    return len(speakers)


def full_transcript_text(transcript: Dict[str, Any]) -> str:
    parts: List[str] = []
    for segment in iter_segments(transcript):
        text = segment.get("text", "")
        if text:
            parts.append(str(text).strip())
    return " ".join(parts)


def parse_explicit_quantity(qty: Optional[str]) -> Optional[int]:
    if not qty:
        return None
    token = qty.strip().lower()
    if token.isdigit():
        return int(token)
    return WORD_TO_NUM.get(token)


def sum_explicit_mentions(text: str, pattern: Pattern[str]) -> int:
    """Sum spoken quantities (e.g. two + three -> 5). Ignore mentions without a number."""
    total = 0
    for match in pattern.finditer(text):
        n = parse_explicit_quantity(match.group(1))
        if n is not None:
            total += n
    return total


def build_analytics(transcript: Dict[str, Any]) -> Dict[str, Any]:
    text = full_transcript_text(transcript)

    return {
        "speaker_count": collect_speaker_count(transcript),
        "people_mentioned_count": sum_explicit_mentions(text, PEOPLE_PATTERN),
        "civilians_mentioned_count": sum_explicit_mentions(text, CIVILIAN_PATTERN),
        "firefighters_mentioned_count": sum_explicit_mentions(text, FIREFIGHTER_PATTERN),
        "civilians_at_safe_distance_mentioned": bool(
            CIVILIANS_AT_SAFE_DISTANCE_PATTERN.search(text)
        ),
        "vehicles_mentioned_count": sum_explicit_mentions(text, NORMAL_VEHICLE_PATTERN),
        "emergency_vehicle_mentioned_count": sum_explicit_mentions(
            text, EMERGENCY_VEHICLE_PATTERN
        ),
    }


def enrich_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(transcript)
    enriched["analytics"] = build_analytics(transcript)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add explicit-quantity mention counts to a transcript JSON."
    )
    parser.add_argument("--input", required=True, help="WhisperX transcript JSON path")
    parser.add_argument("--output", required=True, help="Output path (enriched transcript JSON)")
    args = parser.parse_args()

    transcript = load_json(args.input)
    enriched = enrich_transcript(transcript)
    save_json(enriched, args.output)

    analytics = enriched["analytics"]
    print(f"Speaker count: {analytics['speaker_count']}")
    print(f"People mentioned count: {analytics['people_mentioned_count']}")
    print(f"Civilians mentioned count: {analytics['civilians_mentioned_count']}")
    print(f"Firefighters mentioned count: {analytics['firefighters_mentioned_count']}")
    print(
        "Civilians at safe distance mentioned: "
        f"{analytics['civilians_at_safe_distance_mentioned']}"
    )
    print(f"Vehicles mentioned count: {analytics['vehicles_mentioned_count']}")
    print(f"Emergency vehicle mentioned count: {analytics['emergency_vehicle_mentioned_count']}")
    print(f"Enriched transcript saved to {args.output}")


if __name__ == "__main__":
    main()
