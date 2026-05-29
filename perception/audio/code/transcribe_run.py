"""
WhisperX (terminal) + análise do notebook audioTranscription.ipynb → transcript.json

  python perception/audio/code/transcribe_run.py \\
    --audio perception/audio/input/scenario1_audio1.wav \\
    --run-dir output/scenario1/img0_img12_aud1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set

_CODE_DIR = Path(__file__).resolve().parent
_AUDIO_ROOT = _CODE_DIR.parent
_REPO_ROOT = _CODE_DIR.parents[2]
_TRANSCRIPTIONS_DIR = _AUDIO_ROOT / "transcriptions"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.run_layout import transcript_path

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# --- notebook audioTranscription.ipynb (células 11–13) ---
END_WORDS = {"listening", "finished"}

# --- analytics (contagens para o LLM) ---
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
    r"(?:(?:civilian|private|parked)\s+)?"
    r"(?<!emergency\s)(?<!fire\s)(?<!rescue\s)"
    r"(?:vehicles?|trucks?|cars?)\b",
    re.IGNORECASE,
)
CIVILIAN_PATTERN = re.compile(rf"\b{QTY_CAPTURE}\s+civilians?\b", re.IGNORECASE)
FIREFIGHTER_PATTERN = re.compile(
    rf"\b{QTY_CAPTURE}\s+(?:firefighters?|fire\s+fighters?)\b",
    re.IGNORECASE,
)
CIVILIANS_AT_SAFE_DISTANCE_PATTERN = re.compile(
    r"civilians?.*\bsafe\s+distance\b|\bsafe\s+distance\b.*civilians?",
    re.IGNORECASE,
)
EN_ROUTE_PATTERN = re.compile(
    r"\b(?:en\s+route|on\s+(?:the\s+)?way|responding\s+to|heading\s+(?:to|toward)|"
    r"deployed\s+from|minutes\s+out)\b",
    re.IGNORECASE,
)


def _normalize_unit_id(token: str) -> str:
    word_map = {"one": "1", "two": "2", "three": "3", "four": "4"}
    cleaned = token.strip().lower()
    return word_map.get(cleaned, cleaned)


def _extract_field_speaking_units(text: str) -> Set[str]:
    cleaned = " ".join(text.split()).strip()
    if not cleaned or EN_ROUTE_PATTERN.search(cleaned):
        return set()

    units: Set[str] = set()
    match = re.match(
        r"^command\s+here,?\s*(?:(engine|rescue)\s+(one|two|\d+)|(alpha|beta))\b",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        if match.group(1):
            units.add(f"{match.group(1).lower()}_{_normalize_unit_id(match.group(2))}")
        elif match.group(3):
            units.add(match.group(3).lower())

    match = re.match(
        r"^(?:(rescue|engine)\s+(one|two|\d+)|(alpha|beta))\s+here,?\s*(?:engine|rescue|alpha|beta|command)\b",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        if match.group(1):
            units.add(f"{match.group(1).lower()}_{_normalize_unit_id(match.group(2))}")
        elif match.group(3):
            units.add(match.group(3).lower())
    return units


def _count_on_scene_field_units(transcript: Dict[str, Any]) -> int:
    units: Set[str] = set()
    for segment in transcript.get("segments", []):
        text = (segment.get("text") or "").strip()
        if not text or EN_ROUTE_PATTERN.search(text):
            continue
        units |= _extract_field_speaking_units(text)
    return len(units)


def clean_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def words_to_text(words: List[Dict[str, Any]]) -> str:
    text = " ".join(w["word"] for w in words)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text.strip()


def majority_acoustic_speaker(words: List[Dict[str, Any]]) -> str:
    speakers = [w.get("speaker") for w in words if w.get("speaker")]
    if not speakers:
        return "UNKNOWN"
    return Counter(speakers).most_common(1)[0][0]


def reconstruct_by_end_words(word_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []

    for w in word_segments:
        current_words.append(w)
        word = clean_word(w["word"])

        if word in END_WORDS:
            messages.append(
                {
                    "speaker": majority_acoustic_speaker(current_words),
                    "start": current_words[0].get("start"),
                    "end": current_words[-1].get("end"),
                    "words": current_words,
                }
            )
            current_words = []

    if current_words:
        messages.append(
            {
                "speaker": majority_acoustic_speaker(current_words),
                "start": current_words[0].get("start"),
                "end": current_words[-1].get("end"),
                "words": current_words,
            }
        )

    return messages


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def analyze_transcript(data: Dict[str, Any]) -> Dict[str, Any]:
    word_segments = data.get("word_segments")
    if not word_segments:
        word_segments = []
        for seg in data.get("segments", []):
            word_segments.extend(seg.get("words") or [])

    radio_messages = reconstruct_by_end_words(word_segments)
    for msg in radio_messages:
        msg["message"] = words_to_text(msg["words"])

    speakers = {m.get("speaker") for m in radio_messages if m.get("speaker")}
    segments = [
        {
            "start": m["start"],
            "end": m["end"],
            "text": m["message"],
            "speaker": m["speaker"],
            "words": m["words"],
        }
        for m in radio_messages
    ]

    return {
        "language": data.get("language", "en"),
        "word_segments": word_segments,
        "radio_messages": radio_messages,
        "segments": segments,
        "speaker_count": len(speakers),
    }


def print_analysis(result: Dict[str, Any]) -> None:
    for msg in result["radio_messages"]:
        print(
            {
                "speaker": msg["speaker"],
                "start": msg["start"],
                "end": msg["end"],
                "message": msg["message"],
            }
        )
    print("Número de intervenientes:", result["speaker_count"])


def parse_explicit_quantity(qty: Optional[str]) -> Optional[int]:
    if not qty:
        return None
    token = qty.strip().lower()
    if token.isdigit():
        return int(token)
    return WORD_TO_NUM.get(token)


def sum_explicit_mentions(text: str, pattern: Pattern[str]) -> int:
    total = 0
    for match in pattern.finditer(text):
        n = parse_explicit_quantity(match.group(1))
        if n is not None:
            total += n
    return total


def max_explicit_mentions(text: str, pattern: Pattern[str]) -> int:
    """
    Scene totals should not accumulate repeated radio updates.
    Keep the maximum explicit quantity mentioned across the transcript.
    """
    best = 0
    for match in pattern.finditer(text):
        n = parse_explicit_quantity(match.group(1))
        if n is not None and n > best:
            best = n
    return best


def build_analytics(transcript: Dict[str, Any]) -> Dict[str, Any]:
    parts = [str(s.get("text", "")).strip() for s in transcript.get("segments", []) if s.get("text")]
    text = " ".join(parts)
    speakers: Set[str] = set()
    for segment in transcript.get("segments", []):
        if segment.get("speaker"):
            speakers.add(str(segment["speaker"]))

    return {
        "speaker_count": len(speakers),
        "people_mentioned_count": max_explicit_mentions(text, PEOPLE_PATTERN),
        "civilians_mentioned_count": max_explicit_mentions(text, CIVILIAN_PATTERN),
        "firefighters_mentioned_count": max_explicit_mentions(text, FIREFIGHTER_PATTERN),
        "on_scene_responders_count": _count_on_scene_field_units(transcript),
        "civilians_at_safe_distance_mentioned": bool(
            CIVILIANS_AT_SAFE_DISTANCE_PATTERN.search(text)
        ),
        "vehicles_mentioned_count": sum_explicit_mentions(text, NORMAL_VEHICLE_PATTERN),
        "emergency_vehicle_mentioned_count": sum_explicit_mentions(
            text, EMERGENCY_VEHICLE_PATTERN
        ),
    }


def enrich_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(transcript)
    out["analytics"] = build_analytics(transcript)
    return out


def resolve_whisperx_python() -> Path:
    """Python do whisperx_env (como no terminal do notebook)."""
    if os.environ.get("WHISPERX_PYTHON"):
        return Path(os.environ["WHISPERX_PYTHON"])
    candidates = [
        _REPO_ROOT.parent.parent / "audio_agent" / "whisperx_env" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return Path(sys.executable)


def load_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val.strip()
    env_path = _AUDIO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN") and v.strip():
                return v.strip().strip('"').strip("'")
    return None


def default_whisperx_device_compute(python: Path) -> tuple[str, str]:
    """CPU → int8 (evita float16 em Windows); GPU → float16."""
    result = subprocess.run(
        [str(python), "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
        capture_output=True,
        text=True,
        check=True,
    )
    device = result.stdout.strip() or "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def whisperx_env_with_ffmpeg(python: Path) -> dict[str, str]:
    """WhisperX chama o binário 'ffmpeg'; imageio usa outro nome — criamos shim ffmpeg.exe."""
    import shutil

    env = os.environ.copy()
    if shutil.which("ffmpeg", path=env.get("PATH")):
        return env

    probe = subprocess.run(
        [
            str(python),
            "-c",
            "from pathlib import Path\n"
            "import imageio_ffmpeg\n"
            "print(imageio_ffmpeg.get_ffmpeg_exe(), end='')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    bundled = Path(probe.stdout.strip())
    if not bundled.is_file():
        return env

    shim_dir = _AUDIO_ROOT / "data" / "temp_audio" / "_ffmpeg_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_ffmpeg = shim_dir / "ffmpeg.exe"
    if not shim_ffmpeg.exists() or shim_ffmpeg.stat().st_size != bundled.stat().st_size:
        shutil.copy2(bundled, shim_ffmpeg)

    env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
    return env


def invoke_whisperx_cli(python: Path, args: List[str], hf_token: str) -> None:
    """
    WhisperX com patches para Windows/CPU:
    - torch 2.8 weights_only + pyannote (omegaconf safe globals)
    - VAD silero (notebook) em vez de pyannote VAD
    """
    import json as _json

    argv = ["whisperx", *args]
    script = f"""
import sys
import torch

# PyTorch 2.6+ weights_only=True quebra pyannote/lightning (diarização).
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from torch.torch_version import TorchVersion
    torch.serialization.add_safe_globals([ListConfig, DictConfig, TorchVersion])
except Exception:
    pass

sys.argv = {_json.dumps(argv)}
from whisperx.__main__ import cli
cli()
"""
    env = whisperx_env_with_ffmpeg(python)
    env["HF_TOKEN"] = hf_token
    env["HUGGINGFACE_TOKEN"] = hf_token
    if not __import__("shutil").which("ffmpeg", path=env.get("PATH")):
        raise RuntimeError(
            "ffmpeg em falta. Instala: winget install ffmpeg\n"
            "ou: pip install imageio-ffmpeg (no whisperx_env)"
        )
    subprocess.run([str(python), "-c", script], check=True, env=env)


def run_whisperx(
    audio_path: Path,
    *,
    model: str,
    language: str,
    min_speakers: int,
    max_speakers: int,
    hf_token: str,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Path:
    _TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = _TRANSCRIPTIONS_DIR / f"{audio_path.stem}.json"

    py = resolve_whisperx_python()
    if device is None or compute_type is None:
        auto_device, auto_compute = default_whisperx_device_compute(py)
        device = device or auto_device
        compute_type = compute_type or auto_compute

    wx_args = [
        str(audio_path.resolve()),
        "--output_dir",
        str(_TRANSCRIPTIONS_DIR.resolve()),
        "--output_format",
        "json",
        "--diarize",
        "--vad_method",
        "silero",
        "--device",
        device,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--model",
        model,
        "--min_speakers",
        str(min_speakers),
        "--max_speakers",
        str(max_speakers),
        "--hf_token",
        hf_token,
    ]
    print("=== WhisperX (terminal) ===")
    print(f"{py} -m whisperx {' '.join(wx_args)}")
    try:
        invoke_whisperx_cli(py, wx_args, hf_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "WhisperX falhou. Verifica HF_TOKEN e aceitação dos modelos pyannote no Hugging Face.\n"
            "Se já tens JSON: --skip-whisperx"
        ) from exc

    if not out_json.is_file():
        candidates = list(_TRANSCRIPTIONS_DIR.glob(f"{audio_path.stem}*.json"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"WhisperX não criou {out_json}")
    print(f"WhisperX JSON: {out_json}")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WhisperX (terminal) + notebook → transcript.json"
    )
    parser.add_argument("--audio", required=True, help="WAV/MP3 de entrada")
    parser.add_argument("--run-dir", help="Pasta do run")
    parser.add_argument("--output", help="Caminho de saída alternativo")
    parser.add_argument("--skip-whisperx", action="store_true", help="Usar JSON em transcriptions/")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Auto se omitido")
    parser.add_argument(
        "--compute-type",
        choices=("int8", "float16", "float32"),
        help="Em CPU usar int8 (default auto)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"Not an audio file: {audio_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    whisperx_json = _TRANSCRIPTIONS_DIR / f"{audio_path.stem}.json"

    if args.skip_whisperx:
        if not whisperx_json.is_file():
            raise FileNotFoundError(f"JSON em falta: {whisperx_json}")
        print(f"=== WhisperX (transcriptions/) ===\n{whisperx_json}")
    else:
        token = load_hf_token()
        if not token:
            raise ValueError("HF_TOKEN em falta (perception/audio/.env)")
        whisperx_json = run_whisperx(
            audio_path,
            model=args.model,
            language=args.language,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            hf_token=token,
            device=args.device,
            compute_type=args.compute_type,
        )

    print("\n=== Análise (notebook) ===")
    analyzed = analyze_transcript(load_json(whisperx_json))
    print_analysis(analyzed)

    final = enrich_transcript(analyzed)
    if args.output:
        out_path = Path(args.output)
    elif args.run_dir:
        out_path = transcript_path(args.run_dir)
    else:
        out_path = _TRANSCRIPTIONS_DIR / f"{audio_path.stem}_analyzed.json"

    save_json(final, out_path)
    analytics = final.get("analytics", {})
    print(f"\n=== Guardado ===\n{out_path}")
    print(f"  radio_messages: {len(final.get('radio_messages', []))}")
    print(f"  speakers: {analytics.get('speaker_count')}")


if __name__ == "__main__":
    main()
