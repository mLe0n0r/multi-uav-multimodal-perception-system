# Audio code

Python modules for the edge audio pipeline.

| Script | Purpose |
|--------|---------|
| `transcript_analytics.py` | Speaker count and people/vehicle mention detection from transcript JSON |

Notebooks under `../notebooks/` perform transcription. Save enriched transcripts under `output/<run>/transcript.json` (e.g. `img0_sc1a1`; see `output/README.md`).

If you have not run WhisperX yet, use an existing transcription from `../data/transcriptions/` (e.g. scenario 1 pairs with frame `00000`):

```bash
python perception/audio/code/transcript_analytics.py --input perception/audio/data/transcriptions/scenario1_audio1.json --output output/img0_sc1a1/transcript.json
```

After WhisperX, point `--input` at your transcript JSON and write to `output/<run>/transcript.json`.

The output JSON adds a compact `analytics` block: speaker count plus mention counts for people, vehicles, and emergency vehicles. Full dialogue remains in `segments`.
