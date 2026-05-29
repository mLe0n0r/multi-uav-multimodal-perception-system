# Fusion (edge node)

Fusion integrates **image and audio** perception into a single **Service Level Specification (SLS)** per experiment run. Each SLS encodes the operational scene, participating entities and their inferred roles, salient audio-derived communications, and per-entity **throughput** requirements. 

## Repository structure

```text
fusion/
  ├─ matching/        → cross-view association for dual-UAV runs 
  ├─ llm/             → LLM orchestrator + prompts
  ├─ sls/             → throughput policy + SLS assembly 
  └─ run_layout.py    → run paths and shared loaders used by scripts
```

## Processing stages

| Stage | Implementation | Function |
|-------|----------------|----------|
| Cross-view association | Geometric matching and triangulation | Associates detections across two diferent views |
| Semantic fusion | **Gemma 4** (`gemma4:e2b`) via **Ollama** | Fuses visual and transcript JSON into a structured scene representation |
| Post-processing | Deterministic Python rules | Enforces schema validity, role assignment from audio, audio-only entities, and count consistency |
| Throughput assignment | Fixed Mbps policy (`communication_demand.py`) | Derives `traffic_demand_mbps` from entity role and declared communication services |

**Upstream inputs (outside this directory)**: visual detection and 3D localization (`perception/vision/`), audio transcription with **WhisperX** (`perception/audio/`).

## Execution flow

```text
perception/visual.json       perception/transcript.json
          │                               │
          └──────────────┬────────────────┘
                         │
                         ▼
        fusion/llm/code/llm_orchestrator.py
                (semantic fusion)
                         │
                         ▼
              fusion/llm_output.json
                         │
                         ▼
          fusion/sls/sls_builder.py
            (throughput assignment)
                         │
                         ▼
               fusion/sls.json
```

For **dual‑view** runs, `fusion/matching/cross_view_match.py` executes before the orchestrator and provides `fusion/cross_view.json` (same‑incident decision and cross‑camera associations), which is taken into account in the steps above.

## Output

| Artefact | Produced after | Additional content |
|----------|----------------|-------------------|
| `llm_output.json` | LLM inference and post-processing | Incident fields, `communications`, `objects[]` with geometry, inferred roles, qualitative `throughput_need` |
| `sls.json` | Throughput policy application | Same payload as `llm_output.json`, with **`traffic_demand_mbps`** per object |
