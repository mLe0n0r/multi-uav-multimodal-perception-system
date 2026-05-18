# Pipeline outputs

Per-run folders under `output/`. These are versioned in git (example results for the thesis).

## Run folder naming

`<image>_<scenario><audio>` — e.g. **`img0_sc1a1`**:

| Part | Meaning | This example |
|------|---------|--------------|
| `img0` | Vision frame `00000.png` | `perception/vision/data/daylight_images/00000.png` |
| `sc1` | Scenario 1 (radio dialogue set) | — |
| `a1` | Audio track 1 for that scenario | `perception/audio/data/transcriptions/scenario1_audio1.json` |

Use the same name for every artefact of one fusion run.

## Layout per run

```
output/
  img0_sc1a1/
    visual.json
    transcript.json
    llm_output.json
    sls.json
```

## Example (`img0_sc1a1`)

```powershell
$R = "img0_sc1a1"
$O = "output/$R"

python perception/vision/code/integration_pipeline.py `
  --image perception/vision/data/daylight_images/00000.png `
  --telemetry perception/vision/data/telemetryData/00000.txt `
  --mode day --output "$O/visual.json"

python perception/audio/code/transcript_analytics.py `
  --input perception/audio/data/transcriptions/scenario1_audio1.json `
  --output "$O/transcript.json"

python fusion/llm_orchestrator.py `
  --visual-json "$O/visual.json" `
  --transcript-json "$O/transcript.json" `
  --prompt fusion/prompts/sls_orchestrator_prompt.txt `
  --output "$O/llm_output.json" --model qwen2.5:latest

python fusion/sls_builder.py `
  --llm-output "$O/llm_output.json" `
  --visual-json "$O/visual.json" `
  --output "$O/sls.json"
```
