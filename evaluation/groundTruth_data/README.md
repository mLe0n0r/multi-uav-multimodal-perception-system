# Ground truth data

These annotations were defined separately for the visual and audio components of the pipeline, since each modality provides different types of evidence and supports different evaluation tasks.

## Visual (`visual/`)

| Ground truth data | File(s) | Annotation source | Used for |
|---|---|---|---|
| 2D object bounding boxes | `visual/gt_labels/*.txt` | Exported from **Roboflow**, with one YOLO-format label file per image. | Object detection evaluation, class-level detection metrics, and image-level object counting. |
| Object counts | `visual/objects_count.xlsx` | Manually annotated per image and per scenario. Scenario-level counts are deduplicated to represent the real number of objects in the scene. | Count-error analysis, scene-level counting, and single-view vs multi-view comparison. |
| Object positions | `visual/objects_position.xlsx` | Extracted from **Unreal Engine** using the ground-truth world coordinates of each object. | Localization error analysis and comparison between predicted and real-world object positions. |
| Cross-view object matches | `visual/objects_matches.xlsx` | Manually annotated for each evaluated pair of UAV images. | Object matching evaluation, and triangulated localization. |
| Person roles (GT) | `visual/gt_roles.xlsx` | Two sheets: **`per_scenario`** — person roles with world coordinates per scenario; **`per_img`** — `img`, `id` (from `visual.json`), `gt_role` per detected person. | Role assignment evaluation: match predictions to GT by visual object id; scenario-level counts from `per_scenario`. Scenario 5 has no `per_img` rows (audio-only). Overrides: `scenario3_audio3` (3 ff), `scenario5_audio2` (2 audio-only civ + 1 emergency vehicle). |

## Audio (`audio/`)

| Ground truth data | File(s) | Annotation source | Used for |
|---|---|---|---|
| Speaker count | `audio/speaker_count.xlsx` | Manually annotated for each audio clip. | Evaluation of the predicted number of speakers or intervenients. |
| Scene-relevant audio cues | `audio/audio_cues.xlsx` | Manually annotated based on the main information inferable from each audio clip. | Evaluation of audio-based scene understanding, including inferred people, vehicles, roles, and required services. |
