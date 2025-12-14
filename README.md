# Boxer-Pose-Estimation

This repository implements a complete pipeline for boxer detection, tracking, and 14-keypoint pose estimation. It contains data preparation tools, model training/inference pipelines (YOLO, RF-DETR, DINO, ViTPose integrations), visualization utilities, and supporting scripts used during development and experimentation.

---

## Table of contents
- [Architecture overview](#architecture-overview)
- [Folder structure (high-level)](#folder-structure-high-level)
- [Core components and entry points](#core-components-and-entry-points)
- [Data flow and processing steps](#data-flow-and-processing-steps)
- [Training and inference pipelines](#training-and-inference-pipelines)
- [Utilities and developer tooling](#utilities-and-developer-tooling)
- [How to run (quick commands)](#how-to-run-quick-commands)
- [Notes, conventions, and mappings](#notes-conventions-and-mappings)
- [Author / License](#author--license)

---

## Architecture overview
The codebase is organized around modular pipelines (YOLO detection/pose, RF-DETR detection, DinoV2-based keypoint heads, ViTPose integration), a small set of data-processing tools for preparing COCO/YOLO datasets, and visualization helpers.

- Detection: YOLO (Ultralytics), RF-DETR (custom wrapper), DeepSort tracking (optional).
- Keypoint extraction: either YOLO-pose outputs (14 keypoints) or a two-stage approach:
  1. Detection + crop (RF-DETR or YOLO)
  2. Feature backbone (DinoV2) + custom deconv head to produce 64x64 heatmaps → keypoints
- Utilities: dataset creation, COCO ↔ YOLO conversion, annotation cleaning, visualization to produce videos with boxes & skeletons.

---

## Folder structure (high-level)
- `src/` — All source code and pipelines
  - `src/main.py` — CLI menu/launcher. See file: [src/main.py](src/main.py)
  - `src/training/train.py` — Training dispatcher. See function: [`main`](src/training/train.py)
  - `src/training/inference.py` — Inference dispatcher. See file: [src/training/inference.py](src/training/inference.py)
  - `src/pipelines/`
    - `src/pipelines/yolo_pipeline.py` — YOLO training/inference helpers and dataset utilities. Key symbols: [`run_yolo_training`](src/pipelines/yolo_pipeline.py), [`run_yolo_inference`](src/pipelines/yolo_pipeline.py), [`dataset_yaml_path_for`](src/pipelines/yolo_pipeline.py)
    - `src/pipelines/dino_pipeline.py` — DinoV2 backbone + deconv head training & inference. Key symbols: [`setup_dino_model`](src/pipelines/dino_pipeline.py), [`run_dino_training`](src/pipelines/dino_pipeline.py), [`run_dino_inference`](src/pipelines/dino_pipeline.py), [`ensure_dino_weights`](src/pipelines/dino_pipeline.py)
    - `src/pipelines/vitpose_pipeline.py` — ViTPose wrapper for mmpose-based inference. Key symbols: [`ensure_vitpose_model`](src/pipelines/vitpose_pipeline.py), [`run_vitpose_inference`](src/pipelines/vitpose_pipeline.py)
  - `src/detector_model/` — Detector model experiments, scripts, and a lightweight pipeline framework
    - `src/detector_model/Pipeline/Structure/` — Pipeline abstractions. See: [src/detector_model/Pipeline/Structure/example_usage.py](src/detector_model/Pipeline/Structure/example_usage.py) and [`Operation`](src/detector_model/Pipeline/Structure/Operation.py)
    - `src/detector_model/Scripts/` — Scripted utilities for annotation creation and dataset prep:
      - YOLO annotation generation: [src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model.py](src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model.py)
      - YOLO with tracking: [src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model_with_tracking.py](src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model_with_tracking.py)
      - RF-DETR annotation script: [src/detector_model/Scripts/rf-detr/get_annotations_from_model/get_annotations_from_model.py](src/detector_model/Scripts/rf-detr/get_annotations_from_model/get_annotations_from_model.py)
      - Dataset prep from CVAT export: [src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py](src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py)
    - Notebooks used for experiments and training logs: `src/detector_model/Notebooks/` (multiple `train_boxer_detection-*.ipynb` and tracking pipelines)
  - `src/data_processing/`
    - `src/data_processing/extract_frames.py` — frame extraction helper. See [`extract_frames`](src/data_processing/extract_frames.py)
    - `src/data_processing/split_dataset.py`, `merge_datasets.py`, `annotations_cleaner.py` — dataset management and cleanup. See [`final_cleanup`](src/data_processing/annotations_cleaner.py)
    - `src/data_processing/annotations_cleaner.py` — heavy utilities for converting keypoint order, categories, and final cleanup. Key function: [`final_cleanup`](src/data_processing/annotations_cleaner.py)
  - `src/utils/`
    - `src/utils/visualize.py` — visualization & video creation from annotations. See: [src/utils/visualize.py](src/utils/visualize.py)
    - `src/utils/tensor_to_coco.py` — helper to convert model output tensors → COCO JSON. See: [src/utils/tensor_to_coco.py](src/utils/tensor_to_coco.py)
    - `src/utils/results_to_coco.py` — convert YOLO Results objects to COCO keypoint JSON. See [`save_results_as_coco`](src/utils/results_to_coco.py)
    - `src/utils/coco_to_yolo_pose.py` — COCO → YOLO keypoint label conversion (imported by YOLO pipeline)
    - `src/utils/standalone_app.py` — small tkinter UI to pick a video + run pose extraction
- `data/` — data storage
  - `data/raw/`, `data/processed/`, `data/main_dataset/`, `data/outputs/`
- `models/` — saved model weights per model name (e.g., `dinov2_vits14`, `yolov11m-pose`, ...)
- `runs/` — training run outputs for YOLO runs
- Misc: `requirements.txt`, weights like `yolo11n.pt`, dataset yamls

---

## Core components and entry points

- Top-level CLI / quick launcher
  - [`src/main.py`](src/main.py) — interactive menu. It dispatches to scripts by module name. Uses `run_script` / `run_model_selection` helpers in the same file.
- Training and inference dispatchers
  - [`src/training/train.py`](src/training/train.py) — picks pipeline based on `--model` and calls pipeline functions such as [`run_yolo_training`](src/pipelines/yolo_pipeline.py) or [`run_dino_training`](src/pipelines/dino_pipeline.py).
  - [`src/training/inference.py`](src/training/inference.py) — inference dispatcher, calls pipeline inference functions (YOLO, DINO, ViTPose).
- Pipelines
  - YOLO pipeline: [src/pipelines/yolo_pipeline.py](src/pipelines/yolo_pipeline.py)
    - Useful functions/symbols:
      - [`run_yolo_training`](src/pipelines/yolo_pipeline.py) — training loop and label recreation logic.
      - [`dataset_yaml_path_for`](src/pipelines/yolo_pipeline.py) — generates dataset YAML for ultralytics.
      - [`run_yolo_inference`](src/pipelines/yolo_pipeline.py) — inference wrapper producing COCO JSON via `save_results_as_coco`.
  - DinoV2 pipeline: [src/pipelines/dino_pipeline.py](src/pipelines/dino_pipeline.py)
    - Useful functions/symbols:
      - [`ensure_dino_weights`](src/pipelines/dino_pipeline.py) — downloads backbone weights if missing.
      - [`setup_dino_model`](src/pipelines/dino_pipeline.py) — builds frozen Dino backbone + trainable deconv head.
      - [`run_dino_training`](src/pipelines/dino_pipeline.py), [`run_dino_inference`](src/pipelines/dino_pipeline.py)
      - Internal helper: [`_load_rf_detr_annotation`](src/pipelines/dino_pipeline.py) — loads detection JSON for cropping.
  - ViTPose pipeline: [src/pipelines/vitpose_pipeline.py](src/pipelines/vitpose_pipeline.py)
    - [`ensure_vitpose_model`](src/pipelines/vitpose_pipeline.py), [`run_vitpose_inference`](src/pipelines/vitpose_pipeline.py)
- Data prep & annotation generation
  - Frame extraction: [`src/data_processing/extract_frames.py`](src/data_processing/extract_frames.py) — `extract_frames(video_path, output_dir, target_fps)`
  - CVAT export → YOLO dataset prep: [`src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py`](src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py)
  - Create CVAT XML from model detections: YOLO and RF-DETR scripts under `src/detector_model/Scripts/`
- Annotation cleaning & conversion
  - [`src/data_processing/annotations_cleaner.py`](src/data_processing/annotations_cleaner.py) — utilities to:
    - Fix frame names
    - Normalize categories (keep only person/boxer)
    - Reorder keypoints (current → target mapping)
    - Convert to multi-class (blue/red boxers)
    - Final cleanup helper: [`final_cleanup`](src/data_processing/annotations_cleaner.py)

---

## Data flow and processing steps (typical)
1. Collect raw videos into `data/raw/`.
2. Extract frames:
   - Use [`src/data_processing/extract_frames.py`](src/data_processing/extract_frames.py) or the extractor in `src/detector_model/Pipeline/Functionality/extract_frames.py`.
3. Annotate frames:
   - Manual annotation with CVAT → export YOLO/COCO
   - Optionally generate annotations from an existing detector model using:
     - YOLO script: [src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model.py](src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model.py)
     - RF-DETR script: [src/detector_model/Scripts/rf-detr/get_annotations_from_model/get_annotations_from_model.py](src/detector_model/Scripts/rf-detr/get_annotations_from_model/get_annotations_from_model.py)
4. Prepare YOLO dataset (images + labels + `data.yaml`):
   - Script: [src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py](src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py)
   - `dataset_yaml_path_for` in [`src/pipelines/yolo_pipeline.py`](src/pipelines/yolo_pipeline.py) can be used to generate model-specific dataset YAML automatically.
5. Train detection/pose:
   - YOLO: call [`run_yolo_training`](src/pipelines/yolo_pipeline.py) or use `src/detector_model/Notebooks/*` for experiment logs.
   - Dino/Keypoint head: see [`run_dino_training`](src/pipelines/dino_pipeline.py).
6. Inference:
   - YOLO inference → convert ultralytics results to COCO via [`save_results_as_coco`](src/utils/results_to_coco.py).
   - Dino inference uses saved RF-DETR detections to crop and predict heatmaps: [`run_dino_inference`](src/pipelines/dino_pipeline.py).
   - ViTPose inference produces COCO keypoint JSON via [`run_vitpose_inference`](src/pipelines/vitpose_pipeline.py).
7. Visualization & outputs:
   - Visualize detections/poses with [`src/utils/visualize.py`](src/utils/visualize.py).

---

## Training and inference (details)

- YOLO
  - Config and dataset YAML generation: [`dataset_yaml_path_for`](src/pipelines/yolo_pipeline.py) writes a YAML with `kpt_shape`, `names`, `keypoints`, `skeleton`.
  - Training entrypoint: [`run_yolo_training`](src/pipelines/yolo_pipeline.py) (handles label recreation, retries, training options).
  - Example notebooks: `src/detector_model/Notebooks/yolo/train_boxer_detection-*.ipynb` show training logs and hyperparameter experiments.

- DinoV2-based keypoint head
  - Backbone: DinoV2-ViTS14 from DINO release (`DINO_URL`) downloaded via [`ensure_dino_weights`](src/pipelines/dino_pipeline.py).
  - Head: custom `DeconvolutionalHead` expected to exist in `src/models/keypoint_heads.py` and is attached in [`setup_dino_model`](src/pipelines/dino_pipeline.py).
  - Training: [`run_dino_training`](src/pipelines/dino_pipeline.py) wraps dataloaders (crops, heatmap generation).
  - Inference: [`run_dino_inference`](src/pipelines/dino_pipeline.py) loads RF-DETR detection JSONs (helper [`_load_rf_detr_annotation`](src/pipelines/dino_pipeline.py)) to crop boxer images before feeding through the backbone + head.

- ViTPose (mmpose)
  - Requires `mmpose` and `mmcv`. Setup/download helper: [`ensure_vitpose_model`](src/pipelines/vitpose_pipeline.py).
  - Inference: [`run_vitpose_inference`](src/pipelines/vitpose_pipeline.py) produces COCO keypoint JSON.

---

## Utilities & helpers
- JSON conversion:
  - [`src/utils/tensor_to_coco.py`](src/utils/tensor_to_coco.py) — convert output tensors/heatmaps → COCO JSON (`DEFAULT_OUTPUT_PATH`).
  - [`src/utils/results_to_coco.py`](src/utils/results_to_coco.py) — convert ultralytics `Results` → COCO keypoints JSON (`save_results_as_coco`).
- Visualization:
  - [`src/utils/visualize.py`](src/utils/visualize.py) — draw boxes, keypoints, skeletons, and save annotated videos.
- Annotation cleaning:
  - [`src/data_processing/annotations_cleaner.py`](src/data_processing/annotations_cleaner.py) — functions to reorder keypoints, set category ids, add color attributes, and perform final cleanup (`final_cleanup`).
- Dataset helpers:
  - `src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py` — central script to convert CVAT export and video frames into YOLO dataset structure (images, labels, data.yaml).
  - Train/val/test splitter: [`src/detector_model/Pipeline/Functionality/train_val_test_split.py`](src/detector_model/Pipeline/Functionality/train_val_test_split.py)

---

## How to run (quick commands)
- Install requirements:
  - pip install -r requirements.txt
- Interactive menu:
  - python src/main.py
- Train (example: YOLO pipeline via dispatcher):
  - python -m src.training.train --model yolov8m-pose --epochs 50
    - (calls [`run_yolo_training`](src/pipelines/yolo_pipeline.py))
- Inference (example):
  - python -m src.training.inference --model yolov11m-pose --conf 0.6
    - (calls inference pipeline in [`src/training/inference.py`](src/training/inference.py))
- Prepare YOLO dataset from CVAT export:
  - python src/detector_model/Scripts/dataset_prep_from_cvat_export/prepare_dataset.py --dataset_dir <CVAT_EXPORT> --video_path <VIDEO> --frame_step 1
- Create annotations from model (YOLO):
  - python src/detector_model/Scripts/yolo/generate_annotations_from_model/get_annotations_from_model.py --model <model.pt> --video <video.mp4> --labels "boxer_blue,boxer_red" --fps 2

Refer to the top of each script for more CLI args and usage.

---

## Notes, conventions, and mappings

- Keypoint convention
  - Project uses a custom 14-key mapping (see constants in [`src/pipelines/yolo_pipeline.py`](src/pipelines/yolo_pipeline.py) and [`src/utils/tensor_to_coco.py`](src/utils/tensor_to_coco.py)).
- Skeleton
  - Skeleton edge list is defined in multiple places (`src/pipelines/yolo_pipeline.py`, `src/pipelines/vitpose_pipeline.py`, `src/utils/tensor_to_coco.py`) to ensure visualization and COCO outputs match the expected connections.
- Models storage
  - Model weights & checkpoints are stored under `models/{model_name}/` (helpers in pipelines assume `MODELS_ROOT = PROJECT_ROOT / "models"`).
- Logs & runs
  - YOLO training runs are written to `runs/` (YOLO training logic in [`src/pipelines/yolo_pipeline.py`](src/pipelines/yolo_pipeline.py) writes to `RUNS_ROOT`).
- Notebooks
  - Several Jupyter notebooks under `src/detector_model/Notebooks/` record training logs and interactive experiments. Use them for reference to hyperparameters and training outputs.

---

## Author / License
- Licensed under MIT. See: [LICENSE](LICENSE)

---

<!-- If you want, I can:
- Expand any section into more step-by-step guides (e.g., full example: preparing dataset → training → inference → visualization).
- Create a small CONTRIBUTING.md or developer quickstart with environment setup commands. -->