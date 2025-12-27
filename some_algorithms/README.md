# some_algorithms

## Overview
This folder contains three iterations of motion capture keyframe selection.

## Versions
- v1 (`motion_capture_v1_failed.py`)
  - Method: either geometric pose peaks (elbow/knee angle extrema) or optical flow
    static detection (global motion magnitude).
  - Output: saves detected frames only.
  - Evaluation: not wired to MAE.
- v2 (`motion_capture_v2.py`)
  - Method: distinct stability detector (normalized pose vectors + local-min
    velocity + difference from last captured pose). Optional geometric/optical
    modes are still present.
  - Output: saves frames with frame indices and writes `detected_frames.txt`.
  - Evaluation: nearest-neighbor MAE vs `ground_truth.txt`.
  - MAE: 7.39 frames.
- v3 (`motion_capture_v3.py`)
  - Method: signal-processing pipeline. Scan video to build pose-velocity
    series, smooth it, find local minima, filter by velocity limit, select
    Top-K with suppression and fallback fill to reach target count.
  - Output: saves frames with frame indices and writes `detected_frames.txt`.
  - Evaluation: nearest-neighbor MAE vs `ground_truth.txt`.
  - MAE: 6.38 frames.

## Common Pipeline
- Input: `old.mp4`
- Output folders: `motion_capture_v2_results` / `motion_capture_v3_results`
- Metadata: `detected_frames.txt`
