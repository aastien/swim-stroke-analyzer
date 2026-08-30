# MediaPipe vs RTMPose pose-estimation benchmark

This experiment is **isolated from the swim-analysis application**. It does not
change coaching logic, Flask, or the frontend.

It answers only:

> Which pose model gives better raw body tracking on the same video frames?

## What it produces

Given one **local** video:

| File | Description |
| --- | --- |
| `results/mediapipe_annotated.mp4` | MediaPipe skeleton on the source frames |
| `results/rtmpose_annotated.mp4` | RTMPose skeleton on the same frames |
| `results/comparison_side_by_side.mp4` | MediaPipe left, RTMPose right |
| `results/pose_benchmark.json` | Metrics |
| `results/pose_benchmark.txt` | Human-readable report |

Green = usable landmark (confidence ≥ 0.5 and in-frame). Orange = low confidence.
Red = missing landmark or incomplete bone.

## Isolated environment

Do **not** install these packages into the app virtualenv.

```bash
python3 -m venv experiments/rtmpose/.venv
experiments/rtmpose/.venv/bin/pip install -r experiments/rtmpose/requirements-rtmpose.txt
```

RTMPose uses `rtmlib` + `onnxruntime` on CPU, `mode=balanced`. ONNX weights are
downloaded from the official OpenMMLab RTMPose URLs into the local rtmlib cache.
Video frames are never sent to a network service.

## Run

```bash
# App venv must already have MediaPipe (pip install -r requirements.txt).
python experiments/rtmpose/run_benchmark.py --video /path/to/local/swim.mp4
```

Optional: `--max-frames N` to cap length. Every processed frame is still the
**same** frame for both models (no frame skipping).

## Compared landmarks

nose, shoulders, elbows, wrists, hips, knees, ankles.

Not computed: elbow catch, body rotation, centerline, head lift, knee technique,
stroke rate, technique score, or coaching text.
