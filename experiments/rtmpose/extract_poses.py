#!/usr/bin/env python3
"""Extract per-frame common landmarks from a local video.

Backends:
  mediapipe  — existing src.pose_detector.PoseDetector (app venv)
  rtmpose    — rtmlib Body, onnxruntime CPU balanced (isolated venv)

All inference is local. The video is never uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
sys.path.insert(0, EXPERIMENT_DIR)

from landmarks import (  # noqa: E402
    COCO_INDEX,
    COMMON_LANDMARKS,
    MEDIAPIPE_INDEX,
    PERSON_MIN_USABLE_JOINTS,
    USABLE_CONFIDENCE,
)


def _in_frame(x: float, y: float, width: int, height: int) -> bool:
    return 0.0 <= x < width and 0.0 <= y < height


def _empty_landmarks() -> Dict[str, Optional[Dict[str, Any]]]:
    return {name: None for name in COMMON_LANDMARKS}


def _pack_landmark(x: float, y: float, confidence: float, width: int, height: int) -> Dict[str, Any]:
    usable = float(confidence) >= USABLE_CONFIDENCE and _in_frame(x, y, width, height)
    return {
        'x': float(x),
        'y': float(y),
        'confidence': float(confidence),
        'usable': bool(usable),
    }


def _person_detected_from_landmarks(landmarks: Dict[str, Optional[Dict[str, Any]]]) -> bool:
    usable = sum(1 for item in landmarks.values() if item is not None and item['usable'])
    return usable >= PERSON_MIN_USABLE_JOINTS


def extract_mediapipe(frames: List[np.ndarray], width: int, height: int) -> Dict[str, Any]:
    sys.path.insert(0, REPO_ROOT)
    from src.pose_detector import PoseDetector

    init_start = time.perf_counter()
    detector = PoseDetector()
    init_seconds = time.perf_counter() - init_start

    records = []
    inference_seconds = 0.0
    for index, frame in enumerate(frames):
        t0 = time.perf_counter()
        result = detector.detect_pose(frame)
        inference_seconds += time.perf_counter() - t0

        landmarks = _empty_landmarks()
        person_detected = result is not None
        if result is not None:
            for name in COMMON_LANDMARKS:
                idx = MEDIAPIPE_INDEX[name]
                raw = result['raw_landmarks'].landmark[idx]
                packed = _pack_landmark(raw.x * width, raw.y * height, raw.visibility, width, height)
                landmarks[name] = packed

        records.append({
            'frame_index': index,
            'person_detected': person_detected,
            'landmarks': landmarks,
        })

    return {
        'backend': 'mediapipe',
        'init_seconds': init_seconds,
        'inference_seconds': inference_seconds,
        'frames': records,
    }


def extract_rtmpose(frames: List[np.ndarray], width: int, height: int) -> Dict[str, Any]:
    from rtmlib import Body

    init_start = time.perf_counter()
    body = Body(backend='onnxruntime', device='cpu', mode='balanced', to_openpose=False)
    init_seconds = time.perf_counter() - init_start

    records = []
    inference_seconds = 0.0
    for index, frame in enumerate(frames):
        t0 = time.perf_counter()
        raw_bboxes = body.det_model(frame)
        if raw_bboxes is None:
            bboxes: list = []
        else:
            arr = np.asarray(raw_bboxes, dtype=np.float32)
            if arr.size == 0:
                bboxes = []
            else:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                bboxes = [row[:4].tolist() for row in arr]

        landmarks = _empty_landmarks()
        person_detected = False
        if len(bboxes) > 0:
            keypoints, scores = body.pose_model(frame, bboxes=bboxes)
            if keypoints is not None and len(keypoints) > 0:
                kpts = np.asarray(keypoints)
                scrs = np.asarray(scores)
                if kpts.ndim == 2:
                    kpts = kpts[None, ...]
                    scrs = scrs[None, ...]
                # Pick the person with the highest mean common-landmark score.
                common_idx = [COCO_INDEX[name] for name in COMMON_LANDMARKS]
                means = scrs[:, common_idx].mean(axis=1)
                best = int(np.argmax(means))
                for name in COMMON_LANDMARKS:
                    j = COCO_INDEX[name]
                    packed = _pack_landmark(
                        float(kpts[best, j, 0]),
                        float(kpts[best, j, 1]),
                        float(scrs[best, j]),
                        width,
                        height,
                    )
                    landmarks[name] = packed
                person_detected = _person_detected_from_landmarks(landmarks)

        inference_seconds += time.perf_counter() - t0
        records.append({
            'frame_index': index,
            'person_detected': person_detected,
            'landmarks': landmarks,
        })

    return {
        'backend': 'rtmpose',
        'rtmpose': {
            'library': 'rtmlib',
            'backend': 'onnxruntime',
            'device': 'cpu',
            'mode': 'balanced',
        },
        'init_seconds': init_seconds,
        'inference_seconds': inference_seconds,
        'frames': records,
    }


def read_video_frames(video_path: str, max_frames: Optional[int]) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Unable to open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise ValueError(f'No frames decoded from {video_path}')

    return {
        'video_path': os.path.abspath(video_path),
        'fps': fps if fps > 1e-3 else 30.0,
        'width': width,
        'height': height,
        'reported_frame_count': reported_count,
        'processed_frame_count': len(frames),
        'frames': frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Extract common pose landmarks from a local video.')
    parser.add_argument('--backend', choices=('mediapipe', 'rtmpose'), required=True)
    parser.add_argument('--video', required=True, help='Local video path. Never uploaded.')
    parser.add_argument('--out', required=True, help='JSON output path')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise SystemExit(f'Video not found: {video_path}')

    decoded = read_video_frames(video_path, args.max_frames)
    frames = decoded.pop('frames')

    if args.backend == 'mediapipe':
        result = extract_mediapipe(frames, decoded['width'], decoded['height'])
    else:
        result = extract_rtmpose(frames, decoded['width'], decoded['height'])

    payload = {
        **decoded,
        **result,
        'common_landmarks': COMMON_LANDMARKS,
        'usable_confidence_threshold': USABLE_CONFIDENCE,
        'processed_every_frame': True,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(payload, fh)
    n = payload['processed_frame_count']
    inf = payload['inference_seconds']
    print(
        f"{payload['backend']}: {n} frames, "
        f"inference {inf:.2f}s ({(n / inf) if inf > 0 else 0:.2f} fps)"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
