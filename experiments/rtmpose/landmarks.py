"""Shared landmark names and index maps for the pose benchmark.

Only landmarks present in both MediaPipe Pose (33) and RTMPose COCO-17
are compared. This module is experiment-only and is not used by the app.
"""

from typing import Dict, List, Tuple

COMMON_LANDMARKS: List[str] = [
    'nose',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

# MediaPipe Pose landmark indices (same as src.pose_detector.PoseDetector).
MEDIAPIPE_INDEX: Dict[str, int] = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

# RTMPose / COCO-17 body indices (rtmlib Body, to_openpose=False).
COCO_INDEX: Dict[str, int] = {
    'nose': 0,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

SKELETON_EDGES: List[Tuple[str, str]] = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('nose', 'left_shoulder'),
    ('nose', 'right_shoulder'),
]

# Landmark is usable when confidence is at least this and the point is in-frame.
USABLE_CONFIDENCE = 0.5
# RTMPose person detection: YOLOX returned a box, or enough confident joints.
PERSON_MIN_USABLE_JOINTS = 3
