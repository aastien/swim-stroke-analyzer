#!/usr/bin/env python3
"""Compare MediaPipe and RTMPose landmark tracks on the same frames.

Produces annotated videos, a side-by-side comparison MP4, and a JSON/text
report. Does not compute coaching or stroke-technique metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)

from landmarks import COMMON_LANDMARKS, SKELETON_EDGES, USABLE_CONFIDENCE  # noqa: E402

GREEN = (40, 200, 80)
ORANGE = (0, 165, 255)
RED = (40, 40, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL_BG = (32, 32, 32)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as fh:
        return json.load(fh)


def _landmark(frame: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    item = frame['landmarks'].get(name)
    if item is None:
        return None
    return item


def compute_model_metrics(data: Dict[str, Any], fps: float) -> Dict[str, Any]:
    frames = data['frames']
    n = len(frames)
    detected = sum(1 for fr in frames if fr['person_detected'])

    per_landmark = {}
    all_confidences: List[float] = []
    for name in COMMON_LANDMARKS:
        usable_count = 0
        confs: List[float] = []
        dropouts = 0
        jitter_px: List[float] = []
        prev_usable = False
        prev_xy: Optional[Tuple[float, float]] = None
        for fr in frames:
            item = _landmark(fr, name)
            usable = bool(item and item['usable'])
            if item is not None:
                confs.append(float(item['confidence']))
                all_confidences.append(float(item['confidence']))
            if usable:
                usable_count += 1
                xy = (float(item['x']), float(item['y']))
                if prev_usable and prev_xy is not None:
                    dx = xy[0] - prev_xy[0]
                    dy = xy[1] - prev_xy[1]
                    jitter_px.append(float(np.hypot(dx, dy)))
                prev_xy = xy
            elif prev_usable:
                dropouts += 1
            prev_usable = usable
        per_landmark[name] = {
            'usable_frame_pct': (100.0 * usable_count / n) if n else 0.0,
            'usable_frames': usable_count,
            'mean_confidence': float(np.mean(confs)) if confs else None,
            'dropouts': dropouts,
            'mean_jitter_px': float(np.mean(jitter_px)) if jitter_px else None,
            'jitter_samples': len(jitter_px),
        }

    jitter_values = [
        per_landmark[name]['mean_jitter_px']
        for name in COMMON_LANDMARKS
        if per_landmark[name]['mean_jitter_px'] is not None
    ]
    dropout_total = sum(per_landmark[name]['dropouts'] for name in COMMON_LANDMARKS)
    # Dropouts among landmarks that were actually tracked at least once.
    tracked = [per_landmark[n] for n in COMMON_LANDMARKS if per_landmark[n]['usable_frames'] > 0]
    tracked_dropouts = sum(p['dropouts'] for p in tracked)
    tracked_usable = sum(p['usable_frames'] for p in tracked)
    dropout_rate = (
        tracked_dropouts / (tracked_usable + tracked_dropouts)
        if (tracked_usable + tracked_dropouts) else None
    )
    mean_usable_pct = float(np.mean([per_landmark[n]['usable_frame_pct'] for n in COMMON_LANDMARKS]))
    inference = float(data.get('inference_seconds') or 0.0)
    effective_fps = (n / inference) if inference > 0 else 0.0

    return {
        'backend': data.get('backend'),
        'frames': n,
        'person_detected_frames': detected,
        'person_detected_pct': (100.0 * detected / n) if n else 0.0,
        'mean_usable_landmark_pct': mean_usable_pct,
        'mean_landmark_confidence': float(np.mean(all_confidences)) if all_confidences else None,
        'dropouts_total': dropout_total,
        'dropout_rate_among_tracked': dropout_rate,
        'mean_jitter_px': float(np.mean(jitter_values)) if jitter_values else None,
        'inference_seconds': inference,
        'init_seconds': float(data.get('init_seconds') or 0.0),
        'effective_fps': effective_fps,
        'source_fps': fps,
        'per_landmark': per_landmark,
    }


def _color_for(item: Optional[Dict[str, Any]]) -> Tuple[int, int, int]:
    if item is None:
        return RED
    if item['usable']:
        return GREEN
    return ORANGE


def _draw_label_bar(image: np.ndarray, title: str, detected: bool, missing: List[str]) -> None:
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 36), BLACK, -1)
    status = 'PERSON' if detected else 'NO PERSON'
    color = GREEN if detected else RED
    cv2.putText(image, f'{title}  |  {status}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if missing:
        text = 'missing: ' + ', '.join(missing[:6]) + ('…' if len(missing) > 6 else '')
        cv2.putText(image, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)


def draw_pose(frame: np.ndarray, record: Dict[str, Any], title: str) -> np.ndarray:
    canvas = frame.copy()
    last_missing: List[str] = []
    for a, b in SKELETON_EDGES:
        la = _landmark(record, a)
        lb = _landmark(record, b)
        if la is None or lb is None:
            continue
        pa = (int(round(la['x'])), int(round(la['y'])))
        pb = (int(round(lb['x'])), int(round(lb['y'])))
        if la['usable'] and lb['usable']:
            color = GREEN
            thickness = 2
        else:
            color = RED if (not la['usable'] or not lb['usable']) else ORANGE
            thickness = 1
        cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)

    missing = []
    for name in COMMON_LANDMARKS:
        item = _landmark(record, name)
        if item is None:
            missing.append(name)
            continue
        pt = (int(round(item['x'])), int(round(item['y'])))
        color = _color_for(item)
        if item['usable']:
            cv2.circle(canvas, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 6, WHITE, 1, cv2.LINE_AA)
        else:
            # Low-confidence prediction: orange open circle plus a small cross.
            cv2.circle(canvas, pt, 6, ORANGE, 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, pt, ORANGE, cv2.MARKER_TILTED_CROSS, 10, 1)
        conf = item['confidence']
        cv2.putText(
            canvas,
            f'{conf:.2f}',
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
        if not item['usable']:
            missing.append(name)

    last_missing = missing
    _draw_label_bar(canvas, title, record['person_detected'], last_missing)
    return canvas


def _open_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    for fourcc in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f'Could not open VideoWriter for {path}')


def reencode_h264(path: str) -> None:
    """Best-effort local ffmpeg re-encode; argv list, no shell."""
    base, ext = os.path.splitext(path)
    tmp = f'{base}_reenc{ext}'
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-y', '-i', path,
                '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23',
                '-an', '-movflags', '+faststart', tmp,
            ],
            capture_output=True,
            timeout=180,
            shell=False,
        )
        if result.returncode == 0 and os.path.isfile(tmp):
            os.replace(tmp, path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def write_videos(
    video_path: str,
    mp_data: Dict[str, Any],
    rtm_data: Dict[str, Any],
    out_dir: str,
    fps: float,
) -> Dict[str, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paths = {
        'mediapipe': os.path.join(out_dir, 'mediapipe_annotated.mp4'),
        'rtmpose': os.path.join(out_dir, 'rtmpose_annotated.mp4'),
        'comparison': os.path.join(out_dir, 'comparison_side_by_side.mp4'),
    }
    mp_writer = _open_writer(paths['mediapipe'], fps, width, height)
    rtm_writer = _open_writer(paths['rtmpose'], fps, width, height)
    cmp_writer = _open_writer(paths['comparison'], fps, width * 2, height)

    mp_frames = mp_data['frames']
    rtm_frames = rtm_data['frames']
    n = min(len(mp_frames), len(rtm_frames))
    index = 0
    while index < n:
        ok, frame = cap.read()
        if not ok:
            break
        left = draw_pose(frame, mp_frames[index], 'MediaPipe')
        right = draw_pose(frame, rtm_frames[index], 'RTMPose')
        combo = np.hstack([left, right])
        mp_writer.write(left)
        rtm_writer.write(right)
        cmp_writer.write(combo)
        index += 1

    cap.release()
    mp_writer.release()
    rtm_writer.release()
    cmp_writer.release()
    for path in paths.values():
        reencode_h264(path)
    return paths


def decide_winner(mp_m: Dict[str, Any], rtm_m: Dict[str, Any]) -> Dict[str, Any]:
    """Rank raw tracking only. No technique or coaching claims."""
    criteria = []

    def better(name: str, mp_val, rtm_val, higher_is_better: bool) -> str:
        if mp_val is None and rtm_val is None:
            winner = 'tie'
        elif mp_val is None:
            winner = 'rtmpose'
        elif rtm_val is None:
            winner = 'mediapipe'
        elif abs(mp_val - rtm_val) < 1e-6:
            winner = 'tie'
        elif higher_is_better:
            winner = 'mediapipe' if mp_val > rtm_val else 'rtmpose'
        else:
            winner = 'mediapipe' if mp_val < rtm_val else 'rtmpose'
        criteria.append({
            'metric': name,
            'mediapipe': mp_val,
            'rtmpose': rtm_val,
            'winner': winner,
            'higher_is_better': higher_is_better,
        })
        return winner

    better('person_detected_pct', mp_m['person_detected_pct'], rtm_m['person_detected_pct'], True)
    better('mean_usable_landmark_pct', mp_m['mean_usable_landmark_pct'], rtm_m['mean_usable_landmark_pct'], True)
    better('mean_landmark_confidence', mp_m['mean_landmark_confidence'], rtm_m['mean_landmark_confidence'], True)
    better('dropout_rate_among_tracked', mp_m['dropout_rate_among_tracked'], rtm_m['dropout_rate_among_tracked'], False)
    better('mean_jitter_px', mp_m['mean_jitter_px'], rtm_m['mean_jitter_px'], False)
    better('effective_fps', mp_m['effective_fps'], rtm_m['effective_fps'], True)

    # Tracking quality votes exclude speed (effective_fps).
    quality = [c for c in criteria if c['metric'] != 'effective_fps']
    mp_votes = sum(1 for c in quality if c['winner'] == 'mediapipe')
    rtm_votes = sum(1 for c in quality if c['winner'] == 'rtmpose')
    coverage = next(c['winner'] for c in criteria if c['metric'] == 'mean_usable_landmark_pct')
    stability = [
        c['winner'] for c in criteria
        if c['metric'] in ('dropout_rate_among_tracked', 'mean_jitter_px')
    ]
    stability_set = set(stability)
    if coverage != 'tie' and stability_set and coverage not in stability_set and 'tie' not in stability_set:
        overall = 'mixed'
    elif mp_votes > rtm_votes:
        overall = 'mediapipe'
    elif rtm_votes > mp_votes:
        overall = 'rtmpose'
    else:
        overall = 'tie'

    return {
        'overall_raw_tracking': overall,
        'coverage_winner': coverage,
        'stability_winner': (
            next(iter(stability_set)) if len(stability_set) == 1 else 'mixed'
        ),
        'quality_votes': {'mediapipe': mp_votes, 'rtmpose': rtm_votes},
        'criteria': criteria,
        'note': (
            'Coverage is mean usable-landmark percentage across the 13 shared joints. '
            'Stability is dropout rate (among joints that were tracked) and temporal jitter. '
            'If coverage and stability disagree, overall_raw_tracking is mixed. '
            'Speed is reported but not used as a tracking-quality vote. '
            'This is not a coaching or technique judgment.'
        ),
    }


def format_text_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append('POSE ESTIMATION BENCHMARK: MediaPipe vs RTMPose')
    lines.append('Scope: raw body tracking only. No coaching metrics.')
    lines.append(f"Video: {report['video']}")
    lines.append(
        f"Frames compared: {report['frames_compared']} at {report['source_fps']:.3f} fps "
        f"({report['width']}x{report['height']})"
    )
    lines.append(f"Usable confidence threshold: {report['usable_confidence_threshold']}")
    lines.append('')
    for key in ('mediapipe', 'rtmpose'):
        m = report[key]
        lines.append(f'=== {key.upper()} ===')
        lines.append(f"  Person detected: {m['person_detected_pct']:.2f}% "
                     f"({m['person_detected_frames']}/{m['frames']})")
        lines.append(f"  Mean usable landmark % (13 joints): {m['mean_usable_landmark_pct']:.2f}%")
        mean_c = m['mean_landmark_confidence']
        lines.append(f"  Mean landmark confidence: {mean_c:.4f}" if mean_c is not None else
                     '  Mean landmark confidence: n/a')
        jitter = m['mean_jitter_px']
        dr = m.get('dropout_rate_among_tracked')
        lines.append(f"  Landmark dropouts (usable -> missing): {m['dropouts_total']}")
        lines.append(
            f"  Dropout rate among tracked joints: {100.0 * dr:.2f}%"
            if dr is not None else '  Dropout rate among tracked joints: n/a'
        )
        lines.append(f"  Mean temporal jitter: {jitter:.2f} px" if jitter is not None else
                     '  Mean temporal jitter: n/a')
        lines.append(f"  Inference time: {m['inference_seconds']:.2f}s  "
                     f"init {m['init_seconds']:.2f}s  effective {m['effective_fps']:.2f} FPS")
        lines.append('  Per-landmark usable % / mean conf / dropouts / jitter px:')
        for name in COMMON_LANDMARKS:
            pl = m['per_landmark'][name]
            conf = f"{pl['mean_confidence']:.3f}" if pl['mean_confidence'] is not None else 'n/a'
            jit = f"{pl['mean_jitter_px']:.2f}" if pl['mean_jitter_px'] is not None else 'n/a'
            lines.append(
                f"    {name:16s}  usable {pl['usable_frame_pct']:6.2f}%  "
                f"conf {conf:>6}  dropouts {pl['dropouts']:4d}  jitter {jit:>7}"
            )
        lines.append('')
    winner = report['decision']['overall_raw_tracking']
    lines.append(f'Raw tracking winner: {winner}')
    lines.append(
        f"Coverage (usable joints): {report['decision'].get('coverage_winner')}  |  "
        f"Stability (jitter/dropouts): {report['decision'].get('stability_winner')}"
    )
    lines.append(report['decision']['note'])
    lines.append('')
    lines.append('Per-criterion winners:')
    for c in report['decision']['criteria']:
        lines.append(f"  {c['metric']}: {c['winner']}")
    lines.append('')
    lines.append('Outputs:')
    for k, v in report['outputs'].items():
        lines.append(f'  {k}: {v}')
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare MediaPipe and RTMPose landmark JSON dumps.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--mediapipe', required=True)
    parser.add_argument('--rtmpose', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    mp_data = load_json(args.mediapipe)
    rtm_data = load_json(args.rtmpose)
    n = min(len(mp_data['frames']), len(rtm_data['frames']))
    mp_data['frames'] = mp_data['frames'][:n]
    rtm_data['frames'] = rtm_data['frames'][:n]

    fps = float(mp_data.get('fps') or rtm_data.get('fps') or 30.0)
    os.makedirs(args.out_dir, exist_ok=True)
    outputs = write_videos(os.path.abspath(args.video), mp_data, rtm_data, args.out_dir, fps)

    mp_metrics = compute_model_metrics(mp_data, fps)
    rtm_metrics = compute_model_metrics(rtm_data, fps)
    decision = decide_winner(mp_metrics, rtm_metrics)

    report = {
        'video': os.path.abspath(args.video),
        'frames_compared': n,
        'source_fps': fps,
        'width': mp_data.get('width'),
        'height': mp_data.get('height'),
        'usable_confidence_threshold': USABLE_CONFIDENCE,
        'common_landmarks': COMMON_LANDMARKS,
        'processed_every_frame': True,
        'excluded': [
            'elbow catch quality',
            'body rotation',
            'centerline crossing',
            'head lifting',
            'knee technique',
            'stroke rate',
            'technique score',
            'coaching recommendations',
        ],
        'mediapipe': mp_metrics,
        'rtmpose': rtm_metrics,
        'decision': decision,
        'outputs': outputs,
    }

    json_path = os.path.join(args.out_dir, 'pose_benchmark.json')
    txt_path = os.path.join(args.out_dir, 'pose_benchmark.txt')
    report['outputs']['json'] = os.path.abspath(json_path)
    report['outputs']['text'] = os.path.abspath(txt_path)
    text = format_text_report(report)
    with open(json_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    with open(txt_path, 'w') as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
