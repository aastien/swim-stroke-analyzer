#!/usr/bin/env python3
"""Run the isolated MediaPipe vs RTMPose pose benchmark.

MediaPipe uses the existing application virtualenv.
RTMPose uses experiments/rtmpose/.venv so its deps cannot break the app.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
DEFAULT_OUT = os.path.join(EXPERIMENT_DIR, 'results')
DEFAULT_RTM_VENV = os.path.join(EXPERIMENT_DIR, '.venv', 'bin', 'python')
DEFAULT_APP_VENV = os.path.join(REPO_ROOT, '.venv', 'bin', 'python')


def _python_exists(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def run(cmd: list, env: dict) -> None:
    print('+ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, shell=False)


def main() -> int:
    parser = argparse.ArgumentParser(description='Local MediaPipe vs RTMPose pose benchmark.')
    parser.add_argument('--video', required=True, help='Local swimming video. Never uploaded.')
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--mediapipe-python', default=DEFAULT_APP_VENV)
    parser.add_argument('--rtmpose-python', default=DEFAULT_RTM_VENV)
    args = parser.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        raise SystemExit(f'Video not found: {video}')

    mp_py = args.mediapipe_python if _python_exists(args.mediapipe_python) else sys.executable
    rtm_py = args.rtmpose_python
    if not _python_exists(rtm_py):
        raise SystemExit(
            f'RTMPose interpreter not found: {rtm_py}\n'
            f'Create it with:\n'
            f'  python3 -m venv {os.path.join(EXPERIMENT_DIR, ".venv")}\n'
            f'  {os.path.join(EXPERIMENT_DIR, ".venv/bin/pip")} install '
            f'-r {os.path.join(EXPERIMENT_DIR, "requirements-rtmpose.txt")}'
        )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mp_json = os.path.join(out_dir, 'mediapipe_poses.json')
    rtm_json = os.path.join(out_dir, 'rtmpose_poses.json')
    extract = os.path.join(EXPERIMENT_DIR, 'extract_poses.py')
    compare = os.path.join(EXPERIMENT_DIR, 'compare.py')

    extra = []
    if args.max_frames is not None:
        extra = ['--max-frames', str(args.max_frames)]

    env = os.environ.copy()
    env.pop('PYTHONPATH', None)

    run([mp_py, extract, '--backend', 'mediapipe', '--video', video, '--out', mp_json, *extra], env)
    run([rtm_py, extract, '--backend', 'rtmpose', '--video', video, '--out', rtm_json, *extra], env)
    run([mp_py, compare, '--video', video, '--mediapipe', mp_json, '--rtmpose', rtm_json, '--out-dir', out_dir], env)
    print(f'\nBenchmark complete. Outputs in {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
