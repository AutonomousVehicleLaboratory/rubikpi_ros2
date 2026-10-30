#!/usr/bin/env python3
"""
Measure AprilTag distance-estimation error without moving the robot.

For each trial: place the camera a known distance from a tag, press Enter,
and the script averages the estimate over --frames camera frames, then asks
for the true distance you measured with a tape. Every trial is saved to a
CSV for your report.

Needs the camera running:
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
Then, from this folder:
    python3 tag_distance.py --size 0.163
    python3 tag_distance.py --size 0.163 --frames 30 --out tag_error.csv

What "distance" means here
    range   : straight-line distance from the camera to the tag CENTER
    forward : distance along the camera's viewing direction only
Measure with the tape from the front of the LENS to the tag's printed face.
For a tag straight ahead and facing the camera squarely, range and forward
are almost equal; compare against range otherwise.
"""
import argparse
import csv
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage

from tag_localization import TagDetector


class Cam(Node):
    def __init__(self):
        super().__init__('tag_distance')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.frame = None          # (receive_time, jpeg bytes)
        self.K = self.D = None
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed',
                                 lambda m: setattr(self, 'frame', (time.time(), bytes(m.data))),
                                 qos)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._info, 10)

    def _info(self, m):
        if self.K is None:
            self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(m.d, dtype=np.float64)


def measure_frame(det, img, K, D):
    """Return {tag_id: (range, forward, bearing_deg)} for one image."""
    out = {}
    for d in det.detect(img, K, D):
        r, t, _ = d['candidates'][0]
        t = np.asarray(t, dtype=np.float64)
        out[d['id']] = (float(np.linalg.norm(t)), float(t[2]),
                        math.degrees(math.atan2(-t[0], t[2])))
    return out


def collect(node, det, n_frames, timeout=15.0):
    """Average over n fresh frames. Returns {id: dict of stats}."""
    samples = {}
    t_last = time.time()
    got = 0
    t_end = time.time() + timeout
    while got < n_frames and time.time() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        f = node.frame
        if f is None or f[0] <= t_last or node.K is None:
            continue
        t_last = f[0]
        img = cv2.imdecode(np.frombuffer(f[1], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        got += 1
        for tid, v in measure_frame(det, img, node.K, node.D).items():
            samples.setdefault(tid, []).append(v)
    stats = {}
    for tid, vals in samples.items():
        a = np.array(vals)
        stats[tid] = {'n': len(a), 'range': a[:, 0].mean(), 'range_std': a[:, 0].std(),
                      'forward': a[:, 1].mean(), 'bearing': a[:, 2].mean()}
    return stats, got


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=float, default=0.163, help='black square edge, m')
    p.add_argument('--family', default='36h11')
    p.add_argument('--frames', type=int, default=20, help='frames averaged per trial')
    p.add_argument('--out', default=os.path.expanduser('~/hw2_logs/tag_distance_error.csv'))
    args = p.parse_args()

    rclpy.init()
    node = Cam()
    det = TagDetector(args.size, args.family)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new_file = not os.path.exists(args.out)
    rows = []
    try:
        print('Waiting for camera and camera_info ...')
        t_end = time.time() + 10
        while (node.K is None or node.frame is None) and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.K is None:
            print('No /camera/camera_info. Is the camera running with its calibration?')
            return
        print(f'Tag size {args.size} m, averaging {args.frames} frames per trial.')
        print(f'Saving to {args.out}\n')

        with open(args.out, 'a', newline='') as fcsv:
            w = csv.writer(fcsv)
            if new_file:
                w.writerow(['tag', 'true_m', 'range_m', 'range_std_m', 'forward_m',
                            'bearing_deg', 'error_m', 'error_pct', 'frames'])
            while rclpy.ok():
                cmd = input('Place the robot, then press Enter to measure (q to finish): ')
                if cmd.strip().lower() == 'q':
                    break
                stats, got = collect(node, det, args.frames)
                if not stats:
                    print(f'  no tags in {got} frames; check the view and lighting\n')
                    continue
                for tid, s in sorted(stats.items()):
                    print(f'  tag {tid}: range {s["range"]:.3f} m (std {s["range_std"] * 100:.1f} cm, '
                          f'{s["n"]}/{got} frames)  forward {s["forward"]:.3f} m  '
                          f'bearing {s["bearing"]:+.1f} deg')
                tid = min(stats, key=lambda k: stats[k]['range'])
                if len(stats) > 1:
                    pick = input(f'  Which tag did you measure? [{tid}]: ').strip()
                    if pick:
                        tid = int(pick)
                if tid not in stats:
                    print('  that tag was not detected; skipping\n')
                    continue
                ans = input(f'  True distance to tag {tid} in meters (Enter to skip): ').strip()
                if not ans:
                    print()
                    continue
                try:
                    true = float(ans)
                except ValueError:
                    print('  not a number; skipped\n')
                    continue
                s = stats[tid]
                err = s['range'] - true
                pct = 100 * err / true if true else float('nan')
                w.writerow([tid, true, round(s['range'], 4), round(s['range_std'], 4),
                            round(s['forward'], 4), round(s['bearing'], 2),
                            round(err, 4), round(pct, 2), s['n']])
                fcsv.flush()
                rows.append((true, err, pct))
                print(f'  error {err * 100:+.1f} cm ({pct:+.1f}%)\n')

        if rows:
            e = np.array([r[1] for r in rows])
            pct = np.array([r[2] for r in rows])
            print(f'\n{len(rows)} trials: mean error {e.mean() * 100:+.1f} cm, '
                  f'mean |error| {np.abs(e).mean() * 100:.1f} cm, '
                  f'mean error {pct.mean():+.1f}%')
            if abs(pct.mean()) > 2 and np.std(pct) < abs(pct.mean()):
                print(f'Errors are mostly a constant percentage, which points to the '
                      f'tag size or camera calibration. A tag_size of '
                      f'{args.size / (1 + pct.mean() / 100):.4f} m would cancel it.')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

