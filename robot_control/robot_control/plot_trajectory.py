#!/usr/bin/env python3
"""
Plot a HW2 run: estimated positions (one dot per step), waypoints, and tags.

Usage (on the Pi or after copying the CSVs to your laptop):
    python3 plot_trajectory.py ~/hw2_logs/trajectory_<stamp>.csv
    python3 plot_trajectory.py               # newest file in ~/hw2_logs
Writes trajectory_<stamp>.png next to the CSV.
"""
import csv
import glob
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')                     # no display needed
import matplotlib.pyplot as plt

WAYPOINTS = [(0, 0, 0), (1, 0, 0), (1, 2, math.pi), (0, 0, 0)]
# keep in sync with config/hw2_params.yaml: id: (x, y, facing yaw)
TAGS = {0: (2.0, 0.0, math.pi), 1: (1.0, -3.0, math.pi / 2),
        2: (0.0, -2.0, 0.0), 3: (-0.447, 0.894, -1.107)}


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.expanduser('~/hw2_logs/trajectory_*.csv')))
        if not files:
            sys.exit('No trajectory CSV found; pass a path.')
        path = files[-1]

    rows = list(csv.DictReader(open(path)))
    xs = [float(r['x']) for r in rows]
    ys = [float(r['y']) for r in rows]
    th = [float(r['theta']) for r in rows]
    src = [int(r['source']) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.plot(xs, ys, color='#d62728', lw=0.8, alpha=0.4)
    tag = [i for i, s in enumerate(src) if s != 0]
    odo = [i for i, s in enumerate(src) if s == 0]
    ax.scatter([xs[i] for i in tag], [ys[i] for i in tag], s=22, color='#d62728',
               label='estimate (tag seen)', zorder=3)
    ax.scatter([xs[i] for i in odo], [ys[i] for i in odo], s=22, facecolors='none',
               edgecolors='#d62728', label='estimate (odometry only)', zorder=3)
    ax.quiver(xs, ys, [math.cos(t) for t in th], [math.sin(t) for t in th],
              angles='xy', scale=25, width=0.003, color='#d62728', alpha=0.5)

    for i, (x, y, t) in enumerate(WAYPOINTS[:-1]):
        ax.plot(x, y, 'k*', ms=12, zorder=4)
        ax.annotate(f'WP{i}', (x, y), textcoords='offset points', xytext=(6, 6))
    for tid, (x, y, a) in TAGS.items():
        ax.plot([x - 0.09 * math.sin(a), x + 0.09 * math.sin(a)],
                [y + 0.09 * math.cos(a), y - 0.09 * math.cos(a)], color='#1f77b4', lw=4)
        ax.arrow(x, y, 0.2 * math.cos(a), 0.2 * math.sin(a), color='#1f77b4',
                 head_width=0.05, length_includes_head=True)
        ax.annotate(f'tag {tid}', (x, y), textcoords='offset points', xytext=(6, -12),
                    color='#1f77b4')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Estimated trajectory (one dot per step)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    out = os.path.splitext(path)[0] + '.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
