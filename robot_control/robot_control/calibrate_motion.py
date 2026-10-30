#!/usr/bin/env python3
"""
Quick motion calibration for hw2_navigator. Run it whenever the battery
level changes noticeably.

Needs the motor node running in another terminal:
    ros2 run robot_control motor_control
Then:
    python3 calibrate_motion.py                 # turn test + forward test
    python3 calibrate_motion.py --skip-forward  # turns only
    python3 calibrate_motion.py --manual        # no IMU; you judge the angles

It measures:
  1. Turns: for increasing turn commands, turns left for --turn-time
     seconds and reads the IMU to see how far it really turned. This finds
     the weakest command that reliably turns and its turn rate. It also
     checks the IMU yaw sign.
  2. Forward: drives with drive_left/drive_right for --drive-time seconds.
     You measure the distance with a tape measure and type it in. It
     reports the speed and whether the robot veers.
At the end it prints the lines to paste into hw2_params.yaml.
Give it at least 1.5 m of clear floor ahead and room to spin.
"""
import argparse
import math
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Calib(Node):
    def __init__(self):
        super().__init__('calibrate_motion')
        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, '/imu_data', self._imu_cb, 10)
        self.yaw = None          # unwrapped, raw sign from the IMU
        self._prev = None

    def _imu_cb(self, msg):
        q = msg.orientation
        raw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.yaw = raw if self._prev is None else self.yaw + wrap(raw - self._prev)
        self._prev = raw

    def send(self, left, right):
        m = Float32MultiArray()
        m.data = [float(left), float(right)]
        self.pub.publish(m)

    def run_for(self, left, right, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:
            self.send(left, right)
            time.sleep(0.05)         # motor node stops after 0.15 s of silence
        for _ in range(3):
            self.send(0.0, 0.0)
            time.sleep(0.05)

    def measured(self, left, right, seconds, settle=0.6):
        """Run a command and return the IMU yaw change (including coasting)."""
        y0 = self.yaw
        self.run_for(left, right, seconds)
        time.sleep(settle)
        return self.yaw - y0


def ask(prompt):
    return input(prompt).strip().lower()


def manual(node, args):
    """Timed turns measured by eye, for when the IMU can't be trusted."""
    c, T = args.turn_cmd, 2.0
    try:
        print(f'\nManual turn test at turn_cmd={c} for {T:.1f} s each way.')
        print('Put a strip of tape on the floor along the robot\'s front edge so you')
        print('can see how far it turned. A quarter turn is 90, a half turn 180.')
        rates = []
        for name, (l, r) in [('LEFT', (-c, c)), ('RIGHT', (c, -c))]:
            ask(f'\nPress Enter to turn {name}.')
            node.run_for(l, r, T)
            deg = float(ask(f'  Degrees it turned {name}: '))
            rates.append(math.radians(deg) / T)
            print(f'  {name}: {deg / T:.1f} deg/s')
        if not args.skip_forward:
            ask(f'\nForward test: mark the start, then press Enter '
                f'(L={args.drive_left}, R={args.drive_right}, {args.drive_time:.1f} s).')
            node.run_for(args.drive_left, args.drive_right, args.drive_time)
            dist = float(ask('  Distance forward, in meters: '))
            side = float(ask('  Sideways drift in meters (+ = ended LEFT of the '
                             'straight line, - = right, 0 if none): '))
            speed = dist / args.drive_time
            veer = math.degrees(math.atan2(2 * side, dist)) if dist > 0 else 0.0
            print(f'  speed {speed:.3f} m/s, veers about {veer:+.1f} deg')
        print('\nPaste into hw2_params.yaml (then rebuild):')
        print('    use_imu: false')
        print(f'    turn_cmd: {c}')
        print(f'    search_cmd: {c}')
        print(f'    turn_rate: {sum(rates) / 2:.3f}')
        if abs(rates[0] - rates[1]) > 0.15 * max(rates):
            print('  (left and right differ by >15%; the navigator uses their average)')
        if not args.skip_forward:
            print(f'    drive_speed: {speed:.3f}')
    except (KeyboardInterrupt, EOFError, ValueError):
        pass
    finally:
        node.run_for(0.0, 0.0, 0.1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cmds', default='0.15,0.2,0.25,0.3,0.35,0.4',
                   help='turn commands to try, weakest first')
    p.add_argument('--turn-time', type=float, default=1.0)
    p.add_argument('--drive-left', type=float, default=0.31)
    p.add_argument('--drive-right', type=float, default=0.40)
    p.add_argument('--drive-time', type=float, default=2.0)
    p.add_argument('--skip-forward', action='store_true')
    p.add_argument('--manual', action='store_true',
                   help='no IMU: you measure the turn angles by eye')
    p.add_argument('--turn-cmd', type=float, default=0.30,
                   help='turn strength for --manual')
    args = p.parse_args()

    rclpy.init()
    node = Calib()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    results = {}
    if args.manual:
        manual(node, args)
        node.destroy_node()
        rclpy.shutdown()
        return
    try:
        print('Waiting for IMU on /imu_data ...')
        t_end = time.time() + 5
        while node.yaw is None and time.time() < t_end:
            time.sleep(0.1)
        if node.yaw is None:
            print('No IMU data. Is motor_control running?')
            return

        # ---------- 1. turns ----------
        ask('\n[1/2] Turn test. Clear space around the robot, then press Enter.')
        cmds = [float(c) for c in args.cmds.split(',')]
        sign = None
        rows = []
        for c in cmds:
            dl = node.measured(-c, c, args.turn_time)   # left (CCW)
            dr = node.measured(c, -c, args.turn_time)   # right (CW), returns it home
            if sign is None and max(abs(dl), abs(dr)) > 0.3:
                # left turns should raise yaw; if they lower it, flip the sign
                sign = 1.0 if dl > 0 else -1.0
            s = sign or 1.0
            left_rate = s * dl / args.turn_time
            right_rate = -s * dr / args.turn_time
            rows.append((c, left_rate, right_rate))
            print(f'  cmd {c:.2f}: left {math.degrees(left_rate):6.1f} deg/s, '
                  f'right {math.degrees(right_rate):6.1f} deg/s')

        # weakest command where BOTH directions turn at >= 20 deg/s
        good = [r for r in rows if min(r[1], r[2]) >= math.radians(20)]
        if good:
            # add one step of margin so a weaker battery still turns
            idx = rows.index(good[0])
            pick = rows[min(idx + 1, len(rows) - 1)]
            results['turn_cmd'] = pick[0]
            results['turn_rate'] = round((pick[1] + pick[2]) / 2, 3)
            results['search_cmd'] = good[0][0]
        else:
            print('  None of the commands turned reliably; try stronger --cmds.')
        if sign is not None:
            results['imu_yaw_sign'] = sign
        # how far the robot keeps turning after the stop (for turn_lead)
        if good:
            c = results['turn_cmd']
            node.run_for(-c, c, 0.5)
            y_stop = node.yaw
            time.sleep(0.8)
            coast = abs(node.yaw - y_stop)
            node.measured(c, -c, 0.5)                   # turn back
            results['turn_lead'] = round(min(max(coast, 0.02), 0.2), 3)
            print(f'  coasts {math.degrees(coast):.1f} deg after stopping')

        # ---------- 2. forward ----------
        if not args.skip_forward:
            ask(f'\n[2/2] Forward test: mark the robot\'s position. It will drive for '
                f'{args.drive_time:.1f} s at L={args.drive_left}, R={args.drive_right}. '
                f'Press Enter.')
            dyaw = node.measured(args.drive_left, args.drive_right, args.drive_time)
            dist = float(ask('  Distance traveled, in meters (e.g. 0.95): '))
            results['drive_speed'] = round(dist / args.drive_time, 3)
            veer = (sign or 1.0) * dyaw
            print(f'  speed {results["drive_speed"]:.3f} m/s, heading changed '
                  f'{math.degrees(veer):+.1f} deg ({"left" if veer > 0 else "right"})')
            if abs(veer) > math.radians(3):
                print('  It veers. To drive straighter: if it veers LEFT, raise '
                      'drive_left (or lower drive_right) by ~0.01-0.02; if RIGHT, '
                      'the opposite. Re-run with --drive-left/--drive-right to check.')

        # ---------- summary ----------
        print('\nPaste into hw2_params.yaml (then rebuild):')
        for k in ['turn_cmd', 'turn_rate', 'turn_lead', 'search_cmd', 'imu_yaw_sign',
                  'drive_speed']:
            if k in results:
                print(f'    {k}: {results[k]}')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for _ in range(3):
            node.send(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
