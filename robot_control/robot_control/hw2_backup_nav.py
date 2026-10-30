#!/usr/bin/env python3
"""
CSE 276A HW2: closed-loop waypoint driving with AprilTag localization.

Loop (one "step" is about 1 s of motion):
    1. Plan:   compare the pose estimate with the current waypoint and pick
               one motion: turn toward it, drive forward (<= step_time),
               back up a little after an overshoot, or turn to the final
               heading.
    2. Act:    run that motion, then stop.
    3. Predict: update the pose from the motion (distance from calibrated
               drive speed, heading change from the IMU).
    4. Look:   wait for the robot to settle, grab a fresh frame, detect
               tags, and correct the pose (complementary filter).
    5. Log:    append (t, x, y, theta, source, n_tags, waypoint) to the
               trajectory matrix, saved to CSV for plotting.
If several steps in a row see no tag, rotate slowly clockwise in small
increments, looking after each one, until a tag is seen (or a full turn
passes with nothing, which stops the run).

Needs (in other terminals):
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
    ros2 run robot_control motor_control
Then:
    ros2 run robot_control hw2_navigator --ros-args \
        --params-file ~/ros2_ws/src/rubikpi_ros2/robot_control/config/hw2_params.yaml
"""
import csv
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu
from std_msgs.msg import Float32MultiArray

from robot_control.tag_localization import TagDetector, blend, localize, wrap


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Hw2Navigator(Node):
    def __init__(self):
        super().__init__('hw2_navigator')
        P = self.declare_parameter
        # --- map and waypoints ---
        P('waypoints', [0.0, 0.0, 0.0,  1.0, 0.0, 0.0,
                        1.0, 2.0, math.pi,  0.0, 0.0, 0.0])
        P('tag_ids', [0, 1, 2, 3])
        P('tag_x', [2.0, 1.0, 0.5, -0.224])
        P('tag_y', [0.0, 2.5, 2.0, -0.447])
        P('tag_yaw', [math.pi, -math.pi / 2, 0.0, 1.107])
        P('tag_size', 0.1715)            # black square edge, meters
        P('tag_family', '36h11')
        # --- camera ---
        P('image_topic', '/camera/image_raw/compressed')
        P('camera_info_topic', '/camera/camera_info')
        P('camera_offset_x', 0.0)        # camera ahead of robot center, m
        P('camera_offset_y', 0.0)        # camera left of robot center, m
        P('image_is_rectified', False)   # True if camera runs image_rectify:=true
        # --- motion (HW1 calibration) ---
        P('drive_left', 0.31)            # L/R that drive straight
        P('drive_right', 0.40)
        P('drive_speed', 0.5)            # m/s at those commands: MEASURE THIS
        P('turn_cmd', 0.2)               # |L| = |R| for turning in place
        P('turn_rate', 0.785)            # rad/s at turn_cmd (fallback if no IMU)
        P('turn_rate_left', 0.0)         # rad/s turning left; 0 = use turn_rate
        P('turn_rate_right', 0.0)        # rad/s turning right; 0 = use turn_rate
        P('turn_lead', 0.05)             # stop this early (rad) to allow for coasting
        P('search_cmd', 0.15)            # slower turn while searching
        P('search_step', 0.35)           # rad per search increment (~20 deg)
        P('imu_topic', '/imu_data')
        P('use_imu', True)
        P('imu_yaw_sign', 1.0)           # set -1.0 if IMU yaw grows on RIGHT turns
        # --- control loop ---
        P('step_time', 1.0)              # max seconds of driving per step
        P('min_step_time', 0.15)
        P('settle_time', 0.4)            # wait after stopping before the photo
        P('pos_tol', 0.08)               # m
        P('reach_radius', 0.18)          # m; within this, a mostly SIDEWAYS miss counts as
                                         # arrived (fixing it would need two big turns)
        P('max_jump', 0.75)              # m; reject a tag fix this far from the prediction...
        P('accept_jump_after', 4)        # ...unless we've been blind this many steps
        P('ang_tol', 0.10)               # rad, final heading at a waypoint
        P('heading_tol', 0.12)           # rad, aim error allowed while driving
        P('alpha_pos', 0.8)              # trust in tag position vs prediction
        P('theta_trust_dist', 1.0)       # m, see tag_localization.localize
        P('max_blind_steps', 1)          # steps with no tag before searching
        P('max_steps_per_waypoint', 40)
        P('log_dir', os.path.expanduser('~/hw2_logs'))

        g = lambda n: self.get_parameter(n).value
        self.p = {n: g(n) for n in [
            'tag_size', 'tag_family', 'camera_offset_x', 'camera_offset_y',
            'image_is_rectified', 'drive_left', 'drive_right', 'drive_speed',
            'turn_cmd', 'turn_rate', 'turn_rate_left', 'turn_rate_right', 'turn_lead', 'search_cmd', 'search_step',
            'use_imu', 'imu_yaw_sign', 'step_time', 'min_step_time',
            'settle_time', 'pos_tol', 'reach_radius', 'max_jump', 'accept_jump_after',
            'ang_tol', 'heading_tol', 'alpha_pos',
            'theta_trust_dist', 'max_blind_steps', 'max_steps_per_waypoint',
            'log_dir']}
        wp = list(g('waypoints'))
        self.waypoints = [tuple(wp[i:i + 3]) for i in range(0, len(wp), 3)]
        self.tag_map = {int(i): (x, y, a) for i, x, y, a in
                        zip(g('tag_ids'), g('tag_x'), g('tag_y'), g('tag_yaw'))}
        self.cam_offset = (self.p['camera_offset_x'], self.p['camera_offset_y'])
        self.detector = TagDetector(self.p['tag_size'], self.p['tag_family'])

        # --- ROS I/O ---
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, g('image_topic'), self._img_cb, img_qos)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._info_cb, 10)
        self.create_subscription(Imu, g('imu_topic'), self._imu_cb, 10)

        self._frame = None               # (receive_time, jpeg bytes)
        self.K = None
        self.D = None
        self.imu_yaw = None              # unwrapped, radians, CCW positive
        self._imu_raw_prev = None

        # --- state ---
        self.pose = tuple(self.waypoints[0])
        self.t0 = time.time()
        self.wp_index = 0
        self.blind_steps = 0
        self.dist_traveled = 0.0         # meters (|forward| + |backward|)
        self.angle_turned = 0.0          # radians
        self.traj = []                   # rows: t, x, y, theta, source, n_tags, wp
        self.wp_results = []

        self.get_logger().info(f'Waypoints: {self.waypoints}')
        self.get_logger().info(f'Tag map: {self.tag_map}')

    # ---------------- callbacks ----------------
    def _img_cb(self, msg):
        self._frame = (time.time(), bytes(msg.data))

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64)
            self.D = np.zeros(5) if (self.p['image_is_rectified'] or d.size == 0) else d
            self.get_logger().info(f'Camera intrinsics received: fx={self.K[0, 0]:.1f} '
                                   f'cx={self.K[0, 2]:.1f} cy={self.K[1, 2]:.1f} D={self.D}')

    def _imu_cb(self, msg):
        raw = self.p['imu_yaw_sign'] * yaw_from_quaternion(msg.orientation)
        if self._imu_raw_prev is None:
            self.imu_yaw = raw
        else:
            self.imu_yaw += wrap(raw - self._imu_raw_prev)   # unwrap
        self._imu_raw_prev = raw

    # ---------------- motors ----------------
    def _publish(self, left, right):
        m = Float32MultiArray()
        m.data = [float(left), float(right)]
        self.motor_pub.publish(m)

    def stop(self):
        for _ in range(3):
            self._publish(0.0, 0.0)
            time.sleep(0.05)

    def run_for(self, left, right, duration):
        """Publish at 50 Hz (the motor node stops if commands pause), then stop."""
        t_end = time.time() + duration
        while time.time() < t_end and rclpy.ok():
            self._publish(left, right)
            time.sleep(0.02)
        self.stop()

    def imu_ok(self):
        return self.p['use_imu'] and self.imu_yaw is not None

    def turn(self, angle, cmd=None):
        """Turn in place by angle (rad, + = left/CCW). Returns measured change."""
        cmd = cmd if cmd is not None else self.p['turn_cmd']
        left, right = (-cmd, cmd) if angle > 0 else (cmd, -cmd)
        if self.imu_ok():
            yaw0 = self.imu_yaw
            target = max(0.0, abs(angle) - self.p['turn_lead'])
            # generous timeout in case the IMU stops updating
            timeout = abs(angle) / self.p['turn_rate'] * 2.0 + 1.0
            t_end = time.time() + timeout
            while abs(self.imu_yaw - yaw0) < target and time.time() < t_end and rclpy.ok():
                self._publish(left, right)
                time.sleep(0.02)
            self.stop()
            time.sleep(self.p['settle_time'])
            turned = self.imu_yaw - yaw0
        else:
            rate = self.p['turn_rate_left' if angle > 0 else 'turn_rate_right'] \
                or self.p['turn_rate']
            self.run_for(left, right, abs(angle) / rate)
            time.sleep(self.p['settle_time'])
            turned = angle
        x, y, th = self.pose
        self.pose = (x, y, wrap(th + turned))
        self.angle_turned += abs(turned)
        return turned

    def drive(self, duration, backward=False):
        """Drive straight for duration seconds and dead-reckon the result."""
        duration = max(duration, self.p['min_step_time'])
        sgn = -1.0 if backward else 1.0
        yaw0 = self.imu_yaw if self.imu_ok() else None
        self.run_for(sgn * self.p['drive_left'], sgn * self.p['drive_right'], duration)
        time.sleep(self.p['settle_time'])
        dth = (self.imu_yaw - yaw0) if yaw0 is not None else 0.0
        d = sgn * self.p['drive_speed'] * duration
        x, y, th = self.pose
        mid = th + dth / 2.0             # assume heading changed evenly
        self.pose = (x + d * math.cos(mid), y + d * math.sin(mid), wrap(th + dth))
        self.dist_traveled += abs(d)

    # ---------------- sensing ----------------
    def fresh_frame(self, after, timeout=1.5):
        """Return a decoded BGR frame received after time `after`."""
        t_end = time.time() + timeout
        while time.time() < t_end and rclpy.ok():
            f = self._frame
            if f is not None and f[0] > after:
                img = cv2.imdecode(np.frombuffer(f[1], np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            time.sleep(0.02)
        return None

    def look(self, theta_pred='auto', tries=2):
        """Take up to `tries` fresh frames; correct the pose if a tag is seen."""
        if theta_pred == 'auto':
            theta_pred = self.pose[2]
        t_req = time.time()
        for _ in range(tries):
            img = self.fresh_frame(after=t_req)
            if img is None:
                self.get_logger().warn('No camera frame received')
                break
            t_req = time.time()
            dets = self.detector.detect(img, self.K, self.D)
            # trust our position (and get heading from the tag's bearing)
            # unless we've gone a while without seeing a tag
            pos_pred = (self.pose[:2] if theta_pred is not None
                        and self.blind_steps < self.p['accept_jump_after'] else None)
            meas, used = localize(dets, self.tag_map, self.cam_offset, theta_pred,
                                  theta_trust_dist=self.p['theta_trust_dist'],
                                  pos_pred=pos_pred)
            if meas is not None and theta_pred is not None and len(used) == 1:
                jump = math.hypot(meas[0] - self.pose[0], meas[1] - self.pose[1])
                if jump > self.p['max_jump'] and self.blind_steps < self.p['accept_jump_after']:
                    self.get_logger().warn(
                        f'  rejected tag {used[0][0]} fix ({meas[0]:+.2f}, {meas[1]:+.2f}): '
                        f'{jump:.2f} m from where we think we are')
                    meas = None
            if meas is not None:
                if theta_pred is None:
                    self.pose = meas                  # no prior: take it
                else:
                    a = self.p['alpha_pos']
                    self.pose = blend(self.pose, meas, a, 1.0)   # theta already blended
                tags = ', '.join(f'{i}@{d:.2f}m' for i, d, _ in used)
                self.get_logger().info(
                    f'  tags [{tags}] -> pose ({self.pose[0]:+.2f}, {self.pose[1]:+.2f}, '
                    f'{math.degrees(self.pose[2]):+.0f} deg)')
                self.blind_steps = 0
                self.record(source=1, n_tags=len(used))
                return True
            if dets:
                ids = [d['id'] for d in dets]
                unmapped = sorted(set(i for i in ids if i not in self.tag_map))
                if unmapped:
                    self.get_logger().warn(f'  saw tag(s) {unmapped} that are NOT in tag_ids '
                                           f'{sorted(self.tag_map)}; check the printed IDs')
                else:
                    self.get_logger().info(f'  saw tags {ids} but none usable (too far / bad fit)')
        self.blind_steps += 1
        self.get_logger().info(
            f'  no tag; odometry pose ({self.pose[0]:+.2f}, {self.pose[1]:+.2f}, '
            f'{math.degrees(self.pose[2]):+.0f} deg)')
        self.record(source=0, n_tags=0)
        return False

    def search(self):
        """Rotate clockwise in small steps until a tag is seen (max one turn)."""
        self.get_logger().warn('Lost: searching clockwise for a tag')
        turned = 0.0
        while turned < 2 * math.pi and rclpy.ok():
            turned += abs(self.turn(-self.p['search_step'], cmd=self.p['search_cmd']))
            if self.look():
                self.get_logger().info(f'Found a tag after {math.degrees(turned):.0f} deg')
                return True
        return False

    def record(self, source, n_tags):
        x, y, th = self.pose
        self.traj.append([time.time() - self.t0, x, y, th, source, n_tags, self.wp_index])

    # ---------------- control ----------------
    def step_toward(self, wp):
        """Do one motion toward waypoint wp. Returns True when wp is reached."""
        x, y, th = self.pose
        dx, dy = wp[0] - x, wp[1] - y
        dist = math.hypot(dx, dy)
        p = self.p
        along = dx * math.cos(th) + dy * math.sin(th)    # how far ahead the waypoint is
        sideways_only = dist <= p['reach_radius'] and abs(along) <= p['pos_tol']
        if dist > p['pos_tol'] and not sideways_only:
            err = wrap(math.atan2(dy, dx) - th)
            if dist < 0.35 and abs(err) > 2.0:
                # just overshot: back up instead of spinning around
                along = -(dx * math.cos(th) + dy * math.sin(th))
                self.get_logger().info(f'  back up {along:.2f} m')
                self.drive(min(p['step_time'], along / p['drive_speed']), backward=True)
            elif (abs(err) > p['heading_tol']
                  and not (abs(err) < math.pi / 2
                           and dist * abs(math.sin(err)) < 0.8 * p['pos_tol'])):
                # (close in, a small sideways miss is fine: skip the turn and
                # just drive, so we don't waste rotation on tiny corrections)
                self.get_logger().info(f'  turn {math.degrees(err):+.0f} deg to face waypoint')
                self.turn(err)
            else:
                ahead = dist * math.cos(err)            # distance along our heading
                t = min(p['step_time'], ahead / p['drive_speed'])
                self.get_logger().info(f'  drive {t * p["drive_speed"]:.2f} m ({dist:.2f} m left)')
                self.drive(t)
            return False
        err = wrap(wp[2] - th)
        if abs(err) <= p['ang_tol']:
            return True
        self.get_logger().info(f'  at waypoint; turn {math.degrees(err):+.0f} deg to final heading')
        self.turn(err)
        return False

    def wait_for_inputs(self, timeout=15.0):
        t_end = time.time() + timeout
        while time.time() < t_end and rclpy.ok():
            if self.K is not None and self._frame is not None:
                break
            time.sleep(0.1)
        if self.K is None:
            raise RuntimeError('No camera_info. Is the camera running with a calibration YAML?')
        if self._frame is None:
            raise RuntimeError('No images on the camera topic.')
        time.sleep(0.5)
        if self.p['use_imu'] and self.imu_yaw is None:
            self.get_logger().warn('No IMU data; turns will be timed instead of measured')

    def run(self):
        self.wait_for_inputs()
        self.get_logger().info('Initial look...')
        self.record(source=2, n_tags=0)               # start pose as given
        self.look()
        for i, wp in enumerate(self.waypoints[1:], start=1):
            self.wp_index = i
            self.get_logger().info(f'=== Waypoint {i}: {wp} ===')
            reached = False
            for _ in range(self.p['max_steps_per_waypoint']):
                if not rclpy.ok():
                    return
                if self.step_toward(wp):
                    reached = True
                    break
                if not self.look() and self.blind_steps > self.p['max_blind_steps']:
                    if not self.search():
                        self.get_logger().error('No tag found after a full turn; stopping')
                        return
            x, y, th = self.pose
            e_pos = math.hypot(wp[0] - x, wp[1] - y)
            e_th = wrap(wp[2] - th)
            self.wp_results.append([i, *wp, x, y, th, e_pos, e_th, time.time() - self.t0])
            self.get_logger().info(
                f'Waypoint {i} {"reached" if reached else "GAVE UP"}: est pose '
                f'({x:+.3f}, {y:+.3f}, {math.degrees(th):+.1f} deg), '
                f'est error {e_pos * 100:.1f} cm / {math.degrees(e_th):+.1f} deg')
            time.sleep(1.0)    # pause so the video shows the robot on the marker
        self.get_logger().info(
            f'Done. Distance driven {self.dist_traveled:.2f} m, '
            f'rotation {self.angle_turned:.2f} rad ({math.degrees(self.angle_turned):.0f} deg)')

    def save_logs(self):
        os.makedirs(self.p['log_dir'], exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        traj_path = os.path.join(self.p['log_dir'], f'trajectory_{stamp}.csv')
        with open(traj_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x', 'y', 'theta', 'source', 'n_tags', 'waypoint'])
            w.writerows(self.traj)
        wp_path = os.path.join(self.p['log_dir'], f'waypoints_{stamp}.csv')
        with open(wp_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['waypoint', 'goal_x', 'goal_y', 'goal_theta', 'est_x', 'est_y',
                        'est_theta', 'est_pos_err', 'est_theta_err', 't'])
            w.writerows(self.wp_results)
            w.writerow([])
            w.writerow(['distance_m', self.dist_traveled, 'rotation_rad', self.angle_turned])
        self.get_logger().info(f'Saved {traj_path} and {wp_path}')
        return traj_path


def main(args=None):
    rclpy.init(args=args)
    node = Hw2Navigator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    except Exception as e:
        node.get_logger().error(f'Stopped: {e}')
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.save_logs()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
