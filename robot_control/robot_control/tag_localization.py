"""
Pure-math helpers for HW2: detect AprilTags with OpenCV, get each tag's pose
relative to the camera with solvePnP, and turn that into a robot pose in the
world frame using the known tag map.

No ROS imports here, so this file can be tested on a laptop.

Frames
------
World:  x, y in meters on the floor; theta = yaw, CCW from +x.
Robot:  origin at the robot center; forward = +x_r, left = +y_r.
Camera (OpenCV): x right, y down, z forward (out of the lens).
Tag:    origin at the tag center; x right, y up, z out of the printed face
        (toward whoever is looking at the tag).
A tag's map yaw is the direction its printed face points in the world.
"""
import math

import cv2
import numpy as np


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class TagDetector:
    """AprilTag 36h11 detection + per-tag pose with cv2.aruco and solvePnP."""

    def __init__(self, tag_size, family='36h11'):
        self.tag_size = tag_size
        dict_id = {
            '36h11': cv2.aruco.DICT_APRILTAG_36h11,
            '25h9': cv2.aruco.DICT_APRILTAG_25h9,
            '16h5': cv2.aruco.DICT_APRILTAG_16h5,
        }[family]
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        # OpenCV >= 4.7 has ArucoDetector; older versions use free functions.
        if hasattr(cv2.aruco, 'ArucoDetector'):
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(dictionary, params)
            self._detect = lambda g: self._detector.detectMarkers(g)
        else:
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detect = lambda g: cv2.aruco.detectMarkers(
                g, dictionary, parameters=params)

        # Corner order required by SOLVEPNP_IPPE_SQUARE, which matches the
        # order aruco returns: top-left, top-right, bottom-right, bottom-left.
        s = tag_size / 2.0
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0],
                                 [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    def _reproj_err(self, rvec, tvec, img_pts, K, D):
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, K, D)
        return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))

    def detect(self, image_bgr, K, D):
        """
        Returns a list of dicts, one per detected tag:
          id, corners (4x2 px), dist (m),
          candidates: list of (rvec, tvec, reproj_err_px), best first.

        A small square seen roughly head-on has two poses that fit the
        corners almost equally well (they differ by tilting the tag left or
        right). Both are kept so the caller can pick the one that agrees
        with the robot's predicted heading.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.flatten()):
            img_pts = c.reshape(4, 2).astype(np.float64)
            cands = []
            try:
                n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    self.obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                for r, t in zip(rvecs, tvecs):
                    cands.append((r.ravel(), t.ravel(),
                                  self._reproj_err(r, t, img_pts, K, D)))
            except cv2.error:
                pass
            # IPPE can break down for an exactly frontal view; fall back to
            # the iterative solver when nothing fits well.
            if not cands or min(e for _, _, e in cands) > 2.0:
                ok, r, t = cv2.solvePnP(self.obj_pts, img_pts, K, D,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    cands.append((r.ravel(), t.ravel(),
                                  self._reproj_err(r, t, img_pts, K, D)))
            cands = [c for c in cands if c[1][2] > 0]   # tag must be in front
            if not cands:
                continue
            cands.sort(key=lambda c: c[2])
            out.append({
                'id': int(i),
                'corners': img_pts,
                'dist': float(np.linalg.norm(cands[0][1])),
                'candidates': cands,
            })
        return out


def tag_in_robot_frame(rvec, tvec, cam_offset=(0.0, 0.0)):
    """
    Project a camera-frame tag pose onto the floor plane, in the robot frame.

    Returns (fx, ly, phi):
      fx, ly : tag center position, meters forward / left of the robot center
      phi    : yaw of the tag's face normal in the robot frame
               (pi means the tag faces straight back at the robot)
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).ravel()
    # camera (x right, z forward) -> robot (forward, left)
    fx = t[2] + cam_offset[0]
    ly = -t[0] + cam_offset[1]
    n = R[:, 2]                       # tag face normal, in camera frame
    phi = math.atan2(-n[0], n[2])     # same axis swap as above
    return fx, ly, phi


def robot_pose_from_tag(tag_world, rel):
    """
    tag_world: (x_t, y_t, psi_t) tag center and face yaw in the world
    rel:       (fx, ly, phi) from tag_in_robot_frame
    Returns the robot's world pose (x, y, theta).

    The tag face yaw seen from the robot (phi) plus the robot yaw must equal
    the tag's world yaw, so theta = psi_t - phi. Then undo the rotation to
    get the robot position from the tag position.
    """
    x_t, y_t, psi_t = tag_world
    fx, ly, phi = rel
    return position_from_tag(tag_world, rel, wrap(psi_t - phi))


def position_from_tag(tag_world, rel, theta):
    """
    Robot pose given a heading theta and the tag's position in the robot
    frame (range + bearing). This only uses where the tag center is, which
    is measured much more reliably than which way the tag is tilted.
    """
    x_t, y_t, _ = tag_world
    fx, ly, _ = rel
    c, s = math.cos(theta), math.sin(theta)
    x = x_t - (c * fx - s * ly)
    y = y_t - (s * fx + c * ly)
    return x, y, theta


def fuse_estimates(estimates, weights):
    """Weighted mean of several (x, y, theta), with a circular mean for theta."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    e = np.asarray(estimates, dtype=np.float64)
    x = float(np.dot(w, e[:, 0]))
    y = float(np.dot(w, e[:, 1]))
    theta = math.atan2(float(np.dot(w, np.sin(e[:, 2]))),
                       float(np.dot(w, np.cos(e[:, 2]))))
    return x, y, theta


def localize(detections, tag_map, cam_offset=(0.0, 0.0), theta_pred=None,
             max_reproj_err=3.0, max_dist=4.0, theta_trust_dist=1.0, pos_pred=None):
    """
    Turn detections into one robot pose estimate.

    detections: output of TagDetector.detect
    tag_map:    {id: (x, y, yaw)}
    theta_pred: predicted robot heading (rad). None means trust the tag fully.
    pos_pred:   predicted robot (x, y), if it is trustworthy (e.g. we just
                turned in place after a good fix). Then the heading comes from
                the tag's BEARING: we know where we are and where the tag is,
                so the pixel column of the tag tells us which way we face.
                That is accurate to a degree or two at any distance and does
                not depend on the (ambiguous) tilt of the tag, so it catches
                turns that came up short or long.
    theta_trust_dist: how far away (m) a tag's own tilt is still trusted for
                heading. Tilt is good up close, poor at ~2 m, so its weight
                falls off with distance.
    Returns (pose, used) where pose is (x, y, theta) or None, and used is a
    list of (id, dist, single-tag pose) for logging.
    """
    ests, weights, used = [], [], []
    for d in detections:
        if d['id'] not in tag_map or d['dist'] > max_dist:
            continue
        best_err = d['candidates'][0][2]
        if best_err > max_reproj_err:
            continue
        ok = [c for c in d['candidates'] if c[2] <= max(1.0, 3.0 * best_err)]
        tag = tag_map[d['id']]
        rels = [tag_in_robot_frame(r, t, cam_offset) for r, t, _ in ok]
        poses = [robot_pose_from_tag(tag, rel) for rel in rels]

        # reference heading: from the bearing if we trust our position,
        # otherwise the prediction
        ref = theta_pred
        if pos_pred is not None:
            fx, ly, _ = rels[0]
            to_tag = math.atan2(tag[1] - pos_pred[1], tag[0] - pos_pred[0])
            ref = wrap(to_tag - math.atan2(ly, fx))

        if ref is None:
            pose = poses[0]
        else:
            # pick the tilt solution that agrees with the reference, then
            # blend headings, trusting the tilt less the farther away it is
            k = min(range(len(poses)), key=lambda j: abs(wrap(poses[j][2] - ref)))
            w = 1.0 / (1.0 + (d['dist'] / theta_trust_dist) ** 2)
            theta = wrap(ref + w * wrap(poses[k][2] - ref))
            pose = position_from_tag(tag, rels[k], theta)
        ests.append(pose)
        weights.append(1.0 / max(d['dist'], 0.2) ** 2)   # closer = more trusted
        used.append((d['id'], d['dist'], pose))
    if not ests:
        return None, used
    return fuse_estimates(ests, weights), used


def blend(pred, meas, alpha_pos, alpha_theta):
    """Complementary filter: move the prediction toward the measurement."""
    x = pred[0] + alpha_pos * (meas[0] - pred[0])
    y = pred[1] + alpha_pos * (meas[1] - pred[1])
    theta = wrap(pred[2] + alpha_theta * wrap(meas[2] - pred[2]))
    return x, y, theta

