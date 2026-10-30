#!/usr/bin/env python3
#!/usr/bin/env python3
"""
CSE 276A HW1: drive the waypoint path.

Waypoints (x m, y m, theta rad):
  0,0,0 -> 1,0,0 -> 1,1,1.57 -> 2,1,0 -> 2,2,-1.57 -> 1,1,-0.78 -> 0,0,0

Each leg has its own hard-coded moves, so you can tune every move
separately. Each move is (L, R, seconds):
  L, R    = left and right wheel commands (positive = forward)
  seconds = how long to run before stopping

Starting values come from testing:
  L=0.33, R=0.42  drives straight (time per meter still needs measuring)
  L=-0.2, R=0.2   for 2 s = 90 deg left
  L=0.2,  R=-0.2  for 2 s = 90 deg right
"""

import time
import serial

PORT = '/dev/ttyUSB0'
BAUD = 115200
PAUSE = 0.5   # settle time after each move


# ---------- Leg 1: (0, 0, 0) -> (1, 0, 0) ----------
LEG_1 = [
    (0.31, 0.40, 2.0),     # forward 1 m
]

# ---------- Leg 2: (1, 0, 0) -> (1, 1, 1.57) ----------
LEG_2 = [
    (-0.20, 0.21, 2.0),      # turn left 90 deg
    (0.31, 0.40, 2.0),     # forward 1 m
]

# ---------- Leg 3: (1, 1, 1.57) -> (2, 1, 0) ----------
LEG_3 = [
    (0.20, -0.21, 2.0),      # turn right 90 deg
    (0.32, 0.41, 2.0),     # forward 1 m
]

# ---------- Leg 4: (2, 1, 0) -> (2, 2, -1.57) ----------
LEG_4 = [
    (-0.20, 0.23, 2.0),      # turn left 90 deg
    (0.32, 0.41, 2.0),     # forward 1 m
    (0.22, -0.24, 4.0),      # turn right 180 deg (face -90 deg)
]

# ---------- Leg 5: (2, 2, -1.57) -> (1, 1, -0.78) ----------
LEG_5 = [
    (0.20, -0.21, 1.0),      # turn right 45 deg (face the diagonal)
    (0.32, 0.41, 2.83),    # forward 1.414 m
    (-0.21, 0.21, 2.0),      # turn left 90 deg (face -45 deg)
]

# ---------- Leg 6: (1, 1, -0.78) -> (0, 0, 0) ----------
LEG_6 = [
    (0.2, -0.21, 1.0),      # turn right 90 deg (face the diagonal)
    (0.32, 0.41, 2.83),    # forward 1.414 m
    (-0.2, 0.21, 3.0),      # turn left 135 deg (face 0 deg)
]

LEGS = [LEG_1, LEG_2, LEG_3, LEG_4, LEG_5, LEG_6]


def send(s, left, right):
    s.write(('{"T":1,"L":%s,"R":%s}\n' % (left, right)).encode())


def run_leg(s, leg):
    for left, right, seconds in leg:
        send(s, left, right)
        time.sleep(seconds)
        send(s, 0, 0)
        time.sleep(PAUSE)


def main():
    s = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)

    try:
        for i, leg in enumerate(LEGS, start=1):
            print(f"Leg {i}")
            run_leg(s, leg)
        print("Done!")
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        send(s, 0, 0)
        s.close()


if __name__ == '__main__':
    main()