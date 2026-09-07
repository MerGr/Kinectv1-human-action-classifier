import os

os.environ["NITE2_REDIST"] = r"C:\Program Files\PrimeSense\NiTE2\Redist"
os.add_dll_directory(os.environ["NITE2_REDIST"])
os.add_dll_directory(r"C:\libfreenect\lib\OpenNI2-FreenectDriver")
os.add_dll_directory(r"C:\libfreenect\lib")

import numpy as np
import cv2
import keyboard
from time import sleep
from openni import openni2, nite2
from collections import deque
import warnings
import sys

# Add parent directory to path to import inference_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_engine import (
    model, scaler, WINDOW_SIZE, CONFIDENCE_THRESHOLD,
    ACTION_LABELS, predict, PredictionSmoother
)

# Suppress sklearn feature names warning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MULTI_USER = False

KARD_JOINTS = [
    nite2.JointType.NITE_JOINT_HEAD,
    nite2.JointType.NITE_JOINT_NECK,
    nite2.JointType.NITE_JOINT_RIGHT_SHOULDER,
    nite2.JointType.NITE_JOINT_RIGHT_ELBOW,
    nite2.JointType.NITE_JOINT_RIGHT_HAND,
    nite2.JointType.NITE_JOINT_LEFT_SHOULDER,
    nite2.JointType.NITE_JOINT_LEFT_ELBOW,
    nite2.JointType.NITE_JOINT_LEFT_HAND,
    nite2.JointType.NITE_JOINT_TORSO,
    nite2.JointType.NITE_JOINT_RIGHT_HIP,
    nite2.JointType.NITE_JOINT_RIGHT_KNEE,
    nite2.JointType.NITE_JOINT_RIGHT_FOOT,
    nite2.JointType.NITE_JOINT_LEFT_HIP,
    nite2.JointType.NITE_JOINT_LEFT_KNEE,
    nite2.JointType.NITE_JOINT_LEFT_FOOT,
]

SKELETON_BONES = [
    (0, 1), (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (2, 8), (5, 8),
    (8, 9), (8, 12),
    (9, 12),
    (9, 10), (10, 11),
    (12, 13), (13, 14),
]

JOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
]

DISPLAY_OFFSET = (0, 14)

# ---------------------------------------------------------------------------
# HARDWARE INIT
# ---------------------------------------------------------------------------

def _init_hardware():
    global dev, color_stream, user_tracker
    openni2.initialize()
    nite2.initialize()
    dev = openni2.Device.open_any()
    color_stream = dev.create_color_stream()
    color_stream.start()
    user_tracker = nite2.UserTracker(dev)
    print("[Init] hardware ready")

def _reinit_hardware():
    sleep(3)
    for attempt in range(1, 3):
        try:
            _init_hardware()
            print(f"[Reinit] recovered on attempt {attempt}")
            return
        except Exception as e:
            print(f"[Reinit] attempt {attempt} failed: {e}")
            sleep(2)

_init_hardware()

# ---------------------------------------------------------------------------
# SKELETON DRAWING
# ---------------------------------------------------------------------------

def _joint_px(user, joint_idx, offset):
    j = user.skeleton.joints[KARD_JOINTS[joint_idx]]
    x, y = user_tracker.convert_joint_coordinates_to_depth(
        j.position.x, j.position.y, j.position.z
    )
    ox, oy = offset
    return int(x) + ox, int(y) + oy


def draw_skeleton(canvas, user, offset):
    try:
        pts = [_joint_px(user, i, offset) for i in range(len(KARD_JOINTS))]
    except Exception:
        return
    for i, pt in enumerate(pts):
        cv2.circle(canvas, pt, 4, JOINT_COLORS[i], -1)
    for a, b in SKELETON_BONES:
        cv2.line(canvas, pts[a], pts[b], (0, 220, 0), 2)


def draw_prediction(canvas, label, conf, pos=(10, 30)):
    if label is None:
        return

    # Determine color based on confidence
    if conf >= 0.7:
        color = (0, 255, 0)
    elif conf >= 0.4:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(canvas, f"{label}", pos,
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(canvas, f"{conf:.1%}", (pos[0], pos[1] + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

# ---------------------------------------------------------------------------
# PER-USER STATE
# ---------------------------------------------------------------------------

user_buffers = {}
user_smoothers = {}
user_labels = {}

inference_enabled = True

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

print("[Ready]  SPACE=capture | I=toggle inference | Q=quit")
print(f"[Ready]  multi-user={'on' if MULTI_USER else 'off'}")
print(f"[Ready]  features=161  |  window={WINDOW_SIZE} frames")

while True:
    try:
        frame = user_tracker.read_frame()
        color_frame = color_stream.read_frame()
    except Exception as e:
        print(f"[Capture] read error: {e} — reinitialising")
        _reinit_hardware()
        continue

    rgb = np.frombuffer(color_frame.get_buffer_as_uint8(),
                          dtype=np.uint8).reshape((480, 640, 3))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    canvas = rgb.copy()

    tracked_ids = set()

    for user in frame.users:
        uid = user.id

        if user.is_new():
            user_tracker.start_skeleton_tracking(uid)

        elif user.is_lost():
            user_buffers.pop(uid, None)
            user_smoothers.pop(uid, None)
            user_labels.pop(uid, None)
            continue

        if user.skeleton.state != nite2.SkeletonState.NITE_SKELETON_TRACKED:
            continue

        if not MULTI_USER and tracked_ids:
            continue

        tracked_ids.add(uid)
        user_tracker.set_skeleton_smoothing_factor(0.0)

        # Buffer skeleton frame
        try:
            joints = user.skeleton.joints
            frame_arr = np.array([
                [joints[jt].position.x,
                 joints[jt].position.y,
                 joints[jt].position.z]
                for jt in KARD_JOINTS
            ], dtype=np.float32)
        except Exception:
            continue

        if uid not in user_buffers:
            user_buffers[uid] = deque(maxlen=WINDOW_SIZE)
            user_smoothers[uid] = PredictionSmoother(size=15)
            user_labels[uid] = (None, 0.0)

        if inference_enabled:
            user_buffers[uid].append(frame_arr)
            if len(user_buffers[uid]) == WINDOW_SIZE:
                label, conf, _ = predict(
                    user_buffers[uid],
                    smoother=user_smoothers[uid]
                )
                user_labels[uid] = (label, conf)

        # Draw skeleton
        draw_skeleton(canvas, user, DISPLAY_OFFSET)

        label, conf = user_labels[uid]
        y_pos = 30 + (uid - 1) * 80
        draw_prediction(canvas, label, conf, pos=(10, y_pos))

        if MULTI_USER:
            cv2.putText(canvas, f"U{uid}", (10, y_pos - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Buffer fill indicator
    primary_id = min(tracked_ids) if tracked_ids else None
    if primary_id and primary_id in user_buffers:
        fill = len(user_buffers[primary_id])
        cv2.putText(canvas, f"buf {fill}/{WINDOW_SIZE}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if not inference_enabled:
        cv2.putText(canvas, "INFERENCE OFF", (440, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    cv2.namedWindow("Action Recognition", cv2.WINDOW_NORMAL)
    cv2.imshow("Action Recognition", canvas)

    key = cv2.waitKey(1)
    if key == ord('q') or keyboard.is_pressed('q'):
        break
    if key == ord('i'):
        inference_enabled = not inference_enabled
        print(f"[Inference] {'enabled' if inference_enabled else 'disabled'}")

# ---------------------------------------------------------------------------
# SHUTDOWN
# ---------------------------------------------------------------------------

nite2.unload()
openni2.unload()
cv2.destroyAllWindows()
