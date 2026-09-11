import streamlit as st
import cv2
import numpy as np
import threading
import time
#import os
import sys
from collections import deque

# Add parent directory to path
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_engine import (
    model, scaler, WINDOW_SIZE, CONFIDENCE_THRESHOLD,
    ACTION_LABELS, predict, PredictionSmoother
)

# ---------------------------------------------------------------------------
# KINECT INFERENCE THREAD
# ---------------------------------------------------------------------------

class KinectInferenceThread(threading.Thread):
    """Background thread for Kinect inference."""
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.running = False
        self.show_skeleton = True  # Default to showing skeleton
        self._init_kinect()

    def _init_kinect(self):
        """Initialize Kinect hardware."""
        try:
            from openni import openni2, nite2

            # Set up DLL paths
#            os.environ["NITE2_REDIST"] = r"C:\Program Files\PrimeSense\NiTE2\Redist"
#            os.add_dll_directory(os.environ["NITE2_REDIST"])
#            os.add_dll_directory(r"C:\libfreenect\lib\OpenNI2-FreenectDriver")
#            os.add_dll_directory(r"C:\libfreenect\lib")

            openni2.initialize()
            nite2.initialize()
            self.dev = openni2.Device.open_any()
            self.color_stream = self.dev.create_color_stream()
            self.color_stream.start()
            self.user_tracker = nite2.UserTracker(self.dev)

            # KARD joint definitions
            self.KARD_JOINTS = [
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

            self.SKELETON_BONES = [
                (0, 1), (1, 2), (1, 5), (2, 3), (3, 4),
                (5, 6), (6, 7), (2, 8), (5, 8), (8, 9),
                (8, 12), (9, 12), (9, 10), (10, 11),
                (12, 13), (13, 14),
            ]

            self.JOINT_COLORS = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            ]

            self.DISPLAY_OFFSET = (0, 14)
            self.buffer = deque(maxlen=WINDOW_SIZE)
            self.smoother = PredictionSmoother(size=15)
            self.initialized = True
            print("[Kinect] Initialized successfully")

        except Exception as e:
            print(f"[Kinect] Initialization failed: {e}")
            self.initialized = False

    def _joint_px(self, user, joint_idx, offset):
        j = user.skeleton.joints[self.KARD_JOINTS[joint_idx]]
        x, y = self.user_tracker.convert_joint_coordinates_to_depth(
            j.position.x, j.position.y, j.position.z
        )
        ox, oy = offset
        return int(x) + ox, int(y) + oy

    def _draw_skeleton(self, canvas, user, offset):
        try:
            pts = [self._joint_px(user, i, offset) for i in range(len(self.KARD_JOINTS))]
        except Exception:
            return
        for i, pt in enumerate(pts):
            cv2.circle(canvas, pt, 4, self.JOINT_COLORS[i], -1)
        for a, b in self.SKELETON_BONES:
            cv2.line(canvas, pts[a], pts[b], (0, 220, 0), 2)

    def run(self):
        """Main inference loop."""
        if not self.initialized:
            return

        self.running = True
        print("[Kinect] Starting inference loop")

        while self.running:
            try:
                frame = self.user_tracker.read_frame()
                color_frame = self.color_stream.read_frame()
            except Exception as e:
                print(f"[Kinect] Capture error: {e}")
                time.sleep(0.1)
                continue

            # Get RGB frame
            rgb = np.frombuffer(color_frame.get_buffer_as_uint8(),
                              dtype=np.uint8).reshape((480, 640, 3))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            canvas = rgb.copy()

            # Process users
            user_detected = False
            for user in frame.users:
                if user.is_new():
                    self.user_tracker.start_skeleton_tracking(user.id)
                elif user.is_lost():
                    self.buffer.clear()
                    self.smoother.clear()
                    continue

                if user.skeleton.state != 2:  # NITE_SKELETON_TRACKED
                    continue

                # Extract skeleton
                try:
                    joints = user.skeleton.joints
                    frame_arr = np.array([
                        [joints[jt].position.x,
                         joints[jt].position.y,
                         joints[jt].position.z]
                        for jt in self.KARD_JOINTS
                    ], dtype=np.float32)
                except Exception:
                    continue

                user_detected = True

                # Update buffer and predict
                self.buffer.append(frame_arr)
                if len(self.buffer) == WINDOW_SIZE:
                    label, conf, all_conf = predict(self.buffer, self.smoother)

                    # Draw skeleton if enabled
                    if self.show_skeleton:
                        self._draw_skeleton(canvas, user, self.DISPLAY_OFFSET)

                    # Update shared state
                    self.state.update(canvas, label, conf, all_conf)
                    break  # Only process first user

            # Update state even if no user detected
            if not user_detected:
                self.state.update(canvas, None, 0.0, {})

            # Small sleep to prevent CPU overload
            time.sleep(0.01)

    def stop(self):
        """Stop the inference thread."""
        self.running = False
        if hasattr(self, 'user_tracker'):
            from openni import nite2, openni2
            nite2.unload()
            openni2.unload()
        print("[Kinect] Stopped")

# ════════════════════════════════════════════════════════════════
# ██  SHARED STATE FOR REAL-TIME INFERENCE  ██
# ════════════════════════════════════════════════════════════════

class InferenceState:
    """Shared state for real-time inference."""
    def __init__(self):
        self.current_frame = None
        self.current_label = None
        self.current_conf = 0.0
        self.all_confidences = {}
        self.lock = threading.Lock()

    def update(self, frame, label, conf, all_conf):
        with self.lock:
            self.current_frame = frame
            self.current_label = label
            self.current_conf = conf
            self.all_confidences = all_conf

    def get_state(self):
        with self.lock:
            return (
                self.current_frame,
                self.current_label,
                self.current_conf,
                self.all_confidences.copy()
            )

# Global state and thread
inference_state = InferenceState()
inference_thread = None

# Start inference on module load
if model is not None and scaler is not None:
    try:
        inference_thread = KinectInferenceThread(inference_state)
        inference_thread.start()
        print("[App] Kinect inference started")
    except Exception as e:
        print(f"[App] Failed to start Kinect: {e}")
        inference_thread = None

# ════════════════════════════════════════════════════════════════
# ██  PUBLIC API  ██
# ════════════════════════════════════════════════════════════════

def get_current_frame():
    """Get the current video frame from inference."""
    frame, _, _, _ = inference_state.get_state()
    return frame

def get_confidences():
    """Get current confidence scores for all classes."""
    _, _, _, conf_dict = inference_state.get_state()

    default_confidences = {
        "Wave Horizontal": 0.0,
        "Wave Up": 0.0,
        "Wave Both Hands": 0.0,
        "Bend": 0.0,
        "Clap": 0.0,
        "Walk": 0.0,
        "Call": 0.0,
        "Drink": 0.0,
        "Sit": 0.0,
        "Standing up": 0.0,
        "Stand": 0.0,
        "Unknown": 1.0,
    }

    if conf_dict:
        default_confidences.update(conf_dict)
        # Set "Unknown" to 0 if we have predictions
        if any(v > 0 for k, v in conf_dict.items() if k != "Unknown"):
            default_confidences["Unknown"] = 0.0

    return default_confidences

def is_inference_running():
    """Check if inference is currently running."""
    return inference_thread is not None and inference_thread.is_alive()


def render():
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-dot"></div>
        <div>
            <div class="brand-tag">Live inference</div>
            <div class="brand-sub">Realtime action recognition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show inference status
    running = is_inference_running()
    status_color = "#28a745" if running else "#dc3545"  # Green for running, red for stopped
    status_text = "● RUNNING" if running else "● STOPPED"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px">
        <span style="color:{status_color};font-size:14px;font-weight:bold">{status_text}</span>
        <span style="color:#666;font-size:12px">
            {'Kinect active' if running else 'Error'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_feed, col_conf = st.columns([1.55, 1], gap="large")

    # ── Left : video feed ─────────────────────────────────────
    with col_feed:
        st.markdown('<div class="sec-title">RGB camera feed</div>', unsafe_allow_html=True)

        # Skeleton toggle at the top of the feed
        if 'show_skeleton' not in st.session_state:
            st.session_state.show_skeleton = True

        show_skeleton = st.toggle(
            "Show skeleton",
            value=st.session_state.show_skeleton,
            key="skeleton_toggle"
        )

        # Update inference thread setting
        if inference_thread and inference_thread.is_alive():
            inference_thread.show_skeleton = show_skeleton

        feed_placeholder = st.empty()

        # ── Right : confidence panel ──────────────────────────────
    with col_conf:
        st.markdown('<div class="sec-title">Predicted label</div>', unsafe_allow_html=True)

        pred_placeholder = st.empty()
        conf_placeholder = st.empty()

    # Continuous update loop
    while True:
        # Get current frame from inference
        video_frame = get_current_frame()

        if video_frame is not None:
            # Convert BGR→RGB if needed
            if video_frame.ndim == 3 and video_frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = video_frame
            feed_placeholder.image(frame_rgb, width="stretch", channels="RGB")
        else:
            # Only show placeholder if Kinect is not connected at all
            status_msg = "initializing Kinect..." if running else "Kinect not connected!"
            feed_placeholder.markdown(f"""
            <div class="live-frame">
                <div class="live-badge">
                    <div class="live-indicator" style="background-color:{status_color}"></div>
                    {status_msg}
                </div>
                <div style="text-align:center">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:12px;color:#2a2a2a;
                                letter-spacing:0.08em">
                        {'Error' if not running else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Get current confidences
        class_confidences = get_confidences()

        # Sort confidences DESC
        sorted_conf = sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)
        top_class, top_conf = sorted_conf[0]

        # Current prediction
        pred_placeholder.markdown(f"""
        <div class="panel" style="margin-bottom:18px">
            <div class="panel-title">Current prediction</div>
            <div class="pred-label">{top_class}</div>
            <div class="pred-conf">confidence level {top_conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar chart
        conf_placeholder.markdown('<div class="sec-title">Confidence scores</div>', unsafe_allow_html=True)

        bars_html = ""
        for i, (label, conf) in enumerate(sorted_conf):
            is_top = i == 0
            fill_class = "top" if is_top else ""
            short = label if len(label) <= 22 else label[:21] + "…"
            bars_html += f"""
            <div class="conf-row">
                <div class="conf-label" title="{label}">{short}</div>
                <div class="conf-track">
                    <div class="conf-fill {fill_class}" style="width:{conf*100:.1f}%"></div>
                </div>
                <div class="conf-pct">{conf:.0%}</div>
            </div>"""
        conf_placeholder.markdown(bars_html, unsafe_allow_html=True)

        # Small delay to prevent excessive CPU usage
        time.sleep(0.05)
