import os
import numpy as np
import joblib
from collections import deque
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='daal4py')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_PATH  = "../pretrained_models"
MODEL_NAME  = "rf"
SCALER_NAME = "data_scaler"

CONFIDENCE_THRESHOLD = 0.4
WINDOW_SIZE          = 60

# Bone pairs (source_idx, target_idx)
BONE_PAIRS = [
    (0, 1), (1, 2), (1, 5), (2, 3), (3, 4),
    (5, 6), (6, 7), (8, 9), (8, 12), (9, 10),
    (10, 11), (12, 13), (13, 14)
]

# Discriminative joint pairs
JOINT_PAIRS = [
    (4, 7), (4, 0), (7, 0), (11, 14), (9, 12)
]

ACTION_LABELS = {
    1: "Wave Horizontal", 
    2: "Wave Up", 
    3: "Wave Both Hands",
    4: "Bend", 
    5: "Clap", 
    6: "Walk",
    7: "Call", 
    8: "Drink", 
    9: "Sit",
    10: "Stand up", 
    11: "Standing",
}

# ---------------------------------------------------------------------------
# MODEL & SCALER
# ---------------------------------------------------------------------------

def _load_joblib(path, label):
    """Load model/scaler with error handling."""
    if not os.path.exists(path):
        print(f"[{label}] not found at {path}")
        return None
    obj = joblib.load(path)
    print(f"[{label}] loaded ({type(obj).__name__})")
    return obj

model  = _load_joblib(os.path.join(MODEL_PATH, f"{MODEL_NAME}.joblib"),  "Model")
scaler = _load_joblib(os.path.join(MODEL_PATH, f"{SCALER_NAME}.joblib"), "Scaler")

# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_features(buffer):
    """Extract 161 features from skeleton buffer."""
    if len(buffer) < WINDOW_SIZE:
        return None

    raw = np.array(list(buffer)[-WINDOW_SIZE:], dtype=np.float32)
    num_frames = raw.shape[0]
    frames = raw.reshape(num_frames, 15, 3)

    # Torso-centering + scale normalisation
    frames = frames - frames[:, 8:9, :]
    ref_len = float(np.mean(np.linalg.norm(frames[:, 1, :], axis=1))) + 1e-8
    frames = frames / ref_len

    # Velocity
    velocity = np.linalg.norm(np.diff(frames, axis=0), axis=2)

    # Directional velocity
    vel_xyz = np.diff(frames, axis=0)
    mean_vel_x = np.mean(vel_xyz[:, :, 0], axis=0)
    mean_vel_y = np.mean(vel_xyz[:, :, 1], axis=0)
    mean_vel_z = np.mean(vel_xyz[:, :, 2], axis=0)

    # Distance to torso
    dist_to_torso = np.linalg.norm(frames, axis=2)

    # Bone modality
    bone_lengths = np.stack([
        np.linalg.norm(frames[:, j, :] - frames[:, i, :], axis=1)
        for i, j in BONE_PAIRS
    ], axis=1)
    bone_vel = np.abs(np.diff(bone_lengths, axis=0))

    # Pairwise joint distances
    pair_dists = np.stack([
        np.linalg.norm(frames[:, j, :] - frames[:, i, :], axis=1)
        for i, j in JOINT_PAIRS
    ], axis=1)

    # Aggregation
    if velocity.shape[0] > 0:
        feature_vector = np.concatenate([
            np.mean(dist_to_torso, axis=0),
            np.std(dist_to_torso, axis=0),
            np.mean(velocity, axis=0),
            np.max(velocity, axis=0),
            np.sqrt(np.mean(velocity ** 2, axis=0)),
            mean_vel_x, mean_vel_y, mean_vel_z,
            np.mean(bone_lengths, axis=0),
            np.mean(bone_vel, axis=0),
            np.mean(pair_dists, axis=0),
            np.std(pair_dists, axis=0),
            pair_dists.max(axis=0) - pair_dists.min(axis=0),
        ])
    else:
        feature_vector = np.concatenate([
            np.mean(dist_to_torso, axis=0),
            np.std(dist_to_torso, axis=0),
            np.zeros(15), np.zeros(15), np.zeros(15),
            np.zeros(15), np.zeros(15), np.zeros(15),
            np.mean(bone_lengths, axis=0),
            np.zeros(13),
            np.mean(pair_dists, axis=0),
            np.std(pair_dists, axis=0),
            np.zeros(5),
        ])

    return feature_vector.astype(np.float32)

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """Averages probability distributions over a rolling window."""
    def __init__(self, size=5):
        self._buf = deque(maxlen=size)
        self._classes = None

    def update(self, proba_vec, classes):
        self._buf.append(proba_vec)
        self._classes = classes

    def get_sorted(self):
        if not self._buf or self._classes is None:
            return []
        avg = np.mean(self._buf, axis=0)
        order = np.argsort(avg)[::-1]
        return [(self._classes[i], float(avg[i])) for i in order]

    def clear(self):
        self._buf.clear()
        self._classes = None

def predict(buffer, smoother=None):
    """Run prediction on buffer. Returns (label, confidence, all_confidences)."""
    if model is None or scaler is None:
        return None, 0.0, {}

    feat = extract_features(buffer)
    if feat is None:
        return None, 0.0, {}

    feat = scaler.transform(feat.reshape(1, -1))
    proba = model.predict_proba(feat)[0]
    classes = np.array([ACTION_LABELS.get(int(c), str(c)) for c in model.classes_])

    if smoother:
        smoother.update(proba, classes)
        results = smoother.get_sorted()
        if results and results[0][1] >= CONFIDENCE_THRESHOLD:
            label, conf = results[0]
        else:
            label, conf = 'Inconnue', 0.0
    else:
        top_idx = np.argmax(proba)
        label = classes[top_idx]
        conf = float(proba[top_idx])


    conf_dict = {}
    for i, cls in enumerate(classes):
        conf_dict[cls] = float(proba[i])

    return label, conf, conf_dict
