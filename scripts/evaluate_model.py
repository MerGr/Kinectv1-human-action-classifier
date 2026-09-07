"""
Model Evaluation Script
Evaluates a trained model with inference.txt data using bone length,
directional velocity, and pairwise distance features (161 dimensions)
"""

import numpy as np
import os
import joblib
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_FILE  = "inference.txt"
MODEL_PATH  = "./models"
MODEL_NAME  = "rf_best_model"
SCALER_NAME = "data_scaler"

WINDOW_SIZE = 30
CONFIDENCE_THRESHOLD = 0.0

TORSO_IDX = 8
NECK_IDX  = 1

ACTION_LABELS = {
    1: "Wave Horizontal", 2: "Wave Up", 3: "Wave Both Hands",
    4: "Bend", 5: "Clap", 6: "Walk", 7: "Call",
    8: "Drink", 9: "Sit Down", 10: "Stand Up", 11: "Standing"
}

# Reverse mapping for evaluation
LABEL_TO_ID = {v: k for k, v in ACTION_LABELS.items()}

# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

# Bone pairs for bone length features
BONE_PAIRS = [
    (0, 1),   # Head-Neck
    (1, 2),   # Neck-RShoulder
    (1, 5),   # Neck-LShoulder
    (2, 3),   # RShoulder-RElbow
    (3, 4),   # RElbow-RHand
    (5, 6),   # LShoulder-LElbow
    (6, 7),   # LElbow-LHand
    (8, 9),   # Torso-RHip
    (8, 12),  # Torso-LHip
    (9, 10),  # RHip-RKnee
    (10, 11), # RKnee-RFoot
    (12, 13), # LHip-LKnee
    (13, 14)  # LKnee-LFoot
]

# Key pairwise distances
PAIRS = [
    (4, 7),   # RHand-LHand
    (4, 0),   # RHand-Head
    (7, 0),   # LHand-Head
    (11, 14), # RFoot-LFoot
    (9, 12)   # RHip-LHip
]

def extract_features_from_buffer(buffer, window_size=90):
    """
    Extract features from buffer using the new transformation logic.
    Returns a 161-dimensional feature vector.
    """
    if len(buffer) < window_size:
        return None

    # Convert buffer to numpy array and reshape to (T, 15, 3)
    raw = np.array(list(buffer)[-window_size:], dtype=np.float32)
    num_frames = raw.shape[0]
    frames = raw.reshape(num_frames, 15, 3)

    # Step 1: Torso centering
    frames = frames - frames[:, 8:9, :]

    # Step 2: Scale-invariant coordinates (normalize by neck-to-torso distance)
    ref_len = np.mean(np.linalg.norm(frames[:, 1, :], axis=1)) + 1e-8
    frames = frames / ref_len

    # Step 3: Velocity (norm of diff)
    velocity = np.linalg.norm(np.diff(frames, axis=0), axis=2)  # shape (T-1, 15)

    # Step 4: Directional velocity (mean vel_x, vel_y, vel_z for each joint)
    vel_xyz = np.diff(frames, axis=0)  # shape (T-1, 15, 3)
    mean_vel_x = np.mean(vel_xyz[:, :, 0], axis=0)  # 15 values — horizontal
    mean_vel_y = np.mean(vel_xyz[:, :, 1], axis=0)  # 15 values — vertical (key for sit/stand)
    mean_vel_z = np.mean(vel_xyz[:, :, 2], axis=0)  # 15 values — depth

    # Step 5: Distance to torso for all 15 joints
    dist_to_torso = np.linalg.norm(frames[:, :, :], axis=2)  # shape (T, 15)

    # Step 6: Bone lengths and bone velocity
    bone_lengths = np.array([
        np.linalg.norm(frames[:, j, :] - frames[:, i, :], axis=1)
        for i, j in BONE_PAIRS
    ]).T  # shape (T, 13)
    bone_vel = np.abs(np.diff(bone_lengths, axis=0))  # shape (T-1, 13)

    # Step 7: Pairwise distances
    pair_dists = np.array([
        np.linalg.norm(frames[:, j, :] - frames[:, i, :], axis=1)
        for i, j in PAIRS
    ]).T  # shape (T, 5)

    # Step 8: Concatenate all features
    if velocity.shape[0] > 0:
        feature_vector = np.concatenate([
            np.mean(dist_to_torso, axis=0),      # 15 values
            np.std(dist_to_torso, axis=0),       # 15 values
            np.mean(velocity, axis=0),           # 15 values
            np.max(velocity, axis=0),            # 15 values
            np.sqrt(np.mean(velocity**2, axis=0)), # 15 values (RMS)
            mean_vel_x,                          # 15 values
            mean_vel_y,                          # 15 values
            mean_vel_z,                          # 15 values
            np.mean(bone_lengths, axis=0),       # 13 values (avg bone length)
            np.mean(bone_vel, axis=0),           # 13 values (avg bone motion)
            np.mean(pair_dists, axis=0),         # 5 values
            np.std(pair_dists, axis=0),          # 5 values
            pair_dists.max(axis=0) - pair_dists.min(axis=0),  # 5 values (range)
        ])
    else:
        # Fallback if no velocity data
        feature_vector = np.concatenate([
            np.mean(dist_to_torso, axis=0),      # 15 values
            np.std(dist_to_torso, axis=0),       # 15 values
            np.zeros(15),                         # mean_vel (no data)
            np.zeros(15),                         # max_vel (no data)
            np.zeros(15),                         # rms_vel (no data)
            np.zeros(15),                         # mean_vel_x (no data)
            np.zeros(15),                         # mean_vel_y (no data)
            np.zeros(15),                         # mean_vel_z (no data)
            np.mean(bone_lengths, axis=0),       # 13 values
            np.zeros(13),                         # bone_vel (no data)
            np.mean(pair_dists, axis=0),         # 5 values
            np.std(pair_dists, axis=0),          # 5 values
            pair_dists.max(axis=0) - pair_dists.min(axis=0),  # 5 values
        ])

    return feature_vector.astype(np.float32)


def extract_features_improved_reduced(buffer, window_size=90):
    """Alias for compatibility - uses the new feature extraction."""
    return extract_features_from_buffer(buffer, window_size)


def extract_features_activity_specific(buffer, window_size=90):
    """Alias for compatibility - uses the new feature extraction."""
    return extract_features_from_buffer(buffer, window_size)

# ---------------------------------------------------------------------------
# GROUND TRUTH MAPPING
# ---------------------------------------------------------------------------

def create_ground_truth_mapping():
    ground_truth = {}

    # Define intervals (start_frame, end_frame, activity_name)
    intervals = [
        (0, 330, "Walk"),
        (337, 423, "Wave Both Hands"),
        (423, 493, "Wave Up"),  # right hand
        (493, 586, "Wave Up"),  # left hand
        (586, 668, "Wave Up"),  # right hand
        (674, 774, "Wave Horizontal"),  # right hand
        (774, 853, "Wave Horizontal"),  # left hand
        (853, 1005, "Walk"),
        (1014, 1087, "Bend"),
        (1107, 1152, "Bend"),
        (1323, 1371, "Clap"),
        (1444, 1573, "Clap"),
        (1598, 1716, "Call"),  # right hand
        (1739, 1881, "Call"),  # left hand
        (1895, 2080, "Drink"),  # right hand
        (2089, 2249, "Drink"),  # right hand
        (2253, 2366, "Drink"),  # right hand
        (2370, 2474, "Drink"),  # left hand
        (2480, 2590, "Drink"),  # left hand
        (2720, 2788, "Bend"),
        (2804, 2932, "Walk"),
        (3138, 3210, "Walk"),
        (3236, 3409, "Sit Down"),
        (3410, 3522, "Stand Up"),
        (3567, 3652, "Sit Down"),
        (3652, 3692, "Wave Both Hands"),  # while sitting
        (3692, 3749, "Sit Down"),
        (3750, 3813, "Stand Up"),
        (3843, 3939, "Walk"),
        (3950, 4032, "Wave Up"),  # right hand
        (4044, 4116, "Wave Up"),  # left hand
        (4116, 4219, "Wave Both Hands"),
        (4219, 4248, "Wave Horizontal"),  # right hand
        (4250, 4290, "Wave Horizontal"),  # left hand
        (4300, 4430, "Walk"),
        (4430, 4500, "Bend"),
        (4505, 4567, "Bend"),
        (4579, 4691, "Sit Down"),
        (4692, 4768, "Stand Up"),
        (4858, 4910, "Walk"),
    ]

    # First pass: mark all intervals
    for start, end, activity in intervals:
        for frame in range(start, end + 1):
            ground_truth[frame] = activity

    # Second pass: handle gaps (default to Standing)
    max_frame = max(intervals, key=lambda x: x[1])[1]
    for frame in range(max_frame + 1):
        if frame not in ground_truth:
            ground_truth[frame] = "Standing"

    # Third pass: handle sit down -> stand up transitions
    # If between sit down and stand up, mark as sit down
    sit_down_intervals = [(3236, 3409), (3567, 3652), (3692, 3749), (4579, 4691)]
    stand_up_intervals = [(3410, 3522), (3750, 3813), (4692, 4768)]

    for sit_start, sit_end in sit_down_intervals:
        for stand_start, stand_end in stand_up_intervals:
            # If stand up starts after sit down ends
            if stand_start > sit_end:
                # Mark frames between them as sit down
                for frame in range(sit_end + 1, stand_start):
                    if frame in ground_truth:
                        ground_truth[frame] = "Sit Down"

    return ground_truth

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_model_and_scaler():
    """Load the trained model and scaler."""
    model_path = os.path.join(MODEL_PATH, f"{MODEL_NAME}.joblib")
    scaler_path = os.path.join(MODEL_PATH, f"{SCALER_NAME}.joblib")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"Loaded model: {MODEL_NAME}")
    print(f"Loaded scaler: {SCALER_NAME}")
    print(f"Model expects {scaler.n_features_in_} features")

    return model, scaler

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """Averages probability distributions over a rolling window."""
    def __init__(self, size=15):
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

def predict_frame(buffer, model, scaler, smoother, use_improved_features=True):
    """Predict activity for current frame using improved features."""
    if model is None or scaler is None:
        return None, 0.0

    # Use improved features
    if use_improved_features:
        feat = extract_features_activity_specific(buffer, WINDOW_SIZE)
    else:
        # Fallback to original features
        feat = extract_features_improved_reduced(buffer, WINDOW_SIZE)

    if feat is None:
        return None, 0.0

    # Handle feature dimensionality mismatch
    if feat.shape[0] != scaler.n_features_in_:
        print(f"Warning: Feature dimension mismatch. Expected {scaler.n_features_in_}, got {feat.shape[0]}")
        # Try to pad or truncate
        if feat.shape[0] < scaler.n_features_in_:
            feat = np.pad(feat, (0, scaler.n_features_in_ - feat.shape[0]), 'constant')
        else:
            feat = feat[:scaler.n_features_in_]

    feat = scaler.transform(feat.reshape(1, -1))
    proba = model.predict_proba(feat)[0]
    classes = np.array([ACTION_LABELS.get(int(c), str(c)) for c in model.classes_])

    smoother.update(proba, classes)
    results = smoother.get_sorted()

    # Return top prediction regardless of confidence for evaluation
    if results:
        return results[0]
    return None, 0.0

# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODEL EVALUATION WITH IMPROVED FEATURES")
    print("=" * 60)

    # Load model
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print("Failed to load model. Exiting.")
        return

    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    data = np.loadtxt(INPUT_FILE)
    num_frames = len(data) // 15
    print(f"Total frames: {num_frames}")

    # Create ground truth mapping
    print("\nCreating ground truth mapping...")
    ground_truth = create_ground_truth_mapping()
    print(f"Ground truth labels for {len(ground_truth)} frames")

    # Initialize prediction storage
    predictions = {}
    confidences = {}
    buffer = deque(maxlen=WINDOW_SIZE)
    smoother = PredictionSmoother(size=15)

    # Process each frame
    print(f"\nProcessing frames (window size: {WINDOW_SIZE})...")
    print("Using new feature extraction with bone lengths, directional velocity, and pairwise distances...")
    print("This may take a while...")

    prediction_count = 0
    confidence_scores = []

    for frame_num in range(num_frames):
        # Get skeleton data for this frame
        start_idx = frame_num * 15
        end_idx = (frame_num + 1) * 15
        skeleton = data[start_idx:end_idx]

        # Add to buffer
        buffer.append(skeleton.copy())

        # Predict if buffer is full
        if len(buffer) == WINDOW_SIZE:
            label, conf = predict_frame(buffer, model, scaler, smoother, use_improved_features=True)
            if label is not None:
                predictions[frame_num] = label
                confidences[frame_num] = conf
                prediction_count += 1
                confidence_scores.append(conf)

        # Progress indicator
        if (frame_num + 1) % 500 == 0:
            avg_conf = np.mean(confidence_scores) if confidence_scores else 0.0
            print(f"Processed {frame_num + 1}/{num_frames} frames... (predictions: {prediction_count}, avg conf: {avg_conf:.3f})")

    print(f"\nCompleted! Made predictions for {len(predictions)} frames")

    if confidence_scores:
        print(f"\nConfidence Statistics:")
        print(f"  Average: {np.mean(confidence_scores):.3f}")
        print(f"  Min: {np.min(confidence_scores):.3f}")
        print(f"  Max: {np.max(confidence_scores):.3f}")
        print(f"  Std: {np.std(confidence_scores):.3f}")
        print(f"  Below 0.3: {sum(1 for c in confidence_scores if c < 0.3)} ({sum(1 for c in confidence_scores if c < 0.3)/len(confidence_scores)*100:.1f}%)")
        print(f"  Above 0.6: {sum(1 for c in confidence_scores if c >= 0.6)} ({sum(1 for c in confidence_scores if c >= 0.6)/len(confidence_scores)*100:.1f}%)")

    # Evaluate performance
    print("\n" + "=" * 60)
    print("PERFORMANCE EVALUATION")
    print("=" * 60)

    # Get frames where we have both ground truth and predictions
    eval_frames = []
    y_true = []
    y_pred = []

    for frame_num in range(min(len(ground_truth), num_frames)):
        if frame_num in predictions and frame_num in ground_truth:
            true_label = ground_truth[frame_num]
            pred_label = predictions[frame_num]

            # Convert to numeric IDs
            if true_label in LABEL_TO_ID and pred_label in LABEL_TO_ID:
                eval_frames.append(frame_num)
                y_true.append(LABEL_TO_ID[true_label])
                y_pred.append(LABEL_TO_ID[pred_label])

    print(f"Evaluation frames: {len(eval_frames)}")

    if len(eval_frames) == 0:
        print("No frames to evaluate!")
        return

    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Detailed classification report
    label_names = [ACTION_LABELS[i] for i in sorted(ACTION_LABELS)]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Bone Length + Directional Velocity Features')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Saved confusion matrix to: confusion_matrix.png")
    plt.close()

    # Analyze specific activities
    print("\n" + "=" * 60)
    print("ACTIVITY-SPECIFIC ANALYSIS")
    print("=" * 60)

    activity_performance = {}
    for activity_id in sorted(ACTION_LABELS.keys()):
        activity_name = ACTION_LABELS[activity_id]

        # Get frames for this activity
        true_indices = [i for i, t in enumerate(y_true) if t == activity_id]
        if not true_indices:
            continue

        # Get predictions for this activity
        pred_indices = [y_pred[i] for i in true_indices]
        correct = sum(1 for t, p in zip([y_true[i] for i in true_indices], pred_indices) if t == p)

        activity_performance[activity_name] = {
            'total': len(true_indices),
            'correct': correct,
            'accuracy': correct / len(true_indices) if len(true_indices) > 0 else 0
        }

        print(f"\n{activity_name}:")
        print(f"  Total frames: {len(true_indices)}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct / len(true_indices):.4f}")

        # Show common misclassifications
        if len(true_indices) > 0:
            misclassified = [(i, y_pred[i]) for i in true_indices if y_true[i] != y_pred[i]]
            if misclassified:
                print(f"  Common misclassifications:")
                from collections import Counter
                misclass_counts = Counter([ACTION_LABELS[p] for _, p in misclassified])
                for wrong_act, count in misclass_counts.most_common(3):
                    print(f"    - {wrong_act}: {count} times")

    # Analyze transition periods
    print("\n" + "=" * 60)
    print("TRANSITION ANALYSIS")
    print("=" * 60)

    # Find frames where ground truth changes
    transition_frames = []
    prev_activity = None
    for frame_num in sorted(ground_truth.keys()):
        current_activity = ground_truth[frame_num]
        if prev_activity is not None and current_activity != prev_activity:
            transition_frames.append((frame_num, prev_activity, current_activity))
        prev_activity = current_activity

    print(f"Found {len(transition_frames)} activity transitions")

    # Analyze performance around transitions
    transition_window = 10  # frames before and after transition
    transition_performance = []

    for trans_frame, from_act, to_act in transition_frames:
        # Check performance in transition window
        window_start = max(0, trans_frame - transition_window)
        window_end = min(num_frames, trans_frame + transition_window)

        window_correct = 0
        window_total = 0

        for frame in range(window_start, window_end):
            if frame in predictions and frame in ground_truth:
                true_label = ground_truth[frame]
                pred_label = predictions[frame]
                if true_label in LABEL_TO_ID and pred_label in LABEL_TO_ID:
                    if LABEL_TO_ID[true_label] == LABEL_TO_ID[pred_label]:
                        window_correct += 1
                    window_total += 1

        if window_total > 0:
            transition_performance.append({
                'frame': trans_frame,
                'from': from_act,
                'to': to_act,
                'accuracy': window_correct / window_total,
                'total': window_total
            })

    if transition_performance:
        print("\nPerformance around transitions:")
        for trans in transition_performance[:10]:  # Show first 10
            print(f"  Frame {trans['frame']}: {trans['from']} -> {trans['to']}")
            print(f"    Accuracy: {trans['accuracy']:.4f} ({trans['total']} frames)")

    # Save detailed results
    results = {
        'overall_accuracy': accuracy,
        'activity_performance': activity_performance,
        'transition_performance': transition_performance,
        'total_eval_frames': len(eval_frames),
        'total_frames': num_frames,
        'feature_type': 'bone_length_directional_velocity_pairwise'
    }

    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: evaluation_results.json")
    print(f"Confusion matrix saved to: confusion_matrix.png")

if __name__ == "__main__":
    main()
