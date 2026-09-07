import numpy as np
import cv2
import os

# --- CONFIGURATION ---
INPUT_FILE = "inference.txt"

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (2, 8), (5, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (9, 12)
]

def visualize_manual():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    data = np.loadtxt(INPUT_FILE)
    num_frames = len(data) // 15
    current_frame = 0
    
    scale = 1.5 
    win_w, win_h = 800, 600

    print("Use [D] for Next Frame")
    print("Use [A] for Previous Frame")
    print("Press [Q] to Quit")

    while True:
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        
        # Get current frame data
        skeleton = data[current_frame*15 : (current_frame+1)*15]
        center_x, center_y = np.mean(skeleton[:, 0]), np.mean(skeleton[:, 1])

        projected_points = []
        for x, y, z in skeleton:
            px = int((x - center_x) * scale + (win_w / 2))
            py = int((y - center_y) * scale + (win_h / 2))
            projected_points.append((px, py))

        # Draw Skeleton
        for start_jt, end_jt in SKELETON_CONNECTIONS:
            cv2.line(canvas, projected_points[start_jt], projected_points[end_jt], (0, 255, 0), 2)
        for px, py in projected_points:
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)

        # UI Text
        cv2.putText(canvas, f"Frame: {current_frame + 1}/{num_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Manual Mode: Use A/D to Step", (10, 580),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Manual Skeleton Viewer", canvas)

        # Handle Key Inputs
        key = cv2.waitKeyEx(0) # 0 means wait indefinitely for a key press

        if key == ord('d'):
            current_frame = min(current_frame + 1, num_frames - 1)
        elif  key == ord('a'):
            current_frame = max(current_frame - 1, 0)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_manual()
