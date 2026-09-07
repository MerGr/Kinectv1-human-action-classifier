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
import freenect

# --- CONFIGURATION ---
OUTPUT_FILE = "capture_output.txt"

KARD_JOINTS = [
    nite2.JointType.NITE_JOINT_HEAD, nite2.JointType.NITE_JOINT_NECK,
    nite2.JointType.NITE_JOINT_RIGHT_SHOULDER, nite2.JointType.NITE_JOINT_RIGHT_ELBOW, nite2.JointType.NITE_JOINT_RIGHT_HAND,
    nite2.JointType.NITE_JOINT_LEFT_SHOULDER, nite2.JointType.NITE_JOINT_LEFT_ELBOW, nite2.JointType.NITE_JOINT_LEFT_HAND,
    nite2.JointType.NITE_JOINT_TORSO,
    nite2.JointType.NITE_JOINT_RIGHT_HIP, nite2.JointType.NITE_JOINT_RIGHT_KNEE, nite2.JointType.NITE_JOINT_RIGHT_FOOT,
    nite2.JointType.NITE_JOINT_LEFT_HIP, nite2.JointType.NITE_JOINT_LEFT_KNEE, nite2.JointType.NITE_JOINT_LEFT_FOOT
]

# --- INITIALIZATION ---
openni2.initialize()
nite2.initialize()
dev = openni2.Device.open_any()

# 1. ALIGN DEPTH TO RGB (Hardware Level)

color_stream = dev.create_color_stream()
color_stream.start()
user_tracker = nite2.UserTracker(dev)

def reinit_openni_nite():
    global dev, color_stream, user_tracker
    sleep(3)
    try:
        openni2.initialize()
        nite2.initialize()
        dev = openni2.Device.open_any()
        color_stream = dev.create_color_stream()
        color_stream.start()
        user_tracker = nite2.UserTracker(dev)
        print("[Device] Successfully reinitialized OpenNI/NiTE2")
    except Exception as e:
        print(f"[Device] Error reinitializing: {e}")
        sleep(2)
        openni2.initialize()
        nite2.initialize()
        dev = openni2.Device.open_any()
        color_stream = dev.create_color_stream()
        color_stream.start()
        user_tracker = nite2.UserTracker(dev)

# --- STATE VARIABLES ---
recording = False
frame_id = 0
video_writer = None
video_fps = 30
video_frame_count = 0

def get_accel_from_freenect():
    global dev, color_stream, user_tracker
    nite2.unload() 
    openni2.unload()
    
    ctx = freenect.init()
    free_dev = freenect.open_device(ctx, 0)
    accel = freenect.get_accel(free_dev)
    freenect.close_device(free_dev)
    freenect.shutdown(ctx)
    
    reinit_openni_nite()
    return accel

def save_frame_bundle(active_user, depth, rgb, f_id):
    # KARD uses 15 joints. Format: [x,y,z];[x,y,z]....
    kard_skel = []
    for jt in KARD_JOINTS:
        j = active_user.skeleton.joints[jt]
        sx, sy = user_tracker.convert_joint_coordinates_to_depth(j.position.x, j.position.y, j.position.z)
        kard_skel.append(f"[{sx},{sy},{j.position.z}]")

    # Save skeleton data to output file
    skeleton_line = f"frame_{f_id:04d} " + ";".join(kard_skel) + "\n"
    with open(OUTPUT_FILE, "a") as f:
        f.write(skeleton_line)

def start_video_recording():
    global video_writer, video_frame_count
    video_path = OUTPUT_FILE.replace(".txt", ".mp4")

    # MP4 codec (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (640, 480))
    video_frame_count = 0
    print(f"[Video] Started recording to {video_path}")

def stop_video_recording():
    global video_writer, video_frame_count
    if video_writer:
        video_writer.release()
        video_writer = None
        print(f"[Video] Stopped recording ({video_frame_count} frames)")

def write_video_frame(rgb):
    global video_writer, video_frame_count
    if video_writer:
        video_writer.write(rgb)
        video_frame_count += 1

    

print("READY. [SPACE]: Single Frame | [V]: Toggle Video recording | [Q]: Quit")

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
          (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0)]


accel = get_accel_from_freenect()
accel_file = OUTPUT_FILE.replace(".txt", "_accelerometer.txt")
with open(accel_file, "w") as f:
    f.write(f"Accelerometer (G's): X={accel[0]:.3f}, Y={accel[1]:.3f}, Z={accel[2]:.3f}\n")

while True:
    frame = user_tracker.read_frame()
    color_frame = color_stream.read_frame()
    
    # Process Images
    rgb = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((480, 640, 3))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = np.frombuffer(frame.get_depth_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape((480, 640))
    
    # Alignment Preview
    preview = cv2.applyColorMap((depth/4500.0*255).astype(np.uint8), cv2.COLORMAP_JET)
    
    active_user = None
    for user in frame.users:
        if user.is_new(): user_tracker.start_skeleton_tracking(user.id)
        elif user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED:
            active_user = user

    # Create independent skeleton view (not part of saved data)
    skeleton_rgb = rgb.copy()
    if active_user:
        for joint_type in KARD_JOINTS:
            j = active_user.skeleton.joints[joint_type]
            x, y = user_tracker.convert_joint_coordinates_to_depth(j.position.x, j.position.y, j.position.z)
            cv2.circle(skeleton_rgb, (int(x), int(y)), 3, COLORS[KARD_JOINTS.index(joint_type)], -1)
            
            #HEAD-NECK
            j1 = active_user.skeleton.joints[KARD_JOINTS[0]]
            j2 = active_user.skeleton.joints[KARD_JOINTS[1]]
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            #NECK-RIGHT SHOULDER
            j1 = active_user.skeleton.joints[KARD_JOINTS[1]]
            j2 = active_user.skeleton.joints[KARD_JOINTS[2]]
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            #NECK-LEFT SHOULDER
            j1 = active_user.skeleton.joints[KARD_JOINTS[1]]
            j2 = active_user.skeleton.joints[KARD_JOINTS[5]]
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # RIGHT SHOULDER-ELBOW-HAND
            for i in range(2, 4):
                j1 = active_user.skeleton.joints[KARD_JOINTS[i]]
                j2 = active_user.skeleton.joints[KARD_JOINTS[i+1]]
                x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
                x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
                cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # LEFT SHOULDER-ELBOW-HAND
            for i in range(5, 7):
                j1 = active_user.skeleton.joints[KARD_JOINTS[i]]
                j2 = active_user.skeleton.joints[KARD_JOINTS[i+1]]
                x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
                x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
                cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # RIGHT SHOULDER-TORSO
            j1 = active_user.skeleton.joints[KARD_JOINTS[2]]
            j2 = active_user.skeleton.joints[KARD_JOINTS[8]]
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # LEFT SHOULDER-TORSO
            j1 = active_user.skeleton.joints[KARD_JOINTS[5]]
            j2 = active_user.skeleton.joints[KARD_JOINTS[8]]
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # TORSO-HIPS
            j1 = active_user.skeleton.joints[KARD_JOINTS[8]]
            j2 = active_user.skeleton.joints[KARD_JOINTS[9]] # RIGHT HIP
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            j2 = active_user.skeleton.joints[KARD_JOINTS[12]]
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # HIPS
            j1 = active_user.skeleton.joints[KARD_JOINTS[9]] # RIGHT HIP
            j2 = active_user.skeleton.joints[KARD_JOINTS[12]] # LEFT HIP
            x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
            x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
            cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


            # HIPS-LEGS
            for i in [9, 12]: # RIGHT HIP, LEFT HIP
                j1 = active_user.skeleton.joints[KARD_JOINTS[i]]
                j2 = active_user.skeleton.joints[KARD_JOINTS[i+1]] # KNEE
                x1, y1 = user_tracker.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
                x2, y2 = user_tracker.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)
                cv2.line(skeleton_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                j3 = active_user.skeleton.joints[KARD_JOINTS[i+2]] # FOOT
                x3, y3 = user_tracker.convert_joint_coordinates_to_depth(j3.position.x, j3.position.y, j3.position.z)
                cv2.line(skeleton_rgb, (int(x2), int(y2)), (int(x3), int(y3)), (0, 255, 0), 2)
    cv2.imshow("Skeleton View", skeleton_rgb)

    # UI Feedback
    # if recording: cv2.putText(rgb, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("RGB (Aligned)", rgb)
    cv2.imshow("Depth", preview)

    # INPUT HANDLING
    key = cv2.waitKey(1)
    if keyboard.is_pressed('space') and active_user:
        save_frame_bundle(active_user, depth, rgb, frame_id)
        frame_id += 1
        print(f"Captured Frame {frame_id}")
        sleep(0.2)

    if keyboard.is_pressed('v'):
        recording = not recording
        if recording:
            start_video_recording()
        else:
            stop_video_recording()
        print(f"Recording: {recording}")
        sleep(0.5)

    if recording and active_user:
        save_frame_bundle(active_user, depth, rgb, frame_id)
        write_video_frame(rgb)
        frame_id += 1

    if key == ord('q') or keyboard.is_pressed('q'): break

# Cleanup
if recording:
    stop_video_recording()

nite2.unload(); openni2.unload()

cv2.destroyAllWindows()
