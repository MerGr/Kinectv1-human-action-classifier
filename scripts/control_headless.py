import os
os.add_dll_directory(r"C:\libfreenect\lib")

import freenect
import numpy as np

ctx = freenect.init()
dev = freenect.open_device(ctx, 0)

current_tilt = 0
led_mode = 1  #Green

print("""
--- KINECT FEATURE TEST ---
'w' / 's' : Tilt Up/Down
'l'       : Cycle LED Colors
'a'       : Read Accelerometer
'd'       : Default Tilt/LED
'q'       : Quit
---------------------------
""")

try:
    while True:

        key = input("Command (w/s/l/a/q): ").strip().lower()

        if key == 'q':
            break

        # --- TILT ---
        elif key == 'w':
            current_tilt = min(30, current_tilt + 5)
            freenect.set_tilt_degs(dev, current_tilt)
            print(f"[Motor] Tilt: {current_tilt}°")

        elif key == 's':
            current_tilt = max(-30, current_tilt - 5)
            freenect.set_tilt_degs(dev, current_tilt)
            print(f"[Motor] Tilt: {current_tilt}°")

        # --- LED ---
        elif key == 'l':
            led_mode = (led_mode + 1) % 7
            freenect.set_led(dev, led_mode)
            modes = ["Off", "Green", "Red", "Yellow", "Blink Green", "Blink Red/Yellow", "Blink Yellow"]
            print(f"[LED] Mode {led_mode}: {modes[led_mode]}")

        # --- ACCELEROMETER ---
        elif key == 'a':
            freenect.update_tilt_state(dev)
            state = freenect.get_tilt_state(dev)
            x, y, z = freenect.get_mks_accel(state)
            angle = freenect.get_tilt_degs(state)
            print(f"[Accel] X:{x:.2f} Y:{y:.2f} Z:{z:.2f} | Current Angle: {angle:.1f}°")
        elif key == 'd':
            freenect.set_led(dev, 1)
            freenect.set_tilt_degs(dev, 0)

finally:
    freenect.close_device(dev)
    freenect.shutdown(ctx)