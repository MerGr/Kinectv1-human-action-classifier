import os

filename = "./a11/a11_s03_e01_realworld.txt"
output_filename = "a11_s03_e01_realworld.txt"

# Frames to remove (1-based counting as per your request)
start_frame = 129
end_frame = 280


# Each frame has 15 lines of joints
lines_per_skeleton = 15

# Line indices to exclude (0-based)
# Start line: (22-1) * 15 = 315
# End line: (104 * 15) - 1 = 1559
start_line = (start_frame - 1) * lines_per_skeleton
end_line = (end_frame * lines_per_skeleton) - 1

if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
else:
    with open(filename, 'r') as f:
        all_lines = f.readlines()

    # Filter out lines within the forbidden range
    kept_lines = [
        line for idx, line in enumerate(all_lines) 
        if not (start_line <= idx <= end_line)
    ]

    with open(output_filename, 'w') as f:
        f.writelines(kept_lines)

    removed_count = (end_frame - start_frame + 1)
    print(f"Done! Removed {removed_count} frames ({removed_count * 15} lines).")
    print(f"New file saved as: {output_filename}")
