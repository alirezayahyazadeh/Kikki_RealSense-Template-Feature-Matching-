import cv2
import numpy as np
import os
import pyrealsense2 as rs

# Set template image path
image_path = r"D:\template.jpg"  # Use raw string (r"") to avoid escape issues

# Check if the template file exists
if not os.path.exists(image_path):
    print(f"Error: File {image_path} not found.")
    exit()

# Load the template image
template = cv2.imread(image_path, 0)  # Load in grayscale
if template is None:
    print(f"Error: Failed to read {image_path}. Check if it's a valid image file.")
    exit()

h, w = template.shape[:2]  # Get template dimensions

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    while True:
        # Get frames from RealSense camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.ascontiguousarray(depth_frame.get_data())

        # Normalize depth image for better visibility
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Convert depth image to color map for visualization
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Perform template matching
        res = cv2.matchTemplate(depth_normalized, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Draw rectangle around detected object
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(depth_color, top_left, bottom_right, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Detected', depth_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
