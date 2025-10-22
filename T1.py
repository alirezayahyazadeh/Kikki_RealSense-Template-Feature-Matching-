import cv2
import numpy as np
import pyrealsense2 as rs

# Load the template image
template_path = r'D:\template.jpg'
template = cv2.imread(template_path, 0)

# Check if the template was loaded correctly
if template is None:
    raise FileNotFoundError(f"Error: Template image not found at {template_path}")

# Convert template to uint8 to match depth image format
template = cv2.convertScaleAbs(template)
w, h = template.shape[::-1]

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.ascontiguousarray(depth_frame.get_data(), dtype=np.uint8)

        # Convert depth image to displayable format
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Convert to grayscale for template matching
        depth_gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(depth_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Draw rectangle around matched region
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
