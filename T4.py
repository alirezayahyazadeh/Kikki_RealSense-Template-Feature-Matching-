import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # ----- Setup -----
    # Path to the template image (update this path as needed)
    template_path = r"D:\template.jpg"
    
    # Load the template image in grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("Error: Template image not found at", template_path)
        return
    h, w = template.shape[:2]
    
    # ----- Initialize Intel RealSense Camera -----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        pipeline.start(config)
        print("Press 'q' to quit.")
        while True:
            # Wait for the next set of frames from the camera
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            
            # Convert the depth frame to a numpy array (note: depth data is 16-bit)
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Normalize depth image to 8-bit (0-255) for visualization and matching
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)
            
            # Apply a Gaussian blur to reduce noise
            depth_gray = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
            
            # ----- Template Matching -----
            result = cv2.matchTemplate(depth_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Determine the best match location (using max_loc) and draw a rectangle
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            # For visualization, convert the 8-bit depth image to a color map
            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.rectangle(depth_color, top_left, bottom_right, (0, 255, 0), 2)
            
            # Display the result
            cv2.imshow("Template Matching on Depth Frame", depth_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error:", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
