import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from rembg import remove

def main():
    # ============== 1) Load Template ==============
    template_path = r"D:\template.jpg"  # <-- CHANGE to your template path
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template image from {template_path}")
        return
    t_h, t_w = template.shape[:2]

    # ============== 2) Initialize RealSense (COLOR) ==============
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        print("Starting RealSense stream. Press 'q' to quit.")
        while True:
            # ============== 3) Capture Color Frame ==============
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to BGR NumPy array
            color_image = np.asanyarray(color_frame.get_data())

            # ============== 4) (Optional) Remove Background with rembg ==============
            # a) Convert BGR -> RGB for rembg
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # b) Convert to PIL
            pil_image = Image.fromarray(color_image_rgb)
            # c) Remove background (returns RGBA PIL image)
            out_pil = remove(pil_image)
            # d) Convert RGBA -> BGR for OpenCV
            out_np = np.array(out_pil)  # shape: (H, W, 4)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGBA2BGR)

            # ============== 5) Template Matching ==============
            # Convert background-removed frame to grayscale
            out_gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)

            # Run matchTemplate
            result = cv2.matchTemplate(out_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Print out the max_val for debugging
            print(f"max_val = {max_val:.3f}")

            # ============== 6) Draw a Heatmap for Debugging (Optional) ==============
            # Normalize result to [0..255] so we can see it
            heatmap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # ============== 7) If the match is good enough, draw bounding box ==============
            # You might set a threshold like 0.5 or 0.6
            threshold = 0.5
            if max_val > threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
                cv2.rectangle(out_bgr, top_left, bottom_right, (0, 255, 0), 2)
            else:
                # If below threshold, we won't draw anything
                pass

            # ============== 8) Show Windows ==============
            cv2.imshow("Background Removed + Template Matching", out_bgr)
            cv2.imshow("Match Heatmap", heatmap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error:", e)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
