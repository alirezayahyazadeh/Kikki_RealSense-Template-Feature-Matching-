import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # --------- 1) Load the Template Image ---------
    template_path = r"D:\template.jpg"  # <-- Change this to your template image path
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template image from {template_path}")
        return
    template_h, template_w = template.shape[:2]
    
    # --------- 2) Initialize ORB Detector and Compute Template Features ---------
    orb = cv2.ORB_create(nfeatures=1000)
    kp_template, des_template = orb.detectAndCompute(template, None)
    if des_template is None:
        print("Error: Could not compute features for the template.")
        return
    
    # --------- 3) Setup Intel RealSense for Color Streaming ---------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # BFMatcher for ORB (using Hamming distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    try:
        print("Starting RealSense stream. Press 'q' to quit.")
        while True:
            # --------- 4) Capture a Color Frame from RealSense ---------
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert the scene image to grayscale for feature detection
            gray_scene = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # --------- 5) Detect and Compute Features in the Scene ---------
            kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)
            if des_scene is None:
                cv2.imshow("Detected Template", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # --------- 6) Match Template Features with Scene Features ---------
            matches = bf.match(des_template, des_scene)
            if not matches:
                cv2.imshow("Detected Template", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Sort matches by distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)
            # Use a percentage of the best matches (or at least 10)
            num_good_matches = max(int(len(matches) * 0.2), 10)
            good_matches = matches[:num_good_matches]
            
            # --------- 7) Compute Homography if Enough Matches are Found ---------
            if len(good_matches) >= 10:
                pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
                if M is not None:
                    # Define the template corners and transform them to scene coordinates
                    template_corners = np.float32([[0, 0], [template_w, 0], [template_w, template_h], [0, template_h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(template_corners, M)
                    
                    # Draw the polygon boundary on the scene image
                    cv2.polylines(color_image, [np.int32(transformed_corners)], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            
            # --------- 8) Show the Result ---------
            cv2.imshow("Detected Template", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error:", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
