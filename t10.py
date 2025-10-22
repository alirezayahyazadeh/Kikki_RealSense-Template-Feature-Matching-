import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # ========== 1) Load Template ==========
    template_path = r"D:\template.jpg"  # <-- Update to your template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template image from {template_path}")
        return
    
    t_h, t_w = template.shape[:2]
    
    # ========== 2) Initialize ORB with More Features ==========
    orb = cv2.ORB_create(nfeatures=2000)  # Increased from 1000 to 2000
    kp_template, des_template = orb.detectAndCompute(template, None)
    if des_template is None:
        print("No ORB descriptors in template.")
        return
    
    # ========== 3) Setup RealSense (Color Stream) ==========
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # BFMatcher with default params (no crossCheck)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    print("Press 'q' to quit.")
    try:
        while True:
            # ========== 4) Grab a Frame from RealSense ==========
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            scene_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # ========== 5) Detect/Compute ORB in Scene ==========
            kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
            if des_scene is None:
                cv2.imshow("Detected Template", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ========== 6) Use knnMatch + Lowe's Ratio Test ==========
            # We ask for the 2 nearest neighbors for each descriptor
            knn_matches = bf.knnMatch(des_template, des_scene, k=2)
            good_matches = []
            ratio_thresh = 0.75  # typical ratio threshold
            for m, n in knn_matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            # Sort good matches by distance (lowest = best)
            good_matches = sorted(good_matches, key=lambda x: x.distance)

            # ========== 7) Draw Matches for Debugging (Optional) ==========
            num_draw = min(50, len(good_matches))
            match_img = cv2.drawMatches(
                template, kp_template,
                color_image, kp_scene,
                good_matches[:num_draw],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow("Feature Matches", match_img)

            # ========== 8) Compute Homography if Enough Good Matches ==========
            if len(good_matches) >= 4:
                pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
                if M is not None:
                    # Draw the template corners in the scene
                    corners = np.float32([[0, 0],
                                          [t_w, 0],
                                          [t_w, t_h],
                                          [0, t_h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, M)
                    cv2.polylines(color_image, [np.int32(transformed_corners)],
                                  isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

            # ========== 9) Show the Final Detection ==========
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
   
   