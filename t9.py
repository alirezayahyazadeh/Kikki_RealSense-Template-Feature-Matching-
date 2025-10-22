import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # 1) Load Template
    template_path = r"D:\template.jpg"  # Change to your template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template from {template_path}")
        return
    
    # ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp_template, des_template = orb.detectAndCompute(template, None)
    if des_template is None:
        print("No ORB descriptors in template.")
        return
    
    # 2) RealSense Setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # BFMatcher with default params (no crossCheck)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    t_h, t_w = template.shape[:2]

    print("Press 'q' to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            scene_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 3) Detect/Compute ORB in Scene
            kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
            if des_scene is None:
                cv2.imshow("Detected Template", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 4) knnMatch + Lowe's Ratio Test
            # We ask for the 2 nearest neighbors
            knn_matches = bf.knnMatch(des_template, des_scene, k=2)

            good_matches = []
            ratio_thresh = 0.75
            for m, n in knn_matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            # 5) Draw top matches for debugging (optional)
            # Sort good matches by distance
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            num_draw = min(50, len(good_matches))
            match_img = cv2.drawMatches(template, kp_template,
                                        color_image, kp_scene,
                                        good_matches[:num_draw],
                                        None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Feature Matches", match_img)

            # 6) Compute Homography
            if len(good_matches) >= 4:
                pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
                if M is not None:
                    # 7) Draw Polygon
                    corners = np.float32([[0,0], [t_w,0], [t_w,t_h], [0,t_h]]).reshape(-1,1,2)
                    transformed_corners = cv2.perspectiveTransform(corners, M)
                    cv2.polylines(color_image, [np.int32(transformed_corners)],
                                  isClosed=True, color=(0,255,0), thickness=3, lineType=cv2.LINE_AA)

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
