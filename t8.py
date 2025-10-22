import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # ========== 1) Load Template ==========
    template_path = r"D:\template.jpg"  # <-- Change this to your template file
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template image from {template_path}")
        return

    # Dimensions of the template
    t_h, t_w = template.shape[:2]

    # ========== 2) Initialize ORB Detector ==========
    orb = cv2.ORB_create(nfeatures=1000)
    kp_template, des_template = orb.detectAndCompute(template, None)
    if des_template is None:
        print("Error: No ORB descriptors found in the template.")
        return

    # ========== 3) Initialize RealSense (Color Stream) ==========
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # BFMatcher (Hamming) for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

            # ========== 5) Detect and Compute ORB Features in the Scene ==========
            kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
            if des_scene is None or len(kp_scene) == 0:
                # No features found in the scene, just show the color image
                cv2.imshow("Detected Template", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ========== 6) Match Features ==========
            matches = bf.match(des_template, des_scene)
            # Sort matches by distance (best first)
            matches = sorted(matches, key=lambda x: x.distance)

            # ========== 7) Draw Some Matches for Debugging ==========
            # Let's draw the top 50 matches (or all if fewer than 50)
            num_draw = min(50, len(matches))
            match_img = cv2.drawMatches(
                template, kp_template,
                color_image, kp_scene,
                matches[:num_draw], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Show the matches in a separate window
            cv2.imshow("Feature Matches", match_img)

            # ========== 8) Filter Good Matches and Compute Homography ==========
            # Use 20% of all matches or at least 10
            num_good = max(int(len(matches) * 0.2), 10)
            good_matches = matches[:num_good]

            if len(good_matches) >= 4:
                # Extract matched keypoints
                pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography
                M, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)

                if M is not None:
                    # ========== 9) Draw Polygon for the Template Corners ==========
                    corners = np.float32([[0, 0],
                                          [t_w, 0],
                                          [t_w, t_h],
                                          [0, t_h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, M)
                    # Draw the polygon
                    cv2.polylines(color_image, [np.int32(transformed_corners)],
                                  isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

            # ========== 10) Show the Final Detection ==========
            cv2.imshow("Detected Template", color_image)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error:", e)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
