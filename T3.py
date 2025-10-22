import cv2
import numpy as np

# Set the image path
image_path = r"D:\template.jpg"

# Load the depth image
depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Convert to grayscale if not already
if len(depth_image.shape) == 3:
    gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
else:
    gray = depth_image

# Preprocessing
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.bilateralFilter(gray, 9, 75, 75)

# Define a template (Crop a region)
template = gray[100:200, 100:200]  # Example cropping, adjust as needed
w, h = template.shape[::-1]

# Use ORB feature detector
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(template, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray, None)

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
output = cv2.drawMatches(template, keypoints1, gray, keypoints2, matches[:10], None, flags=2)

# Find best match location using template matching
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw rectangle around detected object
cv2.rectangle(gray, top_left, bottom_right, (0, 255, 0), 2)

# Display results
cv2.imshow("Detected", gray)
cv2.imshow("Feature Matching", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
