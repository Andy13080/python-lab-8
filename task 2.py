import cv2
import matplotlib.pyplot as plt

image_path = "pyramid.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_with_keypoints)
plt.title("Feature-Based Tracking on Pyramid Image")
plt.axis("off")
plt.show().