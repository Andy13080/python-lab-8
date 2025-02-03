import cv2
import numpy as np
import matplotlib.pyplot as plt

pyramid_img_path = "pyramid.jpg"
fly_img_path = "fly64.png"

pyramid_img = cv2.imread(pyramid_img_path)
fly_img = cv2.imread(fly_img_path, cv2.IMREAD_UNCHANGED)

gray = cv2.cvtColor(pyramid_img, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

if len(keypoints) > 0:
    keypoint_coords = np.array([kp.pt for kp in keypoints])
    avg_x, avg_y = np.mean(keypoint_coords, axis=0)
    avg_x, avg_y = int(avg_x), int(avg_y)
    print(f"Mark Center: X = {avg_x}, Y = {avg_y}")
else:
    print("No keypoints detected!")
    exit()

fly_h, fly_w = fly_img.shape[:2]

top_left_x = avg_x - fly_w // 2
top_left_y = avg_y - fly_h // 2

if top_left_x < 0: top_left_x = 0
if top_left_y < 0: top_left_y = 0

roi = pyramid_img[top_left_y:top_left_y + fly_h, top_left_x:top_left_x + fly_w]

if fly_img.shape[2] == 4:
    alpha_channel = fly_img[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = (1 - alpha_channel) * roi[:, :, c] + alpha_channel * fly_img[:, :, c]
else:
    roi[:] = fly_img[:, :, :3]

pyramid_img[top_left_y:top_left_y + fly_h, top_left_x:top_left_x + fly_w] = roi


pyramid_img = cv2.cvtColor(pyramid_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(pyramid_img)
plt.title("Fly Superimposed on Pyramid Image")
plt.axis("off")
plt.show()