import cv2
import matplotlib.pyplot as plt

image_path = 'pyramid.jpg'
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

lower_res1 = cv2.pyrDown(image)
lower_res2 = cv2.pyrDown(lower_res1)
lower_res3 = cv2.pyrDown(lower_res2)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(lower_res1)
axes[1].set_title("Level 1")
axes[2].imshow(lower_res2)
axes[2].set_title("Level 2")
axes[3].imshow(lower_res3)
axes[3].set_title("Level 3")

for ax in axes:
    ax.axis("off")

plt.show().
