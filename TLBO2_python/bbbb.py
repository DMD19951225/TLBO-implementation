import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa

# random example image
# image = np.random.randint(0, 255, size=(600, 600, 3)).astype(np.uint8)
fig = plt.figure(figsize=(1, 1))
image = cv2.imread("E:/Python Task Finished/Abdulaziz_Jordan_TLBO/TLBO2_python/real_and_fake_face/training_fake/easy_1_1110.jpg")
plt.show()
# augment 16 times the example image
images_aug = iaa.Affine(rotate=(-45, 45)).augment_images([image] * 2)

# iterate over every example image and save it to 0.jpg, 1.jpg, 2.jpg, ...
for i, image_aug in enumerate(images_aug):
    imageio.imwrite("%d.jpg" % (i,), image_aug)