import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
image = Image.open("/Users/tomer/PycharmProjects/a/00000199.jpg")
plt.imshow(image)
plt.show()
resized_image = tf.image.resize(image, [299, 299])
resized_image = (np.rint(resized_image)).astype(int)
print(resized_image)
plt.imshow(resized_image)
plt.show()
