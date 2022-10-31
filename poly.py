import cv2
import numpy as np
from matplotlib import pyplot as plt

shapes = np.zeros((1920, 1080))

data = [[0, 0], [1040, 0], [1030, 520], [445, 1080], [0, 1080]]
contours = np.array(data)
cv2.fillPoly(shapes, pts = [contours], color =(255,255,255))

plt.imshow(shapes)
plt.title("matplotlib imshow")
plt.show()