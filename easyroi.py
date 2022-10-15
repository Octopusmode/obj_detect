from EasyROI import EasyROI
import cv2
import numpy as np
from matplotlib import pyplot as plt

# from sympy import Point, Polygon
from shapely.geometry import Polygon

roi_helper = EasyROI(verbose=True)

frame = cv2.imread('image.jpg')

polygon_roi = roi_helper.draw_polygon(frame, 2) # quantity=3 specifies number of polygons to draw

frame_temp = roi_helper.visualize_roi(frame, polygon_roi)

verticles_dict1 = polygon_roi["roi"][0]["vertices"]
verticles_dict2 = polygon_roi["roi"][1]["vertices"]

# creating points using Point()
# p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
  
# creating polygons using Polygon()
poly1 = Polygon(verticles_dict1)
poly2 = Polygon(verticles_dict2)

# poly2 = Polygon(p5, p6, p7)
  
# using intersection()
# isIntersection = poly1.intersection(poly2)
  
# print(poly1)
# print(poly2)

diff = poly2.difference(poly1)  # or difference = polygon2 - polygon1
shapes = np.zeros_like(frame, np.uint8)

int_coords = lambda x: np.array(x).round().astype(np.int32)
exterior = [int_coords(diff.exterior.coords)]
poly1_ex = [int_coords(poly1.exterior.coords)]
# poly2_ex = [int_coords(poly2.exterior.coords)]

cv2.drawContours(shapes, exterior, -1, (0, 255, 255), -1)
cv2.drawContours(shapes, poly1_ex, -1, (255, 0, 255), -1)
# cv2.drawContours(shapes, poly2_ex, -1, (255, 255, 0), -1)

perc = (poly2.area / 100 * (poly2.area - diff.area))

# Generate output by blending image with shapes image, using the shapes
# images also as mask to limit the blending to those parts
out = frame.copy()
alpha = 0.5
mask = shapes.astype(bool)
out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

print(frame.shape[0])
print(frame.shape[1])

plt.imshow(out)
plt.title("matplotlib imshow")
plt.show()