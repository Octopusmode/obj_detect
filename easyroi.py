from EasyROI import EasyROI
import cv2
from sympy import Point, Polygon
from shapely import difference

roi_helper = EasyROI(verbose=True)

frame = cv2.imread('image.jpg')

polygon_roi = roi_helper.draw_polygon(frame, 2) # quantity=3 specifies number of polygons to draw

frame_temp = roi_helper.visualize_roi(frame, polygon_roi)

verticles_dict1 = polygon_roi["roi"][0]["vertices"]
verticles_dict2 = polygon_roi["roi"][1]["vertices"]

# creating points using Point()
# p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
  
# creating polygons using Polygon()
poly1 = Polygon(*verticles_dict1)
poly2 = Polygon(*verticles_dict2)

# poly2 = Polygon(p5, p6, p7)
  
# using intersection()
isIntersection = poly1.intersection(poly2)
  
# print(poly1)
# print(poly2)

diff = poly2.difference(poly1)  # or difference = polygon2 - polygon1

print(isIntersection)
print(diff)