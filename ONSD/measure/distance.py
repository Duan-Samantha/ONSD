import numpy as np
import cv2


img = cv2.imread("SMP20210512165011/左眼测量2.JPG")
bbox = cv2.selectROI('Select ROI', img, False)
print(bbox)

# 126 pixel == 6.4 mm -> 0.05