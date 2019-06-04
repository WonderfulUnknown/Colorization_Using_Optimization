import numpy as np
import cv2

original = cv2.imread("picture\example2.bmp")
marked = cv2.imread("picture\example2_marked.bmp")

# up_img = cv2.pyrUp(original)  # 上采样操作
# down_img = cv2.pyrDown(original)  # 下采样操作

down_img = cv2.resize(marked, (int(y / 2), int(x / 2)))

cv2.namedWindow("src")
cv2.imshow("src",original)
cv2.namedWindow("down")
cv2.imshow("down",down_img)
cv2.imwrite("1.bmp",down_img)
cv2.waitKey (0)