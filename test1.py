import numpy as np
import cv2 as cv

def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    edge_output = cv.Canny(xgrad, ygrad, 10, 30)
    # edge_output = cv.Canny(xgrad, ygrad, 20, 60)
    # edge_output = cv.Canny(gray, 20, 60)
    cv.imshow("Canny Edge", edge_output)
    dst = cv.bitwise_and(image, image, mask= edge_output)
    cv.imshow("Color Edge", dst)
    cv.waitKey()

original = cv.imread("picture\example.bmp")
marked = cv.imread("picture\example_marked.bmp")
edge_demo(original)
