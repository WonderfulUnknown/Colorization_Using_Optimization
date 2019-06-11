import numpy as np
import cv2

def gauss_pyramid(image,level):
    # level = 3      #设置金字塔的层数为3
    temp = image.copy()  #拷贝图像
    pyramid_images = []  #建立一个空列表
    for i in range(level):
        dst = cv2.pyrDown(temp)   #先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
        pyramid_images.append(dst)  #在列表末尾添加新的对象
        cv2.imshow("pyramid"+str(i), dst)
        temp = dst.copy()
    return pyramid_images
#拉普拉斯金字塔
def lapalian_pyramid(image,level):
    pyramid_images = gauss_pyramid(image,level)    #做拉普拉斯金字塔必须用到高斯金字塔的结果
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):#数组下标从0开始 i从金字塔层数-1开始减减
        if (i-1) < 0:#原图
            expand = cv2.pyrUp(pyramid_images[i], dstsize = image.shape[:2])
            lpls = cv2.subtract(image, expand)
            cv.imshow("lapalian_down_"+str(i), lpls)
        else:
            expand = cv2.pyrUp(pyramid_images[i], dstsize = pyramid_images[i-1].shape[:2])
            lpls = cv2.subtract(pyramid_images[i-1], expand)
            cv2.imshow("lapalian_down_"+str(i), lpls)


original = cv2.imread("picture\example2.bmp")
marked = cv2.imread("picture\example2_marked.bmp")
cv2.namedWindow("0")
cv2.imshow("0",marked)

down_img = cv2.pyrDown(marked)  # 下采样操作
cv2.namedWindow("1")
cv2.imshow("1",down_img)

up_img = cv2.pyrUp(down_img)  # 上采样操作
cv2.namedWindow("2")
cv2.imshow("2",up_img)

# img = lapalian_pyramid(original,3)
# down_img = gauss_pyramid(original,1)
# down_img = cv2.resize(marked, (int(y / 2), int(x / 2)))

# cv2.namedWindow("src")
# cv2.imshow("src",original)
# cv2.namedWindow("down")
# cv2.imshow("down",down_img)
# cv2.imwrite("1.bmp",down_img)
cv2.waitKey (0)