import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import colorsys
import cv2


def yiq_to_rgb(y, i, q):
    r = y + 0.948262 * i + 0.624013 * q
    g = y - 0.276066 * i - 0.639810 * q
    b = y - 1.105450 * i + 1.729860 * q
    r[r < 0] = 0
    r[r > 1] = 1
    g[g < 0] = 0
    g[g > 1] = 1
    b[b < 0] = 0
    b[b > 1] = 1
    return (r, g, b)


def colorization(original, marked, is_colored):
    original = original.astype(float) / 255
    marked = marked.astype(float) / 255

    # is_colored = abs(original - marked).sum(2) > 0.01  # 得到上色点的位置

    (Y, _, _) = colorsys.rgb_to_yiq(original[:, :, 0], original[:, :, 1], original[:, :, 2])  # 获取灰度值
    (_, I, Q) = colorsys.rgb_to_yiq(marked[:, :, 0], marked[:, :, 1], marked[:, :, 2])  # 获取I,Q分量

    YUV = np.zeros(original.shape)
    YUV[:, :, 0] = Y
    YUV[:, :, 1] = I
    YUV[:, :, 2] = Q

    n = YUV.shape[0]  # row
    m = YUV.shape[1]  # col
    image_size = n * m

    indices_matrix = np.arange(image_size).reshape(n, m, order='F').copy()  # 0-image_size n行m列 类FORTRAN 按列存储

    window_size = 1
    number = (2 * window_size + 1) ** 2  # 周围需要选取点的数量
    all_number = image_size * number  # 一张图需要选取点的数量

    row_index = np.zeros(all_number, dtype=np.int64)
    col_index = np.zeros(all_number, dtype=np.int64)
    value = np.zeros(all_number)

    length = 0  # 实际计算的点的个数+原图点的个数(因为在边缘不扩展不能用all_number)
    # count = 0#代表在整张图像中的某个点

    # 计算图中的像素
    for j in range(m):
        for i in range(n):
            # 像素未被上色
            if (not is_colored[i, j]):
                window_index = 0
                gray_value = np.zeros(number)

                # 在[i,j]周围的窗口循环计算
                for ii in range(max(0, i - window_size), min(i + window_size + 1, n)):  # 防止越界
                    for jj in range(max(0, j - window_size), min(j + window_size + 1, m)):

                        # 当前的像素位置不是[i,j]
                        if (ii != i or jj != j):
                            row_index[length] = indices_matrix[i, j]  # count
                            col_index[length] = indices_matrix[ii, jj]
                            gray_value[window_index] = YUV[ii, jj, 0]
                            length += 1
                            window_index += 1

                curr_value = YUV[i, j, 0].copy()
                gray_value[window_index] = curr_value
                # 计算像素[i，j]周围窗口的方差
                std = gray_value[0:window_index + 1] - np.mean(gray_value[0:window_index + 1])
                variance = np.mean(std ** 2)
                # sigma过大会导致边界上色异常，过小使得上色有缺口
                sigma = variance * 0.6

                diff = gray_value[0:window_index] - curr_value
                min_diff = min(diff ** 2)
                # 没懂源码为何如此写
                if (sigma < (-min_diff / np.log(0.01))):
                    sigma = -min_diff / np.log(0.01)
                # 避免sigma为0，导致后面除法异常
                if (sigma < 0.000002):
                    sigma = 0.000002
                # 第二个权重函数
                gray_value[0:window_index] = np.exp(-(diff ** 2) / sigma)
                # 权重和为1
                gray_value[0:window_index] = gray_value[0:window_index] / np.sum(gray_value[0:window_index])
                curr_post = length - window_index
                value[curr_post:length] = -gray_value[0:window_index]

            # 记录当前的点
            # row_index[length] = count
            row_index[length] = indices_matrix[i, j]
            col_index[length] = indices_matrix[i, j]
            # debug
            # print("count = ",count)
            # print("indices_matrix[i,j] = ",indices_matrix[i,j])
            value[length] = 1
            length += 1
            # count += 1

    # debug
    # print(length)
    # print(image_size)
    # print(all_number)
    # print(count)

    # 舍弃不需要的值
    value = value[0:length]
    col_index = col_index[0:length]
    row_index = row_index[0:length]

    # A = sparse.csr_matrix((value, (row_index, col_index)), (count, image_size))
    # 构建稀疏列矩阵
    A = csr_matrix((value, (row_index, col_index)), shape=(image_size, image_size))
    b = np.zeros((A.shape[0]))
    colorized = np.zeros(YUV.shape)
    colorized[:, :, 0] = YUV[:, :, 0]

    color_copy = is_colored.reshape(image_size, order='F').copy()
    colored_index = np.nonzero(color_copy)

    for t in [1, 2]:
        curr_image = YUV[:, :, t].reshape(image_size, order='F').copy()
        b[colored_index] = curr_image[colored_index]
        # 求解稀疏线性方程组Ax=b
        result = spsolve(A, b)
        colorized[:, :, t] = result.reshape(n, m, order='F')

    # cv2.namedWindow("1")
    # cv2.imshow("1",colorized)

    # 转回RGB
    (R, G, B) = yiq_to_rgb(colorized[:, :, 0], colorized[:, :, 1], colorized[:, :, 2])
    colorizedRGB = np.zeros(colorized.shape)
    colorizedRGB[:, :, 0] = R
    colorizedRGB[:, :, 1] = G
    colorizedRGB[:, :, 2] = B

    cv2.namedWindow("colored")
    cv2.imshow("colored", colorizedRGB)
    cv2.waitKey(0)
    return colorizedRGB


src1 = cv2.imread("picture\example3.bmp")
src2 = cv2.imread("picture\example3_marked.bmp")
cv2.namedWindow("0")
cv2.imshow("0", src2)

n = src1.shape[0]
m = src1.shape[1]

is_colored = abs(src1 - src2).sum(2) > 0.01  # 得到上色点的位置
color_record = np.zeros((3,3), dtype=np.int64)
#得到具体上色的色彩值
for j in range(m):
    for i in range(n):
        if (is_colored[i,j]):
            color_value = src2[i,j]
            # 颜色没有被记录过,添加
            # if (not (color_record == color_value).any()):
            if (color_value not in color_record):
                temp = list(color_record)
                temp.append(color_value)
                color_record = np.array(temp)

# problem!
# 金字塔导致像素模糊，使得涂鸦的颜色不一致.出现模糊
# 图像金字塔
# original = cv2.pyrDown(original)
# down_img = cv2.pyrDown(src2)  # 下采样操作
# up_img = cv2.pyrUp(down_img)  # 上采样操作

# 图像缩放,像素变为1/4
x, y = src1.shape[0:2]
down_img1 = cv2.resize(src1, (int(y / 2), int(x / 2)))
down_img2 = cv2.resize(src2, (int(y / 2), int(x / 2)))

is_colored = abs(down_img1 - down_img2).sum(2) > 0.01  # 得到上色点的位置
n = down_img1.shape[0]
m = down_img1.shape[1]

for j in range(m):
    for i in range(n):
        if (is_colored[i,j]):
            color_value = down_img2[i,j]
            # 如果颜色没有被记录过，说明颜色被修改，is_colored此位变为false
            # if (not (color_record == color_value).any()):
            if (color_value not in color_record):
                is_colored[i,j] = False

colorization(down_img1, down_img2, is_colored)

# cv2.waitKey(0)