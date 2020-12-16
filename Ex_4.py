from matplotlib import pyplot as plt
import cv2 as cv
import random as rdm
import numpy as np


def sap(src, per):
    noise_img = src
    noise_num = int(per * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        # 随机生成行rdx,随机生成列rdy
        # 椒盐噪声图片边缘不处理，故-1
        rdx = rdm.randint(0, src.shape[0] - 1)
        rdy = rdm.randint(0, src.shape[1] - 1)
        if rdm.random() <= 0.5:
            noise_img[rdx, rdy] = 0
        else:
            noise_img[rdx, rdy] = 255
    return noise_img


def gaussian(src, means, sigma, per):
    # means为均值，sigma为标准差
    noise_img = src
    noise_num = int(per * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        rdx = rdm.randint(0, src.shape[0] - 1)
        rdy = rdm.randint(0, src.shape[1] - 1)
        # 在原像素灰度值上加上随机数
        noise_img[rdx, rdy] = noise_img[rdx, rdy] + rdm.gauss(means, sigma)
        if noise_img[rdx, rdy] < 0:
            noise_img[rdx, rdy] = 0
        elif noise_img[rdx, rdy] > 255:
            noise_img[rdx, rdy] = 255
    return noise_img


# 不同算子
sobel_filter1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_filter2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

roberts_filter1 = np.array([[-1, 0], [0, 1]])
roberts_filter2 = np.array([[0, -1], [1, 0]])

prewitt_filter1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_filter2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
# 拉普拉斯掩膜h1，h2
laplace_h1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
laplace_h2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
laplace_log = np.array(
    [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

img = cv.imread('H:/Digital Image/Ex_4/sharpen/beiwen2.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图

img1 = img.copy()  # 备份原图方便下一步处理

img_sap = sap(img, 0.2)  # 添加椒盐噪声
img_gau = gaussian(img_sap, 2, 1, 0.5)  # 添加高斯噪声


def fun(ori, kernel_1, kernel_2):  # 分别计算x,y方向梯度并融合，返回x,y和最终梯度
    x = cv.filter2D(ori, cv.CV_16S, kernel_1)
    y = cv.filter2D(ori, cv.CV_16S, kernel_2)
    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)
    fin = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return abs_x, abs_y, fin


def fun1(ori, kernel):
    fin = cv.convertScaleAbs(cv.filter2D(ori, cv.CV_16S, kernel))
    return fin


sobel_img_x, sobel_img_y, sobel_img = fun(img1, sobel_filter1, sobel_filter2)

sobel_noise_x, sobel_noise_y, sobel_noise = fun(img_gau, sobel_filter1, sobel_filter2)

roberts_img_x, roberts_img_y, roberts_img = fun(img1, roberts_filter1, roberts_filter2)

roberts_noise_x, roberts_noise_y, roberts_noise = fun(img_gau, roberts_filter1, roberts_filter2)

prewitt_img_x, prewitt_img_y, prewitt_img = fun(img1, prewitt_filter1, prewitt_filter2)

prewitt_noise_x, prewitt_noise_y, prewitt_noise = fun(img_gau, prewitt_filter1, prewitt_filter2)

laplace_h1_img = fun1(img1, laplace_h1)

laplace_h2_img = fun1(img1, laplace_h2)

laplace_log_img = fun1(img1, laplace_log)

laplace_h1_noise = fun1(img_gau, laplace_h1)

laplace_h2_noise = fun1(img_gau, laplace_h2)

laplace_log_noise = fun1(img_gau, laplace_log)

plt.figure(0)
plt.subplot(2, 3, 1)
plt.imshow(sobel_img_x, 'gray')
plt.title('origin_sobel_x')
plt.subplot(2, 3, 2)
plt.imshow(sobel_img_y, 'gray')
plt.title('origin_sobel_y')
plt.subplot(2, 3, 3)
plt.imshow(sobel_img, 'gray')
plt.title('origin_sobel')
plt.subplot(2, 3, 4)
plt.imshow(sobel_noise_x, 'gray')
plt.title('noise_sobel_x')
plt.subplot(2, 3, 5)
plt.imshow(sobel_noise_y, 'gray')
plt.title('noise_sobel_y')
plt.subplot(2, 3, 6)
plt.imshow(sobel_noise, 'gray')
plt.title('noise_sobel')

plt.figure(1)
plt.subplot(2, 3, 1)
plt.imshow(roberts_img_x, 'gray')
plt.title('origin_roberts_x')
plt.subplot(2, 3, 2)
plt.imshow(roberts_img_y, 'gray')
plt.title('origin_roberts_y')
plt.subplot(2, 3, 3)
plt.imshow(roberts_img, 'gray')
plt.title('origin_roberts')
plt.subplot(2, 3, 4)
plt.imshow(roberts_noise_x, 'gray')
plt.title('noise_roberts_x')
plt.subplot(2, 3, 5)
plt.imshow(roberts_noise_y, 'gray')
plt.title('noise_roberts_y')
plt.subplot(2, 3, 6)
plt.imshow(roberts_noise, 'gray')
plt.title('noise_roberts')

plt.figure(2)
plt.subplot(2, 3, 1)
plt.imshow(prewitt_img_x, 'gray')
plt.title('origin_prewitt_x')
plt.subplot(2, 3, 2)
plt.imshow(prewitt_img_y, 'gray')
plt.title('origin_prewitt_y')
plt.subplot(2, 3, 3)
plt.imshow(prewitt_img, 'gray')
plt.title('origin_prewitt')
plt.subplot(2, 3, 4)
plt.imshow(prewitt_noise_x, 'gray')
plt.title('noise_prewitt_x')
plt.subplot(2, 3, 5)
plt.imshow(prewitt_noise_y, 'gray')
plt.title('noise_prewitt_y')
plt.subplot(2, 3, 6)
plt.imshow(prewitt_noise, 'gray')
plt.title('noise_prewitt')

plt.figure(3)
plt.subplot(2, 3, 1)
plt.imshow(laplace_h1_img)
plt.title('laplace_h1_img')
plt.subplot(2, 3, 2)
plt.imshow(laplace_h2_img)
plt.title('laplace_h2_img')
plt.subplot(2, 3, 3)
plt.imshow(laplace_log_img)
plt.title('laplace_loh_img')
plt.subplot(2, 3, 4)
plt.imshow(laplace_h1_noise)
plt.title('laplace_h1_noise')
plt.subplot(2, 3, 5)
plt.imshow(laplace_h2_noise)
plt.title('laplace_h2_noise')
plt.subplot(2, 3, 6)
plt.imshow(laplace_log_noise)
plt.title('laplace_log_noise')

# 膨胀与腐蚀
element = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))  # 定义十字结构元素
# NpKernel = np.uint8(np.zeros((5,5))) #numpy定义
# for i in range(5):
#     NpKernel[2, i] = 1
#     NpKernel[i, 2] = 1

# 腐蚀
eroded_img = cv.erode(img1, element)
eroded_noise = cv.erode(img_gau, element)
# 膨胀
dilated_img = cv.dilate(img1, element)
dilated_noise = cv.dilate(img_gau, element)
# 开运算，先腐蚀再膨胀
opened_img = cv.morphologyEx(img1, cv.MORPH_OPEN, element)
opened_noise = cv.morphologyEx(img1, cv.MORPH_OPEN, element)
# 闭运算，先膨胀再腐蚀
closed_img = cv.morphologyEx(img_gau, cv.MORPH_CLOSE, element)
closed_noise = cv.morphologyEx(img_gau, cv.MORPH_CLOSE, element)

plt.figure(4)
plt.subplot(2, 4, 1)
plt.imshow(eroded_img)
plt.title('eroded_img')
plt.subplot(2, 4, 2)
plt.imshow(dilated_img)
plt.title('dilated_img')
plt.subplot(2, 4, 3)
plt.imshow(opened_img)
plt.title('opened_img')
plt.subplot(2, 4, 4)
plt.imshow(closed_img)
plt.title('closed_img')
plt.subplot(2, 4, 5)
plt.imshow(eroded_noise)
plt.title('eroded_noise')
plt.subplot(2, 4, 6)
plt.imshow(dilated_noise)
plt.title('dilated_noise')
plt.subplot(2, 4, 7)
plt.imshow(opened_noise)
plt.title('opened_noise')
plt.subplot(2, 4, 8)
plt.imshow(closed_noise)
plt.title('closed_noise')

plt.show()
