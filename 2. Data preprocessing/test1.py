
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import MNIST_data
import glob


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


kernel_size = 5  # 가우시안
k = 0
rho = 5
theta = np.pi / 180
threshold = 220
min_line_len = 150
max_line_gap = 150

imges = [cv2.imread(pro) for pro in glob.glob("D:\pro1\*.jpg")]

for i in range(0, len(imges)):

    img = imges[i]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray

    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    for y in range(0,10):
        blur_gray = cv2.GaussianBlur(blur_gray , (kernel_size, kernel_size), 10)

    low_threshold = 10
    high_threshold = 20
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)
    else:
        ignore_mask_color = 255

    vertices1 = np.array([[(500, 1000),
                          (730, 750),
                          (840, 750),
                          (725, 1000)]], dtype=np.int32)
    vertices2 = np.array([[(725, 1000),
                          (840, 700),
                          (950, 700),
                          (950, 1000)]], dtype=np.int32)
    vertices3 = np.array([[(950, 1000),
                          (950, 700),
                          (1060, 700),
                          (1175, 1000)]], dtype=np.int32)
    vertices4 = np.array([[(1175, 1000),
                          (1060, 700),
                          (1180, 700),
                          (1400, 1000)]], dtype=np.int32)


    mask1 = region_of_interest(edges, vertices1)
    mask2 = region_of_interest(edges, vertices2)
    mask3 = region_of_interest(edges, vertices3)
    mask4 = region_of_interest(edges, vertices4)

    try:
        lines1 = hough_lines(mask1, rho, theta, threshold, min_line_len, max_line_gap)
        li1 = True
    except:

        li1 = False
    try:
        lines2 = hough_lines(mask2, rho, theta, threshold, min_line_len, max_line_gap)
        li2 = True
    except:

        li2 = False
    try:
        lines3 = hough_lines(mask3, rho, theta, threshold, min_line_len, max_line_gap)
        li3 = True
    except:

        li3 = False
    try:
        lines4 = hough_lines(mask4, rho, theta, threshold, min_line_len, max_line_gap)
        li4 = True
    except:

        li4 = False





    if ((li1==True)|(li2==True)|(li3==True)|(li4==True)):
        if ((i - k) > 3):
            for t in range(k, (i - 3)):
                cv2.imshow("", imges[t])
                cv2.imwrite('D:\sample2\y' + str(t) + '.jpg', imges[t])
            k = i
        else:
            k = i
    if(i==(len(imges)-1)):
        for t in range(k,i):
            cv2.imwrite('D:\sample2\y' + str(t) + '.jpg', imges[t])
cv2.waitKey(0)
cv2.destroyWindow()