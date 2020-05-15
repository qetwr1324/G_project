import numpy as np
import os
import cv2
import pandas as pd
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

rho = 5
theta = np.pi / 180
threshold = 60
min_line_len = 50
max_line_gap = 50
low_threshold = 10
high_threshold = 20
Loadplace="D:\pro8/"
dir=os.listdir('D:\pro8/')
Saveplace="D:\Smaple 10/"
CSVfile = '20200513134840.csv'


csv_svdata=pd.DataFrame()
for filenum in range(0,len(dir)):
    k=0
    os.makedirs(Saveplace+dir[filenum]+'/')
    imges = [cv2.imread(pro) for pro in glob.glob(Loadplace+ dir[filenum] + "\*.jpg")]
    csv_data = pd.read_csv(Loadplace+ dir[filenum] +'/' + dir[filenum] + '.csv')
    csv_svdata = pd.DataFrame()
    for i in range(0, len(imges)):

        img = imges[i]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray

        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        for y in range(0, 10):
            blur_gray = cv2.GaussianBlur(blur_gray, (kernel_size, kernel_size), 10)

        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        vertices1 = np.array([[(73, 70),
                               (83, 0),
                               (95, 0),
                               (95, 70)]], dtype=np.int32)
        vertices2 = np.array([[(95, 70),
                               (95, 0),
                               (110, 0),
                               (120, 70)]], dtype=np.int32)

        mask1 = region_of_interest(edges, vertices1)
        mask2 = region_of_interest(edges, vertices2)

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

        if ((li1 == True) | (li2 == True)):
            if ((i - k) > 3):
                for t in range((k + 1), (i - 2)):
                    cv2.imwrite(Saveplace + dir[filenum] + '/' + str(csv_data.ix[t - 1][1]), imges[t])
                    csv_svdata = csv_svdata.append(csv_data.iloc[t])
                k = i
            else:
                k = i
        if (i == (len(imges) - 1)):
            for t in range((k + 1), i):
                cv2.imwrite(Saveplace + dir[filenum] + '/' + str(csv_data.ix[t][1]), imges[t])
                csv_svdata = csv_svdata.append(csv_data.iloc[t])
    csv_svdata = csv_svdata.loc[:, ["file_name", "break", "accel", "wheel"]]

    csv_svdata.to_csv(Saveplace + dir[filenum] + '/' + dir[filenum] + '.csv')

print(len(dir))
cv2.waitKey(0)
cv2.destroyWindow()