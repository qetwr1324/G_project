
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import MNIST_data
import glob
def region_of_interest(img, vertices):
    mask= np.zeros_like(img)
    if len(img.shape) >2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices, ignore_mask_color)
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
k=0
imges= [cv2.imread(pro) for pro in glob.glob("D:\pro\*.jpg")]

for i in range(0,len(imges)) :

   img = imges[i]
   cv2.imshow("",img)
   gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray

   blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

   low_threshold = 50
   high_threshold = 200
   edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

   if len(img.shape) > 2:
       channel_count = img.shape[2]
       ignore_mask_color = (255,)
   else:
       ignore_mask_color = 255

   imshape = img.shape

   vertices = np.array([[(100, 100),
                         (100, 101),
                         (100, 101),
                         (100, 100)]], dtype=np.int32)
   vertices1 = np.array([[(255, 1080),
                          (255, 800),
                          (600, 800),
                          (600, 1080)]], dtype=np.int32)

   vertices2 = np.array([[(1550, 1080),
                          (1450, 800),
                          (1750, 800),
                          (1850, 1080)]], dtype=np.int32)
   vertices3 = np.array([[(85, 1080),
                          (420, 780),
                          (680, 700),
                          (250, 1080)]], dtype=np.int32)

   mask = region_of_interest(edges, vertices)
   mask1 = region_of_interest(edges, vertices1)
   mask2 = region_of_interest(edges, vertices2)
   mask3 = region_of_interest(edges, vertices3)

   if (mask2.any()!=None):
       rho = 2
       theta = np.pi / 180
       threshold = 150
       min_line_len = 120
       max_line_gap = 150
       try:
           lines = hough_lines(mask2, rho, theta, threshold, min_line_len, max_line_gap)
       except:
           continue
   if (mask1.any()!=None):
       try:
           lines1 = hough_lines(mask1, rho, theta, threshold, min_line_len, max_line_gap)
       except:
           continue
   if((lines.any()!=None)&(lines1.any()!=None)):
       k+=1
       if(k>3):
           cv2.imwrite('D:\sample1\y' + str(i) + '.jpg', img)
   else:
        k=0




   if cv2.waitKey(33)>0:
     break

cv2.waitKey(0)
cv2.destroyWindow()





