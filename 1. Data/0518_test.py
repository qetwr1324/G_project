import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import cv2
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from PIL import Image
import shutil
from datetime import datetime
import time

#shift 변수 값
shift_par_x = 100
shift_par_y = 10

#폴더 지정 가능 
Loadplace = 'C:/python_test/aug/'

#break, accel 값, n x 4 행렬 유지위해   
break_default = 0 
accel_default = 0

#train, valid 갯수 체크용
len_train = []
len_valid = []

#하위 폴더들의 이름 저장용 배열 
dir_dir=[]

#현재 폴더 기준, 모든 폴더 검색
pwd = "."
for path, dirs, files in os.walk(pwd):
    for dir in dirs:
        dir_dir.append ( dir )
print("현재 폴더 목록:",dir_dir)
print("현재 폴더 갯수:",len(dir_dir),"개")

#테스트 전 폴더 제거, 위험!!!
#shutil.rmtree(r"C:\v324dsf")​

def load_data(labels_file, test_size):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            labels_file: The labels CSV file.
            test_size: The size of the testing set.
    """
    labels = pd.read_csv(labels_file)
    X = labels[['file_name']].values
    Y = labels['wheel'].values
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=test_size,shuffle =False,random_state=0)
    len_train.append( len(X_train) )
    len_valid.append( len(X_valid) )    
    return X_train, Y_train, X_valid, Y_valid

def load_image(data_dir, image_file):
    """
    Load RGB image.
        Parameters:
            data_dir: The directory where the images are.
            image_file: The image file name.
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def display(image, angle, label):
    plt.imshow(image)
    plt.xlabel("Steering angle: {:.10f}".format(angle))
    plt.title(label)

    plt.xticks([])
    plt.yticks([])
    plt.show()

def flip(image, steering_angle):
    """
    Randomly flipping the input image horizontaly, with steering angle adjustment.
        Parameters:
            image: The input image.
            steering_angle: The steering angle related to the input image.
    """
    image = cv2.flip(image, 1)#1은 좌우 반전, 0은 상하 반전
    steering_angle = -steering_angle
    return image, steering_angle

def random_shift(image, steering_angle, range_x, range_y):
    """
    Shifting (Translating) the input images, with steering angle adjustment.
        Parameters:
            image: The input image.
            steering_angle: The steering angle related to the input image.
            range_x: Horizontal translation range.
            range_y: Vertival translation range.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


print("flip_start")
#각 폴더 돌면서 'flip' 
for i in range(0,len(dir_dir)):
    print(i,"번째 폴더")
    # 20%는 검증(validate) 용도로 분리됨
    data = load_data(Loadplace + dir_dir[i] + '/' +dir_dir[i]+ '.csv', 0.2) 
    Saveplace = 'C:/python_test/fliped/'
    now = time.localtime()
    folder_name = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    try:
        if not(os.path.isdir(Saveplace + dir_dir[i])):
            os.makedirs(os.path.join(Saveplace + dir_dir[i]))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    df_train = pd.DataFrame([], columns=["file_name", "break", "accel", "wheel"], index=None)
    df_valid = pd.DataFrame([], columns=["file_name", "break", "accel", "wheel"], index=None)

    #train 파일 flip
    for j in range(0,len_train[i]):
        try:
            if not(os.path.isdir(Saveplace + dir_dir[i]+'/train')):
                os.makedirs(os.path.join(Saveplace + dir_dir[i]+'/train'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

        image = load_image(Loadplace+ '/' + dir_dir[i] + '/', str(data[0][0][0]) )
        #print(j,"th",data[0][j]) #현재 img파일 이름 확인 가능 
        steering_angle = data[1][j] #wheel 값 == image[1]

        image = flip(image, steering_angle)

        now = datetime.now()
        file_name = now.strftime("%Y%m%d%H%M%S%f.jpg")
        cv2.imwrite(Saveplace + dir_dir[i] + '/train/' + file_name,cv2.cvtColor(image[0],cv2.COLOR_BGR2RGB))

        df1 = pd.DataFrame({"file_name":data[0][j], "break":break_default, "accel":accel_default, "wheel":image[1]})
        df_train = pd.concat([df_train,df1],ignore_index = True)
        
        if j == len_train[i]-1:            
            df_train.to_csv(Saveplace + dir_dir[i] + '/train/' + folder_name+".csv")#train_csv
            print("csv_train 파일 저장 완료")
            
    #valid 파일 flip 
    for k in range(0,len_valid[i]):
        try:
            if not(os.path.isdir(Saveplace + dir_dir[i]+'/valid')):
                os.makedirs(os.path.join(Saveplace + dir_dir[i]+'/valid'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
        image = load_image(Loadplace + '/' + dir_dir[i] + '/', str(data[0][0][0]) )
        
        #print(k,"th",data[2][k])#현재 img파일 이름 확인 가능 
        steering_angle = data[3][k]#wheel 값 == image[1]
        
        image = flip(image, steering_angle)

        now = datetime.now()
        file_name = now.strftime("%Y%m%d%H%M%S%f.jpg")
        cv2.imwrite(Saveplace + dir_dir[i] + '/valid/' + file_name,cv2.cvtColor(image[0],cv2.COLOR_BGR2RGB))
        df2 = pd.DataFrame({"file_name":data[0][k], "break":break_default, "accel":accel_default, "wheel":image[1]})
        df_valid = pd.concat([df_valid,df2],ignore_index = True)

        if k == len_valid[i]-1:
            df_valid.to_csv(Saveplace + dir_dir[i] + '/valid/' + folder_name + ".csv")#train_csv
            print("csv_valid 파일 저장 완료")
    

print("flip_finish")
#shift 함수 참고용
"""
def random_shift(image, steering_angle, range_x, range_y):
    Shifting (Translating) the input images, with steering angle adjustment.

        Parameters:
            image: The input image.
            steering_angle: The steering angle related to the input image.
            range_x: Horizontal translation range.
            range_y: Vertival translation range.

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle
"""
print("shift_start")
#각 폴더 돌면서 'shift' 
for i in range(0,len(dir_dir)):
    print(i,"번째 폴더")
    # 20%는 검증(validate) 용도로 분리됨
    data = load_data(Loadplace + dir_dir[i] + '/' +dir_dir[i]+ '.csv', 0.2)
    #Loadplace = 'C:/python_test/aug/'
    Saveplace = 'C:/python_test/shifted/'
    now = time.localtime()
    folder_name = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    try:
        if not(os.path.isdir(Saveplace + dir_dir[i])):
            os.makedirs(os.path.join(Saveplace + dir_dir[i]))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    df_train = pd.DataFrame([], columns=["file_name", "break", "accel", "wheel"], index=None)
    df_valid = pd.DataFrame([], columns=["file_name", "break", "accel", "wheel"], index=None)

    #train 파일 shift
    for j in range(0,len_train[i]):
        try:
            if not(os.path.isdir(Saveplace + dir_dir[i]+'/train')):
                os.makedirs(os.path.join(Saveplace + dir_dir[i]+'/train'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

        image = load_image(Loadplace + dir_dir[i] + '/', str(data[0][0][0]) )
        #print(j,"th",data[0][j]) #현재 img파일 이름 확인 가능 
        steering_angle = data[1][j] #wheel, image[1]은 wheel 값!!!

        image = random_shift(image, steering_angle, shift_par_x, shift_par_y)

        now = datetime.now()
        file_name = now.strftime("%Y%m%d%H%M%S%f.jpg")
        
        cv2.imwrite(Saveplace + dir_dir[i] + '/train/' + file_name, cv2.cvtColor(image[0],cv2.COLOR_BGR2RGB))

        df1 = pd.DataFrame({"file_name":data[0][j], "break":break_default, "accel":accel_default, "wheel":image[1]})
        df_train = pd.concat([df_train,df1],ignore_index = True)
        
        if j == len_train[i]-1:            
            df_train.to_csv(Saveplace + dir_dir[i] + '/train/' + folder_name + ".csv")#train_csv
            print("csv_train 파일 저장 완료")
            
    #valid 파일 shift 
    for k in range(0,len_valid[i]):
        try:
            if not(os.path.isdir(Saveplace + dir_dir[i]+'/valid')):
                os.makedirs(os.path.join(Saveplace + dir_dir[i]+'/valid'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
        image = load_image(Loadplace + dir_dir[i] + '/', str(data[0][0][0]) )
        
        #print(k,"th",data[2][k])#현재 img파일 이름 확인 가능  
        steering_angle = data[3][k]#wheel, image[1]은 wheel 값!!!
        
        image = random_shift(image, steering_angle, shift_par_x, shift_par_y)
 
        now = datetime.now()
        file_name = now.strftime("%Y%m%d%H%M%S%f.jpg")
        cv2.imwrite(Saveplace + dir_dir[i] + '/valid/' + file_name, cv2.cvtColor(image[0],cv2.COLOR_BGR2RGB))
        df2 = pd.DataFrame({"file_name":data[0][k], "break":break_default, "accel":accel_default, "wheel":image[1]})
        df_valid = pd.concat([df_valid,df2],ignore_index = True)

        if k == len_valid[i]-1:
            df_valid.to_csv(Saveplace + dir_dir[i] + '/valid/' +folder_name + ".csv")#train_csv
            print("csv_valid 파일 저장 완료")
    

print("shift_finish")
