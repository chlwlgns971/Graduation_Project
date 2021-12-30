#학습된 모델 성능테스트(테스트 이미지 사용해서 얼마나 판별을 잘 하는지)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #로그레벨(쓸데없는 오류 뜨는걸 안보이게 처리)
#from keras.models import load_model #gpu환경에선 안돌아감
from tensorflow.keras.models import load_model
import cv2
import numpy as np

#모델 불러오기
model=load_model('C:\\Users\\wkdeh\\Desktop\\강의자료\\졸업작품\\졸작\\MaskCheckModel0712.hdf5')

def change_brightness(img, percentage):
    if percentage < 0 :
        print("잘못된 피라미터 입니다.")
        return None

    adjusted = img.copy()
    for row in range(len(img)):
        for col in range(len(img[0])):
            #픽셀의 값에 백분율을 곱하고 100으로 정수나눗셈을 함.
            #이부분을 넘파이를 이용하면 성능이 대폭 향상됨.
            r = img[row][col][0] * percentage // 100
            g = img[row][col][1] * percentage // 100
            b = img[row][col][2] * percentage // 100

            #최댓값을 초과하는 경우 255로 보정
            if r > 255 : r = 255
            if g > 255 : g = 255
            if b > 255 : b = 255

            #처리된 RGB값을 픽셀에 할당
            adjusted[row][col][0] = r
            adjusted[row][col][1] = g
            adjusted[row][col][2] = b

    return adjusted

#테스트 데이터 불러오기
X=[]
Y=[]
#전처리 과정 x
for i in range(0,1):
    # path='C:\\Users\\wkdeh\\Pictures\\TestPhoto\\n'+str(i)+'.jpg'
    path = 'C:\\Users\\wkdeh\\Pictures\\TestModel2\\selfie.png'
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    resized=cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imshow('resize',resized)
    cv2.waitKey(0)
    X.append(resized/255)
#전처리 과정 ㅇ
# for i in range(0,4):
#     path='C:\\Users\\wkdeh\\Pictures\\TestModel2\\' + str(i) + '.jpg'
#     img=cv2.imread(path, cv2.IMREAD_COLOR)
#     img = change_brightness(img, 200)
#
#     #균일화
#     # convert from RGB color-space to YCrCb
#     ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     # equalize the histogram of the Y channel
#     ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
#     # convert back to RGB color-space from YCrCb
#     equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
#
#     # equalized_img=cv2.bilateralFilter(equalized_img,5,20,20)
#     # cv2.imwrite('C:\\Users\\wkdeh\\Pictures\\TestModel2\\a' + str(i) + '.jpg', equalized_img)
#     X.append(equalized_img/255)




#테스트
test1=np.array(X)
answer1=model.predict(test1)
print(answer1)
for i in range(0,1):
    ans1=''
    if(np.argmax(answer1[i])==0):
        ans1='mask'
    elif(np.argmax(answer1[i])==1):
        ans1='nomask'
    elif (np.argmax(answer1[i]) == 2):
        ans1 = 'wrongmask'
    print(str(i+1) + '번째:',ans1)