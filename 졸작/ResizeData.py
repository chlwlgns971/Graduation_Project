#학습데이터를 일괄적으로 같은 사이즈로 리사이징 시키는 코드

import numpy as np
import cv2
# #한글경로
# for i in range(0,282):
#     full_path="C:\\Users\\wkdeh\\Desktop\\강의자료\\졸업작품\\자료\\mask(1)\\" + str(i) + ".jpg"
#     path_array1=np.fromfile(full_path,np.uint8)
#     img = cv2.imdecode(path_array1, cv2.IMREAD_COLOR)
#     resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#     # cv2.imshow('ex',resized)
#     # cv2.waitKey
#     cv2.imwrite("C:\\Users\\wkdeh\\ResizedPhoto\\trainm" + str(i+718) + ".jpg", resized)

#한글경로x
for i in range(0,12):
    path = 'C:\\Users\\wkdeh\\Pictures\\data\\' + str(i+1) + '.jpg'
    # path='C:\\Users\\wkdeh\\Pictures\\TestModel2\\selfie.png'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    # cv2.imshow('ex',resized)
    # cv2.waitKey(0)
    cv2.imwrite('C:\\Users\\wkdeh\\Pictures\\data\\n' + str(i+958) + '.jpg',resized)
