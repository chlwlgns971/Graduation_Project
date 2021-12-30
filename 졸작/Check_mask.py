#마스크 착용상태 판별 학습모델 코드

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #로그레벨(쓸데없는 오류 뜨는걸 안보이게 처리)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# tf.debugging.set_log_device_placement(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#학습데이터 불러오기
groups_folder_path1 = 'C:\\Users\\LSP\\CheckMask\\train_dataset'
categories = ['mask','nomask','wrongmask']
num_classes = len(categories)

X = []
Y = []

for idex, categorie in enumerate(categories):
    print(idex,categorie)
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path1+'\\'+ categorie + '\\'
    print(image_dir)

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            full_path=image_dir + filename
            img_array=np.fromfile(full_path,np.uint8) #한글경로를 읽지 못해 바이너리 데이터를 넘파이 행렬로 읽는다.
            img = cv2.imdecode(img_array,cv2.IMREAD_COLOR) #indecode함수로 복호화 해줌으로 opencv에서 사용할 수 있는 형태로 바꿔준다.
            # print(img) #데이터가 제대로 들어오는지 확인용
            # print(label)
            X.append(img/255) #정규화
            Y.append(label)

X_train = np.array(X)
Y_train = np.array(Y)

#테스트데이터 불러오기
groups_folder_path2 = 'C:\\Users\\LSP\\CheckMask\\test_dataset'
categories = ['mask','nomask','wrongmask']
num_classes = len(categories)

X = []
Y = []

for idex, categorie in enumerate(categories):
    print(idex, categorie)
    label = [0 for i in range(num_classes)]
    label[idex] = 1 #one-hot인코딩 (mask=[1,0,0], nomask=[0,1,0], wrongmask=[0,0,1])
    image_dir = groups_folder_path2+'\\'+ categorie + '\\'
    print(image_dir)

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            full_path=image_dir + filename
            img_array=np.fromfile(full_path,np.uint8) #한글경로를 읽지 못해 바이너리 데이터를 넘파이 행렬로 읽는다.
            img = cv2.imdecode(img_array,cv2.IMREAD_COLOR) #indecode함수로 복호화 해줌으로 opencv에서 사용할 수 있는 형태로 바꿔준다.
            # print(img) #데이터가 제대로 들어오는지 확인용
            # print(label)
            X.append(img/255) #정규화
            Y.append(label)

X_test = np.array(X)
Y_test = np.array(Y)

# #데이터들이 train변수에 잘 들어갔는지 확인
# cv2.imshow("trainfile",X_train[1])
# cv2.waitKey(0)
# cv2.imshow("testfile",X_test[1])
# cv2.waitKey(0)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#신경망 구성(5계층)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), padding='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3),  activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

#모델 최적화
modelpath='C:\\Users\\LSP\\CheckMask\\MaskCheckModel1012.hdf5' #학습된 모델 저장경로와 이름
checkpointer=ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1,save_best_only=True)
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=20)

#모델실행
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(num_classes,activation = 'softmax')) #마스크를 착용했을 확률을 구하기 위해 Softmax사용

model.summary() #배치사이즈가 클수록 학습시간이 오래걸리지만 결과가 좋게 수렴한다?
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#클래스가 늘어나면서 categorical_crossentropy 손실함수 사용 기존엔 마스크를 쓰고 안쓰고 경우가 2가지므로 loss함수로 binary사용
history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), batch_size=64, epochs=100, verbose=0, callbacks=[early_stopping_callback,checkpointer])
#history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), batch_size=128, epochs=100, verbose=0)

#테스트 정확도 출력
print('\n Test Accuracy: %.4f'%(model.evaluate(X_test,Y_test)[1]))

#그래프표현
y_loss=history.history['loss'] #학습셋의 오차
y_vloss=history.history['val_loss'] #테스트셋의 오차
x_len=np.arange(len(y_loss))
plt.plot(x_len,y_loss,marker='.',c='blue',label='Trainset_loss')
plt.plot(x_len,y_vloss,marker='.',c='red',label='Testset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()