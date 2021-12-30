#라즈베리파이 동작코드(영상 촬영중 사람 얼굴이 인식되면 사진을 찍어 서버(PC)로 전송하고 서버에서 판별값을 받아 LED로 결과출력
# 초록Led: 마스크착용, 노랑Led: 마스크 오착용(코스크, 턱스크), 빨강Led: 마스크 미착용


# -*- coding: utf8 -*-
import cv2
import socket
import numpy as np
from keras.models import load_model
import RPi.GPIO as GPIO
import time

led_r = 13
led_g = 19
led_b = 26

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(led_r, GPIO.OUT)
GPIO.setup(led_g, GPIO.OUT)
GPIO.setup(led_b, GPIO.OUT)


def led_off():
    GPIO.output(led_r, GPIO.LOW)
    GPIO.output(led_g, GPIO.LOW)
    GPIO.output(led_b, GPIO.LOW)


def led_on(rgb):
    if rgb == "R":
        GPIO.output(led_r, GPIO.HIGH)
    if rgb == "G":
        GPIO.output(led_g, GPIO.HIGH)
    if rgb == "Y":
        GPIO.output(led_r, GPIO.HIGH)
        GPIO.output(led_g, GPIO.HIGH)


## TCP 사용
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
## server ip, port
# my pc
# c.connect(('192.168.10.2', 8484))
# jahoon pc
# c.connect(('121.155.160.92' ,8123))
# jahoon pc my room
c.connect(('192.168.10.6', 8123))
# jahoon pc hotspot
# c.connect(('192.168.43.203', 8123))
# CBNU
# c.connect(('203.255.73.209', 8123))


## webcam 이미지 capture
cam = cv2.VideoCapture(0)
## 이미지 속성 변경 3 = width, 4 = height
cam.set(3, 800);
cam.set(4, 480);
## 0~100에서 90의 이미지 품질로 설정 (default = 95)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

model = load_model('DNN/FaceModel1022.hdf5') #얼굴인식모델 경로

led_off()  # while문 시작전에 led 초기화
# count to check 1/n frame
count = 1
# CFS : 얼굴이 인식된 한 프레임만 보내기 위한 변수
CFS = 0
while cv2.waitKey(33) < 0:  # 스페이스바 누르면 종료
    msg_rcv = ""
    count += 1
    # 비디오의 한 프레임씩 읽는다.
    # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
    ret, frame = cam.read()
    cv2.rectangle(frame, (250, 90), (550, 390), (0, 0, 255), 2)  # bounding box출력 - red
    cv2.putText(frame, "Fill in the square with your face", (140, 440), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    frame_box = frame[90:390, 250:550]
    if not count % 8 == 0:
        continue

    face = frame_box
    face = face / 256
    face_input = cv2.resize(face, (256, 256))
    face_input = np.expand_dims(face_input, axis=0)
    face_input = np.array(face_input)

    if np.argmax(model.predict(face_input)) == 0:
        cv2.rectangle(frame, (250, 90), (550, 390), (0, 255, 0), 2)  # bounding box출력 - green
        cv2.putText(frame, "Fill in the square with your face", (140, 440), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        if CFS == 4:  # 5번째의 Face가 인식된 프레임을 서버로 송신
            # cv2. imencode(ext, img [, params])
            # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
            result, frame_box = cv2.imencode('.jpg', frame_box, encode_param)
            # frame을 String 형태로 변환
            data = np.array(frame_box)
            # stringData = data.tostring() (오류 때문에 byte로 사용)
            stringData = data.tobytes()

            try:
                # 서버에 데이터 전송
                # (str(len(stringData))).encode().ljust(16)
                c.sendall((str(len(stringData))).encode().ljust(16) + stringData)

                # 서버에서 반환받은 결과값
                msg_rcv = c.recv(1024)
                msg_rcv = (msg_rcv.decode())
                print(msg_rcv)

            except:
                c.close()
                print("연결이 끊겼습니다.")
                break

            if msg_rcv == "Mask":
                led_on('G')
            elif msg_rcv == "NoMask":
                led_on('R')
            elif msg_rcv == "WrongMask":
                led_on('Y')

    elif np.argmax(model.predict(face_input)) == 1:
        led_off()
        CFS = 0  # NoneFace가 감지될 경우 CFS를 초기화
    CFS += 1  # 0~4번째 Face에서는 대기

    # full screen in raspberry pi
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # fullscreen
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", frame)

# c.close()
GPIO.cleanup()
cam.release()
cv2.destroyAllWindows()