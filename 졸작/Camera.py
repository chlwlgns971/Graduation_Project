#오픈소스 dnn기반 사람얼굴인식 코드활용
#사용한 dnn모델:ResNet(Residual Neural Network)
#출처:https://blog.naver.com/dldudcks1779/222180883779

#PC에서 웹캠을 이용해 실시간 영상 촬영 및 모델구동 테스트 코드

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #로그레벨(쓸데없는 오류 뜨는걸 안보이게 처리)
#from keras.models import load_model #gpu환경에선 안돌아감
from tensorflow.keras.models import load_model
from imutils.video import FPS
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # opencv 모듈

model=load_model('C:\\Users\\wkdeh\\Desktop\\강의자료\\졸업작품\\졸작\\MaskCheckModel1004.hdf5')

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser()  # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로")  # 비디오 저장 경로
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# 얼굴 인식 모델 로드
print("[얼굴 인식 모델 로딩]")
face_detector = "C:\\Users\\wkdeh\\Desktop\\graduateWorkData\\" #네트워크파일 경로
prototxt = face_detector + "deploy.prototxt"  # prototxt 파일 : 모델의 레이어 구성 및 속성 정의
weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"  # caffemodel 파일 : 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는 SSD(Single Shot Detector) 프레임워크를 통해 사전 훈련된 모델 가중치 사용
net = cv2.dnn.readNet(prototxt, weights)  # cv2.dnn.readNet() : 네트워크를 메모리에 로드

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# 인식할 최소 확률
minimum_confidence = 0.5

writer = None

# fps 정보 초기화
fps = FPS().start()

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()

    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break

    # 프레임 resize
    frame = imutils.resize(frame, width=400)

    # 이미지 크기
    (H, W) = frame.shape[:2]

    # blob 이미지 생성
    # 파라미터
    # 1) image : 사용할 이미지
    # 2) scalefactor : 이미지 크기 비율 지정
    # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 얼굴 인식
    net.setInput(blob)  # setInput() : blob 이미지를 네트워크의 입력으로 설정
    detections = net.forward()  # forward() : 네트워크 실행(얼굴 인식)

    # 얼굴 번호
    number = 0
    # 마스크 번호
    nnum=0
    # 잘못된 마스크 번호
    wnum=0

    # 얼굴 인식을 위한 반복
    for i in range(0, detections.shape[2]):
        # 얼굴 인식 확률 추출
        confidence = detections[0, 0, i, 2]

        # 얼굴 인식 확률이 최소 확률보다 큰 경우
        if confidence > minimum_confidence:
            # bounding box 위치 계산
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # bounding box 가 전체 좌표 내에 있는지 확인
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(W - 1, endX), min(H - 1, endY))

            try:
                time.sleep(1)
                face = frame[startY-30:endY+5, startX-20:endX+20] #기존 좌표가 얼굴이 너무 짤려서 테스트에 적합한 사이즈로 변경
                face = face / 255
                face_input = cv2.resize(face, (256, 256))
                # cv2.imshow("trainfile", face_input)
                # cv2.waitKey(0)
                face_input = np.expand_dims(face_input, axis=0)
                face_input = np.array(face_input)

            except Exception as e:
                #print(str(e))
                print('')


            if np.argmax(model.predict(face_input))==0:
                cv2.putText(frame, "Mask[{}]".format(nnum + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)  # 얼굴 번호 출력
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # bounding box 출력
            elif np.argmax(model.predict(face_input))==1:
                cv2.putText(frame, "NoMask[{}]".format(nnum + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)  # 얼굴 번호 출력
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)  # bounding box 출력
            elif np.argmax(model.predict(face_input))==2:
                cv2.putText(frame, "WrongMask[{}]".format(wnum + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)  # 얼굴 번호 출력
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)  # bounding box 출력

            number = number + 1  # 얼굴 번호 증가
            nnum=nnum+1 #마스크 번호 증가
            wnum=wnum+1 #잘못된 마스크 번호 증가
        #얼굴 정면을 쳐다보고 있지 않으면 정면을 응시하라는 문구 출력
        # elif (confidence<minimum_confidence) :
        #     cv2.putText(frame, "Look straight at the camera.", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (0, 0, 255), 2)
        #수정사항: frame 박스가 없을 때 이 문구가 출력되게 하면 어떨까?
        # if frame is None:
        #     cv2.putText(frame, "Look straight at the camera.", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (0, 0, 255), 2)


    # 프레임 resize
    frame = imutils.resize(frame, width=500)

    # 프레임 출력
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break

    # fps 정보 업데이트
    fps.update()

    # output video 설정
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # 비디오 저장
    if writer is not None:
        writer.write(frame)

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
vs.release()
cv2.destroyAllWindows()
