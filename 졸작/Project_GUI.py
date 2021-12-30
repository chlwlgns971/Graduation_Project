#서버측(PC) Gui코드

import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #로그레벨(쓸데없는 오류 뜨는걸 안보이게 처리)
import socket
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import datetime

uiclass=uic.loadUiType("Project_GUI.ui")[0]
model=load_model('C:\\Users\\wkdeh\\Desktop\\강의자료\\졸업작품\\졸작\\MaskCheckModel1012.hdf5')

HOST=""
PORT=8123
text_count=0
s=""
conn=""

#QThread 클래스 선언하기, QThread 클래스를 쓰려면 QtCore 모듈을 import 해야함.
class Thread1(QThread): #초기화 메서드 구현
    def __init__(self, parent): #parent는 WndowClass에서 전달하는 self이다.(WidnowClass의 인스턴스)
     super().__init__(parent)
     self.parent = parent #self.parent를 사용하여 WindowClass 위젯을 제어할 수 있다.
    def run(self): #쓰레드로 동작시킬 함수 내용 구현
        global HOST
        global PORT
        global s
        global conn
        self.parent.pushButton_2.setEnabled(False)
        self.parent.pushButton_4.setEnabled(False)
        self.parent.radioButton.setEnabled(False)
        self.parent.radioButton_2.setEnabled(False)

        # TCP 사용
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(s)
        print('Socket created')

        # 서버의 아이피와 포트번호 지정
        s.bind((HOST, PORT))
        print('Socket bind complete')
        print(s)
        # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
        s.listen(10)
        print('Socket now listening')

        # socket에서 수신한 버퍼를 반환하는 함수
        def recvall(sock, count):
            # 바이트 문자열
            buf = b''
            while count:
                newbuf = sock.recv(count)
                if not newbuf: return None
                buf += newbuf
                count -= len(newbuf)

            return buf

        # 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
        conn, addr = s.accept()
        if not self.parent.started:
            self.parent.started = True
        while self.parent.started:
            msg_sd = ""
            current_time = ""
            result = ""

            # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
            try:
                length = recvall(conn, 16)
                stringData = recvall(conn, int(length))
                data = np.fromstring(stringData, dtype='uint8')
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except:
                print("연결이 끊겼습니다.")
                break

            face = frame
            face = face / 256
            face_input = cv2.resize(face, (256, 256))
            face_input = np.expand_dims(face_input, axis=0)
            face_input = np.array(face_input)

            if np.argmax(model.predict(face_input)) == 0:
                cv2.putText(frame, "Mask", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)  # 얼굴 번호 출력
                current_time = datetime.datetime.now()
                result = (str(current_time.replace(microsecond=0)) + " Mask")
                msg_sd = "Mask"
                conn.send((msg_sd).encode())
            elif np.argmax(model.predict(face_input)) == 1:
                cv2.putText(frame, "NoMask", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)  # 얼굴 번호 출력
                current_time = datetime.datetime.now()
                result = (str(current_time.replace(microsecond=0)) + " NoMask")
                msg_sd = "NoMask"
                conn.send((msg_sd).encode())
            elif np.argmax(model.predict(face_input)) == 2:
                cv2.putText(frame, "WrongMask", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)  # 얼굴 번호 출력
                current_time = datetime.datetime.now()
                result = (str(current_time.replace(microsecond=0)) + " WrongMask")
                msg_sd = "WrongMask"
                conn.send((msg_sd).encode())
            self.parent.textBrowser.append(result)
            

#ui호출용 Class 생성
class GUIWindow(QMainWindow,uiclass):
    started=False
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.radioButton.setChecked(True)
        self.setWindowTitle("Detection Mask")
        self.setWindowIcon(QIcon("camera_icon.png"))

        #버튼 클릭했을 때 이벤트 연결
        #버튼1: ip주소 입력 확인버튼
        #버튼2: ip주소 삭제버튼
        #라디오버튼1: 포트8123
        #라디오버튼2: 포트4321
        #버튼4: 소켓통신 시작버튼
        #버튼5: 통신해제버튼
        #버튼6: 마스크판별기록 저장버튼
        #버튼7: 마스크판별기록 삭제버튼
        self.pushButton.clicked.connect(self.btn1Clicked)
        self.pushButton_2.clicked.connect(self.btn2Clicked)
        self.radioButton.pressed.connect(self.rbtPressed)
        self.radioButton_2.pressed.connect(self.rbt2Pressed)
        self.pushButton_4.clicked.connect(self.btn4Clicked)
        self.pushButton_5.clicked.connect(self.btn5Clicked)
        self.pushButton_6.clicked.connect(self.btn6Clicked)
        self.pushButton_7.clicked.connect(self.btn7Clicked)

    def btn1Clicked(self):
        global HOST
        reply = QMessageBox.question(self, 'Message', 'Is your ip address ('+self.textEdit.toPlainText()+') right?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            HOST = self.textEdit.toPlainText()
            self.textEdit.setEnabled(False)
            self.pushButton.setEnabled(False)
            print('HOST:'+HOST)
        else:
            self.textEdit.clear()

    def btn2Clicked(self):
        global HOST
        HOST=""
        self.textEdit.setEnabled(True)
        self.pushButton.setEnabled(True)
        
    def rbtPressed(self):
        global PORT
        PORT=8123

    def rbt2Pressed(self):
        global PORT
        PORT=4321

    def btn4Clicked(self):
        h1 = Thread1(self)
        h1.start()

    def btn5Clicked(self):
        global s
        global conn
        if self.started:
            self.started = False  # 소켓통신에서 self.started가 True일 때 무한루프이므로 False로 만들어 루프 탈출
        self.pushButton_2.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.radioButton.setEnabled(True)
        self.radioButton_2.setEnabled(True)
        s.close()
        conn.close()

    def btn6Clicked(self):
        global text_count  # 전역변수를 지역변수처럼 사용할 수 있게 해줌
        text = str(self.textBrowser.toPlainText())
        text_count = 1
        save_time = datetime.datetime.now()
        # st = str(save_time.replace(microsecond=0)) #st=save_time=현재시간(초단위까지)
        with open(save_time.strftime("%Y-%m-%d_%H%M") + '.txt', 'w') as file:
            file.write(text)
        file.close()


    def btn7Clicked(self):
        global text_count #전역변수를 지역변수처럼 사용할 수 있게 해줌
        if text_count==0:
            reply = QMessageBox.question(self, 'Message', 'The record has not been saved.\nAre you sure to quit?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                print("삭제합니다.")
                self.textBrowser.clear()
            else:
                print("취소합니다.")
        else:
            print("삭제합니다.")
            self.textBrowser.clear()
            text_count=0

if __name__=="__main__":
    app=QApplication(sys.argv)
    myApp=GUIWindow()
    myApp.show()
    app.exec_()