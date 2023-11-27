import cv2
import time
import PoseModule as pm
import socket

pTime = 0

def computeFPS():
    global pTime  # 全局变量，用于存储上一帧的时间
    cTime = time.time()  # 获取当前时间
    fps = 1 / (cTime - pTime)  # 计算帧率
    pTime = cTime  # 更新上一帧的时间为当前时间
    # 图片上添加文字
    # 参数：图片 要添加的文字 文字添加到图片上的位置 字体的类型 字体大小 字体颜色 字体粗细
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

cap = cv2.VideoCapture(0)  # 调用电脑摄像头
# cap = cv2.VideoCapture('video/2.mp4')  # 视频

# 构建一个实例，去连接服务端的监听端口。
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #基于TCP
# client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #基于UDP
client.connect(('127.0.0.1', 9999))
# msg = client.recv(1024)
# print('New message from server: %s' % msg.decode('utf-8'))

detector = pm.poseDetector()
strdata = ""  # 定义字符串变量
while True:
    # success代表有没有读取到图片True/False
    # img表示截取到一帧的图片，读进来直接是BGR格式数据格式
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    #     print(lmList)
    if len(lmList) != 0:
        for data in lmList:
            print(data)  # print(lmList[n]) 可以打印第n个
            for i in range(1, 4):
                if i == 2:
                    strdata = strdata + str(img.shape[0] - data[i]) + ','
                else:
                    strdata = strdata + str(data[i]) + ','
        print(strdata)

        client.send(strdata.encode('utf-8'))
        strdata = ""

    computeFPS()  # 计算帧率
    cv2.imshow("Image", img)  # 显示图片

    # 按Esc键退出
    if cv2.waitKey(5) & 0xFF == 27:
        break

client.close()