import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # pose对象 1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、是否分割，5、减少抖动，6、检测阈值，7、跟踪阈值
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, False, True, # 这里的False 和True为默认
                                     self.detectionCon, self.trackCon)
        '''
        STATIC_IMAGE_MODE：如果设置为 false，该解决方案会将输入图像视为视频流。它将尝试在第一张图像中检测最突出的人，并在成功检测后进一步定位姿势地标。在随后的图像中，它只是简单地跟踪那些地标，而不会调用另一个检测，直到它失去跟踪，以减少计算和延迟。如果设置为 true，则人员检测会运行每个输入图像，非常适合处理一批静态的、可能不相关的图像。默认为false。
        MODEL_COMPLEXITY：姿势地标模型的复杂度：0、1 或 2。地标准确度和推理延迟通常随着模型复杂度的增加而增加。默认为 1。
        SMOOTH_LANDMARKS：如果设置为true，解决方案过滤不同的输入图像上的姿势地标以减少抖动，但如果static_image_mode也设置为true则忽略。默认为true。
        UPPER_BODY_ONLY：是要追踪33个地标的全部姿势地标还是只有25个上半身的姿势地标。
        ENABLE_SEGMENTATION：如果设置为 true，除了姿势地标之外，该解决方案还会生成分割掩码。默认为false。
        SMOOTH_SEGMENTATION：如果设置为true，解决方案过滤不同的输入图像上的分割掩码以减少抖动，但如果 enable_segmentation设置为false或者static_image_mode设置为true则忽略。默认为true。
        MIN_DETECTION_CONFIDENCE：来自人员检测模型的最小置信值 ([0.0, 1.0])，用于将检测视为成功。默认为 0.5。
        MIN_TRACKING_CONFIDENCE：来自地标跟踪模型的最小置信值 ([0.0, 1.0])，用于将被视为成功跟踪的姿势地标，否则将在下一个输入图像上自动调用人物检测。将其设置为更高的值可以提高解决方案的稳健性，但代价是更高的延迟。如果 static_image_mode 为 true，则忽略，人员检测在每个图像上运行。默认为 0.5。
        '''
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换成灰度图片
        self.results = self.pose.process(imgRGB)  # 处理 RGB 图像并返回检测到的最突出人物的姿势特征点。
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # results.pose_landmarks画点 mpPose.POSE_CONNECTIONS连线
        return img


    def findPosition(self, img, draw = True):
    #print(results.pose_landmarks)
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):  # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                h, w, c = img.shape  # 返回图片的(高,宽,位深)

                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)  # lm.x  lm.y是比例  乘上总长度就是像素点位置
                lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # 画蓝色圆圈
        return lmList


def main():
    cap = cv2.VideoCapture(0)  # 调用电脑摄像头
    # cap = cv2.VideoCapture('video/2.mp4')  # 视频
    # cap = cv2.VideoCapture('video/3.png')
    # cap = cv2.VideoCapture('video/ASOUL.mp4')
    pTime = 0

    detector = poseDetector()
    while True:
        success, img = cap.read()  # 第一个参数代表有没有读取到图片True/False 第二个参数frame表示截取到一帧的图片  读进来直接是BGR 格式数据格式
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)  # print(lmList[n]) 可以打印第n个
        # 计算帧率
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)  # 图片上添加文字  参数：图片 要添加的文字 文字添加到图片上的位置 字体的类型 字体大小 字体颜色 字体粗细

        cv2.imshow("Image", img)  # 显示图片

        cv2.waitKey(3)  # 等待按键


if __name__ == "__main__":
    main()
