import cv2
import mediapipe as mp

# 初始化Mediapipe的手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 存储五个指尖点
key_point = []

# 初始化摄像头捕获
cap = cv2.VideoCapture(0)

# 使用Mediapipe库的手部模型，初始化手部检测
with mp_hands.Hands(
        model_complexity=0,  # 模型复杂度（0表示基本模型）
        min_detection_confidence=0.5,  # 最小检测置信度
        min_tracking_confidence=0.5) as hands:
    while True:
        # 初始化0关键点的坐标
        lst = [0, 0, 0]
        # 初始化距离字典
        distanceDict = {}

        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理图像，检测手部
        results = hands.process(imgRGB)

        scissors_dis = 0  # 剪刀
        # 如果检测到手部关键点
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # id代表21个不同结点，lm为每个结点的xyz坐标
                for id, lm in enumerate(handLms.landmark):
                    # 存储0关键点的三个坐标（手掌跟）
                    if id == 0:
                        lst = [round(lm.x, 2), round(lm.y, 2), round(lm.z, 2)]
                    if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                        row = [round(lm.x, 2), round(lm.y, 2), round(lm.z, 2)]
                        key_point.append(row)
                    # 绘制手部关键结点
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

                # 标注手部关键点和连接线
                mp_drawing.draw_landmarks(
                    img,
                    handLms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        print(key_point)
        distance = 0
        # 计算指尖到手掌跟的距离
        for triple in key_point:
            distance += (lst[0] - triple[0]) ** 2 + (lst[1] - triple[1]) ** 2 + (lst[2] - triple[2]) ** 2

        print('dis', distance)

        # 通过距离来判断手势
        if 0.1 < distance < 0.25:
            command = 'shitou'
        elif 0.45 < distance < 0.6:
            command = 'jiandao'
        elif 0.8 < distance < 1.1:
            command = 'bu'
        else:
            command = 'unKnown'

        cv2.putText(img, command, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("MediaPipe Hands", img)

        # 重置数据
        key_point.clear()

        # 如果按下"Esc"键，退出循环
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
