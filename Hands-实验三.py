import cv2
import mediapipe as mp

# 初始化Mediapipe库中的绘图工具和手部检测模型
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# 定义要处理的静态图像文件的列表
IMAGE_FILES = []


# 使用Mediapipe库的手部模型，初始化手部检测
with mp_hands.Hands(
    static_image_mode=True,  # 静态图像模式，用于处理静态图像
    max_num_hands=2,          # 最多检测2只手
    min_detection_confidence=0.5) as hands:

    # 遍历图像文件列表
    for idx, file in enumerate(IMAGE_FILES):
        # 读取图像并翻转以适应正确的手性
        image = cv2.flip(cv2.imread(file), 1)

        # 处理图像，检测手部
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 打印手的左右信息
        print('Handedness:', results.multi_handedness)

        # 如果未检测到手，继续下一个图像
        if not results.multi_hand_landmarks:
            continue

        image_height, image_width, _ = image.shape
        # 复制原图像用于标注
        annotated_image = image.copy()

        for hand_landmarks in results.multi_hand_landmarks:
            # 打印手部关键点坐标
            print('hand_landmarks:', hand_landmarks)

            # 打印食指指尖的坐标
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )

            # 标注手部关键点和连接线
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # 保存标注后的图像
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

        # 如果未检测到手部的世界坐标系关键点，继续下一个图像
        if not results.multi_hand_world_landmarks:
            continue

        for hand_world_landmarks in results.multi_hand_world_landmarks:
            # 绘制手部世界坐标系关键点
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# 初始化摄像头
cap = cv2.VideoCapture(0)


# 使用Mediapipe库的手部模型，初始化手部检测
with mp_hands.Hands(
    model_complexity=0,            # 模型复杂度（0表示基本模型）
    min_detection_confidence=0.5,  # 最小检测置信度
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        # 读取摄像头捕获的图像
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 处理图像，检测手部
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        # 将图像从RGB格式切换回BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 标注手部关键点和连接线
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 显示带有手部标注的图像
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # 如果按下"Esc"键，退出循环
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
