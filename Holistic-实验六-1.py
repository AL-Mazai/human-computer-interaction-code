import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 打开摄像头
cap = cv2.VideoCapture(0)

# 使用Holistic模型进行姿势和面部关键点检测
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        # 读取摄像头帧
        success, image = cap.read()
        if not success:
            print("忽略空的摄像头帧.")
            continue

        # 将图像设置为只读模式
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # 将图像设置为可写模式
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 画面部关键点
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        # 画身体关键点和连接
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        # 显示图像
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

        # 按下ESC键退出循环
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 释放摄像头
cap.release()
