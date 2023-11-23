import cv2
import mediapipe as mp

# 创建 MediaPipe 的人脸检测模块和绘图工具模块
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 要处理的静态图片文件的列表
IMAGE_FILES = []

# 使用人脸检测模块，选择模型并设置最小检测置信度
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
        # 读取图片文件
        image = cv2.imread(file)

        # 将 BGR 图像转换为 RGB，并使用 MediaPipe 人脸检测处理它
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 绘制每个检测到的人脸的边界框
        if not results.detections:
            continue

        annotated_image = image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# 打开摄像头并使用人脸检测模块进行实时人脸检测
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # 如果加载视频，使用 'break' 而不是 'continue'。
            continue

        # 为了提高性能，将图像标记为不可写，以通过引用传递
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # 在图像上绘制人脸检测的标注
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # 水平翻转图像以进行自拍显示
        cv2.imshow('face detection', cv2.flip(image, 1))

        # 按下 ESC 键退出循环
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
