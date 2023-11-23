# 导入必要的库
import cv2
import mediapipe as mp

# 初始化 MediaPipe 库的相关组件
mp_drawing = mp.solutions.drawing_utils  # 用于绘制关键点和连接线
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh  # 人脸网格模型

# 对于静态图像的处理:
IMAGE_FILES = []  # 存储图像文件的列表
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 使用 FaceMesh 模型处理静态图像
with mp_face_mesh.FaceMesh(
        static_image_mode=True,  # 静态图像模式
        max_num_faces=1,  # 最多处理一张脸
        refine_landmarks=True,  # 优化关键点的精度
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)  # 读取图像
        # 在处理之前，将图像从 BGR 格式转换为 RGB 格式
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 打印并在图像上绘制人脸关键点
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # 绘制网格连接线
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,  # 绘制轮廓连接线
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,  # 绘制虹膜连接线
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# 对于摄像头输入:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)  # 打开摄像头

with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # 最多处理一张脸
        refine_landmarks=True,  # 优化关键点的精度
        min_detection_confidence=0.5,  # 最小检测置信度
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()  # 读取摄像头帧
        if not success:
            print("Ignoring empty camera frame.")
            # 如果加载视频，请使用'break'而不是'continue'。
            continue

        # 为了提高性能，将图像标记为不可写入，以通过引用传递。
        image.flags.writeable = False
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)  # 处理图像

        # 在图像上绘制人脸网格注释
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # 绘制网格连接线
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,  # 绘制轮廓连接线
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,  # 绘制虹膜连接线
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        # 水平翻转图像以进行自拍视图显示。
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:  # 按 'Esc' 键退出循环
            break

cap.release()  # 释放摄像头
