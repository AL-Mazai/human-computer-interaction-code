import cv2
import mediapipe as mp

# 导入MediaPipe的绘图和Holistic模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 使用Holistic模型进行静态图片的姿势和面部关键点检测
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    # 读取图片
    image = cv2.imread(r'images/6-1.png')
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 在图片上画面部、左右手、身体关节点
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()
    )
    mp_drawing.draw_landmarks(
        annotated_image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        annotated_image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS
    )

    # 保存带有标注的图片
    cv2.imwrite('images/result.png', annotated_image)
