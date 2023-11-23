import cv2
import mediapipe as mp
import numpy as np

# 导入绘制姿势标注的工具模块
mp_drawing = mp.solutions.drawing_utils
# 导入姿势标注的绘制样式模块
mp_drawing_styles = mp.solutions.drawing_styles
# 导入姿势估计模块
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)  # gray
# BG_COLOR = (0, 0, 255)  # red

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        enable_segmentation=True,  # 启用分割（背景替换）
        min_detection_confidence=0.5,  # 最小检测置信度
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 为了提高性能，可以将图像标记为不可写，以通过引用传递
        image.flags.writeable = False
        # 将图像从BGR颜色空间转换为RGB颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 对图像进行姿势处理和分析
        results = pose.process(image)

        # 在图像上绘制姿势标注
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 创建一个布尔条件，根据分割掩码将人体部分保留在图像中，其他部分用背景颜色进行替换
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # 创建一个与图像相同形状的全黑图像作为背景图像
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        # 将背景图像的所有像素值设置为指定的背景颜色
        # bg_image[:] = BG_COLOR
        # 读取背景图像
        bg_image = cv2.imread("images/123123.jpg")
        # 调整背景图像的大小与原始图像相同
        bg_image = cv2.resize(bg_image, image.shape[:2][::-1])

        # 根据条件，将图像中满足条件的像素值替换为图像本身，不满足条件的像素值替换为背景图像的像素值
        image = np.where(condition, image, bg_image)

        # 在图像上绘制姿势标注
        mp_drawing.draw_landmarks(
            image,  # 要在其上绘制姿势标注的图像
            results.pose_landmarks,  # 检测到的姿势关键点信息
            mp_pose.POSE_CONNECTIONS,  # 姿势关键点之间的连接关系
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()  # 姿势标注的绘制样式
        )

        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
