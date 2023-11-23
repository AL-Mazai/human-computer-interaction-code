import cv2
import numpy as np
import mediapipe as mp

# 初始化FaceMesh 模块
mp_face_mesh = mp.solutions.face_mesh # 人脸网格模型
mp_drawing = mp.solutions.drawing_utils # 用于绘制关键点和连接线
mp_drawing_styles = mp.solutions.drawing_styles
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# 计算眼睛纵横比(EAR)的函数
def calculate_ear(land_marks, eye_index):
    left_eye_pts = np.array([land_marks[i - 1].x for i in eye_index])
    right_eye_pts = np.array([land_marks[i - 1].x for i in eye_index])

    # 计算眼睛的宽度和高度
    left_eye_width = np.linalg.norm(left_eye_pts[3] - left_eye_pts[0])
    left_eye_height = np.linalg.norm(left_eye_pts[1] - left_eye_pts[5])

    right_eye_width = np.linalg.norm(right_eye_pts[3] - right_eye_pts[0])
    right_eye_height = np.linalg.norm(right_eye_pts[1] - right_eye_pts[5])

    # 计算左眼和右眼的EAR
    left_eye_ear = left_eye_height / left_eye_width
    right_eye_ear = right_eye_height / right_eye_width

    return (left_eye_ear + right_eye_ear) / 2

# 获取所有左眼特征点的索引
all_left_eye_index = list(mp_face_mesh.FACEMESH_LEFT_EYE)
all_left_eye_index = set(np.ravel(all_left_eye_index))
# 获取所有右眼特征点的索引
all_right_eye_index = list(mp_face_mesh.FACEMESH_RIGHT_EYE)
all_right_eye_index = set(np.ravel(all_right_eye_index))

# 获取所有特征点的索引
all_index = all_left_eye_index.union(all_right_eye_index)

# 选择用于计算眼睛EAR的特征点索引
chosen_left_eye_index = [362, 385, 387, 263, 373, 380]
chosen_right_eye_index = [33, 160, 158, 133, 153, 144]

all_chosen_index = chosen_left_eye_index + chosen_right_eye_index

# 打开摄像头（默认摄像头）
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，可以根据需要更改摄像头索引

# 使用FaceMesh 模块处理图像
with mp_face_mesh.FaceMesh(
        static_image_mode=True,  # 静态图像模式
        max_num_faces=1,  # 最多处理1张脸
        refine_landmarks=True,  # 优化关键点的精度
        min_detection_confidence=0.5,  # 检测置信度阈值
        min_tracking_confidence=0.5,  # 跟踪置信度阈值
) as face_mesh:
    while True:
        # 读取摄像头帧
        success, image = cap.read()
        if not success:
            print("摄像头未启动成功！")
            break

        # 为了提高性能，将图像标记为不可写入，以通过引用传递。
        image.flags.writeable = False
        # # 将图像从BGR格式转换为RGB格式
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # 将图像转换为连续数组以供后续处理
        image = np.ascontiguousarray(image)
        # 获取图像的高度和宽度
        imgH, imgW, _ = image.shape
        # 处理图像
        results = face_mesh.process(image)

        # 定义用于绘制图像的函数
        def plot_pic(
                *,
                img_dt,
                img_eye_lmks=None,
                img_eye_lmks_chosen=None,
                face_landmarks=None,
                ts_thickness=1,
                ts_circle_radius=2,
                lmk_circle_radius=1,
                name="1",
        ):
            # 复制输入图像以进行绘制
            img_eye_lmks_chosen = img_dt.copy() if img_eye_lmks_chosen is None else img_eye_lmks_chosen

            # 获取面部特征点列表
            landmarks = face_landmarks.landmark

            # 遍历所有特征点
            for landmark_idx, landmark in enumerate(landmarks):
                # 绘制选择的特征点
                if landmark_idx in all_chosen_index:
                    pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
                    cv2.circle(img_eye_lmks_chosen, pred_cord, lmk_circle_radius, (255, 255, 255), -1)

            # 水平翻转图像以进行自拍视图显示
            cv2.imshow('sleepDetect', cv2.flip(img_eye_lmks_chosen, 1))

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                plot_pic(img_dt=image.copy(), face_landmarks=face_landmarks)

                landmarks = face_landmarks.landmark
                left_eye_ear = calculate_ear(landmarks, chosen_left_eye_index)
                right_eye_ear = calculate_ear(landmarks, chosen_right_eye_index)
                avg_ear = (left_eye_ear + right_eye_ear) / 2.0

                if avg_ear < 0.48:
                    print(avg_ear)
                    print("警告......您睡着了....")
                else:
                    closed_eyes_counter = 0

        if cv2.waitKey(5) & 0xFF == 27:  # 按 'Esc' 键退出循环
            break

cap.release()
cv2.destroyAllWindows()
