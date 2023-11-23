import cv2
import mediapipe as mp
import numpy as np

# 导入绘图工具和样式
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# # 导入姿势相关的模块，创建姿势估计模型对象，并设置为静态图像模式
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 计算两个向量之间的夹角（以度为单位）
def get_angle(v1, v2):
    # 计算两个向量的点积，并除以两个向量的模的乘积，得到夹角的余弦值
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    # 将余弦值转换为弧度，并乘以180/3.14转换为角度
    angle = np.arccos(angle) / 3.14 * 180
    # 计算两个向量的叉乘，并判断叉乘的方向
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    # 返回计算得到的夹角
    return angle

# 姿势识别：主要对举左手、举右手、举双手进行识别
def get_pos(keypoints):
    str_pose = ""
    # 计算左臂与水平方向的夹角
    keypoints = np.array(keypoints)
    v1 = keypoints[12] - keypoints[11]
    v2 = keypoints[13] - keypoints[11]
    angle_left_arm = get_angle(v1, v2)
    # 计算右臂与水平方向的夹角
    v1 = keypoints[11] - keypoints[12]
    v2 = keypoints[14] - keypoints[12]
    angle_right_arm = get_angle(v1, v2)

    #结果
    if angle_left_arm < 0 and angle_right_arm < 0:
        str_pose = "left_hand_up"
    elif angle_left_arm > 0 and angle_right_arm > 0:
        str_pose = "right_hand_up"
    elif angle_left_arm < 0 < angle_right_arm:
        str_pose = "all_hands_up"
    elif angle_left_arm > 0 > angle_right_arm:
        str_pose = "other"
    return str_pose

# 处理摄像头拍摄的视频，这里以帧为单位进行处理
def process_frame(img):
    # 高和宽
    h, w = img.shape[0], img.shape[1]
    # BRG-->RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取 关键点 预测结果
    results = pose.process(img_RGB)
    keypoints = ['' for i in range(33)]
    if results.pose_landmarks:
        # 绘制姿势关键点和连接线
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        for i in range(33):
            # 获取关键点的相对坐标并转换为绝对坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)  # 关键点在图像中的x坐标
            cy = int(results.pose_landmarks.landmark[i].y * h)  # 关键点在图像中的y坐标
            # 将33个关键点的坐标保存到keypoints列表中
            keypoints[i] = (cx, cy)

    # 获取姿态
    str_pose = get_pos(keypoints)
    cv2.putText(img, "current pose: {}".format(str_pose), (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)
    return img

# 识别结果展示
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # frame就是视频截取的帧，process_frame表示对其检测并展示
        frame = process_frame(frame)
        cv2.imshow("frame", frame)
        # 退出
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
