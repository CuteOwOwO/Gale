import cv2
import mediapipe as mp
import numpy as np

# --- 設定區 ---
VIDEO_PATH = 'old.mp4'  # 老師的影片
TARGET_FRAME = 268        # 挑戰的幀數
VISIBILITY_THRESHOLD = 0.5 # 信心度門檻 (低於此值視為沒抓到)
# -------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 定義要評分的部位 (10組向量)
BODY_PARTS_VECTORS = [
    (11, 13), (13, 15), # 左手 (肩-肘, 肘-腕)
    (12, 14), (14, 16), # 右手
    (11, 23), (12, 24), # 軀幹 (左邊, 右邊)
    (23, 25), (25, 27), # 左腿 (臀-膝, 膝-踝)
    (24, 26), (26, 28)  # 右腿
]

def get_landmarks_from_video_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = cap.read()
    landmark_list = None
    if success:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                landmark_list = results.pose_landmarks.landmark
    cap.release()
    return landmark_list

def calculate_cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def compare_poses_strict(user_landmarks, target_landmarks):
    """ 
    嚴格評分模式：
    如果肢體沒被定位出來 (Visibility 低)，直接該部位給 0 分。
    分母固定為 BODY_PARTS_VECTORS 的總數。
    """
    total_score = 0
    missing_parts = 0 # 記錄有幾個部位沒抓到
    
    # 分母固定：我們就是要檢查這幾樣東西
    total_parts = len(BODY_PARTS_VECTORS) 
    
    for idx_a, idx_b in BODY_PARTS_VECTORS:
        # 1. 檢查使用者可見度 (User Visibility)
        vis_a = user_landmarks[idx_a].visibility
        vis_b = user_landmarks[idx_b].visibility
        
        # 如果兩點都清晰可見，才進行比對
        if vis_a > VISIBILITY_THRESHOLD and vis_b > VISIBILITY_THRESHOLD:
            # 老師的向量
            t_a = np.array([target_landmarks[idx_a].x, target_landmarks[idx_a].y])
            t_b = np.array([target_landmarks[idx_b].x, target_landmarks[idx_b].y])
            target_vec = t_b - t_a

            # 學生的向量
            u_a = np.array([user_landmarks[idx_a].x, user_landmarks[idx_a].y])
            u_b = np.array([user_landmarks[idx_b].x, user_landmarks[idx_b].y])
            user_vec = u_b - u_a
            
            # 計算分數
            score = calculate_cosine_similarity(target_vec, user_vec)
            score = max(0, score) # 確保不扣成負分
            total_score += score
        else:
            # --- 懲罰區 ---
            # 沒抓到，這個部位得分就是 0，且計入缺漏
            missing_parts += 1
            # print(f"部位 {idx_a}-{idx_b} 遺失，扣分！") 

    # 最終分數 = 總得分 / 總部位數 (包含沒抓到的)
    final_score = (total_score / total_parts) * 100
    
    return final_score, missing_parts

def main_static_showdown():
    print(f"正在讀取 Frame {TARGET_FRAME} 的標準骨架...")
    target_landmarks = get_landmarks_from_video_frame(VIDEO_PATH, TARGET_FRAME)
    
    if not target_landmarks:
        print("錯誤：無法讀取目標幀的骨架。")
        return

    print("啟動 Webcam... (嚴格模式：肢體出鏡會扣分)")
    cap = cv2.VideoCapture(0) 
    window_name = f"Strict Showdown: Frame {TARGET_FRAME}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            current_score = 0
            missing_count = 0
            
            if results.pose_landmarks:
                # 使用嚴格評分函數
                current_score, missing_count = compare_poses_strict(
                    results.pose_landmarks.landmark, 
                    target_landmarks
                )
                
                # 顏色邏輯
                if current_score > 85:
                    pose_color = (0, 255, 0) # 綠
                    status = "Perfect!"
                elif current_score > 60:
                    pose_color = (0, 255, 255) # 黃
                    status = "Okay"
                else:
                    pose_color = (0, 0, 255) # 紅
                    status = "Bad"

                # 畫骨架
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2)
                )

                # 顯示評語
                cv2.putText(image, status, (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, pose_color, 3)

                # --- 如果有部位缺失，顯示警告 ---
                if missing_count > 0:
                    cv2.putText(image, f"MISSING PARTS: {missing_count}", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "Please show full body!", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # UI 顯示
            cv2.rectangle(image, (50, 50), (450, 100), (50, 50, 50), -1)
            cv2.putText(image, f"Score: {int(current_score)}%", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            bar_width = int((current_score / 100) * 400)
            bar_color = (0, 0, 255)
            if current_score > 60: bar_color = (0, 255, 255)
            if current_score > 85: bar_color = (0, 255, 0)
            cv2.rectangle(image, (50, 110), (50 + bar_width, 130), bar_color, -1)
            cv2.rectangle(image, (50, 110), (450, 130), (255, 255, 255), 2)

            cv2.imshow(window_name, image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_static_showdown()