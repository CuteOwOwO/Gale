import cv2
import mediapipe as mp
import numpy as np

# --- 📁 設定區 ---
VIDEO_PATH = 'old.mp4'   
TARGET_FRAME = 15         
VISIBILITY_THRESHOLD = 0.5 
# ---------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BODY_PARTS_VECTORS = [
    (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 25), (25, 27), 
    (24, 26), (26, 28)
]

def extract_reference_data(video_path, frame_number):
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
    return image, landmark_list

def calculate_cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def compare_poses_strict(user_landmarks, target_landmarks):
    total_score = 0
    missing_parts = 0
    total_parts = len(BODY_PARTS_VECTORS) 
    for idx_a, idx_b in BODY_PARTS_VECTORS:
        vis_a = user_landmarks[idx_a].visibility
        vis_b = user_landmarks[idx_b].visibility
        if vis_a > VISIBILITY_THRESHOLD and vis_b > VISIBILITY_THRESHOLD:
            t_vec = np.array([target_landmarks[idx_b].x - target_landmarks[idx_a].x,
                              target_landmarks[idx_b].y - target_landmarks[idx_a].y])
            u_vec = np.array([user_landmarks[idx_b].x - user_landmarks[idx_a].x,
                              user_landmarks[idx_b].y - user_landmarks[idx_a].y])
            score = max(0, calculate_cosine_similarity(t_vec, u_vec))
            total_score += score
        else:
            missing_parts += 1
    return (total_score / total_parts) * 100, missing_parts

def prepare_teacher_panel(teacher_img, target_height):
    """
    【關鍵優化】：這個函式只會執行一次！
    1. 計算縮放比例，讓老師的高度跟 Webcam 一樣高。
    2. 使用高品質插值 + 銳化濾鏡。
    """
    h, w, _ = teacher_img.shape
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height # 其實就是 int(h * scale)
    
    # 1. 高品質縮放 (INTER_AREA 抗鋸齒)
    resized_teacher = cv2.resize(teacher_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. 影像銳化 (讓細節更清楚)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    resized_teacher = cv2.filter2D(resized_teacher, -1, kernel)
    
    # 加個標題
    cv2.putText(resized_teacher, "Target Pose", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return resized_teacher

def main_split_screen():
    print("系統啟動：載入老師數據中...")
    ref_image, ref_landmarks = extract_reference_data(VIDEO_PATH, TARGET_FRAME)
    if ref_image is None or ref_landmarks is None:
        print("❌ 錯誤：無法讀取參考影像。")
        return

    cap = cv2.VideoCapture(0)
    
    # --- 第一次讀取 Webcam，取得高度資訊 ---
    success, frame = cap.read()
    if not success: return
    webcam_h, webcam_w, _ = frame.shape
    
    print("✅ 預處理：正在製作高畫質老師面板 (只做這一次)...")
    # *** 這裡就是你要的：直接把圖準備好，之後就不動了 ***
    teacher_panel = prepare_teacher_panel(ref_image, webcam_h)
    
    window_name = "Rehab Assistant (Split Screen)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 自動調整視窗大小 (寬度 = Webcam + 老師)
    total_w = webcam_w + teacher_panel.shape[1]
    cv2.resizeWindow(window_name, total_w, webcam_h)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1) # 鏡像
            
            # 轉 RGB 偵測
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            current_score = 0
            missing_count = 0
            pose_color = (200, 200, 200) 
            status = "Ready"

            if results.pose_landmarks:
                current_score, missing_count = compare_poses_strict(
                    results.pose_landmarks.landmark, ref_landmarks
                )
                
                # 顏色邏輯
                if current_score > 85:
                    pose_color = (0, 255, 0); status = "Perfect!"
                elif current_score > 60:
                    pose_color = (0, 255, 255); status = "Good"
                else:
                    pose_color = (0, 0, 255); status = "Adjust..."

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2)
                )

            # --- UI 繪製 (畫在 Webcam 畫面左側) ---
            # 因為右邊被老師佔據了，我們把分數放在左上角，更清楚
            cv2.rectangle(frame, (20, 20), (300, 100), (0, 0, 0), -1) # 黑底
            cv2.putText(frame, f"{int(current_score)}", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.6, pose_color, 3)
            cv2.putText(frame, "Score", (160, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, status, (160, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pose_color, 2)
            
            if missing_count > 0:
                 cv2.putText(frame, "BODY MISSING!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- 關鍵：左右拼接 (Horizontal Concatenation) ---
            # 這裡不需 resize，直接把準備好的 teacher_panel 接在右邊
            # 這運算極快，幾乎不消耗效能
            combined_view = cv2.hconcat([frame, teacher_panel])

            cv2.imshow(window_name, combined_view)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_split_screen()