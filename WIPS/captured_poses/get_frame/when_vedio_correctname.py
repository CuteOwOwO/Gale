# when_vedio.py
import cv2
import mediapipe as mp
import numpy as np
import os

# --- 設定區 ---
VIDEO_ID = "towel"                     # 影片 ID (會用在檔名上)
VIDEO_PATH = '../mp4_home/towel.mp4'      # 測試影片路徑
OUTPUT_DIR = "../UI_and_interface/static/frames"           # 直接存到前端讀取的位置
# 演算法參數
VELOCITY_THRESHOLD = 0.008  
COOLDOWN_FRAMES = 30        
SMOOTHING_FACTOR = 0.5      
# -------------

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[系統] 存檔資料夾已準備: {OUTPUT_DIR}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_pose_velocity(current_landmarks, prev_landmarks):
    """計算手腕與腳踝的平均移動速度"""
    if prev_landmarks is None:
        return 1.0
    
    key_indices = [15, 16, 27, 28] 
    total_movement = 0
    for idx in key_indices:
        c_x, c_y = current_landmarks[idx].x, current_landmarks[idx].y
        p_x, p_y = prev_landmarks[idx].x, prev_landmarks[idx].y
        dist = np.sqrt((c_x - p_x)**2 + (c_y - p_y)**2)
        total_movement += dist
    
    return total_movement / len(key_indices)

def save_captured_images(clean_img, overlay_img, frame_number):
    """
    統一存檔函式，使用影格號碼命名
    """
    # 組合檔名：例如 towel_45.jpg
    filename = f"{VIDEO_ID}_{frame_number}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # 這裡我們存「乾淨的原圖」，因為前端選關卡和右側對照圖不需要骨架線條
    cv2.imwrite(filepath, overlay_img)
    
    print(f"[✅ 存檔成功] 影格 {frame_number} -> {filepath}")
    return frame_number

def process_pose_extractor(video_source):
    cap = cv2.VideoCapture(video_source)
    
    window_name = 'Pose Capture Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 550)

    prev_landmarks_list = None 
    current_velocity = 1.0     
    cooldown_counter = 0      
    is_paused = False
    
    # 用來記錄抓到了哪些影格，最後可以印出來貼到 JSON 裡
    captured_frames = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if is_paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): is_paused = not is_paused
                elif key == ord('q'): break
                continue

            success, image = cap.read()
            if not success:
                print("影片結束")
                break
                
            # 取得目前的影格號碼 (Frame Number)
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 備份乾淨原圖 (重要！這張才是我們要存給前端用的)
            clean_image = image.copy()

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            skeleton_image = np.zeros(image.shape, dtype=np.uint8)
            is_pose_detected = False 

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                mp_drawing.draw_landmarks(
                    skeleton_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                landmarks = results.pose_landmarks.landmark
                raw_velocity = calculate_pose_velocity(landmarks, prev_landmarks_list)
                current_velocity = (raw_velocity * (1 - SMOOTHING_FACTOR)) + (current_velocity * SMOOTHING_FACTOR)
                prev_landmarks_list = landmarks

                # 判斷是否自動抓拍
                if current_velocity < VELOCITY_THRESHOLD and cooldown_counter == 0:
                    is_pose_detected = True
                    cooldown_counter = COOLDOWN_FRAMES
                    
                    # 存檔並記錄影格
                    saved_frame = save_captured_images(clean_image, image, current_frame_idx)
                    captured_frames.append(saved_frame)

                if cooldown_counter > 0:
                    cooldown_counter -= 1

            # UI 繪製
            bar_len = int(current_velocity * 5000)
            color = (0, 255, 0) if current_velocity < VELOCITY_THRESHOLD else (0, 0, 255)
            cv2.rectangle(image, (30, 30), (30 + bar_len, 50), color, -1)
            cv2.putText(image, f"Vel: {current_velocity:.4f}", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, f"Frame: {current_frame_idx}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if is_pose_detected:
                cv2.putText(image, "Auto Captured!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            combined_view = cv2.hconcat([image, skeleton_image])
            cv2.imshow(window_name, combined_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                is_paused = not is_paused
            elif key == ord('s'):
                # 手動強制存檔
                saved_frame = save_captured_images(clean_image, image, current_frame_idx)
                captured_frames.append(saved_frame)

    cap.release()
    cv2.destroyAllWindows()
    
    # 程式結束時，印出可以貼到 JSON 的陣列格式！
    print("\n" + "="*40)
    print("🎉 影片處理完畢！")
    print("請將以下陣列貼到 static/config.json 裡面 'towel' 的 'key_frames' 中：")
    print(f"[{', '.join(map(str, captured_frames))}]")
    print("="*40 + "\n")

if __name__ == "__main__":
    process_pose_extractor(VIDEO_PATH)