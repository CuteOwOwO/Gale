import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

# --- 設定區 ---
VIDEO_PATH = 'old.mp4'   # 測試影片路徑
OUTPUT_DIR = "captured_poses"     # 存檔資料夾
# 演算法參數
VELOCITY_THRESHOLD = 0.008  # 速度閾值 (越小越嚴格，只抓靜止瞬間)
COOLDOWN_FRAMES = 30        # 抓拍冷卻時間 (幀數)
SMOOTHING_FACTOR = 0.5      # 速度平滑化係數
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
    
    # 關注點: 15,16(手腕), 27,28(腳踝)
    key_indices = [15, 16, 27, 28] 
    total_movement = 0
    for idx in key_indices:
        c_x, c_y = current_landmarks[idx].x, current_landmarks[idx].y
        p_x, p_y = prev_landmarks[idx].x, prev_landmarks[idx].y
        dist = np.sqrt((c_x - p_x)**2 + (c_y - p_y)**2)
        total_movement += dist
    
    return total_movement / len(key_indices)

def save_captured_images( overlay_img, skel_img, prefix="auto"):
    """
    統一處理存檔功能的函式
    Args:
        clean_img: 乾淨的原圖
        overlay_img: 畫了骨架的原圖
        skel_img: 黑底骨架圖
        prefix: 檔名前綴 (auto 或 manual)
    """
    timestamp = datetime.now().strftime("%H%M%S_%f")
    
    # 1. 存疊加圖 (教學用: 看得到人 + 線條)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_{timestamp}_overlay.jpg", overlay_img)
    
    # 2. 存黑底骨架 (AI用: 純線條)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_{timestamp}_skel.png", skel_img)
    
    # 3. 存乾淨原圖 (備份用)
    #cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_{timestamp}_raw.jpg", clean_img)
    
    print(f"[{prefix.upper()} 存檔] 已儲存三組圖片 -> 時間戳: {timestamp}")

def process_pose_extractor(video_source):
    cap = cv2.VideoCapture(video_source)
    
    window_name = 'Pose Capture Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 550)

    # 狀態變數
    prev_landmarks_list = None 
    current_velocity = 1.0     
    cooldown_counter = 0      
    is_paused = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # 暫停控制
            if is_paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): is_paused = not is_paused
                elif key == ord('q'): break
                continue

            success, image = cap.read()
            if not success:
                print("影片結束")
                break
            
            # --- 影像前處理 ---
            # 1. 備份乾淨原圖 (重要：這要在畫線之前做！)
            clean_image = image.copy()

            # 2. 轉 RGB 給 MediaPipe 吃
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # 3. 轉回 BGR 準備畫圖 (這是要做 Overlay 的圖)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 4. 準備黑底圖
            skeleton_image = np.zeros(image.shape, dtype=np.uint8)

            is_pose_detected = False # 標記這一幀有沒有被抓到

            if results.pose_landmarks:
                # 繪製 - 疊加圖 (畫在 image 上)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # 繪製 - 黑底圖 (畫在 skeleton_image 上)
                mp_drawing.draw_landmarks(
                    skeleton_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # --- 演算法判斷 ---
                landmarks = results.pose_landmarks.landmark
                raw_velocity = calculate_pose_velocity(landmarks, prev_landmarks_list)
                current_velocity = (raw_velocity * (1 - SMOOTHING_FACTOR)) + (current_velocity * SMOOTHING_FACTOR)
                prev_landmarks_list = landmarks

                # 判斷是否自動抓拍
                if current_velocity < VELOCITY_THRESHOLD and cooldown_counter == 0:
                    is_pose_detected = True
                    cooldown_counter = COOLDOWN_FRAMES
                    # *** 呼叫存檔函式 ***
                    save_captured_images( image, skeleton_image, prefix="auto")

                if cooldown_counter > 0:
                    cooldown_counter -= 1

            # --- 介面繪製 (UI) ---
            # 畫速度條
            bar_len = int(current_velocity * 5000)
            color = (0, 255, 0) if current_velocity < VELOCITY_THRESHOLD else (0, 0, 255)
            cv2.rectangle(image, (30, 30), (30 + bar_len, 50), color, -1)
            cv2.putText(image, f"Vel: {current_velocity:.4f}", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_pose_detected:
                cv2.putText(image, "Auto Captured!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 左右拼接顯示
            combined_view = cv2.hconcat([image, skeleton_image])
            cv2.imshow(window_name, combined_view)

            # --- 按鍵控制 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                is_paused = not is_paused
                print(f"狀態: {'暫停' if is_paused else '播放'}")
            elif key == ord('s'):
                # 手動強制存檔 (也使用同樣的三圖邏輯)
                save_captured_images(image, skeleton_image, prefix="manual")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_pose_extractor(VIDEO_PATH)