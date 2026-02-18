#播影片

import cv2
import mediapipe as mp
import numpy as np

import time
import os
from datetime import datetime

# --- 設定區 ---
VIDEO_PATH = 'old.mp4'  # 若用 WebCam 請改 0
OUTPUT_DIR = "captured_poses"    # 圖片存檔的資料夾名稱
# -------------

# 1. 建立存檔資料夾
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[系統] 已建立資料夾: {OUTPUT_DIR}")

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_pose_extractor(video_source):
    cap = cv2.VideoCapture(video_source)
    
    # 設定視窗大小 (避免太長)
    window_name = 'Left: Preview | Right: AI Input (Press SPACE to Pause, S to Save)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 550) # 寬度設寬一點因為是兩張併排

    is_paused = False # 暫停狀態標記

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:

        while cap.isOpened():
            # 只有在「沒有暫停」的時候才讀取下一幀
            if not is_paused:
                success, image = cap.read()
                if not success:
                    print("影片結束")
                    break
                
                # 備份一份乾淨的原圖 (存檔用)
                clean_image = image.copy()

                # 轉 RGB 處理
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                # 準備畫布
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 這是左邊的預覽圖
                
                # --- 關鍵魔法：建立黑底圖 ---
                # np.zeros 產生一個跟原圖一樣大小的全黑陣列
                skeleton_image = np.zeros(image.shape, dtype=np.uint8)
                
                # 如果有抓到骨架
                if results.pose_landmarks:
                    # 1. 畫在左邊 (原圖預覽)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 2. 畫在右邊 (黑底 AI 用圖)
                    # 這裡我們可以畫得比較簡單乾淨，或者跟原本一樣
                    mp_drawing.draw_landmarks(
                        skeleton_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # --- 畫面拼接 ---
                # 把 原圖 和 骨架圖 左右(horizontal) 接起來
                combined_view = cv2.hconcat([image, skeleton_image])

            # 顯示畫面
            cv2.imshow(window_name, combined_view)

            # --- 按鍵控制區 ---
            key = cv2.waitKey(1) & 0xFF

            # 按 'q' 離開
            if key == ord('q'):
                break
            
            # 按 'SPACE' 暫停/繼續
            elif key == ord(' '):
                is_paused = not is_paused
                status = "暫停中" if is_paused else "播放中"
                print(f"[控制] {status}")

            # 按 's' 存檔
            elif key == ord('s'):
                # 產生時間戳記檔名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # 檔名設定
                filename_skel = f"{OUTPUT_DIR}/pose_{timestamp}.png"
                filename_orig = f"{OUTPUT_DIR}/orig_{timestamp}.jpg"
                
                # 存檔 (存乾淨的黑底骨架圖，不要存合併的)
                cv2.imwrite(filename_skel, skeleton_image)
                cv2.imwrite(filename_orig, clean_image)
                
                print(f"[存檔成功] 已儲存兩張圖片到 {OUTPUT_DIR}/ 資料夾")
                print(f"          -> 骨架圖: pose_{timestamp}.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_pose_extractor(VIDEO_PATH)