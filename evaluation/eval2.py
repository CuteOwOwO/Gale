#評估篹法好壞

import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import argrelextrema # 核心：用來找波谷的工具

# --- 設定區 ---
VIDEO_PATH = 'old.mp4'
GT_FILE = 'ground_truth.txt'
OUTPUT_DIR = 'final_best_poses'

# [演算法參數]
SMOOTHING_WINDOW = 5    # 平滑化窗口：數值越大越平滑，但反應會變慢 (建議 3~7)
MINIMA_ORDER = 15       # 波谷範圍：前後 N 幀內，我是最低的才算 (這取代了之前的 NMS)
MAX_VELOCITY_LIMIT = 0.15 # 絕對門檻：雖然你是波谷，但速度如果還是太快(>0.15)也不要
# -------------

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def load_ground_truth(filepath):
    if not os.path.exists(filepath):
        print(f"[系統] 尚未建立標記檔 ({filepath})，將無法進行評分，僅執行抓取。")
        return []
    gt_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                gt_list.append(int(line))
    gt_list.sort()
    return gt_list

def get_pose_velocity(current_landmarks, prev_landmarks):
    """計算特定部位(手腕腳踝)的移動速度"""
    if prev_landmarks is None: return 1.0
    key_indices = [15, 16, 27, 28] 
    total_movement = 0
    for idx in key_indices:
        c = current_landmarks[idx]
        p = prev_landmarks[idx]
        dist = np.sqrt((c.x - p.x)**2 + (c.y - p.y)**2)
        total_movement += dist
    return total_movement / len(key_indices)

def analyze_video_with_local_minima(video_path, target_k):
    """
    核心演算法：
    1. 播放並收集全片速度數據
    2. 使用 Signal Processing 找出真正的波谷
    """
    print(f"[階段一] 掃描影片並收集數據...")
    
    cap = cv2.VideoCapture(video_path)
    velocities = []     # 存每一幀的速度
    frame_indices = []
    
    # 視覺化視窗
    window_name = "Scanning (Collecting Data)..."
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    prev_landmarks = None
    frame_idx = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            # 1. 姿勢偵測
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # 2. 計算速度
            vel = 1.0 # 預設值
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                vel = get_pose_velocity(landmarks, prev_landmarks)
                prev_landmarks = landmarks
                
                # 畫骨架給你看 (比較不無聊)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            velocities.append(vel)
            frame_indices.append(frame_idx)
            
            # 3. 畫面顯示資訊
            bar_len = int(min(vel, 0.2) * 2000)
            cv2.rectangle(image, (30, 30), (30 + bar_len, 50), (0, 255, 255), -1)
            cv2.putText(image, f"Live Vel: {vel:.4f}", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("使用者中斷掃描")
                break
                
            frame_idx += 1
            
    cap.release()
    cv2.destroyAllWindows()
    
    # --- [數學運算區] ---
    print(f"[運算中] 正在分析 {len(velocities)} 筆速度數據...")
    
    # 1. 轉 numpy
    vel_array = np.array(velocities)
    
    # 2. 平滑化 (解決手抖雜訊)
    # 建立一個 [0.2, 0.2, 0.2, 0.2, 0.2] 的濾波器做移動平均
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smoothed_vel = np.convolve(vel_array, kernel, mode='same')
    
    # 3. 尋找局部極小值 (Local Minima)
    # order=MINIMA_ORDER 代表: "我必須是前後 N 幀裡面最小的"
    local_minima_indices = argrelextrema(smoothed_vel, np.less, order=MINIMA_ORDER)[0]
    
    # 4. 篩選候選點
    candidates = []
    for idx in local_minima_indices:
        v = smoothed_vel[idx]
        # 即使是低點，速度也不能快到太離譜 (過濾掉快速移動中的相對低點)
        if v < MAX_VELOCITY_LIMIT:
            candidates.append((idx, v))
            
    print(f" -> 找到 {len(candidates)} 個波谷候選點")
    
    # 5. Top-K 選拔
    # 按照速度由小到大排序，選前 K 個
    candidates.sort(key=lambda x: x[1])
    
    best_candidates = candidates[:target_k]
    
    # 6. 最後依照時間重新排序回傳
    best_candidates.sort(key=lambda x: x[0])
    
    return [x[0] for x in best_candidates]

def evaluate_performance(ground_truth, detected_frames):
    if not ground_truth: return

    print(f"\n=== 最終評測結果 (波谷偵測版) ===")
    n = min(len(ground_truth), len(detected_frames))
    total_error = 0
    
    print(f"{'GT':^6} | {'Pred':^6} | {'Error':^6}")
    print("-" * 24)
    
    for i in range(n):
        gt = ground_truth[i]
        pred = detected_frames[i]
        err = abs(gt - pred)
        total_error += err
        print(f"{gt:6d} | {pred:6d} | {err:6d}")
        
    mae = total_error / n if n > 0 else 0
    print("-" * 24)
    print(f"MAE (平均誤差): {mae:.2f} 幀")
    
    if mae < 5: print("🏆 評級 S：神準！解決了錯位問題！")
    elif mae < 15: print("🥈 評級 A：非常優秀。")
    else: print("🔧 評級 B：還有進步空間。")

def save_result_images(video_path, frames_to_save):
    if not frames_to_save: return
    print(f"\n[存檔中] 正在儲存 {len(frames_to_save)} 張圖片到 {OUTPUT_DIR}...")
    
    cap = cv2.VideoCapture(video_path)
    
    with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1) as pose:
        for target_frame in frames_to_save:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, image = cap.read()
            if not success: continue
            
            # 重畫骨架
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # 存檔
            filename = f"{OUTPUT_DIR}/pose_{target_frame}.jpg"
            cv2.imwrite(filename, image)
            print(f"  -> Saved: {filename}")
            
    cap.release()
    print("[完成] 所有工作結束。")

if __name__ == "__main__":
    # 1. 讀取 Ground Truth
    ground_truth = load_ground_truth(GT_FILE)
    
    # 如果有 GT，我們就抓一樣多的數量；如果沒有，預設抓 15 張來玩玩
    target_k = len(ground_truth) if ground_truth else 15
    
    # 2. 執行波谷偵測演算法
    best_frames = analyze_video_with_local_minima(VIDEO_PATH, target_k)
    
    # 3. 評分
    evaluate_performance(ground_truth, best_frames)
    
    # 4. 存檔
    save_result_images(VIDEO_PATH, best_frames)