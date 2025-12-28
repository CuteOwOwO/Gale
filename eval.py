import cv2
import mediapipe as mp
import numpy as np
import os

# --- 設定區 ---
VIDEO_PATH = 'old.mp4'   # 請確認檔名是否正確
GT_FILE = 'ground_truth.txt'
OUTPUT_DIR = 'final_best_poses'

# 演算法參數 (使用原本的閾值邏輯)
SCAN_THRESHOLD = 0.05    
SUPPRESSION_RANGE = 20   
# -------------

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def load_ground_truth(filepath):
    if not os.path.exists(filepath):
        print(f"[錯誤] 找不到檔案: {filepath}")
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
    if prev_landmarks is None: return 1.0
    key_indices = [15, 16, 27, 28] 
    total_movement = 0
    for idx in key_indices:
        c = current_landmarks[idx]
        p = prev_landmarks[idx]
        dist = np.sqrt((c.x - p.x)**2 + (c.y - p.y)**2)
        total_movement += dist
    return total_movement / len(key_indices)

def scan_and_visualize(video_path, target_k, suppression):
    """
    第一階段：播放影片 + 視覺化 + 收集數據 (原本的演算法)
    """
    print(f"正在掃描影片... (目標尋找 {target_k} 個關鍵動作)")
    
    candidates = [] 
    cap = cv2.VideoCapture(video_path)
    
    # 設定視窗
    window_name = "Algorithm View (Threshold Method)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    prev_landmarks = None
    frame_idx = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            velocity = 1.0
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                velocity = get_pose_velocity(landmarks, prev_landmarks)
                prev_landmarks = landmarks
                
                # 只要速度低於門檻，就加入候選
                if velocity < SCAN_THRESHOLD:
                    candidates.append((frame_idx, velocity))

                # --- 視覺化 ---
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 儀表板
            bar_len = int(min(velocity, 0.1) * 3000)
            bar_color = (0, 255, 0) if velocity < SCAN_THRESHOLD else (0, 0, 255)
            cv2.rectangle(image, (30, 30), (30 + bar_len, 50), bar_color, -1)
            cv2.putText(image, f"Vel: {velocity:.4f}", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
            cv2.putText(image, f"Frame: {frame_idx}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow(window_name, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_idx += 1
            
    cap.release()
    cv2.destroyAllWindows()

    # --- Top-K 選拔 ---
    print("掃描完成，正在計算最佳 Top-K...")
    candidates.sort(key=lambda x: x[1]) # 速度越小越好
    
    final_selection = []
    for frame, vel in candidates:
        if len(final_selection) >= target_k: break
        
        # NMS 去重
        is_too_close = False
        for selected_frame, _ in final_selection:
            if abs(frame - selected_frame) < suppression:
                is_too_close = True
                break
        
        if not is_too_close:
            final_selection.append((frame, vel))
            
    final_selection.sort(key=lambda x: x[0])
    return [x[0] for x in final_selection]

def evaluate_nearest_neighbor(ground_truth, detected_frames):
    """
    更新後的評分邏輯：每一個 Pred 去找離它最近的 GT
    """
    if not ground_truth: return

    print(f"\n=== 最終評測結果 (Nearest Neighbor) ===")
    print(f"{'Pred':^6} | {'Closest GT':^10} | {'Error':^6}")
    print("-" * 32)
    
    total_error = 0
    
    for pred in detected_frames:
        # 核心邏輯：在所有 GT 中找到跟目前這個 Pred 距離最小的
        closest_gt = min(ground_truth, key=lambda x: abs(x - pred))
        err = abs(pred - closest_gt)
        
        total_error += err
        
        # 狀態圖示
        status = ""
        if err == 0: status = "✨" 
        elif err <= 5: status = "✅"
        elif err > 20: status = "⚠️"
        
        print(f"{pred:6d} | {closest_gt:10d} | {err:6d} {status}")

    # 計算平均
    mae = total_error / len(detected_frames) if detected_frames else 0
    
    print("-" * 32)
    print(f"MAE (平均誤差): {mae:.2f} 幀")
    print("-" * 32)
    
    if mae < 5: print("🏆 評級 S：完美對齊！")
    elif mae < 10: print("🥈 評級 A：非常精準。")
    else: print("🔧 評級 B：仍有誤差。")

def save_overlay_images(video_path, frames_to_save):
    """
    存檔功能
    """
    if not frames_to_save: return
    print(f"\n[存檔中] 正在儲存 {len(frames_to_save)} 張最佳圖片到 {OUTPUT_DIR}/ ...")
    cap = cv2.VideoCapture(video_path)
    
    with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1) as pose:
        for target_frame in frames_to_save:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, image = cap.read()
            if not success: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            filename = f"{OUTPUT_DIR}/best_pose_frame_{target_frame}.jpg"
            cv2.imwrite(filename, image)
            print(f"  -> 已儲存: {filename}")
            
    cap.release()
    print("[完成] 工作結束。")

if __name__ == "__main__":
    # 1. 讀取 GT
    ground_truth = load_ground_truth(GT_FILE)
    
    if not ground_truth:
        print("請先執行標記工具產生 ground_truth.txt")
    else:
        # 確保 target_k 存在 (如果沒 GT 就預設抓 15 個)
        target_k = len(ground_truth) if ground_truth else 15

        # 2. 執行第一階段：掃描 + 視覺化 (原算法)
        best_frames = scan_and_visualize(VIDEO_PATH, target_k, SUPPRESSION_RANGE)
        
        # 3. 執行評測 (新評分邏輯)
        evaluate_nearest_neighbor(ground_truth, best_frames)
        
        # 4. 執行第二階段：存圖
        save_overlay_images(VIDEO_PATH, best_frames)