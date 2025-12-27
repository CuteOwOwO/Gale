import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import argrelextrema
import collections
import os
from datetime import datetime

VIDEO_PATH = "old.mp4"
OUTPUT_DIR = "motion_capture_v1_results"
GT_FILE = "ground_truth.txt"
DETECTED_FRAMES_FILE = os.path.join(OUTPUT_DIR, "detected_frames.txt")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[系統] 存檔資料夾已準備: {OUTPUT_DIR}")

def save_detected_frame(frame, prefix, frame_idx):
    if frame is None:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(OUTPUT_DIR, f"{prefix}_f{frame_idx:06d}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[CAPTURE] 已儲存: {filename}")

# --- 輔助函式：計算角度 ---
def calculate_angle(a, b, c):
    """
    計算三個點之間的夾角 (b 為頂點)
    a, b, c: [x, y] 座標
    return: 角度 (0~180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

class GeometricPoseDetector:
    """
    方法 1: 幾何特徵法
    偵測肢體展開與收縮的「極值點」，適用於抓取動作的頂點 (Peak)。
    """
    def __init__(self, window_size=15):
        self.window_size = window_size
        # 儲存最近 N 幀的角度數據，用於尋找局部最大值
        self.right_elbow_angles = collections.deque(maxlen=window_size)
        self.left_knee_angles = collections.deque(maxlen=window_size)
        self.frame_buffer = collections.deque(maxlen=window_size) # 存影像以供回溯
        self.frame_index_buffer = collections.deque(maxlen=window_size)
        
    def process(self, landmarks, frame, frame_idx):
        """
        傳入 landmarks，回傳是否為關鍵姿勢 (bool) 以及該關鍵姿勢的影像
        """
        h, w, _ = frame.shape
        self.frame_buffer.append(frame.copy())
        self.frame_index_buffer.append(frame_idx)
        
        # 取得關鍵點座標 (正規化座標轉像素座標)
        
        coords = {}
        for idx in [12, 14, 16, 23, 25, 27]:
            coords[idx] = [landmarks[idx].x * w, landmarks[idx].y * h]
            
        # 計算角度
        # 12:右肩, 14:右肘, 16:右腕
        # 23:左臀, 25:左膝, 27:左踝
        r_elbow_ang = calculate_angle(coords[12], coords[14], coords[16])
        l_knee_ang = calculate_angle(coords[23], coords[25], coords[27])
        
        self.right_elbow_angles.append(r_elbow_ang)
        self.left_knee_angles.append(l_knee_ang)
        
        # 當 window 滿了，檢查中間那幀是否為局部最大/最小值
        if len(self.right_elbow_angles) == self.window_size:
            mid_idx = self.window_size // 2
            
            r_data = np.array(self.right_elbow_angles)
            l_data = np.array(self.left_knee_angles)
            
            # 局部最大值 (手臂伸最直的時候) 
            # 條件: 中間點是最大值 且 角度大於 160 度
            is_elbow_peak = r_data[mid_idx] == np.max(r_data) and r_data[mid_idx] > 160
            
            # 局部最小值 (膝蓋彎曲, 蹲最低的時候)
            # 條件: 中間點是最小值 且 角度小於 120 度
            is_knee_valley = l_data[mid_idx] == np.min(l_data) and l_data[mid_idx] < 120
            
            # 回傳造成峰值的那幀
            if is_elbow_peak or is_knee_valley:
                return True, self.frame_buffer[mid_idx], self.frame_index_buffer[mid_idx] 
                
        return False, None, None

class OpticalFlowDetector:
    """
    方法 2: 稠密光流法 (不依賴 MediaPipe 骨架)
    計算整體畫面的像素流動量。當流動量低於閾值，視為靜止。
    """
    def __init__(self, threshold=0.01):
        self.prev_gray = None
        self.threshold = threshold
        
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_keyframe = False
        mag_mean = 0
        
        if self.prev_gray is not None:
            # 計算 Farneback 光流
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 計算流動向量的大小與角度
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 計算全畫面平均移動量
            mag_mean = np.mean(mag)

            if mag_mean < self.threshold:
                is_keyframe = True
                
        self.prev_gray = gray
        return is_keyframe, mag_mean

# --- 主程式整合 ---
def save_detected_indices(indices, path):
    if not indices:
        return
    with open(path, "w") as f:
        for idx in indices:
            f.write(f"{idx}\n")
    print(f"[INFO] Detected frames saved: {path}")

def load_ground_truth(filepath):
    if not os.path.exists(filepath):
        print(f"[WARN] Ground truth not found: {filepath}")
        return []
    gt_list = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                gt_list.append(int(line))
    gt_list.sort()
    return gt_list

def evaluate_nearest_neighbor(ground_truth, detected_frames):
    if not ground_truth or not detected_frames:
        return

    print("\n=== Evaluation (Nearest Neighbor) ===")
    print(f"{'Pred':^6} | {'Closest':^7} | {'Error':^6}")
    print("-" * 28)

    total_error = 0
    for pred in detected_frames:
        closest_gt = min(ground_truth, key=lambda x: abs(x - pred))
        err = abs(pred - closest_gt)
        total_error += err

        status = "OK" if err == 0 else "CLOSE" if err <= 5 else "OFF" if err > 20 else ""
        print(f"{pred:6d} | {closest_gt:7d} | {err:6d} {status}")

    mae = total_error / len(detected_frames) if detected_frames else 0
    print("-" * 28)
    print(f"MAE: {mae:.2f} frames")

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    
    # 選擇你要使用的演算法
    algorithm = "GEOMETRIC" 
    # algorithm = "OPTICAL_FLOW" 
    
    geo_detector = GeometricPoseDetector(window_size=15)
    opt_detector = OpticalFlowDetector(threshold=1.5)
    detected_frames = []
    frame_idx = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            raw_frame = frame.copy()
            display_frame = frame.copy()
            detected = False
            info_text = ""
            
            if algorithm == "GEOMETRIC":
                # 需要先跑 MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 畫骨架
                    mp.solutions.drawing_utils.draw_landmarks(
                        display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # 演算法判斷
                    is_peak, peak_frame, peak_idx = geo_detector.process(
                        results.pose_landmarks.landmark, frame, frame_idx
                    )
                    
                    if is_peak:
                        detected = True
                        info_text = "Pose Peak Detected!"
                        save_detected_frame(peak_frame, "geometric", peak_idx)
                        if peak_idx is not None:
                            detected_frames.append(peak_idx)
                        # 這裡的 detected 是指發現了波峰
                        # 如果要顯示當時的畫面，應該用 peak_frame
            
            elif algorithm == "OPTICAL_FLOW":
                # 不需要 MediaPipe，直接算
                is_static, magnitude = opt_detector.process(frame)
                
                # 視覺化光流大小
                cv2.putText(display_frame, f"Motion Mag: {magnitude:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if is_static:
                    detected = True
                    info_text = "Static Scene Detected!"
                    save_detected_frame(raw_frame, "optical", frame_idx)
                    detected_frames.append(frame_idx)

            # 顯示偵測結果
            if detected:
                cv2.circle(display_frame, (50, 80), 20, (0, 0, 255), -1)
                cv2.putText(display_frame, info_text, (80, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Advanced Pose Detector', display_frame)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
            frame_idx += 1
                
    cap.release()
    cv2.destroyAllWindows()
    
    if detected_frames:
        detected_frames = sorted(set(detected_frames))
        save_detected_indices(detected_frames, DETECTED_FRAMES_FILE)
        ground_truth = load_ground_truth(GT_FILE)
        evaluate_nearest_neighbor(ground_truth, detected_frames)

if __name__ == "__main__":
    # 請將此處換成你的影片路徑，或使用 0 開啟 web-cam
    main(VIDEO_PATH)
