import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import argrelextrema
import collections
import copy
import os
from datetime import datetime

VIDEO_PATH = "old.mp4"
OUTPUT_DIR = "motion_capture_v2_results"
GT_FILE = "ground_truth.txt"
DETECTED_FRAMES_FILE = os.path.join(OUTPUT_DIR, "detected_frames.txt")
MIN_POSE_DIFF = 1.5
VELOCITY_WINDOW = 10
VERY_SLOW_THRESHOLD = 0.05
SUPPRESSION_RANGE = 8

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[INFO] Output dir ready: {OUTPUT_DIR}")

def save_detected_frame(frame, prefix, frame_idx=None):
    if frame is None:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_tag = f"_f{frame_idx:06d}" if frame_idx is not None else ""
    filename = os.path.join(OUTPUT_DIR, f"{prefix}{frame_tag}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[CAPTURE] Saved: {filename}")

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

def suppress_close_frames(frame_indices, min_gap):
    if not frame_indices:
        return []
    kept = []
    last = None
    for idx in sorted(frame_indices):
        if last is None or idx - last >= min_gap:
            kept.append(idx)
            last = idx
    return kept

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

# --- 新增：姿勢正規化函式 ---
def get_normalized_landmarks(landmarks):
    """
    將 MediaPipe 的 landmarks 轉換為以臀部為中心、軀幹長度為單位的相對座標。
    這能排除「人站得遠近」或「人在畫面位置」的影響，只專注於「姿勢形狀」。
    """
    # 提取關鍵點座標 (我們只取 33 個點中的 x, y)
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # 1. 找臀部中心 (點 23:左臀, 24:右臀)
    hip_center = (coords[23] + coords[24]) / 2.0
    
    # 2. 找肩膀中心 (點 11:左肩, 12:右肩) 用來計算軀幹長度
    shoulder_center = (coords[11] + coords[12]) / 2.0
    torso_size = np.linalg.norm(shoulder_center - hip_center)
    
    # 避免除以 0
    if torso_size < 0.01: torso_size = 1.0
        
    # 3. 正規化：(所有點 - 臀部中心) / 軀幹長度
    normalized_coords = (coords - hip_center) / torso_size
    
    # 展平成一維陣列以便計算距離
    return normalized_coords.flatten()

class DistinctPoseDetector:
    """
    方法 3: 差異穩定度偵測法 (推薦用於太極/慢舞)
    邏輯：
    1. 計算當前姿勢與「上一次抓拍姿勢」的差異距離。
    2. 當差異夠大 (代表換動作了)，且速度進入「局部極小值」時，視為關鍵姿勢。
    """
    
    def __init__(self, min_pose_diff=0.8, velocity_window=10, very_slow_threshold=0.05):
        self.min_pose_diff = min_pose_diff # 兩個姿勢之間的最小差異閾值
        self.last_captured_pose = None     # 上一次抓拍的正規化姿勢
        self.very_slow_threshold = very_slow_threshold
        
        # 用於尋找局部最小速度的緩衝區
        self.velocity_window = velocity_window
        self.velocity_buffer = collections.deque(maxlen=velocity_window)
        self.pose_buffer = collections.deque(maxlen=velocity_window)
        self.frame_buffer = collections.deque(maxlen=velocity_window)
        self.frame_index_buffer = collections.deque(maxlen=velocity_window)
        
        self.prev_landmarks = None

    def process(self, landmarks, frame, frame_idx):
        # 1. 姿勢正規化
        current_pose_vec = get_normalized_landmarks(landmarks)
        
        # 2. 計算瞬時速度 (基於正規化後的座標變化，比原始像素速度更準)
        current_velocity = 0.0
        if self.prev_landmarks is not None:
            current_velocity = np.linalg.norm(current_pose_vec - self.prev_landmarks)
        self.prev_landmarks = current_pose_vec
        
        # 存入緩衝區 (因為我們要回頭看過去幾幀是不是最低速)
        self.velocity_buffer.append(current_velocity)
        self.pose_buffer.append(current_pose_vec)
        self.frame_buffer.append(frame.copy())
        self.frame_index_buffer.append(frame_idx)
        
        # 緩衝區還沒滿，先不判斷
        if len(self.velocity_buffer) < self.velocity_window:
            return False, None, 0.0, None

        # --- 核心邏輯 ---
        
        # 我們檢查緩衝區的中間那一幀 (mid_idx)
        mid_idx = self.velocity_window // 2
        mid_velocity = self.velocity_buffer[mid_idx]
        mid_pose = self.pose_buffer[mid_idx]
        mid_frame = self.frame_buffer[mid_idx]
        mid_frame_idx = self.frame_index_buffer[mid_idx]
        
        # 條件 A: 速度是局部極小值 (前後都比它快，代表動作頓點)
        # 或者是速度極低 (趨近於靜止)
        velocities = np.array(self.velocity_buffer)
        is_local_min = (mid_velocity == np.min(velocities))
        is_very_slow = mid_velocity < self.very_slow_threshold # 絕對靜止閾值
        
        is_stable_moment = is_local_min or is_very_slow

        # 條件 B: 與上一次抓拍的姿勢差異夠大 (避免重複抓拍)
        if self.last_captured_pose is None:
            # 第一幀直接存
            self.last_captured_pose = mid_pose
            return True, mid_frame, mid_velocity, mid_frame_idx
        
        dist_from_last = np.linalg.norm(mid_pose - self.last_captured_pose)
        
        # 綜合判斷：動作穩定了 + 姿勢改變了
        if is_stable_moment and dist_from_last > self.min_pose_diff:
            self.last_captured_pose = mid_pose # 更新基準姿勢
            return True, mid_frame, dist_from_last, mid_frame_idx
            
        return False, None, dist_from_last, None


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
        # 12:右肩, 14:右肘, 16:右腕
        # 23:左臀, 25:左膝, 27:左踝
        coords = {}
        for idx in [12, 14, 16, 23, 25, 27]:
            coords[idx] = [landmarks[idx].x * w, landmarks[idx].y * h]
            
        # 計算角度
        r_elbow_ang = calculate_angle(coords[12], coords[14], coords[16])
        l_knee_ang = calculate_angle(coords[23], coords[25], coords[27])
        
        self.right_elbow_angles.append(r_elbow_ang)
        self.left_knee_angles.append(l_knee_ang)
        
        # 當緩衝區滿了，檢查中間那幀是否為局部最大/最小值
        if len(self.right_elbow_angles) == self.window_size:
            # 取得中間索引 (因為我們要判斷過去發生的峰值，所以會有幾幀的延遲)
            mid_idx = self.window_size // 2
            
            # 轉換為 numpy array 以便運算
            r_data = np.array(self.right_elbow_angles)
            l_data = np.array(self.left_knee_angles)
            
            # 判斷是否為局部最大值 (手臂伸最直的時候) 
            # 條件: 中間點是最大值 且 角度大於 160 度 (避免微小的抖動被誤判)
            is_elbow_peak = r_data[mid_idx] == np.max(r_data) and r_data[mid_idx] > 160
            
            # 也可以偵測膝蓋彎曲 (Local Minima, 蹲最低的時候)
            # 條件: 中間點是最小值 且 角度小於 120 度
            is_knee_valley = l_data[mid_idx] == np.min(l_data) and l_data[mid_idx] < 120
            
            if is_elbow_peak or is_knee_valley:
                return True, self.frame_buffer[mid_idx], self.frame_index_buffer[mid_idx] # 回傳造成峰值的那一幀
                
        return False, None, None

class OpticalFlowDetector:
    """
    方法 2: 稠密光流法 (不依賴 MediaPipe 骨架)
    計算整體畫面的像素流動量。當流動量低於閾值，視為靜止。
    """
    def __init__(self, threshold=2.0):
        self.prev_gray = None
        self.threshold = threshold
        
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_keyframe = False
        mag_mean = 0
        
        if self.prev_gray is not None:
            # 計算 Farneback 光流
            # 參數說明: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 
                                              0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 計算流動向量的大小 (Magnitude) 與 角度
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 計算全畫面平均移動量
            mag_mean = np.mean(mag)
            
            # 簡單邏輯：移動量小於閾值視為靜止
            # 進階邏輯：可以加入「移動量由大變小」的趨勢判斷
            if mag_mean < self.threshold:
                is_keyframe = True
                
        self.prev_gray = gray
        return is_keyframe, mag_mean

# --- 主程式整合 ---
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    
    # 選擇你要使用的演算法
    # algorithm = "GEOMETRIC" 
    # algorithm = "OPTICAL_FLOW"
    algorithm = "DISTINCT_STABILITY" # 新方法：差異穩定度
    
    geo_detector = GeometricPoseDetector(window_size=15)
    opt_detector = OpticalFlowDetector(threshold=1.5)
    # 參數 min_pose_diff 越大，抓到的姿勢越少但越獨特
    distinct_detector = DistinctPoseDetector(
        min_pose_diff=MIN_POSE_DIFF,
        velocity_window=VELOCITY_WINDOW,
        very_slow_threshold=VERY_SLOW_THRESHOLD
    )
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
            val_to_show = 0.0
            
            if algorithm == "GEOMETRIC":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    is_peak, peak_frame, peak_idx = geo_detector.process(
                        results.pose_landmarks.landmark, frame, frame_idx
                    )
                    if is_peak:
                        detected = True
                        info_text = "Geometric Peak!"
                        save_detected_frame(peak_frame, "geometric", peak_idx)
                        if peak_idx is not None:
                            detected_frames.append(peak_idx)
            
            elif algorithm == "OPTICAL_FLOW":
                is_static, magnitude = opt_detector.process(frame)
                cv2.putText(display_frame, f"Motion: {magnitude:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_static:
                    detected = True
                    info_text = "Static Scene!"
                    save_detected_frame(raw_frame, "optical", frame_idx)
                    detected_frames.append(frame_idx)
            
            elif algorithm == "DISTINCT_STABILITY":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    is_distinct, distinct_frame, diff_val, distinct_idx = distinct_detector.process(
                        results.pose_landmarks.landmark, frame, frame_idx
                    )
                    
                    # 顯示當前姿勢與上一張存檔姿勢的差異度
                    cv2.putText(display_frame, f"Diff from last: {diff_val:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if is_distinct:
                        detected = True
                        info_text = "New Stable Pose!"
                        save_detected_frame(distinct_frame, "distinct", distinct_idx)
                        if distinct_idx is not None:
                            detected_frames.append(distinct_idx)
                        # 如果需要存檔，這裡可以使用 distinct_frame
                        # cv2.imwrite(f"pose_{int(time.time())}.jpg", distinct_frame)

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
        if SUPPRESSION_RANGE > 0:
            detected_frames = suppress_close_frames(detected_frames, SUPPRESSION_RANGE)
        save_detected_indices(detected_frames, DETECTED_FRAMES_FILE)
        ground_truth = load_ground_truth(GT_FILE)
        evaluate_nearest_neighbor(ground_truth, detected_frames)

if __name__ == "__main__":
    # 請將此處換成你的影片路徑，或使用 0 開啟 web-cam
    main(VIDEO_PATH)
