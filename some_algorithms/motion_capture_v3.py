import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import argrelextrema
import os
from datetime import datetime

VIDEO_PATH = "old.mp4"
OUTPUT_DIR = "motion_capture_v3_results"
GT_FILE = "ground_truth.txt"
DETECTED_FRAMES_FILE = os.path.join(OUTPUT_DIR, "detected_frames.txt")

# 訊號參數（門檻式篩選）
SMOOTHING_WINDOW = 5
MINIMA_ORDER = 8
VELOCITY_PERCENTILE = 30
MIN_VELOCITY_THRESHOLD = 0.20
MAX_VELOCITY_THRESHOLD = 0.40
SUPPRESSION_RANGE = 8

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[INFO] Output dir ready: {OUTPUT_DIR}")

def save_detected_frame(frame, prefix, frame_idx):
    if frame is None:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(OUTPUT_DIR, f"{prefix}_f{frame_idx:06d}_{timestamp}.jpg")
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

def get_normalized_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks])

    # 以臀部中心作為原點
    hip_center = (coords[23] + coords[24]) / 2.0
    # 用肩到臀距離當作尺度
    shoulder_center = (coords[11] + coords[12]) / 2.0
    torso_size = np.linalg.norm(shoulder_center - hip_center)
    if torso_size < 0.01:
        torso_size = 1.0

    # 正規化到相對座標
    normalized_coords = (coords - hip_center) / torso_size
    return normalized_coords.flatten()

# --- 第一階段：掃描影片收集速度 ---
def collect_pose_velocities(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return [], []

    # 初始化姿勢模型與暫存
    mp_pose = mp.solutions.pose
    prev_pose_vec = None
    velocities = []
    valid_mask = []
    frame_idx = 0

    window_name = "Motion Capture V3 - Scan"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 轉成 RGB 供 MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            velocity = 1.0
            is_valid = False
            if results.pose_landmarks:
                # 計算相鄰幀姿勢差當作速度
                pose_vec = get_normalized_landmarks(results.pose_landmarks.landmark)
                if prev_pose_vec is not None:
                    velocity = np.linalg.norm(pose_vec - prev_pose_vec)
                prev_pose_vec = pose_vec
                is_valid = True

                # 視覺化骨架
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # 記錄速度序列與有效標記
            velocities.append(velocity)
            valid_mask.append(is_valid)

            cv2.putText(frame, f"Vel: {velocity:.4f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 即時顯示
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return velocities, valid_mask

# --- 第二階段：局部極小值 + 門檻篩選 ---
def select_keyframes(velocities, valid_mask):
    if not velocities:
        return []

    # 無效幀設為高速度避免入選
    vel_array = np.array(velocities, dtype=float)
    for i, valid in enumerate(valid_mask):
        if not valid:
            vel_array[i] = 1.0

    # 平滑速度序列
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smoothed = np.convolve(vel_array, kernel, mode="same")

    # 找局部極小值
    minima_indices = argrelextrema(smoothed, np.less, order=MINIMA_ORDER)[0]

    valid_smoothed = smoothed[np.array(valid_mask, dtype=bool)]
    if valid_smoothed.size == 0:
        return []

    # 以百分位數計算門檻並夾限
    threshold = np.percentile(valid_smoothed, VELOCITY_PERCENTILE)
    threshold = max(threshold, MIN_VELOCITY_THRESHOLD)
    threshold = min(threshold, MAX_VELOCITY_THRESHOLD)

    # 取門檻內候選
    candidates = []
    for idx in minima_indices:
        if valid_mask[idx] and smoothed[idx] <= threshold:
            candidates.append((idx, smoothed[idx]))

    # 候選過少時放寬到所有局部極小值
    if not candidates:
        candidates = [(idx, smoothed[idx]) for idx in minima_indices if valid_mask[idx]]
        if not candidates:
            return []

    # 依速度排序後做間隔抑制
    candidates.sort(key=lambda x: x[1])
    selected = []
    for idx, _ in candidates:
        if SUPPRESSION_RANGE > 0:
            too_close = any(abs(idx - sel) < SUPPRESSION_RANGE for sel in selected)
            if too_close:
                continue
        selected.append(idx)

    selected.sort()
    return selected

# --- 存檔功能 ---
def save_keyframes(video_path, frame_indices):
    if not frame_indices:
        return
    # 逐幀跳轉並存圖
    cap = cv2.VideoCapture(video_path)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        save_detected_frame(frame, "v3", idx)
    cap.release()

# --- 主程式 ---
def main(video_path):
    # 先掃完整影片取得速度
    velocities, valid_mask = collect_pose_velocities(video_path)
    if not velocities:
        return

    ground_truth = load_ground_truth(GT_FILE)

    # 用門檻法選關鍵幀
    keyframes = select_keyframes(velocities, valid_mask)
    if not keyframes:
        print("[WARN] No keyframes detected.")
        return

    # 存檔並評估
    save_keyframes(video_path, keyframes)
    save_detected_indices(keyframes, DETECTED_FRAMES_FILE)
    evaluate_nearest_neighbor(ground_truth, keyframes)

if __name__ == "__main__":
    main(VIDEO_PATH)
