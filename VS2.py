#可以看到實際畫面的版本，並且有骨架顯示，以及老師疊圖

import cv2
import mediapipe as mp
import numpy as np


# --- 📁 設定區 ---
VIDEO_PATH = 'mp4_home/old.mp4'   
TARGET_FRAME = 151         
VISIBILITY_THRESHOLD = 0.5 
GHOST_ALPHA = 0.1  # 老師原圖的透明度
# ---------------------

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 評分向量
BODY_PARTS_VECTORS = [
    (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 25), (25, 27), 
    (24, 26), (26, 28)
]

def get_torso_geometry(landmarks, image_shape):
    h, w = image_shape[:2]
    if (landmarks[11].visibility < 0.5 or landmarks[12].visibility < 0.5 or
        landmarks[23].visibility < 0.5 or landmarks[24].visibility < 0.5):
        return None, None 

    ls = np.array([landmarks[11].x * w, landmarks[11].y * h])
    rs = np.array([landmarks[12].x * w, landmarks[12].y * h])
    lh = np.array([landmarks[23].x * w, landmarks[23].y * h])
    rh = np.array([landmarks[24].x * w, landmarks[24].y * h])
    
    center = (ls + rs + lh + rh) / 4
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    size = np.linalg.norm(mid_shoulder - mid_hip)
    
    return center, size

def pre_process_teacher(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = cap.read()
    cap.release()
    
    if not success: return None, None, None, None

    landmark_list = None
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            landmark_list = results.pose_landmarks.landmark

    mask = None
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_seg.process(img_rgb)
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

    edges = cv2.Canny(mask, 100, 200)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    neon_outline = np.zeros_like(image)
    neon_outline[edges_dilated > 0] = [0, 255, 255] # 黃色霓虹

    return image, landmark_list, mask, neon_outline

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
            
            norm_t = np.linalg.norm(t_vec)
            norm_u = np.linalg.norm(u_vec)
            if norm_t == 0 or norm_u == 0: score = 0
            else: score = np.dot(t_vec, u_vec) / (norm_t * norm_u)
            
            total_score += max(0, score)
        else:
            missing_parts += 1
    return (total_score / total_parts) * 100

def main_ar_ghost_with_skeleton():
    print("正在準備 AR 素材...")
    t_img, t_landmarks, t_mask, t_neon = pre_process_teacher(VIDEO_PATH, TARGET_FRAME)
    
    if t_landmarks is None:
        print("❌ 老師影像處理失敗")
        return

    t_center, t_size = get_torso_geometry(t_landmarks, t_img.shape)
    if t_center is None:
        print("❌ 無法計算老師軀幹中心")
        return

    cap = cv2.VideoCapture(0)
    window_name = "AR Ghost Coach + Skeleton"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            current_score = 0
            
            if results.pose_landmarks:
                u_landmarks = results.pose_landmarks.landmark
                current_score = compare_poses_strict(u_landmarks, t_landmarks)
                
                # --- AR 疊加 (老師圖層) ---
                u_center, u_size = get_torso_geometry(u_landmarks, frame.shape)
                
                if u_center is not None and u_size > 0:
                    scale = u_size / t_size
                    tx = u_center[0] - t_center[0] * scale
                    ty = u_center[1] - t_center[1] * scale
                    M = np.float32([[scale, 0, tx], [0, scale, ty]])
                    
                    warped_t_img = cv2.warpAffine(t_img, M, (w, h))
                    warped_t_mask = cv2.warpAffine(t_mask, M, (w, h))
                    warped_t_neon = cv2.warpAffine(t_neon, M, (w, h))
                    
                    mask_3ch = cv2.merge([warped_t_mask, warped_t_mask, warped_t_mask])
                    mask_indices = np.where(mask_3ch > 128)
                    
                    # 疊加半透明老師
                    overlay = frame.copy()
                    overlay[mask_indices] = warped_t_img[mask_indices]
                    cv2.addWeighted(overlay, GHOST_ALPHA, frame, 1 - GHOST_ALPHA, 0, frame)
                    
                    # 疊加霓虹邊框
                    neon_indices = np.where(warped_t_neon > 0)
                    frame[neon_indices] = warped_t_neon[neon_indices]

                # --- 這裡！畫出你自己的骨架 (疊在最上面) ---
                # 根據分數改變骨架顏色
                skel_color = (0, 0, 255) # 紅
                if current_score > 60: skel_color = (0, 255, 255) # 黃
                if current_score > 85: skel_color = (0, 255, 0)   # 綠

                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    # 點的樣式
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=skel_color, thickness=2, circle_radius=3),
                    # 線的樣式
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=skel_color, thickness=2)
                )

            # --- UI ---
            color = (0, 0, 255)
            if current_score > 60: color = (0, 255, 255)
            if current_score > 85: color = (0, 255, 0)
            
            cv2.putText(frame, f"Score: {int(current_score)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            if current_score > 85:
                cv2.putText(frame, "HOLD IT!", (w//2-100, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_ar_ghost_with_skeleton()