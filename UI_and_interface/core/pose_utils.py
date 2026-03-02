# core/pose_utils.py
import numpy as np


VISIBILITY_THRESHOLD = 0.5 


BODY_PARTS_VECTORS = [
    (11, 13), (13, 15), (12, 14), (14, 16), # 手部：左大臂、左小臂、右大臂、右小臂
    (11, 23), (12, 24), (23, 25), (25, 27), # 軀幹與大腿：左軀幹、右軀幹、左大腿、左小腿
    (24, 26), (26, 28)                      # 右腿：右大腿、右小腿
]

def get_torso_geometry(landmarks, image_shape):
    """
    計算軀幹的中心點與大小 (用於 AR 疊圖的縮放與對齊)
    
    輸入: 
        landmarks: MediaPipe 偵測到的 33 個身體節點
        image_shape: 影像的 shape (height, width, channels)
    輸出:
        center: 軀幹中心點的 (x, y) 座標
        size: 軀幹的長度 (用於縮放比例)
    """
    h, w = image_shape[:2]
    
    # 確保肩膀(11, 12)和臀部(23, 24)這四個關鍵點的信心度足夠
    if (landmarks[11].visibility < VISIBILITY_THRESHOLD or 
        landmarks[12].visibility < VISIBILITY_THRESHOLD or
        landmarks[23].visibility < VISIBILITY_THRESHOLD or 
        landmarks[24].visibility < VISIBILITY_THRESHOLD):
        return None, None 

    # 將正規化的座標轉換為實際像素座標
    ls = np.array([landmarks[11].x * w, landmarks[11].y * h])
    rs = np.array([landmarks[12].x * w, landmarks[12].y * h])
    lh = np.array([landmarks[23].x * w, landmarks[23].y * h])
    rh = np.array([landmarks[24].x * w, landmarks[24].y * h])
    
    # 計算軀幹中心點
    center = (ls + rs + lh + rh) / 4
    # 計算兩肩中點與兩臀中點
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    # 軀幹大小 = 兩肩中點到兩臀中點的直線距離
    size = np.linalg.norm(mid_shoulder - mid_hip)
    
    return center, size

def compare_poses_strict(user_landmarks, target_landmarks):
    """
    計算使用者骨架與目標(老師)骨架的相似度分數
    """
    total_score = 0
    missing_parts = 0
    total_parts = len(BODY_PARTS_VECTORS) 
    
    for idx_a, idx_b in BODY_PARTS_VECTORS:
        vis_a = user_landmarks[idx_a].visibility
        vis_b = user_landmarks[idx_b].visibility
        
        # 只有當該部位的兩端點都清晰可見時才計分
        if vis_a > VISIBILITY_THRESHOLD and vis_b > VISIBILITY_THRESHOLD:
            # 計算老師的向量
            t_vec = np.array([target_landmarks[idx_b].x - target_landmarks[idx_a].x,
                              target_landmarks[idx_b].y - target_landmarks[idx_a].y])
            # 計算使用者的向量
            u_vec = np.array([user_landmarks[idx_b].x - user_landmarks[idx_a].x,
                              user_landmarks[idx_b].y - user_landmarks[idx_a].y])
            
            norm_t = np.linalg.norm(t_vec)
            norm_u = np.linalg.norm(u_vec)
            
            if norm_t == 0 or norm_u == 0: 
                score = 0
            else: 
                # 利用 Cosine Similarity 計算向量夾角的相似度
                score = np.dot(t_vec, u_vec) / (norm_t * norm_u)
            
            # 確保分數不會是負的 (完全反向時)
            total_score += max(0, score)
        else:
            missing_parts += 1
            
    # 將總分轉換為百分制 (最高 100 分)
    return (total_score / total_parts) * 100