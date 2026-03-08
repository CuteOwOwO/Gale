import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def overlay_image(background, overlay, x, y, size):
    """
    在背景圖的 (x, y) 位置疊加指定大小的圖片
    """
    # 縮放疊加圖
    overlay = cv2.resize(overlay, (size, size))
    h, w, _ = overlay.shape
    
    # 確保座標在畫面範圍內
    if y + h > background.shape[0] or x + w > background.shape[1] or x < 0 or y < 0:
        return background

    # 處理透明度 (Alpha Channel)
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] +
                                           (1.0 - alpha) * background[y:y+h, x:x+w, c])
    else:
        background[y:y+h, x:x+w] = overlay[:, :, :3]
        
    return background

# 讀取要疊加的圖片 (例如 pet.png)
pet_img = cv2.imread('pet.png', cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 轉為 RGB 供 MediaPipe 處理
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # 取得右肩座標 (Landmark 12)
        # Landmark 11 為左肩，可依需求更換
        r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 轉換為像素座標
        h, w, _ = frame.shape
        cx, cy = int(r_shoulder.x * w), int(r_shoulder.y * h)
        
        # 設定寵物大小 (根據肩膀寬度動態調整更佳，這裡先給固定值)
        pet_size = 500
        
        # 疊加圖片 (讓圖片中心對準肩膀或稍微偏上)
        frame = overlay_image(frame, pet_img, cx - pet_size // 2, cy - pet_size // 2, pet_size)

    cv2.imshow('Just Pose - Shoulder Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()