# core/game_engine.py
import cv2
import mediapipe as mp
import numpy as np
import base64
import json


from core.pose_utils import get_torso_geometry, compare_poses_strict

class GameEngine:
    def __init__(self, video_id: str, frame_id: int):
        self.mp_pose = mp.solutions.pose
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ghost_alpha = 0.3 

        # 讀取 config.json
        with open("static/config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # 檢查是否有這個影片的設定，沒有就給預設值防呆
        if video_id in config_data:
            self.video_path = config_data[video_id]["video_path"]
        else:
            print(f"⚠️ 找不到影片 {video_id} 的設定，使用預設影片")
            self.video_path = "../mp4_home/old.mp4"

        self.target_frame = frame_id
        
        print(f"[{video_id}] 正在準備 AR 素材 (影片: {self.video_path}, 影格: {self.target_frame})...")
        self._pre_process_teacher()

    def _pre_process_teacher(self):
        """
        內部方法：讀取老師的指定影格，擷取骨架、去背遮罩與邊框
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
        success, image = cap.read()
        cap.release()
        
        if not success:
            print("❌ 老師影像讀取失敗，請確認路徑！")
            self.t_landmarks = None
            return

        # 1. 抓取老師骨架
        with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            self.t_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        # 2. 抓取老師去背遮罩
        with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
            results = selfie_seg.process(img_rgb)
            self.t_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

        # 3. 製作霓虹邊框
        edges = cv2.Canny(self.t_mask, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        self.t_neon = np.zeros_like(image)
        self.t_neon[edges_dilated > 0] = [0, 255, 255] # 黃色霓虹

        self.t_img = image
        
        # 4. 預先計算老師軀幹中心與大小
        if self.t_landmarks:
            self.t_center, self.t_size = get_torso_geometry(self.t_landmarks, self.t_img.shape)
        else:
            self.t_center, self.t_size = None, None

    def process_frame(self, user_frame):
        """
        處理使用者的單張影像：計算分數、AR疊圖、畫骨架，並回傳 Base64
        """
        # 鏡像翻轉 (非常重要，不然使用者會左右錯亂)
        frame = cv2.flip(user_frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 進行姿勢辨識
        results = self.pose.process(frame_rgb)
        current_score = 0
        
        if results.pose_landmarks and self.t_landmarks and self.t_center is not None:
            u_landmarks = results.pose_landmarks.landmark
            
            # 1. 計算分數
            current_score = compare_poses_strict(u_landmarks, self.t_landmarks)
            
            # 2. 計算使用者的軀幹並進行 AR 疊圖
            u_center, u_size = get_torso_geometry(u_landmarks, frame.shape)
            
            if u_center is not None and u_size > 0:
                # 計算縮放與平移矩陣
                scale = u_size / self.t_size
                tx = u_center[0] - self.t_center[0] * scale
                ty = u_center[1] - self.t_center[1] * scale
                M = np.float32([[scale, 0, tx], [0, scale, ty]])
                
                # 變形老師的素材以貼合使用者
                warped_t_img = cv2.warpAffine(self.t_img, M, (w, h))
                warped_t_mask = cv2.warpAffine(self.t_mask, M, (w, h))
                warped_t_neon = cv2.warpAffine(self.t_neon, M, (w, h))
                
                mask_3ch = cv2.merge([warped_t_mask, warped_t_mask, warped_t_mask])
                mask_indices = np.where(mask_3ch > 128)
                
                # 疊加半透明老師
                overlay = frame.copy()
                overlay[mask_indices] = warped_t_img[mask_indices]
                cv2.addWeighted(overlay, self.ghost_alpha, frame, 1 - self.ghost_alpha, 0, frame)
                
                # 疊加霓虹邊框
                neon_indices = np.where(warped_t_neon > 0)
                frame[neon_indices] = warped_t_neon[neon_indices]

            # 3. 畫上使用者的骨架 (根據分數變色)
            skel_color = (0, 0, 255) # 預設紅 (BGR)
            if current_score > 60: skel_color = (0, 255, 255) # 黃
            if current_score > 85: skel_color = (0, 255, 0)   # 綠

            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=skel_color, thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=skel_color, thickness=2)
            )

        # 4. 把 OpenCV 的 BGR 圖片轉換成 Base64 格式的 JPG
        # 這樣網頁的 <img> 標籤才能直接顯示
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return current_score, jpg_as_text